# rebalance.py

import os
import json
import yaml
import time
import requests
import yfinance as yf
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open("config/portfolio_config.yaml") as f:
        return yaml.safe_load(f)

def load_current_positions() -> dict:
    f = Path("data/positions/current.json")
    return json.load(open(f)) if f.exists() else {}

def load_universe() -> list:
    f = Path("data/universe/tickers.json")
    return json.load(open(f)) if f.exists() else []

# ── OpenRouter client ─────────────────────────────────────────────────────────

MODELS_BY_TASK = {
    "screening": [
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen-2.5-72b-instruct:free",
        "google/gemma-2-9b-it:free",
    ],
    "scenario": [
        "deepseek/deepseek-r1:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen-2.5-72b-instruct:free",
    ],
    "thesis": [
        "qwen/qwen-2.5-72b-instruct:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek/deepseek-r1:free",
    ],
    "macro": [
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen-2.5-72b-instruct:free",
    ],
}

last_request_time = {}

def call_openrouter(
    prompt: str,
    task: str,
    system: str = "",
    max_tokens: int = 1000,
    temperature: float = 0.1,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY no encontrada")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/portfolio-autopilot",
        "X-Title": "Portfolio Autopilot",
    }

    models = MODELS_BY_TASK.get(task, MODELS_BY_TASK["macro"])

    for model in models:
        try:
            # Rate limiting: mínimo 3s entre requests al mismo modelo
            now = time.time()
            last = last_request_time.get(model, 0)
            wait = 3 - (now - last)
            if wait > 0:
                time.sleep(wait)

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )

            last_request_time[model] = time.time()

            if response.status_code == 429:
                print(f"  Rate limit en {model}, rotando...")
                time.sleep(5)
                continue

            if response.status_code != 200:
                print(f"  Error {response.status_code} en {model}: {response.text[:200]}")
                continue

            content = response.json()["choices"][0]["message"]["content"]
            return content

        except Exception as e:
            print(f"  Excepción en {model}: {e}")
            continue

    raise Exception(f"Todos los modelos fallaron para tarea: {task}")


def extract_json(text: str) -> dict:
    """Extrae JSON aunque el modelo añada texto alrededor"""
    # Intentar parsear directamente
    try:
        return json.loads(text)
    except Exception:
        pass

    # Buscar bloque ```json ... ```
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        try:
            return json.loads(text[start:end].strip())
        except Exception:
            pass

    # Buscar primer { hasta último }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass

    raise Exception(f"No se encontró JSON válido en:\n{text[:500]}")

# ── Macro ─────────────────────────────────────────────────────────────────────

def get_macro_context() -> str:
    print("  Obteniendo datos de mercado...")

    market_data = f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d')}\n"

    try:
        proxies = {
            "SPY":     ("S&P 500",        "SPY"),
            "VIX":     ("Volatilidad",    "^VIX"),
            "TNX":     ("10Y Yield",      "^TNX"),
            "QQQ":     ("Nasdaq",         "QQQ"),
        }

        for label, (name, ticker) in proxies.items():
            try:
                hist = yf.Ticker(ticker).history(period="5d")
                if not hist.empty:
                    price = hist["Close"].iloc[-1]
                    ret_5d = (price / hist["Close"].iloc[0] - 1) * 100
                    market_data += f"- {name} ({ticker}): {price:.2f} ({ret_5d:+.1f}% 5d)\n"
            except Exception:
                pass

    except Exception as e:
        print(f"  Warning: error obteniendo datos de mercado: {e}")

    prompt = f"""
{market_data}

Describe el contexto macro actual para un portfolio long-only equity 
en exactamente 150 palabras. Incluye:
1. Fed stance y expectativas de tipos (cuantificado)
2. Estado del ciclo económico
3. Risk appetite del mercado (referencia VIX)
4. Sectores con viento de cola vs headwind
5. Top 2 riesgos macro próximos 3 meses

Sé específico y cuantitativo. Sin frases genéricas.
"""

    return call_openrouter(prompt, task="macro", max_tokens=400)


# ── Scoring ───────────────────────────────────────────────────────────────────

def get_stock_data(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker":         ticker,
            "price":          info.get("currentPrice") or info.get("regularMarketPrice"),
            "forward_pe":     info.get("forwardPE"),
            "trailing_pe":    info.get("trailingPE"),
            "revenue_growth": info.get("revenueGrowth"),
            "gross_margins":  info.get("grossMargins"),
            "free_cashflow":  info.get("freeCashflow"),
            "market_cap":     info.get("marketCap"),
            "52w_high":       info.get("fiftyTwoWeekHigh"),
            "52w_low":        info.get("fiftyTwoWeekLow"),
            "sector":         info.get("sector"),
            "industry":       info.get("industry"),
            "description":    (info.get("longBusinessSummary") or "")[:300],
        }
    except Exception as e:
        print(f"    Error datos {ticker}: {e}")
        return {"ticker": ticker, "price": None}


def score_stock(ticker: str, macro_context: str) -> dict:
    data = get_stock_data(ticker)

    if not data.get("price"):
        return {"ticker": ticker, "composite_score": 0, "data_snapshot": data}

    prompt = f"""
Analiza este stock y puntúalo en DOS dimensiones de 0 a 100.

CONTEXTO MACRO:
{macro_context[:300]}

DATOS:
{json.dumps({k: v for k, v in data.items() if k != 'description'}, indent=2)}

DESCRIPCIÓN: {data.get('description', '')}

DIMENSIÓN 1 - FUNDAMENTAL SCORE (0-100):
Calidad del negocio, márgenes, FCF, moat, balance.

DIMENSIÓN 2 - FORWARD SETUP SCORE (0-100):
Valoración vs histórico propio, catalizadores próximos,
posicionamiento mercado, asimetría riesgo/recompensa actual.

Responde SOLO en JSON:
{{
    "fundamental_score": <0-100>,
    "forward_setup_score": <0-100>,
    "fundamental_rationale": "<2 frases>",
    "setup_rationale": "<2 frases>",
    "key_risk": "<1 frase>",
    "key_catalyst": "<1 frase>"
}}
"""

    try:
        result = extract_json(
            call_openrouter(prompt, task="screening", max_tokens=500)
        )

        composite = (
            result.get("fundamental_score", 0) * 0.40
            + result.get("forward_setup_score", 0) * 0.60
        )

        return {
            "ticker":             ticker,
            "fundamental_score":  result.get("fundamental_score", 0),
            "forward_setup_score":result.get("forward_setup_score", 0),
            "composite_score":    composite,
            "data_snapshot":      {**data, **result},
        }

    except Exception as e:
        print(f"    Error scoring {ticker}: {e}")
        return {"ticker": ticker, "composite_score": 0, "data_snapshot": data}


def score_universe(tickers: list, macro_context: str, top_n: int = 40) -> list:
    scores = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{total}] {ticker}...")
        score = score_stock(ticker, macro_context)
        scores.append(score)
        time.sleep(1)   # Respetar rate limits

    scores.sort(key=lambda x: x["composite_score"], reverse=True)
    return scores[:top_n]


# ── Scenarios ─────────────────────────────────────────────────────────────────

def build_scenario(ticker: str, stock_data: dict, macro_context: str) -> dict:
    price = stock_data.get("price") or 0

    prompt = f"""
Construye un análisis de escenarios completo para {ticker}.
Precio actual: ${price:.2f}

DATOS:
{json.dumps({k: v for k, v in stock_data.items() 
             if k not in ['description', 'fundamental_rationale', 
                          'setup_rationale']}, indent=2)}

MACRO:
{macro_context[:300]}

Construye 3 escenarios con price targets en 4 horizontes.
Las probabilidades DEBEN sumar exactamente 1.0.
El kill_condition debe ser CONCRETO y VERIFICABLE (evento específico,
no una frase genérica).

Responde SOLO en JSON:
{{
    "prob_bull": <0.0-1.0>,
    "prob_base": <0.0-1.0>,
    "prob_bear": <0.0-1.0>,
    "targets_1m":  {{"bull": <price>, "base": <price>, "bear": <price>}},
    "targets_3m":  {{"bull": <price>, "base": <price>, "bear": <price>}},
    "targets_6m":  {{"bull": <price>, "base": <price>, "bear": <price>}},
    "targets_12m": {{"bull": <price>, "base": <price>, "bear": <price>}},
    "bull_thesis": "<3-4 frases con drivers específicos>",
    "base_thesis": "<3-4 frases con drivers específicos>",
    "bear_thesis": "<3-4 frases con drivers específicos>",
    "kill_condition": "<evento concreto y verificable>",
    "key_catalyst": "<próximo catalizador con fecha si existe>"
}}
"""

    try:
        r = extract_json(
            call_openrouter(prompt, task="scenario", max_tokens=1000)
        )

        # Normalizar probabilidades
        total_prob = r["prob_bull"] + r["prob_base"] + r["prob_bear"]
        if abs(total_prob - 1.0) > 0.01:
            r["prob_bull"] /= total_prob
            r["prob_base"] /= total_prob
            r["prob_bear"] /= total_prob

        # Calcular expected values
        def ev(targets):
            return (
                r["prob_bull"] * targets["bull"]
                + r["prob_base"] * targets["base"]
                + r["prob_bear"] * targets["bear"]
            )

        ev_12m = ev(r["targets_12m"])
        bear_12m = r["targets_12m"]["bear"]
        bear_downside = (bear_12m - price) / price if price else 0
        weighted_upside = (ev_12m - price) / price if price else 0
        ud_ratio = (
            abs(weighted_upside / bear_downside)
            if bear_downside != 0 else 0
        )

        return {
            "ticker":                  ticker,
            "current_price":           price,
            "prob_bull":               r["prob_bull"],
            "prob_base":               r["prob_base"],
            "prob_bear":               r["prob_bear"],
            "targets_1m":              r["targets_1m"],
            "targets_3m":              r["targets_3m"],
            "targets_6m":              r["targets_6m"],
            "targets_12m":             r["targets_12m"],
            "ev_1m":                   ev(r["targets_1m"]),
            "ev_3m":                   ev(r["targets_3m"]),
            "ev_6m":                   ev(r["targets_6m"]),
            "ev_12m":                  ev_12m,
            "bear_case_downside_12m":  bear_downside,
            "upside_downside_ratio":   ud_ratio,
            "bull_thesis":             r["bull_thesis"],
            "base_thesis":             r["base_thesis"],
            "bear_thesis":             r["bear_thesis"],
            "kill_condition":          r["kill_condition"],
            "key_catalyst":            r["key_catalyst"],
        }

    except Exception as e:
        print(f"    Error scenario {ticker}: {e}")
        return {
            "ticker":                 ticker,
            "current_price":          price,
            "ev_12m":                 price,
            "bear_case_downside_12m": -0.30,
            "upside_downside_ratio":  0,
            "kill_condition":         "Error en generación",
            "key_catalyst":           "N/A",
        }


# ── Optimizer ─────────────────────────────────────────────────────────────────

def optimize_portfolio(
    scenarios: list,
    current_weights: dict,
    config: dict,
) -> dict:
    import numpy as np
    from scipy.optimize import minimize

    max_pos   = config["portfolio"]["max_positions"]
    max_w     = config["portfolio"]["max_position_size"]
    min_w     = config["portfolio"]["min_position_size"]
    max_to    = config["turnover"]["max_one_sided_turnover"]
    min_chg   = config["turnover"]["min_position_change"]

    # Solo candidatos con EV positivo y ratio > 0
    candidates = [
        s for s in scenarios
        if s.get("ev_12m", 0) > s.get("current_price", 0)
        and s.get("upside_downside_ratio", 0) > 0
    ]
    candidates.sort(key=lambda s: s["upside_downside_ratio"], reverse=True)
    candidates = candidates[: max_pos * 2]

    if not candidates:
        print("  Warning: sin candidatos válidos, manteniendo posiciones")
        return {
            "weights":         current_weights,
            "expected_return": 0,
            "risk_score":      0,
            "risk_adjusted_return": 0,
            "turnover_used":   0,
            "added_names":     [],
            "dropped_names":   [],
        }

    tickers = [s["ticker"] for s in candidates]
    n = len(tickers)

    ev_returns = np.array([
        (s["ev_12m"] - s["current_price"]) / s["current_price"]
        if s["current_price"] > 0 else 0
        for s in candidates
    ])

    bear_downs = np.array([
        abs(s["bear_case_downside_12m"]) for s in candidates
    ])

    current_w = np.array([current_weights.get(t, 0.0) for t in tickers])

    def objective(w):
        port_ev   = np.dot(w, ev_returns)
        port_risk = np.dot(w, bear_downs)
        return -(port_ev / (port_risk + 0.001))

    constraints = [
        {"type": "eq",  "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq","fun": lambda w: max_to - np.sum(np.maximum(w - current_w, 0))},
    ]
    bounds = [(0, max_w)] * n

    x0 = np.zeros(n)
    x0[:min(max_pos, n)] = 1.0 / min(max_pos, n)

    result = minimize(
        objective, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    w_opt = result.x.copy()
    w_opt[w_opt < min_w] = 0
    if w_opt.sum() > 0:
        w_opt /= w_opt.sum()

    final_weights = {
        tickers[i]: float(w_opt[i])
        for i in range(n)
        if w_opt[i] >= min_w
    }

    # Métricas
    scen_map = {s["ticker"]: s for s in candidates}
    port_ev = sum(
        w * (scen_map[t]["ev_12m"] - scen_map[t]["current_price"])
        / scen_map[t]["current_price"]
        for t, w in final_weights.items()
        if scen_map.get(t) and scen_map[t]["current_price"] > 0
    )
    port_risk = sum(
        w * abs(scen_map[t]["bear_case_downside_12m"])
        for t, w in final_weights.items()
        if scen_map.get(t)
    )

    all_tickers = set(list(final_weights.keys()) + list(current_weights.keys()))
    turnover = sum(
        max(final_weights.get(t, 0) - current_weights.get(t, 0), 0)
        for t in all_tickers
    )

    return {
        "weights":              final_weights,
        "expected_return":      port_ev,
        "risk_score":           port_risk,
        "risk_adjusted_return": port_ev / (port_risk + 0.001),
        "turnover_used":        turnover,
        "added_names":   [t for t in final_weights if t not in current_weights],
        "dropped_names": [t for t in current_weights if t not in final_weights],
    }


# ── Thesis ────────────────────────────────────────────────────────────────────

def generate_thesis(
    ticker: str,
    scenario: dict,
    weight: float,
    action: str,
    macro_summary: str,
) -> dict:
    price    = scenario.get("current_price", 0)
    ev_12m   = scenario.get("ev_12m", 0)
    bear_12m = scenario.get("targets_12m", {}).get("bear", 0)

    ev_pct   = (ev_12m - price) / price * 100 if price else 0
    bear_pct = (bear_12m - price) / price * 100 if price else 0
    ratio    = scenario.get("upside_downside_ratio", 0)

    prompt = f"""
Genera la thesis de posición para {ticker}.

PARÁMETROS:
- Acción: {action}
- Peso: {weight:.1%}
- Precio: ${price:.2f}
- Bull ({scenario.get('prob_bull',0):.0%}): ${scenario.get('targets_12m',{}).get('bull',0):.2f}
- Base ({scenario.get('prob_base',0):.0%}): ${scenario.get('targets_12m',{}).get('base',0):.2f}
- Bear ({scenario.get('prob_bear',0):.0%}): ${bear_12m:.2f}
- EV 12M: ${ev_12m:.2f} ({ev_pct:+.1f}%)
- Bear downside: {bear_pct:.1f}%
- U/D ratio: {ratio:.2f}x

BULL: {scenario.get('bull_thesis','')}
BASE: {scenario.get('base_thesis','')}
BEAR: {scenario.get('bear_thesis','')}
KILL: {scenario.get('kill_condition','')}
CATALIZADOR: {scenario.get('key_catalyst','')}

MACRO: {macro_summary[:200]}

Genera la thesis en este formato EXACTO:

---
THESIS: {ticker} | {action} | {weight:.1%} | {datetime.now().isoformat()}
---

**Setup:** [por qué existe la oportunidad ahora]

**Bull case ({scenario.get('prob_bull',0):.0%}):** [drivers. Target ${scenario.get('targets_12m',{}).get('bull',0):.2f}]

**Base case ({scenario.get('prob_base',0):.0%}):** [execution. Target ${scenario.get('targets_12m',{}).get('base',0):.2f}]

**Bear case ({scenario.get('prob_bear',0):.0%}):** [riesgos. Target ${bear_12m:.2f}]

**Expected value:** ${ev_12m:.2f} ({ev_pct:+.1f}%) vs bear {bear_pct:.1f}%. Ratio {ratio:.2f}x.

**Sizing:** [por qué {weight:.1%} y no más o menos]

**Kill condition:** {scenario.get('kill_condition','')}

**Next checkpoint:** [cuándo y qué mirar]
---
"""

    timestamp = datetime.now().isoformat()

    try:
        thesis_text = call_openrouter(
            prompt, task="thesis", max_tokens=1500
        )
    except Exception as e:
        thesis_text = f"Error generando thesis: {e}"

    thesis = {
        "ticker":               ticker,
        "action":               action,
        "weight":               weight,
        "timestamp":            timestamp,
        "price_at_thesis":      price,
        "ev_12m":               ev_12m,
        "expected_return_pct":  ev_pct,
        "bear_downside_pct":    bear_pct,
        "upside_downside_ratio":ratio,
        "kill_condition":       scenario.get("kill_condition", ""),
        "key_catalyst":         scenario.get("key_catalyst", ""),
        "thesis_text":          thesis_text,
        "macro_snapshot":       macro_summary[:300],
    }

    # Guardar
    Path("data/thesis").mkdir(parents=True, exist_ok=True)
    filename = f"{timestamp[:10]}_{ticker}_{action}.json"
    with open(f"data/thesis/{filename}", "w") as f:
        json.dump(thesis, f, indent=2, ensure_ascii=False)

    return thesis


def generate_rebalance_summary(
    result: dict,
    all_thesis: list,
    macro_summary: str,
) -> str:
    portfolio_str = "\n".join(
        f"  {t}: {w:.1%}"
        for t, w in sorted(result["weights"].items(), key=lambda x: -x[1])
    )

    prompt = f"""
Genera el commentary de rebalanceo semanal del portfolio.
Estilo: directo, cuantitativo, primera persona, sin bullshit.
Máximo 350 palabras.

PORTFOLIO:
{portfolio_str}

CAMBIOS:
- Añadidos:  {', '.join(result['added_names'])  or 'Ninguno'}
- Eliminados:{', '.join(result['dropped_names']) or 'Ninguno'}
- Turnover:  {result['turnover_used']:.1%} de 30% máximo

MÉTRICAS:
- EV 12M:         {result['expected_return']:.1%}
- Risk score:     {result['risk_score']:.1%}
- Risk-adj ratio: {result['risk_adjusted_return']:.2f}x

MACRO:
{macro_summary[:300]}

ESTRUCTURA:
1. Qué cambió en macro y cómo afecta al book
2. Cambios en portfolio y por qué
3. Posiciones que se mantienen y por qué no se tocaron
4. Métricas resultantes
5. Qué vigilar hasta el próximo rebalanceo

Termina con: "Not advice, just how I'm sizing my own book."
"""

    try:
        return call_openrouter(prompt, task="thesis", max_tokens=800)
    except Exception as e:
        return f"Error generando summary: {e}"


# ── Email report ──────────────────────────────────────────────────────────────

def generate_email_report(
    result: dict,
    all_thesis: list,
    summary: str,
    positions: dict,
) -> None:
    today    = datetime.now().strftime("%Y-%m-%d")
    added    = result["added_names"]
    dropped  = result["dropped_names"]

    changes_str = ""
    if added:   changes_str += f"+{','.join(added)}"
    if dropped: changes_str += f" -{','.join(dropped)}"
    if not changes_str: changes_str = "sin cambios"

    subject = (
        f"📊 Portfolio {today} | "
        f"EV {result['expected_return']:.1%} | {changes_str}"
    )

    # Portfolio rows
    rows = ""
    for ticker, w in sorted(result["weights"].items(), key=lambda x: -x[1]):
        pos = positions.get(ticker, {})
        ev  = pos.get("ev_12m")
        rows += f"""
        <tr style="border-bottom:1px solid #eee;">
            <td style="padding:8px;"><strong>{ticker}</strong></td>
            <td style="padding:8px;">{w:.1%}</td>
            <td style="padding:8px;">${pos.get('entry_price') or 0:.2f}</td>
            <td style="padding:8px;">{"$"+f"{ev:.2f}" if ev else "-"}</td>
        </tr>"""

    # Thesis del día
    thesis_html = ""
    for t in all_thesis:
        ev_pct = t.get("expected_return_pct", 0)
        color  = "#28a745" if ev_pct > 0 else "#dc3545"
        thesis_html += f"""
        <div style="border:1px solid #ddd;padding:15px;
                    margin:10px 0;border-radius:5px;">
            <h3 style="margin:0 0 8px 0;">
                {t['ticker']} | {t['action']} | {t['weight']:.1%}
            </h3>
            <p>
                EV 12M: <strong style="color:{color}">
                    {ev_pct:+.1f}%
                </strong> &nbsp;|&nbsp;
                Bear: {t.get('bear_downside_pct',0):.1f}% &nbsp;|&nbsp;
                U/D: {t.get('upside_downside_ratio',0):.2f}x
            </p>
            <p style="background:#fff3cd;padding:10px;border-radius:3px;">
                <strong>Kill:</strong> {t.get('kill_condition','N/A')}
            </p>
            <div style="white-space:pre-wrap;font-family:Georgia,serif;
                        line-height:1.6;font-size:14px;">
{t.get('thesis_text','N/A')}
            </div>
        </div>"""

    body = f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;max-width:800px;margin:0 auto;padding:20px;">

<h1 style="color:#1a1a2e;">📊 Portfolio Rebalance — {today}</h1>

<div style="display:flex;gap:15px;margin:20px 0;flex-wrap:wrap;">
    <div style="background:#f8f9fa;padding:15px;border-radius:8px;
                flex:1;min-width:120px;text-align:center;">
        <div style="font-size:22px;font-weight:bold;color:#28a745;">
            {result['expected_return']:.1%}
        </div>
        <div style="color:#666;font-size:13px;">EV 12M</div>
    </div>
    <div style="background:#f8f9fa;padding:15px;border-radius:8px;
                flex:1;min-width:120px;text-align:center;">
        <div style="font-size:22px;font-weight:bold;">
            {result['risk_adjusted_return']:.2f}x
        </div>
        <div style="color:#666;font-size:13px;">Risk-Adj</div>
    </div>
    <div style="background:#f8f9fa;padding:15px;border-radius:8px;
                flex:1;min-width:120px;text-align:center;">
        <div style="font-size:22px;font-weight:bold;color:#fd7e14;">
            {result['turnover_used']:.1%}
        </div>
        <div style="color:#666;font-size:13px;">Turnover</div>
    </div>
    <div style="background:#f8f9fa;padding:15px;border-radius:8px;
                flex:1;min-width:120px;text-align:center;">
        <div style="font-size:22px;font-weight:bold;">
            {len(result['weights'])}
        </div>
        <div style="color:#666;font-size:13px;">Posiciones</div>
    </div>
</div>

<h2>📝 Commentary</h2>
<div style="background:#f8f9fa;padding:20px;border-radius:8px;
            white-space:pre-wrap;font-family:Georgia,serif;line-height:1.8;">
{summary}
</div>

<h2>📈 Portfolio actual</h2>
<table style="width:100%;border-collapse:collapse;">
    <thead>
        <tr style="background:#1a1a2e;color:white;">
            <th style="padding:10px;text-align:left;">Ticker</th>
            <th style="padding:10px;text-align:left;">Weight</th>
            <th style="padding:10px;text-align:left;">Entry</th>
            <th style="padding:10px;text-align:left;">EV 12M</th>
        </tr>
    </thead>
    <tbody>{rows}</tbody>
</table>

{"<h2>🎯 Thesis generadas</h2>" + thesis_html if thesis_html else ""}

<hr style="margin:30px 0;">
<p style="color:#999;font-size:12px;">
    Not advice, just how I'm sizing my own book.
</p>

</body>
</html>"""

    report = {"subject": subject, "body": body}
    with open("data/email_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("✓ Email report guardado en data/email_report.json")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(result: dict, scenarios: dict) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    ts    = datetime.now().isoformat()

    positions = {}
    for ticker, weight in result["weights"].items():
        s = scenarios.get(ticker, {})
        positions[ticker] = {
            "weight":         weight,
            "entry_date":     today,
            "entry_price":    s.get("current_price"),
            "ev_12m":         s.get("ev_12m"),
            "kill_condition": s.get("kill_condition"),
        }

    Path("data/positions").mkdir(parents=True, exist_ok=True)
    with open("data/positions/current.json", "w") as f:
        json.dump(positions, f, indent=2)

    rebalance = {
        "timestamp": ts,
        "portfolio": result["weights"],
        "changes": {
            "added":    result["added_names"],
            "dropped":  result["dropped_names"],
            "turnover": result["turnover_used"],
        },
        "metrics": {
            "expected_return":      result["expected_return"],
            "risk_score":           result["risk_score"],
            "risk_adjusted_return": result["risk_adjusted_return"],
        },
    }

    Path("data/rebalances").mkdir(parents=True, exist_ok=True)
    with open(f"data/rebalances/{today}_rebalance.json", "w") as f:
        json.dump(rebalance, f, indent=2)

    print("✓ Resultados guardados")
    return positions


# ── Main ──────────────────────────────────────────────────────────────────────

def run_rebalance():
    print(f"\n{'='*60}")
    print(f"PORTFOLIO REBALANCE — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")

    config           = load_config()
    current_positions = load_current_positions()
    universe         = load_universe()

    print(f"Universe:   {len(universe)} tickers")
    print(f"Posiciones: {len(current_positions)} nombres\n")

    # 1. Macro
    print("📊 Macro context...")
    macro = get_macro_context()
    print(f"✓ Macro listo\n")

    # 2. Scoring
    print("🔍 Scoring universe...")
    scores = score_universe(universe, macro, top_n=40)
    if scores:
        print(f"✓ Top: {scores[0]['ticker']} ({scores[0]['composite_score']:.1f})\n")

    # 3. Scenarios
    print("📐 Building scenarios...")
    scenarios = {}
    for i, score in enumerate(scores):
        ticker = score["ticker"]
        print(f"  [{i+1}/{len(scores)}] {ticker}...")
        scenario = build_scenario(ticker, score["data_snapshot"], macro)
        scenarios[ticker] = scenario
        time.sleep(1.5)
    print(f"✓ {len(scenarios)} scenarios\n")

    # 4. Optimize
    print("⚙️  Optimizing...")
    current_weights = {t: p["weight"] for t, p in current_positions.items()}
    result = optimize_portfolio(list(scenarios.values()), current_weights, config)

    print(f"✓ {len(result['weights'])} nombres | "
          f"Turnover {result['turnover_used']:.1%} | "
          f"EV {result['expected_return']:.1%}\n")

    # 5. Thesis
    print("📝 Generating thesis...")
    all_thesis = []

    for ticker, weight in result["weights"].items():
        old_w = current_weights.get(ticker, 0)
        diff  = weight - old_w

        if ticker in result["added_names"]:
            action = "OPEN"
        elif abs(diff) >= config["turnover"]["min_position_change"]:
            action = "ADD" if diff > 0 else "TRIM"
        else:
            action = "HOLD"

        if action != "HOLD" and ticker in scenarios:
            thesis = generate_thesis(
                ticker, scenarios[ticker], weight, action, macro
            )
            all_thesis.append(thesis)
            print(f"  ✓ {ticker} [{action}]")
            time.sleep(2)

    for ticker in result["dropped_names"]:
        if ticker in scenarios:
            thesis = generate_thesis(
                ticker, scenarios[ticker], 0.0, "CLOSE", macro
            )
            all_thesis.append(thesis)
            print(f"  ✓ {ticker} [CLOSE]")
            time.sleep(2)

    # 6. Summary
    print("\n📣 Commentary...")
    summary = generate_rebalance_summary(result, all_thesis, macro)
    result["commentary"] = summary

    # Guardar commentary en el rebalance JSON
    today = datetime.now().strftime("%Y-%m-%d")
    rb_file = Path(f"data/rebalances/{today}_rebalance.json")
    if rb_file.exists():
        rb = json.load(open(rb_file))
        rb["commentary"] = summary
        with open(rb_file, "w") as f:
            json.dump(rb, f, indent=2)

    # 7. Guardar posiciones
    positions = save_results(result, scenarios)

    # 8. Email report
    generate_email_report(result, all_thesis, summary, positions)

    print(f"\n{'='*60}")
    print("✅ Rebalance completo")
    print(f"{'='*60}\n")
    print(summary)

    return result


if __name__ == "__main__":
    run_rebalance()
