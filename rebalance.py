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

# ── Configuración de rate limiting ────────────────────────────────────────────

# Segundos mínimos entre cualquier request a OpenRouter
# Con tier gratuito: 20 req/min = 1 req cada 3s mínimo
# Usamos 4s para ir seguros
MIN_SECONDS_BETWEEN_REQUESTS = 4

# Segundos a esperar cuando hay rate limit global
RATE_LIMIT_COOLDOWN = 30

# Máximo de reintentos totales antes de rendirse en una tarea
MAX_TOTAL_ATTEMPTS = 5

last_any_request_time = 0   # Tiempo del último request a cualquier modelo
last_request_time     = {}  # Por modelo
request_counts        = {}
FREE_MODELS_CACHE     = []

EXCLUDE_KEYWORDS = [
    "audio", "image", "vision", "clip", "embed", "lyria",
    "dall", "whisper", "tts", "ocr", "stable", "midjourney",
    "flux", "video", "music", "speech", "rerank", "diffusion",
]

PREFERRED_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-70b-instruct:free",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "mistralai/mistral-small:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "google/gemma-3-12b-it:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
]


def is_text_model(model: dict) -> bool:
    model_id   = model.get("id", "").lower()
    model_name = model.get("name", "").lower()

    for kw in EXCLUDE_KEYWORDS:
        if kw in model_id or kw in model_name:
            return False

    if not model.get("context_length"):
        return False

    return True


def fetch_free_models() -> list:
    api_key = os.getenv("OPENROUTER_API_KEY")
    print("  Consultando modelos en OpenRouter...")

    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}")

        all_models = response.json().get("data", [])
        free_text  = []

        for m in all_models:
            model_id = m.get("id", "")
            pricing  = m.get("pricing", {})

            is_free = (
                ":free" in model_id
                or str(pricing.get("prompt", "1")) == "0"
                or pricing.get("prompt") == 0
            )
            if not is_free:
                continue
            if not is_text_model(m):
                continue

            free_text.append({
                "id":             model_id,
                "context_length": m.get("context_length", 4096),
            })

        # Preferidos primero, luego por contexto
        def sort_key(m):
            if m["id"] in PREFERRED_MODELS:
                return (0, PREFERRED_MODELS.index(m["id"]))
            return (1, -m["context_length"])

        free_text.sort(key=sort_key)
        ids = [m["id"] for m in free_text]

        print(f"  ✓ {len(ids)} modelos de texto gratuitos:")
        for m in free_text:
            tag = "★" if m["id"] in PREFERRED_MODELS else "·"
            print(f"    {tag} {m['id']}")

        return ids

    except Exception as e:
        print(f"  ⚠ Error: {e}. Usando fallback.")
        return [
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "google/gemma-2-9b-it:free",
        ]


def global_rate_limit_wait():
    """
    Espera global entre requests independientemente del modelo.
    Evita el rate limit compartido entre todos los modelos :free.
    """
    global last_any_request_time
    now  = time.time()
    wait = MIN_SECONDS_BETWEEN_REQUESTS - (now - last_any_request_time)
    if wait > 0:
        time.sleep(wait)


def call_openrouter(
    prompt: str,
    task: str,
    system: str = "",
    max_tokens: int = 1000,
    temperature: float = 0.1,
) -> str:
    global last_any_request_time

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY no encontrada")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/portfolio-autopilot",
        "X-Title":      "Portfolio Autopilot",
    }

    models      = FREE_MODELS_CACHE or ["meta-llama/llama-3.1-8b-instruct:free"]
    last_error  = ""
    attempts    = 0

    for model in models:
        if attempts >= MAX_TOTAL_ATTEMPTS:
            break

        try:
            # Rate limit global (entre todos los modelos)
            global_rate_limit_wait()

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model":       model,
                "messages":    messages,
                "max_tokens":  max_tokens,
                "temperature": temperature,
            }

            print(f"    → {model}...")

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=90,
            )

            last_any_request_time        = time.time()
            last_request_time[model]     = time.time()
            request_counts[model]        = request_counts.get(model, 0) + 1
            attempts                    += 1

            # ── Rate limit ───────────────────────────────────
            if response.status_code == 429:
                retry_after = int(
                    response.headers.get("Retry-After", RATE_LIMIT_COOLDOWN)
                )
                wait = max(retry_after, RATE_LIMIT_COOLDOWN)
                print(f"    Rate limit global, enfriando {wait}s...")
                time.sleep(wait)
                # Intentar con el siguiente modelo tras el cooldown
                continue

            # ── Modelo no disponible → siguiente ─────────────
            if response.status_code == 404:
                print(f"    No disponible, rotando...")
                last_error = "404"
                continue

            # ── Error servidor → siguiente ────────────────────
            if response.status_code in (502, 503, 504):
                print(f"    Error {response.status_code}, rotando...")
                last_error = str(response.status_code)
                time.sleep(5)
                continue

            # ── Otro error → siguiente ────────────────────────
            if response.status_code != 200:
                print(f"    Error {response.status_code}, rotando...")
                last_error = response.text[:100]
                continue

            # ── Éxito ─────────────────────────────────────────
            content = response.json()["choices"][0]["message"]["content"]
            if not content or not content.strip():
                print(f"    Respuesta vacía, rotando...")
                continue

            print(f"    ✓ OK ({model.split('/')[-1]})")
            return content.strip()

        except requests.exceptions.Timeout:
            print(f"    Timeout, rotando...")
            last_error = "timeout"
            attempts  += 1
            continue

        except Exception as e:
            print(f"    Error: {e}")
            last_error = str(e)
            attempts  += 1
            continue

    raise Exception(
        f"Sin respuesta válida para '{task}' "
        f"tras {attempts} intentos. "
        f"Último error: {last_error}"
    )


def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass

    for marker in ["```json", "```"]:
        if marker in text:
            start = text.find(marker) + len(marker)
            end   = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except Exception:
                    pass

    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass

    raise Exception(f"JSON no encontrado en:\n{text[:300]}")


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


# ── Macro ─────────────────────────────────────────────────────────────────────

def get_macro_context() -> str:
    print("  Obteniendo datos de mercado...")

    market_data = f"Fecha: {datetime.now().strftime('%Y-%m-%d')}\n"

    for label, ticker in {
        "SPY": "^GSPC", "VIX": "^VIX",
        "TNX": "^TNX",  "QQQ": "QQQ",
    }.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                price  = hist["Close"].iloc[-1]
                ret_5d = (price / hist["Close"].iloc[0] - 1) * 100
                market_data += (
                    f"- {label}: {price:.2f} ({ret_5d:+.1f}% 5d)\n"
                )
        except Exception:
            pass

    prompt = f"""
{market_data}

Describe el contexto macro para un portfolio long-only equity
en 120 palabras máximo. Incluye:
1. Fed stance y tipos (cuantificado)
2. Ciclo económico
3. Risk appetite (VIX)
4. Sectores con viento de cola vs headwind
5. Top 2 riesgos próximos 3 meses

Específico y cuantitativo.
"""
    return call_openrouter(prompt, task="macro", max_tokens=300)


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
            "description":    (info.get("longBusinessSummary") or "")[:200],
        }
    except Exception as e:
        print(f"    Error datos {ticker}: {e}")
        return {"ticker": ticker, "price": None}


def score_stock(ticker: str, macro_context: str) -> dict:
    data = get_stock_data(ticker)

    if not data.get("price"):
        return {"ticker": ticker, "composite_score": 0, "data_snapshot": data}

    # Prompt compacto para consumir menos tokens
    prompt = f"""
Puntúa {ticker} de 0-100 en dos dimensiones.

MACRO (resumen): {macro_context[:150]}

DATOS: precio=${data['price']}, fwd_PE={data['forward_pe']},
revenue_growth={data['revenue_growth']}, margins={data['gross_margins']},
sector={data['sector']}, 52w_low={data['52w_low']}, 52w_high={data['52w_high']}

DESCRIPCIÓN: {data.get('description','')[:150]}

JSON sin texto adicional:
{{
  "fundamental_score": <0-100>,
  "forward_setup_score": <0-100>,
  "key_risk": "<1 frase corta>",
  "key_catalyst": "<1 frase corta>"
}}
"""

    try:
        result    = extract_json(
            call_openrouter(prompt, task="screening", max_tokens=200)
        )
        composite = (
            result.get("fundamental_score", 0) * 0.40
            + result.get("forward_setup_score", 0) * 0.60
        )
        return {
            "ticker":              ticker,
            "fundamental_score":   result.get("fundamental_score", 0),
            "forward_setup_score": result.get("forward_setup_score", 0),
            "composite_score":     composite,
            "data_snapshot":       {**data, **result},
        }
    except Exception as e:
        print(f"    Error scoring {ticker}: {e}")
        return {"ticker": ticker, "composite_score": 0, "data_snapshot": data}


def score_universe(tickers: list, macro_context: str, top_n: int = 20) -> list:
    scores = []
    total  = len(tickers)

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{total}] {ticker}...")
        score = score_stock(ticker, macro_context)
        scores.append(score)
        # Pausa extra cada 10 stocks para dejar enfriar el rate limit
        if (i + 1) % 10 == 0:
            print("  ⏸ Pausa anti-rate-limit (15s)...")
            time.sleep(15)

    valid = [s for s in scores if s["composite_score"] > 0]
    valid.sort(key=lambda x: x["composite_score"], reverse=True)
    print(f"  ✓ {len(valid)} válidos de {total}")
    return valid[:top_n]


# ── Scenarios ─────────────────────────────────────────────────────────────────

def build_scenario(ticker: str, stock_data: dict, macro_context: str) -> dict:
    price = stock_data.get("price") or 0

    prompt = f"""
Escenarios para {ticker} a precio ${price:.2f}.

Datos clave: PE={stock_data.get('forward_pe')},
growth={stock_data.get('revenue_growth')},
margins={stock_data.get('gross_margins')},
52w_range=[{stock_data.get('52w_low')}, {stock_data.get('52w_high')}]

Macro: {macro_context[:150]}

JSON sin texto adicional (probabilidades suman 1.0):
{{
  "prob_bull": <float>,
  "prob_base": <float>,
  "prob_bear": <float>,
  "targets_1m":  {{"bull": <price>, "base": <price>, "bear": <price>}},
  "targets_3m":  {{"bull": <price>, "base": <price>, "bear": <price>}},
  "targets_6m":  {{"bull": <price>, "base": <price>, "bear": <price>}},
  "targets_12m": {{"bull": <price>, "base": <price>, "bear": <price>}},
  "bull_thesis": "<2 frases>",
  "base_thesis": "<2 frases>",
  "bear_thesis": "<2 frases>",
  "kill_condition": "<evento concreto verificable>",
  "key_catalyst": "<catalizador con fecha si existe>"
}}
"""

    try:
        r = extract_json(
            call_openrouter(prompt, task="scenario", max_tokens=600)
        )

        total_p = r["prob_bull"] + r["prob_base"] + r["prob_bear"]
        if abs(total_p - 1.0) > 0.01:
            r["prob_bull"] /= total_p
            r["prob_base"] /= total_p
            r["prob_bear"] /= total_p

        def ev(t):
            return (
                r["prob_bull"] * t["bull"]
                + r["prob_base"] * t["base"]
                + r["prob_bear"] * t["bear"]
            )

        ev_12m   = ev(r["targets_12m"])
        bear_12m = r["targets_12m"]["bear"]
        bd       = (bear_12m - price) / price if price else 0
        wu       = (ev_12m - price) / price if price else 0
        ratio    = abs(wu / bd) if bd != 0 else 0

        return {
            "ticker":                 ticker,
            "current_price":          price,
            "prob_bull":              r["prob_bull"],
            "prob_base":              r["prob_base"],
            "prob_bear":              r["prob_bear"],
            "targets_1m":             r["targets_1m"],
            "targets_3m":             r["targets_3m"],
            "targets_6m":             r["targets_6m"],
            "targets_12m":            r["targets_12m"],
            "ev_1m":                  ev(r["targets_1m"]),
            "ev_3m":                  ev(r["targets_3m"]),
            "ev_6m":                  ev(r["targets_6m"]),
            "ev_12m":                 ev_12m,
            "bear_case_downside_12m": bd,
            "upside_downside_ratio":  ratio,
            "bull_thesis":            r.get("bull_thesis", "N/A"),
            "base_thesis":            r.get("base_thesis", "N/A"),
            "bear_thesis":            r.get("bear_thesis", "N/A"),
            "kill_condition":         r.get("kill_condition", "N/A"),
            "key_catalyst":           r.get("key_catalyst", "N/A"),
        }

    except Exception as e:
        print(f"    Error scenario {ticker}: {e}")
        return {
            "ticker":                 ticker,
            "current_price":          price,
            "ev_12m":                 price,
            "bear_case_downside_12m": -0.30,
            "upside_downside_ratio":  0,
            "kill_condition":         "Error",
            "key_catalyst":           "N/A",
            "targets_12m":            {"bull": price*1.2, "base": price, "bear": price*0.7},
            "prob_bull": 0.25, "prob_base": 0.50, "prob_bear": 0.25,
            "bull_thesis": "N/A", "base_thesis": "N/A", "bear_thesis": "N/A",
        }


# ── Optimizer ─────────────────────────────────────────────────────────────────

def optimize_portfolio(scenarios: list, current_weights: dict, config: dict) -> dict:
    import numpy as np
    from scipy.optimize import minimize

    max_pos = config["portfolio"]["max_positions"]
    max_w   = config["portfolio"]["max_position_size"]
    min_w   = config["portfolio"]["min_position_size"]
    max_to  = config["turnover"]["max_one_sided_turnover"]

    candidates = [
        s for s in scenarios
        if s.get("ev_12m", 0) > s.get("current_price", 0)
        and s.get("upside_downside_ratio", 0) > 0
    ]
    candidates.sort(key=lambda s: s["upside_downside_ratio"], reverse=True)
    candidates = candidates[: max_pos * 2]

    if not candidates:
        print("  ⚠ Sin candidatos, manteniendo posiciones")
        return {
            "weights": current_weights, "expected_return": 0,
            "risk_score": 0, "risk_adjusted_return": 0,
            "turnover_used": 0, "added_names": [], "dropped_names": [],
        }

    tickers   = [s["ticker"] for s in candidates]
    n         = len(tickers)
    ev_ret    = np.array([
        (s["ev_12m"] - s["current_price"]) / s["current_price"]
        if s["current_price"] > 0 else 0 for s in candidates
    ])
    bear_d    = np.array([abs(s["bear_case_downside_12m"]) for s in candidates])
    current_w = np.array([current_weights.get(t, 0.0) for t in tickers])

    def objective(w):
        return -(np.dot(w, ev_ret) / (np.dot(w, bear_d) + 0.001))

    result = minimize(
        objective,
        np.full(n, 1.0 / min(max_pos, n)),
        method="SLSQP",
        bounds=[(0, max_w)] * n,
        constraints=[
            {"type": "eq",   "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: max_to - np.sum(np.maximum(w - current_w, 0))},
        ],
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    w_opt = result.x.copy()
    w_opt[w_opt < min_w] = 0
    if w_opt.sum() > 0:
        w_opt /= w_opt.sum()

    fw = {tickers[i]: float(w_opt[i]) for i in range(n) if w_opt[i] >= min_w}
    sm = {s["ticker"]: s for s in candidates}

    port_ev   = sum(w * (sm[t]["ev_12m"] - sm[t]["current_price"]) / sm[t]["current_price"]
                    for t, w in fw.items() if sm.get(t) and sm[t]["current_price"] > 0)
    port_risk = sum(w * abs(sm[t]["bear_case_downside_12m"])
                    for t, w in fw.items() if sm.get(t))
    all_t     = set(list(fw.keys()) + list(current_weights.keys()))
    turnover  = sum(max(fw.get(t, 0) - current_weights.get(t, 0), 0) for t in all_t)

    return {
        "weights":              fw,
        "expected_return":      port_ev,
        "risk_score":           port_risk,
        "risk_adjusted_return": port_ev / (port_risk + 0.001),
        "turnover_used":        turnover,
        "added_names":          [t for t in fw if t not in current_weights],
        "dropped_names":        [t for t in current_weights if t not in fw],
    }


# ── Thesis ────────────────────────────────────────────────────────────────────

def generate_thesis(ticker: str, scenario: dict, weight: float,
                    action: str, macro_summary: str) -> dict:
    price    = scenario.get("current_price", 0)
    ev_12m   = scenario.get("ev_12m", 0)
    bear_12m = scenario.get("targets_12m", {}).get("bear", 0)
    ev_pct   = (ev_12m - price) / price * 100 if price else 0
    bear_pct = (bear_12m - price) / price * 100 if price else 0
    ratio    = scenario.get("upside_downside_ratio", 0)
    ts       = datetime.now().isoformat()

    prompt = f"""
Thesis de posición para {ticker} | {action} | {weight:.1%}.

Precio: ${price:.2f} | EV 12M: ${ev_12m:.2f} ({ev_pct:+.1f}%)
Bear target: ${bear_12m:.2f} ({bear_pct:.1f}%) | U/D ratio: {ratio:.2f}x
Bull ({scenario.get('prob_bull',0):.0%}): {scenario.get('bull_thesis','')}
Base ({scenario.get('prob_base',0):.0%}): {scenario.get('base_thesis','')}
Bear ({scenario.get('prob_bear',0):.0%}): {scenario.get('bear_thesis','')}
Kill: {scenario.get('kill_condition','')}
Macro: {macro_summary[:150]}

Formato EXACTO:
---
THESIS: {ticker} | {action} | {weight:.1%} | {ts}
---
**Setup:** [oportunidad en 2 frases]
**Bull ({scenario.get('prob_bull',0):.0%}):** [drivers. Target ${scenario.get('targets_12m',{}).get('bull',0):.2f}]
**Base ({scenario.get('prob_base',0):.0%}):** [execution. Target ${scenario.get('targets_12m',{}).get('base',0):.2f}]
**Bear ({scenario.get('prob_bear',0):.0%}):** [riesgos. Target ${bear_12m:.2f}]
**EV:** ${ev_12m:.2f} ({ev_pct:+.1f}%) vs bear {bear_pct:.1f}%. Ratio {ratio:.2f}x.
**Sizing:** [por qué {weight:.1%}]
**Kill:** {scenario.get('kill_condition','')}
**Checkpoint:** [cuándo y qué mirar]
---
"""

    try:
        thesis_text = call_openrouter(prompt, task="thesis", max_tokens=800)
    except Exception as e:
        thesis_text = f"Error: {e}"

    thesis = {
        "ticker": ticker, "action": action, "weight": weight,
        "timestamp": ts, "price_at_thesis": price,
        "ev_12m": ev_12m, "expected_return_pct": ev_pct,
        "bear_downside_pct": bear_pct,
        "upside_downside_ratio": ratio,
        "kill_condition": scenario.get("kill_condition", ""),
        "key_catalyst": scenario.get("key_catalyst", ""),
        "thesis_text": thesis_text,
        "macro_snapshot": macro_summary[:200],
    }

    Path("data/thesis").mkdir(parents=True, exist_ok=True)
    with open(f"data/thesis/{ts[:10]}_{ticker}_{action}.json",
              "w", encoding="utf-8") as f:
        json.dump(thesis, f, indent=2, ensure_ascii=False)

    return thesis


def generate_rebalance_summary(result: dict, macro_summary: str) -> str:
    portfolio_str = "\n".join(
        f"  {t}: {w:.1%}"
        for t, w in sorted(result["weights"].items(), key=lambda x: -x[1])
    )

    prompt = f"""
Commentary de rebalanceo. Directo, cuantitativo, primera persona.
Máximo 300 palabras.

Portfolio:
{portfolio_str}

Cambios: +{result['added_names']} -{result['dropped_names']}
Turnover: {result['turnover_used']:.1%}/30% | EV: {result['expected_return']:.1%}
Macro: {macro_summary[:200]}

Estructura: macro→cambios→posiciones mantenidas→métricas→qué vigilar.
Termina: "Not advice, just how I'm sizing my own book."
"""

    try:
        return call_openrouter(prompt, task="thesis", max_tokens=500)
    except Exception as e:
        return f"Error generando summary: {e}"


# ── Email ─────────────────────────────────────────────────────────────────────

def generate_email_report(result: dict, all_thesis: list,
                           summary: str, positions: dict) -> None:
    today       = datetime.now().strftime("%Y-%m-%d")
    added       = result["added_names"]
    dropped     = result["dropped_names"]

    changes_parts = []
    if added:
        changes_parts.append(f"+{','.join(added)}")
    if dropped:
        changes_parts.append(f"-{','.join(dropped)}")
    changes_str = " ".join(changes_parts) if changes_parts else "sin cambios"

    subject = (
        f"📊 Portfolio {today} | "
        f"EV {result['expected_return']:.1%} | {changes_str}"
    )

    # ── Portfolio rows ────────────────────────────────────
    rows = ""
    sorted_weights = sorted(
        result["weights"].items(), key=lambda x: -x[1]
    )
    for t, w in sorted_weights:
        pos      = positions.get(t, {})
        ev       = pos.get("ev_12m")
        entry    = pos.get("entry_price") or 0
        ev_str   = f"${ev:.2f}" if ev else "-"
        rows += (
            f"<tr style='border-bottom:1px solid #eee'>"
            f"<td style='padding:8px'><strong>{t}</strong></td>"
            f"<td style='padding:8px'>{w:.1%}</td>"
            f"<td style='padding:8px'>${entry:.2f}</td>"
            f"<td style='padding:8px'>{ev_str}</td>"
            f"</tr>"
        )

    # ── Kill conditions ───────────────────────────────────
    kills = ""
    for t, p in positions.items():
        kc = p.get("kill_condition", "")
        if kc:
            w_str = f"{p['weight']:.1%}"
            kills += (
                f"<tr style='border-bottom:1px solid #eee'>"
                f"<td style='padding:8px'><strong>{t}</strong></td>"
                f"<td style='padding:8px'>{w_str}</td>"
                f"<td style='padding:8px'>{kc}</td>"
                f"</tr>"
            )

    # ── Thesis ────────────────────────────────────────────
    thesis_html = ""
    for th in all_thesis:
        ev_pct    = th.get("expected_return_pct", 0)
        bear_pct  = th.get("bear_downside_pct", 0)
        ratio     = th.get("upside_downside_ratio", 0)
        kill      = th.get("kill_condition", "N/A")
        text      = th.get("thesis_text", "N/A")
        ticker    = th["ticker"]
        action    = th["action"]
        weight    = th["weight"]
        ev_color  = "#28a745" if ev_pct > 0 else "#dc3545"

        thesis_html += (
            f"<div style='border:1px solid #ddd;padding:15px;"
            f"margin:10px 0;border-radius:5px'>"
            f"<h3>{ticker} | {action} | {weight:.1%}</h3>"
            f"<p>EV: <strong style='color:{ev_color}'>"
            f"{ev_pct:+.1f}%</strong> | "
            f"Bear: {bear_pct:.1f}% | "
            f"U/D: {ratio:.2f}x</p>"
            f"<p style='background:#fff3cd;padding:10px;"
            f"border-radius:3px'>"
            f"<strong>Kill:</strong> {kill}</p>"
            f"<div style='white-space:pre-wrap;"
            f"font-family:Georgia,serif;line-height:1.6'>"
            f"{text}</div></div>"
        )

    # ── Kill conditions table ─────────────────────────────
    kills_section = ""
    if kills:
        kills_section = (
            "<h2>⚠️ Kill Conditions</h2>"
            "<table style='width:100%;border-collapse:collapse'>"
            "<thead><tr style='background:#dc3545;color:white'>"
            "<th style='padding:10px;text-align:left'>Ticker</th>"
            "<th style='padding:10px;text-align:left'>Weight</th>"
            "<th style='padding:10px;text-align:left'>Condición</th>"
            "</tr></thead>"
            f"<tbody>{kills}</tbody></table>"
        )

    thesis_section = ""
    if thesis_html:
        thesis_section = f"<h2>🎯 Thesis</h2>{thesis_html}"

    ev_pct_port   = result["expected_return"]
    risk_adj      = result["risk_adjusted_return"]
    turnover      = result["turnover_used"]
    n_positions   = len(result["weights"])

    body = f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;max-width:800px;
             margin:0 auto;padding:20px">

<h1 style="color:#1a1a2e">📊 Portfolio Rebalance — {today}</h1>

<div style="display:flex;gap:15px;margin:20px 0;flex-wrap:wrap">
  <div style="background:#f8f9fa;padding:15px;border-radius:8px;
              flex:1;min-width:110px;text-align:center">
    <div style="font-size:22px;font-weight:bold;color:#28a745">
      {ev_pct_port:.1%}
    </div>
    <div style="color:#666;font-size:13px">EV 12M</div>
  </div>
  <div style="background:#f8f9fa;padding:15px;border-radius:8px;
              flex:1;min-width:110px;text-align:center">
    <div style="font-size:22px;font-weight:bold">
      {risk_adj:.2f}x
    </div>
    <div style="color:#666;font-size:13px">Risk-Adj</div>
  </div>
  <div style="background:#f8f9fa;padding:15px;border-radius:8px;
              flex:1;min-width:110px;text-align:center">
    <div style="font-size:22px;font-weight:bold;color:#fd7e14">
      {turnover:.1%}
    </div>
    <div style="color:#666;font-size:13px">Turnover</div>
  </div>
  <div style="background:#f8f9fa;padding:15px;border-radius:8px;
              flex:1;min-width:110px;text-align:center">
    <div style="font-size:22px;font-weight:bold">
      {n_positions}
    </div>
    <div style="color:#666;font-size:13px">Posiciones</div>
  </div>
</div>

<h2>📝 Commentary</h2>
<div style="background:#f8f9fa;padding:20px;border-radius:8px;
            white-space:pre-wrap;font-family:Georgia,serif;
            line-height:1.8">
{summary}
</div>

<h2>📈 Portfolio</h2>
<table style="width:100%;border-collapse:collapse">
  <thead>
    <tr style="background:#1a1a2e;color:white">
      <th style="padding:10px;text-align:left">Ticker</th>
      <th style="padding:10px;text-align:left">Weight</th>
      <th style="padding:10px;text-align:left">Entry</th>
      <th style="padding:10px;text-align:left">EV 12M</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

{kills_section}

{thesis_section}

<hr style="margin:30px 0">
<p style="color:#999;font-size:12px">
  Not advice, just how I'm sizing my own book.
</p>

</body>
</html>"""

    Path("data").mkdir(parents=True, exist_ok=True)
    with open("data/email_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {"subject": subject, "body": body},
            f, indent=2, ensure_ascii=False
        )
    print("✓ Email report guardado")


# ── Guardar ───────────────────────────────────────────────────────────────────

def save_results(result: dict, scenarios: dict) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    ts    = datetime.now().isoformat()

    positions = {
        ticker: {
            "weight":         weight,
            "entry_date":     today,
            "entry_price":    scenarios.get(ticker, {}).get("current_price"),
            "ev_12m":         scenarios.get(ticker, {}).get("ev_12m"),
            "kill_condition": scenarios.get(ticker, {}).get("kill_condition"),
        }
        for ticker, weight in result["weights"].items()
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
    rb_path = Path(f"data/rebalances/{today}_rebalance.json")
    with open(rb_path, "w") as f:
        json.dump(rebalance, f, indent=2)

    print("✓ Resultados guardados")
    return positions


# ── Main ──────────────────────────────────────────────────────────────────────

def run_rebalance():
    global FREE_MODELS_CACHE

    print(f"\n{'='*60}")
    print(f"PORTFOLIO REBALANCE — {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}\n")

    # 0. Modelos disponibles
    print("🔍 Descubriendo modelos gratuitos...")
    FREE_MODELS_CACHE = fetch_free_models()
    if not FREE_MODELS_CACHE:
        raise Exception("Sin modelos disponibles. Verifica API key.")
    print(f"  {len(FREE_MODELS_CACHE)} modelos en rotación\n")

    # 1. Config y estado
    config            = load_config()
    current_positions = load_current_positions()
    universe          = load_universe()
    print(f"Universe: {len(universe)} | Posiciones: {len(current_positions)}\n")

    # 2. Macro
    print("📊 Macro context...")
    macro = get_macro_context()
    print(f"✓ Macro listo\n")

    # 3. Scoring
    print("🔍 Scoring universe...")
    scores = score_universe(universe, macro, top_n=20)
    if scores:
        print(f"✓ Top: {scores[0]['ticker']} ({scores[0]['composite_score']:.1f})\n")

    # 4. Scenarios
    print("📐 Building scenarios...")
    scenarios = {}
    for i, score in enumerate(scores):
        ticker = score["ticker"]
        print(f"  [{i+1}/{len(scores)}] {ticker}...")
        scenarios[ticker] = build_scenario(ticker, score["data_snapshot"], macro)
        time.sleep(3)
    print(f"✓ {len(scenarios)} scenarios\n")

    # 5. Optimize
    print("⚙️  Optimizing...")
    current_weights = {t: p["weight"] for t, p in current_positions.items()}
    result = optimize_portfolio(list(scenarios.values()), current_weights, config)
    print(f"✓ {len(result['weights'])} nombres | "
          f"Turnover {result['turnover_used']:.1%} | "
          f"EV {result['expected_return']:.1%}\n")

    # 6. Thesis (solo cambios)
    print("📝 Generating thesis...")
    all_thesis = []
    min_change = config["turnover"]["min_position_change"]

    for ticker, weight in result["weights"].items():
        old_w  = current_weights.get(ticker, 0)
        diff   = weight - old_w
        action = (
            "OPEN"  if ticker in result["added_names"] else
            "ADD"   if diff >= min_change else
            "TRIM"  if diff <= -min_change else
            "HOLD"
        )
        if action != "HOLD" and ticker in scenarios:
            thesis = generate_thesis(ticker, scenarios[ticker], weight, action, macro)
            all_thesis.append(thesis)
            print(f"  ✓ {ticker} [{action}]")
            time.sleep(3)

    for ticker in result["dropped_names"]:
        if ticker in scenarios:
            thesis = generate_thesis(ticker, scenarios[ticker], 0.0, "CLOSE", macro)
            all_thesis.append(thesis)
            print(f"  ✓ {ticker} [CLOSE]")
            time.sleep(3)

    # 7. Commentary
    print("\n📣 Commentary...")
    summary              = generate_rebalance_summary(result, macro)
    result["commentary"] = summary

    today   = datetime.now().strftime("%Y-%m-%d")
    rb_file = Path(f"data/rebalances/{today}_rebalance.json")
    if rb_file.exists():
        rb = json.load(open(rb_file))
        rb["commentary"] = summary
        with open(rb_file, "w") as f:
            json.dump(rb, f, indent=2)

    # 8. Guardar
    positions = save_results(result, scenarios)

    # 9. Email
    generate_email_report(result, all_thesis, summary, positions)

    # 10. API usage
    print(f"\n📊 API calls:")
    for model, count in sorted(request_counts.items(), key=lambda x: -x[1]):
        print(f"   {model}: {count}")

    print(f"\n{'='*60}")
    print("✅ Rebalance completo")
    print(f"{'='*60}\n")
    print(summary)
    return result


if __name__ == "__main__":
    run_rebalance()
