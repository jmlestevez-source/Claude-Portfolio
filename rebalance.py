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

# ── Configuración LLM ─────────────────────────────────────────────────────────

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash-lite",
]

request_counts = {}


# ── Clientes LLM ──────────────────────────────────────────────────────────────

def call_groq(
    prompt: str,
    system: str = "",
    max_tokens: int = 1000,
    temperature: float = 0.1,
) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY no encontrada")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for model in GROQ_MODELS:
        try:
            time.sleep(2)

            messages = []
            if system:
                messages.append(
                    {"role": "system", "content": system}
                )
            messages.append(
                {"role": "user", "content": prompt}
            )

            r = requests.post(
                "https://api.groq.com/openai/v1/"
                "chat/completions",
                headers=headers,
                json={
                    "model":       model,
                    "messages":    messages,
                    "max_tokens":  max_tokens,
                    "temperature": temperature,
                },
                timeout=60,
            )

            key = f"groq/{model}"
            request_counts[key] = (
                request_counts.get(key, 0) + 1
            )

            if r.status_code == 429:
                retry = int(
                    r.headers.get("Retry-After", 15)
                )
                print(
                    f"    Groq rate limit, "
                    f"esperando {retry}s..."
                )
                time.sleep(retry + 2)
                continue

            if r.status_code == 200:
                content = (
                    r.json()
                    ["choices"][0]["message"]["content"]
                )
                if content and content.strip():
                    short = model.split("-")[0]
                    print(f"    groq/{short} OK")
                    return content.strip()

            print(
                f"    Groq {model}: "
                f"HTTP {r.status_code}"
            )

        except requests.exceptions.Timeout:
            print(f"    Groq {model}: timeout")
            continue
        except Exception as e:
            print(f"    Groq {model}: {e}")
            continue

    raise ValueError("Groq: todos los modelos fallaron")


def call_gemini(
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.1,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY no encontrada")

    for model in GEMINI_MODELS:
        try:
            time.sleep(1)

            r = requests.post(
                "https://generativelanguage.googleapis.com"
                f"/v1beta/models/{model}:generateContent"
                f"?key={api_key}",
                json={
                    "contents": [
                        {"parts": [{"text": prompt}]}
                    ],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature":     temperature,
                    },
                },
                timeout=60,
            )

            key = f"gemini/{model}"
            request_counts[key] = (
                request_counts.get(key, 0) + 1
            )

            if r.status_code == 429:
                print(
                    "    Gemini rate limit, "
                    "esperando 15s..."
                )
                time.sleep(15)
                continue

            if r.status_code == 200:
                data    = r.json()
                content = (
                    data
                    .get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                if content and content.strip():
                    print(f"    gemini/{model} OK")
                    return content.strip()

            print(
                f"    Gemini {model}: "
                f"HTTP {r.status_code}"
            )

        except requests.exceptions.Timeout:
            print(f"    Gemini {model}: timeout")
            continue
        except Exception as e:
            print(f"    Gemini {model}: {e}")
            continue

    raise ValueError(
        "Gemini: todos los modelos fallaron"
    )


def call_llm(
    prompt: str,
    task: str = "general",
    system: str = "",
    max_tokens: int = 1000,
    temperature: float = 0.1,
) -> str:
    groq_key   = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    errors     = []

    if groq_key:
        try:
            return call_groq(
                prompt, system, max_tokens, temperature
            )
        except Exception as e:
            errors.append(f"Groq: {e}")
            print("    Groq fallo, probando Gemini...")

    if gemini_key:
        try:
            full = (
                f"{system}\n\n{prompt}"
                if system else prompt
            )
            return call_gemini(
                full, max_tokens, temperature
            )
        except Exception as e:
            errors.append(f"Gemini: {e}")

    raise Exception(
        f"Todos los LLMs fallaron para '{task}'.\n"
        f"Errores: {errors}"
    )


def call_llm_json(
    prompt: str,
    task: str = "general",
    max_tokens: int = 300,
    temperature: float = 0.1,
) -> dict:
    """
    Llamada LLM que garantiza respuesta JSON.
    Usa system prompt estricto y reintenta con
    Gemini si Groq no devuelve JSON valido.
    """
    system = (
        "You are a financial analyst assistant. "
        "You MUST respond with ONLY a valid JSON object. "
        "Do NOT include any explanation, markdown, "
        "or text before or after the JSON. "
        "Start your response directly with { "
        "and end with }. "
        "All numbers must be numeric, not strings."
    )

    full_prompt = (
        f"{prompt}\n\n"
        "IMPORTANT: Your entire response must be "
        "a single valid JSON object. "
        "Start with { and end with }. "
        "No other text."
    )

    groq_key   = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if groq_key:
        try:
            text = call_groq(
                full_prompt, system, max_tokens,
                temperature
            )
            return extract_json(text)
        except Exception as e:
            print(f"    Groq JSON fallo: {e}")

    if gemini_key:
        try:
            full = f"{system}\n\n{full_prompt}"
            text = call_gemini(
                full, max_tokens, temperature
            )
            return extract_json(text)
        except Exception as e:
            print(f"    Gemini JSON fallo: {e}")

    raise Exception(
        f"No se pudo obtener JSON valido "
        f"para '{task}'"
    )


def extract_json(text: str) -> dict:
    """Extrae JSON aunque el modelo añada texto."""
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
                    return json.loads(
                        text[start:end].strip()
                    )
                except Exception:
                    pass

    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except Exception:
            pass

    raise Exception(
        f"JSON no encontrado en:\n{text[:300]}"
    )


# ── Config y estado ───────────────────────────────────────────────────────────

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
    market_data = (
        f"Fecha: {datetime.now().strftime('%Y-%m-%d')}\n"
    )

    for label, ticker in {
        "SPY": "^GSPC",
        "VIX": "^VIX",
        "TNX": "^TNX",
        "QQQ": "QQQ",
    }.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                price  = hist["Close"].iloc[-1]
                ret_5d = (
                    price / hist["Close"].iloc[0] - 1
                ) * 100
                market_data += (
                    f"- {label}: {price:.2f}"
                    f" ({ret_5d:+.1f}% 5d)\n"
                )
        except Exception:
            pass

    system = (
        "Eres un analista macro experto. "
        "Responde siempre en espanol. "
        "Se conciso, especifico y cuantitativo."
    )

    prompt = (
        f"{market_data}\n"
        "Describe el contexto macro actual para un "
        "portfolio long-only de renta variable "
        "en maximo 120 palabras.\n"
        "Incluye:\n"
        "1. Postura de la Fed y tipos (cuantificado)\n"
        "2. Fase del ciclo economico\n"
        "3. Apetito por el riesgo (referencia VIX)\n"
        "4. Sectores con viento de cola vs en contra\n"
        "5. Top 2 riesgos macro proximos 3 meses\n"
        "Se especifico y cuantitativo. "
        "Sin frases genericas."
    )

    return call_llm(
        prompt,
        task="macro",
        system=system,
        max_tokens=300,
    )


# ── Scoring ───────────────────────────────────────────────────────────────────

def get_stock_data(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker":         ticker,
            "price": (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
            ),
            "forward_pe":     info.get("forwardPE"),
            "trailing_pe":    info.get("trailingPE"),
            "revenue_growth": info.get("revenueGrowth"),
            "gross_margins":  info.get("grossMargins"),
            "free_cashflow":  info.get("freeCashflow"),
            "market_cap":     info.get("marketCap"),
            "52w_high":       info.get("fiftyTwoWeekHigh"),
            "52w_low":        info.get("fiftyTwoWeekLow"),
            "sector":         info.get("sector"),
            "description": (
                info.get("longBusinessSummary") or ""
            )[:200],
        }
    except Exception as e:
        print(f"    Error datos {ticker}: {e}")
        return {"ticker": ticker, "price": None}


def score_stock(ticker: str, macro_context: str) -> dict:
    data = get_stock_data(ticker)

    if not data.get("price"):
        return {
            "ticker":          ticker,
            "composite_score": 0,
            "data_snapshot":   data,
        }

    price   = data["price"]
    fpe     = data["forward_pe"]
    growth  = data["revenue_growth"]
    margins = data["gross_margins"]
    sector  = data["sector"]
    low_52  = data["52w_low"]
    high_52 = data["52w_high"]
    desc    = data.get("description", "")[:150]
    macro_s = macro_context[:150]

    prompt = (
        f"Puntua {ticker} de 0 a 100 en dos "
        f"dimensiones.\n\n"
        f"CONTEXTO MACRO: {macro_s}\n\n"
        f"DATOS: precio={price}, fwd_PE={fpe}, "
        f"crecimiento_ingresos={growth}, "
        f"margenes={margins}, sector={sector}, "
        f"rango_52s=[{low_52}, {high_52}]\n\n"
        f"DESCRIPCION: {desc}\n\n"
        "Devuelve SOLO este JSON:\n"
        "{{\n"
        '  "fundamental_score": <entero 0-100>,\n'
        '  "forward_setup_score": <entero 0-100>,\n'
        '  "key_risk": "<una frase>",\n'
        '  "key_catalyst": "<una frase>"\n'
        "}}"
    )

    try:
        result = call_llm_json(
            prompt, task="screening", max_tokens=200
        )
        composite = (
            result.get("fundamental_score", 0) * 0.40
            + result.get("forward_setup_score", 0) * 0.60
        )
        return {
            "ticker":              ticker,
            "fundamental_score":   result.get(
                "fundamental_score", 0
            ),
            "forward_setup_score": result.get(
                "forward_setup_score", 0
            ),
            "composite_score":     composite,
            "data_snapshot":       {**data, **result},
        }
    except Exception as e:
        print(f"    Error scoring {ticker}: {e}")
        return {
            "ticker":          ticker,
            "composite_score": 0,
            "data_snapshot":   data,
        }


def score_universe(
    tickers: list,
    macro_context: str,
    top_n: int = 20,
) -> list:
    scores = []
    total  = len(tickers)

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{total}] {ticker}...")
        score = score_stock(ticker, macro_context)
        scores.append(score)

    valid = [
        s for s in scores
        if s["composite_score"] > 0
    ]
    valid.sort(
        key=lambda x: x["composite_score"],
        reverse=True,
    )
    print(f"  {len(valid)} validos de {total}")
    return valid[:top_n]


# ── Scenarios ─────────────────────────────────────────────────────────────────

def build_scenario(
    ticker: str,
    stock_data: dict,
    macro_context: str,
) -> dict:
    price   = stock_data.get("price") or 0
    fpe     = stock_data.get("forward_pe")
    growth  = stock_data.get("revenue_growth")
    margins = stock_data.get("gross_margins")
    low_52  = stock_data.get("52w_low")
    high_52 = stock_data.get("52w_high")
    macro_s = macro_context[:150]

    # Calcular rangos razonables para anclar al modelo
    bear_floor  = round(price * 0.65, 2)  # Max -35%
    bull_ceil   = round(price * 1.60, 2)  # Max +60%
    bear_min    = round(price * 0.75, 2)  # Min bear -25%

    prompt = (
        f"Construye 3 escenarios para {ticker} "
        f"a precio actual ${price:.2f}.\n\n"
        f"Datos clave: PE={fpe}, "
        f"crecimiento_ingresos={growth}, "
        f"margenes={margins}, "
        f"rango_52s=[{low_52}, {high_52}]\n\n"
        f"Macro: {macro_s}\n\n"
        "REGLAS OBLIGATORIAS:\n"
        "1. Las probabilidades DEBEN sumar 1.0\n"
        f"2. El target bajista DEBE estar entre "
        f"${bear_floor} y ${bear_min} "
        f"(entre -25% y -35% del precio actual). "
        f"Un bear case de menos del 10% de caida "
        f"no es creible.\n"
        f"3. El target alcista DEBE estar por debajo "
        f"de ${bull_ceil} (+60% maximo)\n"
        "4. La kill_condition debe ser un evento "
        "ESPECIFICO y VERIFICABLE PARA ESTA EMPRESA "
        "en concreto. No usar frases genericas como "
        "'caida en guidance' sin especificar que "
        "metrica concreta, que umbral numerico "
        "y en que fecha o evento. "
        "Ejemplo correcto: 'Reduccion del NRR "
        "por debajo del 115% en Q2 2025 combinada "
        "con corte del guidance anual de ingresos'. "
        "Ejemplo incorrecto: 'caida en guidance "
        "de la empresa'.\n"
        "5. Los catalizadores deben ser eventos "
        "reales con fecha aproximada conocida "
        "(earnings, conferencias, vencimientos "
        "regulatorios). No inventar fechas.\n"
        "6. Todos los textos en espanol.\n\n"
        "Devuelve SOLO este JSON:\n"
        "{{\n"
        '  "prob_bull": <float>,\n'
        '  "prob_base": <float>,\n'
        '  "prob_bear": <float>,\n'
        '  "targets_1m": {{"bull": <precio>, '
        '"base": <precio>, "bear": <precio>}},\n'
        '  "targets_3m": {{"bull": <precio>, '
        '"base": <precio>, "bear": <precio>}},\n'
        '  "targets_6m": {{"bull": <precio>, '
        '"base": <precio>, "bear": <precio>}},\n'
        '  "targets_12m": {{"bull": <precio>, '
        '"base": <precio>, "bear": <precio>}},\n'
        '  "bull_thesis": "<2 frases especificas>",\n'
        '  "base_thesis": "<2 frases especificas>",\n'
        '  "bear_thesis": "<2 frases especificas>",\n'
        '  "kill_condition": "<metrica concreta + '
        'umbral numerico + evento especifico>",\n'
        '  "key_catalyst": "<evento real con fecha '
        'aproximada, sin inventar>"\n'
        "}}"
    )

    try:
        r = call_llm_json(
            prompt, task="scenario", max_tokens=700
        )

        # Normalizar probabilidades
        total_p = (
            r["prob_bull"]
            + r["prob_base"]
            + r["prob_bear"]
        )
        if abs(total_p - 1.0) > 0.01:
            r["prob_bull"] /= total_p
            r["prob_base"] /= total_p
            r["prob_bear"] /= total_p

        # Validar y corregir bear case si es irreal
        bear_12m = r["targets_12m"]["bear"]
        if bear_12m > price * 0.90:
            # Bear case menor al 10%: forzar a -25%
            factor = bear_12m / price
            for h in ["targets_1m", "targets_3m",
                      "targets_6m", "targets_12m"]:
                r[h]["bear"] = round(
                    r[h]["bear"] / factor * 0.75, 2
                )

        def ev(t: dict) -> float:
            return (
                r["prob_bull"] * t["bull"]
                + r["prob_base"] * t["base"]
                + r["prob_bear"] * t["bear"]
            )

        ev_12m   = ev(r["targets_12m"])
        bear_12m = r["targets_12m"]["bear"]
        bd       = (
            (bear_12m - price) / price
            if price else 0
        )
        wu       = (
            (ev_12m - price) / price
            if price else 0
        )
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
            "bull_thesis":  r.get("bull_thesis",  "N/D"),
            "base_thesis":  r.get("base_thesis",  "N/D"),
            "bear_thesis":  r.get("bear_thesis",  "N/D"),
            "kill_condition": r.get(
                "kill_condition", "N/D"
            ),
            "key_catalyst": r.get(
                "key_catalyst", "N/D"
            ),
        }

    except Exception as e:
        print(f"    Error scenario {ticker}: {e}")
        return {
            "ticker":                 ticker,
            "current_price":          price,
            "ev_12m":                 price,
            "bear_case_downside_12m": -0.30,
            "upside_downside_ratio":  0,
            "kill_condition":         "Error en generacion",
            "key_catalyst":           "N/D",
            "targets_12m": {
                "bull": price * 1.2,
                "base": price,
                "bear": price * 0.7,
            },
            "prob_bull":   0.25,
            "prob_base":   0.50,
            "prob_bear":   0.25,
            "bull_thesis": "N/D",
            "base_thesis": "N/D",
            "bear_thesis": "N/D",
        }

# ── Optimizer ─────────────────────────────────────────────────────────────────

def optimize_portfolio(
    scenarios: list,
    current_weights: dict,
    config: dict,
) -> dict:
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
    candidates.sort(
        key=lambda s: s["upside_downside_ratio"],
        reverse=True,
    )
    candidates = candidates[: max_pos * 2]

    if not candidates:
        print(
            "  Sin candidatos validos, "
            "manteniendo posiciones"
        )
        return {
            "weights":              current_weights,
            "expected_return":      0,
            "risk_score":           0,
            "risk_adjusted_return": 0,
            "turnover_used":        0,
            "added_names":          [],
            "dropped_names":        [],
        }

    tickers   = [s["ticker"] for s in candidates]
    n         = len(tickers)

    ev_ret = np.array([
        (s["ev_12m"] - s["current_price"])
        / s["current_price"]
        if s["current_price"] > 0 else 0
        for s in candidates
    ])
    bear_d = np.array([
        abs(s["bear_case_downside_12m"])
        for s in candidates
    ])
    current_w = np.array([
        current_weights.get(t, 0.0) for t in tickers
    ])

    def objective(w):
        port_ev   = np.dot(w, ev_ret)
        port_risk = np.dot(w, bear_d)
        return -(port_ev / (port_risk + 0.001))

    # ── FIX: turnover incluye también posiciones
    # que se cierran (no están en candidates)
    def turnover_constraint(w):
        buys = np.sum(np.maximum(w - current_w, 0))
        # Añadir el peso de posiciones que se cierran
        closed = sum(
            v for t, v in current_weights.items()
            if t not in tickers
        )
        return max_to - buys - closed

    result = minimize(
        objective,
        np.full(n, 1.0 / min(max_pos, n)),
        method="SLSQP",
        bounds=[(0, max_w)] * n,
        constraints=[
            {
                "type": "eq",
                "fun":  lambda w: np.sum(w) - 1.0,
            },
            {
                "type": "ineq",
                "fun":  turnover_constraint,
            },
        ],
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    w_opt = result.x.copy()
    w_opt[w_opt < min_w] = 0
    if w_opt.sum() > 0:
        w_opt /= w_opt.sum()

    fw = {
        tickers[i]: float(w_opt[i])
        for i in range(n)
        if w_opt[i] >= min_w
    }
    sm = {s["ticker"]: s for s in candidates}

    port_ev = sum(
        w
        * (sm[t]["ev_12m"] - sm[t]["current_price"])
        / sm[t]["current_price"]
        for t, w in fw.items()
        if sm.get(t) and sm[t]["current_price"] > 0
    )
    port_risk = sum(
        w * abs(sm[t]["bear_case_downside_12m"])
        for t, w in fw.items()
        if sm.get(t)
    )
    all_t = set(
        list(fw.keys()) + list(current_weights.keys())
    )
    turnover = sum(
        max(fw.get(t, 0) - current_weights.get(t, 0), 0)
        for t in all_t
    )

    return {
        "weights":              fw,
        "expected_return":      port_ev,
        "risk_score":           port_risk,
        "risk_adjusted_return": port_ev / (
            port_risk + 0.001
        ),
        "turnover_used":        turnover,
        "added_names":  [
            t for t in fw
            if t not in current_weights
        ],
        "dropped_names": [
            t for t in current_weights
            if t not in fw
        ],
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
    bear_12m = scenario.get(
        "targets_12m", {}
    ).get("bear", 0)
    ev_pct   = (
        (ev_12m - price) / price * 100
        if price else 0
    )
    bear_pct = (
        (bear_12m - price) / price * 100
        if price else 0
    )
    ratio    = scenario.get("upside_downside_ratio", 0)
    ts       = datetime.now().isoformat()

    pb      = scenario.get("prob_bull", 0)
    pba     = scenario.get("prob_base", 0)
    pbe     = scenario.get("prob_bear", 0)
    t12     = scenario.get("targets_12m", {})
    bull_t  = t12.get("bull", 0)
    base_t  = t12.get("base", 0)
    kill    = scenario.get("kill_condition", "")
    cat     = scenario.get("key_catalyst", "")
    macro_s = macro_summary[:150]

    accion_es = {
        "OPEN":  "ABRIR",
        "ADD":   "ANADIR",
        "TRIM":  "REDUCIR",
        "CLOSE": "CERRAR",
        "HOLD":  "MANTENER",
    }.get(action, action)

    system = (
        "Eres un gestor de portfolio profesional. "
        "Escribe en primera persona. "
        "Se directo, cuantitativo y especifico. "
        "Cada afirmacion debe ser falsable y "
        "verificable. "
        "Sin lenguaje vago ni generalizaciones. "
        "No inventes fechas ni datos. "
        "Responde SIEMPRE en espanol."
    )

    prompt = (
        f"Escribe la tesis de posicion para "
        f"{ticker} | {accion_es} | {weight:.1%}.\n\n"
        f"Precio actual: ${price:.2f}\n"
        f"Valor esperado 12M: ${ev_12m:.2f} "
        f"({ev_pct:+.1f}%)\n"
        f"Objetivo bajista: ${bear_12m:.2f} "
        f"({bear_pct:.1f}%)\n"
        f"Ratio U/D: {ratio:.2f}x\n\n"
        f"Caso alcista ({pb:.0%}): "
        f"{scenario.get('bull_thesis', '')}\n"
        f"Caso base ({pba:.0%}): "
        f"{scenario.get('base_thesis', '')}\n"
        f"Caso bajista ({pbe:.0%}): "
        f"{scenario.get('bear_thesis', '')}\n\n"
        f"Kill condition ya definida: {kill}\n"
        f"Catalizador: {cat}\n"
        f"Contexto macro: {macro_s}\n\n"
        "REGLAS PARA LA TESIS:\n"
        "1. La kill condition en la tesis debe ser "
        "IDENTICA a la ya definida arriba, "
        "no la cambies ni la generalices.\n"
        "2. El sizing debe explicar por que ese peso "
        "especifico considerando el ratio U/D y el "
        "bear case en dolares absolutos.\n"
        "3. El checkpoint debe mencionar un evento "
        "real y proximo (siguiente earnings, "
        "conferencia conocida, dato macro concreto). "
        "No inventes fechas.\n"
        "4. No uses frases como 'podria', 'quizas', "
        "'posiblemente'. Usa afirmaciones directas.\n\n"
        "Usa este formato EXACTO:\n"
        "---\n"
        f"TESIS: {ticker} | {accion_es} | "
        f"{weight:.1%} | {ts}\n"
        "---\n"
        "**Oportunidad:** [por que existe la "
        "oportunidad ahora, 2 frases concretas "
        "con datos especificos]\n\n"
        f"**Caso alcista ({pb:.0%}):** "
        f"[drivers especificos con metricas. "
        f"Objetivo ${bull_t:.2f}]\n\n"
        f"**Caso base ({pba:.0%}):** "
        f"[ejecucion esperada con metricas. "
        f"Objetivo ${base_t:.2f}]\n\n"
        f"**Caso bajista ({pbe:.0%}):** "
        f"[riesgos especificos con metricas. "
        f"Objetivo ${bear_12m:.2f}]\n\n"
        f"**Valor esperado:** ${ev_12m:.2f} "
        f"({ev_pct:+.1f}%) vs bajista "
        f"{bear_pct:.1f}%. Ratio {ratio:.2f}x.\n\n"
        f"**Sizing:** [por que {weight:.1%}: "
        f"relacion con ratio U/D de {ratio:.2f}x "
        f"y bear case de ${bear_12m:.2f}]\n\n"
        f"**Kill condition:** {kill}\n\n"
        "**Proximo checkpoint:** "
        "[evento real y proximo con fecha "
        "aproximada conocida]\n"
        "---"
    )

    try:
        thesis_text = call_llm(
            prompt,
            task="thesis",
            system=system,
            max_tokens=800,
        )
    except Exception as e:
        thesis_text = f"Error: {e}"

    thesis = {
        "ticker":                ticker,
        "action":                action,
        "accion":                accion_es,
        "weight":                weight,
        "timestamp":             ts,
        "price_at_thesis":       price,
        "ev_12m":                ev_12m,
        "expected_return_pct":   ev_pct,
        "bear_downside_pct":     bear_pct,
        "upside_downside_ratio": ratio,
        "kill_condition":        kill,
        "key_catalyst":          cat,
        "thesis_text":           thesis_text,
        "macro_snapshot":        macro_summary[:200],
    }

    Path("data/thesis").mkdir(parents=True, exist_ok=True)
    fname = f"{ts[:10]}_{ticker}_{action}.json"
    with open(
        f"data/thesis/{fname}", "w", encoding="utf-8"
    ) as f:
        json.dump(
            thesis, f, indent=2, ensure_ascii=False
        )

    return thesis

def generate_rebalance_summary(
    result: dict,
    macro_summary: str,
) -> str:
    weights = result["weights"]
    added   = result["added_names"]
    dropped = result["dropped_names"]

    lines = "\n".join(
        f"  {t}: {w:.1%}"
        for t, w in sorted(
            weights.items(), key=lambda x: -x[1]
        )
    )
    added_str   = str(added)
    dropped_str = str(dropped)
    turnover    = result["turnover_used"]
    ev          = result["expected_return"]
    risk_adj    = result["risk_adjusted_return"]
    macro_s     = macro_summary[:200]

    system = (
        "Eres un gestor de portfolio profesional. "
        "Escribe en primera persona. "
        "Se directo y cuantitativo. "
        "Sin lenguaje vago ni frases genericas. "
        "Responde SIEMPRE en espanol."
    )

    prompt = (
        "Escribe el commentary del rebalanceo semanal "
        "del portfolio. Maximo 300 palabras. "
        "Primera persona, directo, cuantitativo. "
        "Cada afirmacion debe ser especifica "
        "y verificable.\n\n"
        f"Portfolio resultante:\n{lines}\n\n"
        f"Cambios: "
        f"posiciones nuevas={added_str}, "
        f"eliminadas={dropped_str}\n"
        f"Turnover utilizado: {turnover:.1%} "
        f"de 30% maximo\n"
        f"EV 12M del portfolio: {ev:.1%}\n"
        f"Retorno ajustado por riesgo: {risk_adj:.2f}x\n\n"
        f"Contexto macro: {macro_s}\n\n"
        "Estructura obligatoria:\n"
        "1. Cambio macro esta semana y efecto "
        "en el portfolio\n"
        "2. Que se compro/vendio y por que\n"
        "3. Que se mantiene y por que no se toco\n"
        "4. Metricas del portfolio resultante\n"
        "5. Que vigilar hasta el proximo rebalanceo\n\n"
        'Termina con: "No es consejo de inversion, '
        'es como estoy gestionando mi propio capital."'
    )

    try:
        return call_llm(
            prompt,
            task="thesis",
            system=system,
            max_tokens=500,
        )
    except Exception as e:
        return f"Error generando commentary: {e}"


# ── Email ─────────────────────────────────────────────────────────────────────

def generate_email_report(
    result: dict,
    all_thesis: list,
    summary: str,
    positions: dict,
) -> None:
    today   = datetime.now().strftime("%Y-%m-%d")
    added   = result["added_names"]
    dropped = result["dropped_names"]

    parts = []
    if added:
        parts.append(f"+{','.join(added)}")
    if dropped:
        parts.append(f"-{','.join(dropped)}")
    changes_str = (
        " ".join(parts) if parts else "sin cambios"
    )

    subject = (
        f"Portfolio {today} | "
        f"EV {result['expected_return']:.1%} | "
        f"{changes_str}"
    )

    # Portfolio rows
    rows = ""
    for t, w in sorted(
        result["weights"].items(),
        key=lambda x: -x[1],
    ):
        pos    = positions.get(t, {})
        ev     = pos.get("ev_12m")
        entry  = pos.get("entry_price") or 0
        ev_str = f"${ev:.2f}" if ev else "-"
        rows += (
            "<tr style='border-bottom:"
            "1px solid #eee'>"
            f"<td style='padding:8px'>"
            f"<strong>{t}</strong></td>"
            f"<td style='padding:8px'>{w:.1%}</td>"
            f"<td style='padding:8px'>"
            f"${entry:.2f}</td>"
            f"<td style='padding:8px'>"
            f"{ev_str}</td>"
            "</tr>"
        )

    # Kill conditions
    kills = ""
    for t, p in positions.items():
        kc = p.get("kill_condition", "")
        if kc:
            kills += (
                "<tr style='border-bottom:"
                "1px solid #eee'>"
                f"<td style='padding:8px'>"
                f"<strong>{t}</strong></td>"
                f"<td style='padding:8px'>"
                f"{p['weight']:.1%}</td>"
                f"<td style='padding:8px'>{kc}</td>"
                "</tr>"
            )

    # Thesis
    thesis_html = ""
    for th in all_thesis:
        ev_pct   = th.get("expected_return_pct", 0)
        bear_pct = th.get("bear_downside_pct", 0)
        ratio    = th.get("upside_downside_ratio", 0)
        kill     = th.get("kill_condition", "N/D")
        text     = th.get("thesis_text", "N/D")
        accion   = th.get("accion", th.get("action", ""))
        ev_color = (
            "#28a745" if ev_pct > 0 else "#dc3545"
        )
        thesis_html += (
            "<div style='border:1px solid #ddd;"
            "padding:15px;margin:10px 0;"
            "border-radius:5px'>"
            f"<h3>{th['ticker']} | {accion} | "
            f"{th['weight']:.1%}</h3>"
            "<p>VE 12M: "
            f"<strong style='color:{ev_color}'>"
            f"{ev_pct:+.1f}%</strong> | "
            f"Bajista: {bear_pct:.1f}% | "
            f"Ratio U/D: {ratio:.2f}x</p>"
            "<p style='background:#fff3cd;"
            "padding:10px;border-radius:3px'>"
            f"<strong>Kill condition:</strong> "
            f"{kill}</p>"
            "<div style='white-space:pre-wrap;"
            "font-family:Georgia,serif;"
            "line-height:1.6'>"
            f"{text}</div></div>"
        )

    kills_section = ""
    if kills:
        kills_section = (
            "<h2>Kill Conditions</h2>"
            "<table style='width:100%;"
            "border-collapse:collapse'>"
            "<thead>"
            "<tr style='background:#dc3545;"
            "color:white'>"
            "<th style='padding:10px;"
            "text-align:left'>Ticker</th>"
            "<th style='padding:10px;"
            "text-align:left'>Peso</th>"
            "<th style='padding:10px;"
            "text-align:left'>Condicion</th>"
            "</tr></thead>"
            f"<tbody>{kills}</tbody></table>"
        )

    thesis_section = ""
    if thesis_html:
        thesis_section = (
            f"<h2>Tesis de posicion</h2>"
            f"{thesis_html}"
        )

    ev_port  = result["expected_return"]
    risk_adj = result["risk_adjusted_return"]
    turnover = result["turnover_used"]
    n_pos    = len(result["weights"])

    body = (
        "<!DOCTYPE html><html>"
        "<body style='font-family:Arial,sans-serif;"
        "max-width:800px;margin:0 auto;padding:20px'>"
        "<h1 style='color:#1a1a2e'>"
        f"Rebalanceo Portfolio {today}</h1>"
        "<div style='display:flex;gap:15px;"
        "margin:20px 0;flex-wrap:wrap'>"
        "<div style='background:#f8f9fa;"
        "padding:15px;border-radius:8px;"
        "flex:1;min-width:110px;text-align:center'>"
        "<div style='font-size:22px;"
        "font-weight:bold;color:#28a745'>"
        f"{ev_port:.1%}</div>"
        "<div style='color:#666;font-size:13px'>"
        "VE 12M</div></div>"
        "<div style='background:#f8f9fa;"
        "padding:15px;border-radius:8px;"
        "flex:1;min-width:110px;text-align:center'>"
        "<div style='font-size:22px;"
        "font-weight:bold'>"
        f"{risk_adj:.2f}x</div>"
        "<div style='color:#666;font-size:13px'>"
        "Ret. Ajustado</div></div>"
        "<div style='background:#f8f9fa;"
        "padding:15px;border-radius:8px;"
        "flex:1;min-width:110px;text-align:center'>"
        "<div style='font-size:22px;"
        "font-weight:bold;color:#fd7e14'>"
        f"{turnover:.1%}</div>"
        "<div style='color:#666;font-size:13px'>"
        "Turnover</div></div>"
        "<div style='background:#f8f9fa;"
        "padding:15px;border-radius:8px;"
        "flex:1;min-width:110px;text-align:center'>"
        "<div style='font-size:22px;"
        "font-weight:bold'>"
        f"{n_pos}</div>"
        "<div style='color:#666;font-size:13px'>"
        "Posiciones</div></div></div>"
        "<h2>Commentary</h2>"
        "<div style='background:#f8f9fa;"
        "padding:20px;border-radius:8px;"
        "white-space:pre-wrap;"
        "font-family:Georgia,serif;line-height:1.8'>"
        f"{summary}</div>"
        "<h2>Portfolio actual</h2>"
        "<table style='width:100%;"
        "border-collapse:collapse'>"
        "<thead>"
        "<tr style='background:#1a1a2e;color:white'>"
        "<th style='padding:10px;text-align:left'>"
        "Ticker</th>"
        "<th style='padding:10px;text-align:left'>"
        "Peso</th>"
        "<th style='padding:10px;text-align:left'>"
        "Entrada</th>"
        "<th style='padding:10px;text-align:left'>"
        "VE 12M</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"{kills_section}"
        f"{thesis_section}"
        "<hr style='margin:30px 0'>"
        "<p style='color:#999;font-size:12px'>"
        "No es consejo de inversion, es como estoy "
        "gestionando mi propio capital."
        "</p></body></html>"
    )

    Path("data").mkdir(parents=True, exist_ok=True)
    with open(
        "data/email_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {"subject": subject, "body": body},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("✓ Email report guardado")


# ── Guardar resultados ────────────────────────────────────────────────────────

def save_results(
    result: dict, scenarios: dict
) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    ts    = datetime.now().isoformat()

    positions = {
        ticker: {
            "weight":         weight,
            "entry_date":     today,
            "entry_price":    scenarios.get(
                ticker, {}
            ).get("current_price"),
            "ev_12m":         scenarios.get(
                ticker, {}
            ).get("ev_12m"),
            "kill_condition": scenarios.get(
                ticker, {}
            ).get("kill_condition"),
        }
        for ticker, weight in result["weights"].items()
    }

    Path("data/positions").mkdir(
        parents=True, exist_ok=True
    )
    with open(
        "data/positions/current.json", "w"
    ) as f:
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
            "expected_return":      result[
                "expected_return"
            ],
            "risk_score":           result["risk_score"],
            "risk_adjusted_return": result[
                "risk_adjusted_return"
            ],
        },
    }

    Path("data/rebalances").mkdir(
        parents=True, exist_ok=True
    )
    rb_path = Path(
        f"data/rebalances/{today}_rebalance.json"
    )
    with open(rb_path, "w") as f:
        json.dump(rebalance, f, indent=2)

    print("✓ Resultados guardados")
    return positions


# ── Main ──────────────────────────────────────────────────────────────────────

def run_rebalance():
    print(f"\n{'='*60}")
    print(
        "REBALANCEO PORTFOLIO — "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    print(f"{'='*60}\n")

    groq_key   = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not groq_key and not gemini_key:
        raise Exception(
            "Necesitas al menos una API key:\n"
            "  GROQ_API_KEY   → console.groq.com\n"
            "  GEMINI_API_KEY → aistudio.google.com"
        )

    if groq_key:
        print("✓ Groq disponible")
    if gemini_key:
        print("✓ Gemini disponible (backup)")
    print()

    config            = load_config()
    current_positions = load_current_positions()
    universe          = load_universe()
    print(
        f"Universo: {len(universe)} tickers | "
        f"Posiciones actuales: {len(current_positions)}\n"
    )

    # 1. Macro
    print("📊 Contexto macro...")
    macro = get_macro_context()
    print("✓ Macro listo\n")

    # 2. Scoring
    print("🔍 Scoring del universo...")
    scores = score_universe(universe, macro, top_n=20)
    if scores:
        top = scores[0]
        print(
            f"✓ Top: {top['ticker']}"
            f" ({top['composite_score']:.1f})\n"
        )

    # 3. Scenarios
    print("📐 Construyendo escenarios...")
    scenarios = {}
    for i, score in enumerate(scores):
        ticker = score["ticker"]
        print(f"  [{i+1}/{len(scores)}] {ticker}...")
        scenarios[ticker] = build_scenario(
            ticker, score["data_snapshot"], macro
        )
    print(f"✓ {len(scenarios)} escenarios listos\n")

    # 4. Optimize
    print("⚙️  Optimizando portfolio...")
    current_weights = {
        t: p["weight"]
        for t, p in current_positions.items()
    }
    result = optimize_portfolio(
        list(scenarios.values()),
        current_weights,
        config,
    )
    print(
        f"✓ {len(result['weights'])} posiciones | "
        f"Turnover {result['turnover_used']:.1%} | "
        f"EV {result['expected_return']:.1%}\n"
    )

    # 5. Thesis (solo cambios reales)
    print("📝 Generando tesis...")
    all_thesis = []
    min_change = config["turnover"]["min_position_change"]

    for ticker, weight in result["weights"].items():
        old_w = current_weights.get(ticker, 0)
        diff  = weight - old_w

        if ticker in result["added_names"]:
            action = "OPEN"
        elif diff >= min_change:
            action = "ADD"
        elif diff <= -min_change:
            action = "TRIM"
        else:
            action = "HOLD"

        if action != "HOLD" and ticker in scenarios:
            thesis = generate_thesis(
                ticker,
                scenarios[ticker],
                weight,
                action,
                macro,
            )
            all_thesis.append(thesis)
            accion = {
                "OPEN":  "ABRIR",
                "ADD":   "ANADIR",
                "TRIM":  "REDUCIR",
                "CLOSE": "CERRAR",
            }.get(action, action)
            print(f"  ✓ {ticker} [{accion}]")

    for ticker in result["dropped_names"]:
        if ticker in scenarios:
            thesis = generate_thesis(
                ticker,
                scenarios[ticker],
                0.0,
                "CLOSE",
                macro,
            )
            all_thesis.append(thesis)
            print(f"  ✓ {ticker} [CERRAR]")

    # 6. Commentary
    print("\n📣 Generando commentary...")
    summary              = generate_rebalance_summary(
        result, macro
    )
    result["commentary"] = summary

    today   = datetime.now().strftime("%Y-%m-%d")
    rb_file = Path(
        f"data/rebalances/{today}_rebalance.json"
    )
    if rb_file.exists():
        rb = json.load(open(rb_file))
        rb["commentary"] = summary
        with open(rb_file, "w") as f:
            json.dump(rb, f, indent=2)

    # 7. Guardar posiciones
    positions = save_results(result, scenarios)

    # 8. Email report
    generate_email_report(
        result, all_thesis, summary, positions
    )

    # 9. Resumen uso API
    print("\n📊 Llamadas API:")
    for model, count in sorted(
        request_counts.items(), key=lambda x: -x[1]
    ):
        print(f"   {model}: {count}")

    print(f"\n{'='*60}")
    print("✅ Rebalanceo completado")
    print(f"{'='*60}\n")
    print(summary)

    return result


if __name__ == "__main__":
    run_rebalance()
