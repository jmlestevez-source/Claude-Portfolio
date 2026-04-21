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

# ── Modelos: descubrimiento dinámico con filtrado robusto ─────────────────────

last_request_time = {}
request_counts    = {}
FREE_MODELS_CACHE = []

# Palabras clave para EXCLUIR modelos que no son LLM de texto
EXCLUDE_KEYWORDS = [
    "audio", "image", "vision", "clip", "embed", "lyria",
    "dall", "whisper", "tts", "ocr", "stable", "midjourney",
    "flux", "video", "music", "speech", "rerank",
]

# Modelos conocidos que funcionan bien para análisis de texto
PREFERRED_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "mistralai/mistral-small:free",
    "google/gemma-2-9b-it:free",
    "google/gemma-3-12b-it:free",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-r1-distill-llama-70b:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
]


def is_text_model(model: dict) -> bool:
    """
    Devuelve True solo si el modelo es un LLM de texto.
    Excluye modelos de audio, imagen, embeddings, etc.
    """
    model_id   = model.get("id", "").lower()
    model_name = model.get("name", "").lower()
    arch       = model.get("architecture", {})

    # Excluir por palabras clave en el ID o nombre
    for kw in EXCLUDE_KEYWORDS:
        if kw in model_id or kw in model_name:
            return False

    # Excluir si la arquitectura indica que no es texto
    modality = arch.get("modality", "")
    if modality and "text->text" not in modality and modality != "":
        # Aceptar si no hay modality definida o es text->text
        if "text" not in modality:
            return False

    # Excluir si no tiene context_length (señal de que no es LLM)
    if not model.get("context_length"):
        return False

    return True


def fetch_free_models() -> list:
    """
    Consulta OpenRouter, filtra modelos de texto gratuitos
    y los ordena priorizando los conocidos buenos.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY no encontrada")

    print("  Consultando modelos disponibles en OpenRouter...")

    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )

        if response.status_code != 200:
            raise Exception(
                f"Error {response.status_code}: {response.text[:200]}"
            )

        all_models = response.json().get("data", [])

        free_text_models = []
        for m in all_models:
            model_id = m.get("id", "")
            pricing  = m.get("pricing", {})

            # Solo gratuitos
            is_free = (
                ":free" in model_id
                or str(pricing.get("prompt", "1")) == "0"
                or pricing.get("prompt") == 0
            )
            if not is_free:
                continue

            # Solo texto
            if not is_text_model(m):
                print(f"    Excluido (no texto): {model_id}")
                continue

            free_text_models.append({
                "id":             model_id,
                "context_length": m.get("context_length", 4096),
                "name":           m.get("name", model_id),
            })

        # Ordenar: preferidos primero, luego por context_length
        def sort_key(m):
            is_preferred = m["id"] in PREFERRED_MODELS
            pref_index   = (
                PREFERRED_MODELS.index(m["id"])
                if is_preferred else 999
            )
            return (0 if is_preferred else 1,
                    pref_index,
                    -m["context_length"])

        free_text_models.sort(key=sort_key)

        ids = [m["id"] for m in free_text_models]

        print(
            f"  ✓ {len(ids)} modelos de texto gratuitos encontrados:"
        )
        for m in free_text_models:
            tag = "★" if m["id"] in PREFERRED_MODELS else " "
            print(
                f"    {tag} {m['id']} "
                f"(ctx: {m['context_length']:,})"
            )

        return ids

    except Exception as e:
        print(f"  ⚠ Error consultando modelos: {e}")
        print("  Usando lista de fallback...")
        return [
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "google/gemma-2-9b-it:free",
        ]


def get_models_for_task(task: str) -> list:
    """
    Para thesis/scenario: modelos grandes primero.
    Para screening/macro: todos, empezando por los más rápidos.
    """
    all_free = FREE_MODELS_CACHE

    if not all_free:
        return ["meta-llama/llama-3.1-8b-instruct:free"]

    if task in ("thesis", "scenario"):
        # Top mitad (mayor contexto/capacidad)
        top_n     = max(3, len(all_free) // 2)
        preferred = all_free[:top_n]
        rest      = all_free[top_n:]
        return preferred + rest

    # screening y macro: todos
    return all_free


def call_openrouter(
    prompt: str,
    task: str,
    system: str = "",
    max_tokens: int = 1000,
    temperature: float = 0.1,
    max_retries_per_model: int = 2,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY no encontrada")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/portfolio-autopilot",
        "X-Title":      "Portfolio Autopilot",
    }

    models     = get_models_for_task(task)
    last_error = ""

    for model in models:
        for attempt in range(max_retries_per_model):
            try:
                # Rate limiting: mínimo 5s entre requests al mismo modelo
                now  = time.time()
                last = last_request_time.get(model, 0)
                wait = 5 - (now - last)
                if wait > 0:
                    time.sleep(wait)

                messages = []
                if system:
                    messages.append(
                        {"role": "system", "content": system}
                    )
                messages.append(
                    {"role": "user", "content": prompt}
                )

                payload = {
                    "model":       model,
                    "messages":    messages,
                    "max_tokens":  max_tokens,
                    "temperature": temperature,
                }

                attempt_str = (
                    f" (intento {attempt+1})"
                    if attempt > 0 else ""
                )
                print(f"    → {model}{attempt_str}...")

                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=90,
                )

                last_request_time[model] = time.time()
                request_counts[model] = (
                    request_counts.get(model, 0) + 1
                )

                # Rate limit → esperar y reintentar MISMO modelo
                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get("Retry-After", 15)
                    )
                    wait_time = retry_after + 5
                    print(
                        f"    Rate limit, "
                        f"esperando {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue  # reintenta el mismo modelo

                # Modelo no disponible → siguiente modelo
                if response.status_code == 404:
                    print(f"    No disponible: {model}")
                    last_error = f"404 en {model}"
                    break  # sal del retry loop, prueba siguiente

                # Error servidor (502, 503...) → esperar y reintentar
                if response.status_code in (502, 503, 504):
                    wait_time = 10 * (attempt + 1)
                    print(
                        f"    Error {response.status_code}, "
                        f"esperando {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue  # reintenta el mismo modelo

                # Otro error → siguiente modelo
                if response.status_code != 200:
                    msg = response.text[:150]
                    print(
                        f"    Error {response.status_code}: {msg}"
                    )
                    last_error = msg
                    break

                # Respuesta OK
                content = (
                    response.json()["choices"][0]["message"]["content"]
                )

                if not content or not content.strip():
                    print(f"    Respuesta vacía, rotando...")
                    break

                print(f"    ✓ OK")
                return content.strip()

            except requests.exceptions.Timeout:
                print(f"    Timeout, reintentando...")
                last_error = "timeout"
                time.sleep(5)
                continue

            except Exception as e:
                print(f"    Excepción: {e}")
                last_error = str(e)
                break

    raise Exception(
        f"Todos los modelos fallaron para '{task}'.\n"
        f"Último error: {last_error}\n"
        f"Modelos probados: {models[:5]}..."
    )


def extract_json(text: str) -> dict:
    """Extrae JSON aunque el modelo añada texto alrededor."""
    try:
        return json.loads(text)
    except Exception:
        pass

    if "```json" in text:
        start = text.find("```json") + 7
        end   = text.find("```", start)
        try:
            return json.loads(text[start:end].strip())
        except Exception:
            pass

    if "```" in text:
        start = text.find("```") + 3
        end   = text.find("```", start)
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

    raise Exception(
        f"No se encontró JSON válido en:\n{text[:500]}"
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
        f"Fecha de análisis: "
        f"{datetime.now().strftime('%Y-%m-%d')}\n"
    )

    proxies = {
        "SPY": "^GSPC",
        "VIX": "^VIX",
        "TNX": "^TNX",
        "QQQ": "QQQ",
        "DXY": "DX-Y.NYB",
    }

    for label, ticker in proxies.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                price  = hist["Close"].iloc[-1]
                ret_5d = (
                    price / hist["Close"].iloc[0] - 1
                ) * 100
                market_data += (
                    f"- {label} ({ticker}): "
                    f"{price:.2f} ({ret_5d:+.1f}% 5d)\n"
                )
        except Exception:
            pass

    prompt = f"""
{market_data}

Describe el contexto macro actual para un portfolio long-only
equity en exactamente 150 palabras. Incluye:
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
            "industry":       info.get("industry"),
            "description": (
                info.get("longBusinessSummary") or ""
            )[:300],
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

    prompt = f"""
Analiza este stock y puntúalo en DOS dimensiones de 0 a 100.

CONTEXTO MACRO:
{macro_context[:300]}

DATOS:
{json.dumps(
    {k: v for k, v in data.items() if k != "description"},
    indent=2
)}

DESCRIPCIÓN: {data.get("description", "")}

DIMENSIÓN 1 - FUNDAMENTAL SCORE (0-100):
Calidad del negocio, márgenes, FCF, moat, balance.

DIMENSIÓN 2 - FORWARD SETUP SCORE (0-100):
Valoración vs histórico propio, catalizadores próximos,
posicionamiento mercado, asimetría riesgo/recompensa actual.

Responde SOLO en JSON sin texto adicional:
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
            call_openrouter(
                prompt, task="screening", max_tokens=400
            )
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
    top_n: int = 40,
) -> list:
    scores = []
    total  = len(tickers)
    errors = 0

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{total}] {ticker}...")
        score = score_stock(ticker, macro_context)
        scores.append(score)

        # Pausa adaptativa: más larga si hay muchos errores
        pause = 3 if errors < 5 else 6
        time.sleep(pause)

        if score["composite_score"] == 0:
            errors += 1

    scores.sort(key=lambda x: x["composite_score"], reverse=True)

    # Filtrar los que tienen score 0 (sin datos o error)
    valid = [s for s in scores if s["composite_score"] > 0]
    print(
        f"  ✓ {len(valid)} stocks con score válido "
        f"de {total} analizados"
    )

    return valid[:top_n]


# ── Scenarios ─────────────────────────────────────────────────────────────────

def build_scenario(
    ticker: str,
    stock_data: dict,
    macro_context: str,
) -> dict:
    price = stock_data.get("price") or 0

    prompt = f"""
Construye un análisis de escenarios para {ticker}.
Precio actual: ${price:.2f}

DATOS:
{json.dumps(
    {k: v for k, v in stock_data.items()
     if k not in [
         "description",
         "fundamental_rationale",
         "setup_rationale",
     ]},
    indent=2
)}

MACRO:
{macro_context[:300]}

3 escenarios con price targets en 4 horizontes temporales.
Las probabilidades DEBEN sumar exactamente 1.0.
Kill condition: evento CONCRETO y VERIFICABLE.

Responde SOLO en JSON sin texto adicional:
{{
    "prob_bull": <0.0-1.0>,
    "prob_base": <0.0-1.0>,
    "prob_bear": <0.0-1.0>,
    "targets_1m":  {{"bull": <price>, "base": <price>, "bear": <price>}},
    "targets_3m":  {{"bull": <price>, "base": <price>, "bear": <price>}},
    "targets_6m":  {{"bull": <price>, "base": <price>, "bear": <price>}},
    "targets_12m": {{"bull": <price>, "base": <price>, "bear": <price>}},
    "bull_thesis": "<3-4 frases>",
    "base_thesis": "<3-4 frases>",
    "bear_thesis": "<3-4 frases>",
    "kill_condition": "<evento concreto y verificable>",
    "key_catalyst": "<próximo catalizador con fecha si existe>"
}}
"""

    try:
        r = extract_json(
            call_openrouter(
                prompt, task="scenario", max_tokens=1000
            )
        )

        # Normalizar probabilidades
        total_prob = (
            r["prob_bull"] + r["prob_base"] + r["prob_bear"]
        )
        if abs(total_prob - 1.0) > 0.01:
            r["prob_bull"] /= total_prob
            r["prob_base"] /= total_prob
            r["prob_bear"] /= total_prob

        def ev(targets: dict) -> float:
            return (
                r["prob_bull"] * targets["bull"]
                + r["prob_base"] * targets["base"]
                + r["prob_bear"] * targets["bear"]
            )

        ev_12m    = ev(r["targets_12m"])
        bear_12m  = r["targets_12m"]["bear"]
        bear_down = (bear_12m - price) / price if price else 0
        w_upside  = (ev_12m - price) / price if price else 0
        ud_ratio  = (
            abs(w_upside / bear_down) if bear_down != 0 else 0
        )

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
            "bear_case_downside_12m": bear_down,
            "upside_downside_ratio":  ud_ratio,
            "bull_thesis":            r["bull_thesis"],
            "base_thesis":            r["base_thesis"],
            "bear_thesis":            r["bear_thesis"],
            "kill_condition":         r["kill_condition"],
            "key_catalyst":           r["key_catalyst"],
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
            "targets_12m": {
                "bull": price * 1.20,
                "base": price,
                "bear": price * 0.70,
            },
            "prob_bull":   0.25,
            "prob_base":   0.50,
            "prob_bear":   0.25,
            "bull_thesis": "N/A",
            "base_thesis": "N/A",
            "bear_thesis": "N/A",
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
        print("  ⚠ Sin candidatos válidos, manteniendo posiciones")
        return {
            "weights":              current_weights,
            "expected_return":      0,
            "risk_score":           0,
            "risk_adjusted_return": 0,
            "turnover_used":        0,
            "added_names":          [],
            "dropped_names":        [],
        }

    tickers = [s["ticker"] for s in candidates]
    n       = len(tickers)

    ev_returns = np.array([
        (s["ev_12m"] - s["current_price"]) / s["current_price"]
        if s["current_price"] > 0 else 0
        for s in candidates
    ])

    bear_downs = np.array([
        abs(s["bear_case_downside_12m"]) for s in candidates
    ])

    current_w = np.array([
        current_weights.get(t, 0.0) for t in tickers
    ])

    def objective(w):
        port_ev   = np.dot(w, ev_returns)
        port_risk = np.dot(w, bear_downs)
        return -(port_ev / (port_risk + 0.001))

    constraints = [
        {
            "type": "eq",
            "fun":  lambda w: np.sum(w) - 1.0,
        },
        {
            "type": "ineq",
            "fun":  lambda w: max_to - np.sum(
                np.maximum(w - current_w, 0)
            ),
        },
    ]
    bounds = [(0, max_w)] * n

    x0 = np.zeros(n)
    x0[: min(max_pos, n)] = 1.0 / min(max_pos, n)

    result = minimize(
        objective,
        x0,
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

    scen_map = {s["ticker"]: s for s in candidates}

    port_ev = sum(
        w
        * (scen_map[t]["ev_12m"] - scen_map[t]["current_price"])
        / scen_map[t]["current_price"]
        for t, w in final_weights.items()
        if scen_map.get(t) and scen_map[t]["current_price"] > 0
    )

    port_risk = sum(
        w * abs(scen_map[t]["bear_case_downside_12m"])
        for t, w in final_weights.items()
        if scen_map.get(t)
    )

    all_tickers = set(
        list(final_weights.keys()) + list(current_weights.keys())
    )
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
        "added_names":  [
            t for t in final_weights if t not in current_weights
        ],
        "dropped_names": [
            t for t in current_weights if t not in final_weights
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
    bear_12m = scenario.get("targets_12m", {}).get("bear", 0)
    ev_pct   = (ev_12m - price) / price * 100 if price else 0
    bear_pct = (bear_12m - price) / price * 100 if price else 0
    ratio    = scenario.get("upside_downside_ratio", 0)
    ts       = datetime.now().isoformat()

    prompt = f"""
Genera la thesis de posición para {ticker}.

PARÁMETROS:
- Acción: {action}
- Peso: {weight:.1%}
- Precio: ${price:.2f}
- Bull  ({scenario.get('prob_bull', 0):.0%}): \
${scenario.get('targets_12m', {}).get('bull', 0):.2f}
- Base  ({scenario.get('prob_base', 0):.0%}): \
${scenario.get('targets_12m', {}).get('base', 0):.2f}
- Bear  ({scenario.get('prob_bear', 0):.0%}): ${bear_12m:.2f}
- EV 12M: ${ev_12m:.2f} ({ev_pct:+.1f}%)
- Bear downside: {bear_pct:.1f}%
- U/D ratio: {ratio:.2f}x

BULL:  {scenario.get('bull_thesis', '')}
BASE:  {scenario.get('base_thesis', '')}
BEAR:  {scenario.get('bear_thesis', '')}
KILL:  {scenario.get('kill_condition', '')}
CAT:   {scenario.get('key_catalyst', '')}

MACRO: {macro_summary[:200]}

Genera la thesis en este formato EXACTO:

---
THESIS: {ticker} | {action} | {weight:.1%} | {ts}
---

**Setup:** [por qué existe la oportunidad ahora]

**Bull case ({scenario.get('prob_bull', 0):.0%}):** \
[drivers. Target \
${scenario.get('targets_12m', {}).get('bull', 0):.2f}]

**Base case ({scenario.get('prob_base', 0):.0%}):** \
[execution. Target \
${scenario.get('targets_12m', {}).get('base', 0):.2f}]

**Bear case ({scenario.get('prob_bear', 0):.0%}):** \
[riesgos. Target ${bear_12m:.2f}]

**Expected value:** ${ev_12m:.2f} ({ev_pct:+.1f}%) \
vs bear {bear_pct:.1f}%. Ratio {ratio:.2f}x.

**Sizing:** [por qué {weight:.1%} y no más o menos]

**Kill condition:** {scenario.get('kill_condition', '')}

**Next checkpoint:** [cuándo y qué mirar]
---
"""

    try:
        thesis_text = call_openrouter(
            prompt, task="thesis", max_tokens=1500
        )
    except Exception as e:
        thesis_text = f"Error generando thesis: {e}"

    thesis = {
        "ticker":                ticker,
        "action":                action,
        "weight":                weight,
        "timestamp":             ts,
        "price_at_thesis":       price,
        "ev_12m":                ev_12m,
        "expected_return_pct":   ev_pct,
        "bear_downside_pct":     bear_pct,
        "upside_downside_ratio": ratio,
        "kill_condition":        scenario.get("kill_condition", ""),
        "key_catalyst":          scenario.get("key_catalyst", ""),
        "thesis_text":           thesis_text,
        "macro_snapshot":        macro_summary[:300],
    }

    Path("data/thesis").mkdir(parents=True, exist_ok=True)
    filename = f"{ts[:10]}_{ticker}_{action}.json"
    with open(
        f"data/thesis/{filename}", "w", encoding="utf-8"
    ) as f:
        json.dump(thesis, f, indent=2, ensure_ascii=False)

    return thesis


def generate_rebalance_summary(
    result: dict,
    all_thesis: list,
    macro_summary: str,
) -> str:
    portfolio_str = "\n".join(
        f"  {t}: {w:.1%}"
        for t, w in sorted(
            result["weights"].items(), key=lambda x: -x[1]
        )
    )

    prompt = f"""
Genera el commentary de rebalanceo semanal del portfolio.
Estilo: directo, cuantitativo, primera persona, sin lenguaje vago.
Máximo 350 palabras.

PORTFOLIO:
{portfolio_str}

CAMBIOS:
- Añadidos:   {', '.join(result['added_names'])  or 'Ninguno'}
- Eliminados: {', '.join(result['dropped_names']) or 'Ninguno'}
- Turnover:   {result['turnover_used']:.1%} de 30% máximo

MÉTRICAS:
- EV 12M:         {result['expected_return']:.1%}
- Risk score:     {result['risk_score']:.1%}
- Risk-adj ratio: {result['risk_adjusted_return']:.2f}x

MACRO:
{macro_summary[:300]}

ESTRUCTURA:
1. Qué cambió en macro y cómo afecta al book
2. Cambios en portfolio y por qué
3. Posiciones que se mantienen y por qué
4. Métricas resultantes
5. Qué vigilar hasta el próximo rebalanceo

Termina con: "Not advice, just how I'm sizing my own book."
"""

    try:
        return call_openrouter(
            prompt, task="thesis", max_tokens=800
        )
    except Exception as e:
        return f"Error generando summary: {e}"


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

    changes_str = ""
    if added:   changes_str += f"+{','.join(added)}"
    if dropped: changes_str += f" -{','.join(dropped)}"
    if not changes_str: changes_str = "sin cambios"

    subject = (
        f"📊 Portfolio {today} | "
        f"EV {result['expected_return']:.1%} | {changes_str}"
    )

    rows = ""
    for ticker, w in sorted(
        result["weights"].items(), key=lambda x: -x[1]
    ):
        pos = positions.get(ticker, {})
        ev  = pos.get("ev_12m")
        rows += f"""
        <tr style="border-bottom:1px solid #eee;">
            <td style="padding:8px;">
                <strong>{ticker}</strong>
            </td>
            <td style="padding:8px;">{w:.1%}</td>
            <td style="padding:8px;">
                ${pos.get('entry_price') or 0:.2f}
            </td>
            <td style="padding:8px;">
                {"$" + f"{ev:.2f}" if ev else "-"}
            </td>
        </tr>"""

    kills = ""
    for ticker, pos in positions.items():
        kc = pos.get("kill_condition", "")
        if kc:
            kills += f"""
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:8px;">
                    <strong>{ticker}</strong>
                </td>
                <td style="padding:8px;">{pos['weight']:.1%}</td>
                <td style="padding:8px;">{kc}</td>
            </tr>"""

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
                EV 12M:
                <strong style="color:{color}">
                    {ev_pct:+.1f}%
                </strong>
                &nbsp;|&nbsp;
                Bear: {t.get('bear_downside_pct', 0):.1f}%
                &nbsp;|&nbsp;
                U/D: {t.get('upside_downside_ratio', 0):.2f}x
            </p>
            <p style="background:#fff3cd;padding:10px;
                      border-radius:3px;">
                <strong>Kill:</strong>
                {t.get('kill_condition', 'N/A')}
            </p>
            <div style="white-space:pre-wrap;
                        font-family:Georgia,serif;
                        line-height:1.6;font-size:14px;">
{t.get('thesis_text', 'N/A')}
            </div>
        </div>"""

    body = f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;
             max-width:800px;margin:0 auto;padding:20px;">

<h1 style="color:#1a1a2e;">
    📊 Portfolio Rebalance — {today}
</h1>

<div style="display:flex;gap:15px;margin:20px 0;flex-wrap:wrap;">
    <div style="background:#f8f9fa;padding:15px;
                border-radius:8px;flex:1;
                min-width:120px;text-align:center;">
        <div style="font-size:22px;font-weight:bold;
                    color:#28a745;">
            {result['expected_return']:.1%}
        </div>
        <div style="color:#666;font-size:13px;">EV 12M</div>
    </div>
    <div style="background:#f8f9fa;padding:15px;
                border-radius:8px;flex:1;
                min-width:120px;text-align:center;">
        <div style="font-size:22px;font-weight:bold;">
            {result['risk_adjusted_return']:.2f}x
        </div>
        <div style="color:#666;font-size:13px;">Risk-Adj</div>
    </div>
    <div style="background:#f8f9fa;padding:15px;
                border-radius:8px;flex:1;
                min-width:120px;text-align:center;">
        <div style="font-size:22px;font-weight:bold;
                    color:#fd7e14;">
            {result['turnover_used']:.1%}
        </div>
        <div style="color:#666;font-size:13px;">Turnover</div>
    </div>
    <div style="background:#f8f9fa;padding:15px;
                border-radius:8px;flex:1;
                min-width:120px;text-align:center;">
        <div style="font-size:22px;font-weight:bold;">
            {len(result['weights'])}
        </div>
        <div style="color:#666;font-size:13px;">Posiciones</div>
    </div>
</div>

<h2>📝 Commentary</h2>
<div style="background:#f8f9fa;padding:20px;border-radius:8px;
            white-space:pre-wrap;font-family:Georgia,serif;
            line-height:1.8;">
{summary}
</div>

<h2>📈 Portfolio actual</h2>
<table style="width:100%;border-collapse:collapse;">
    <thead>
        <tr style="background:#1a1a2e;color:white;">
            <th style="padding:10px;text-align:left;">
                Ticker
            </th>
            <th style="padding:10px;text-align:left;">
                Weight
            </th>
            <th style="padding:10px;text-align:left;">
                Entry
            </th>
            <th style="padding:10px;text-align:left;">
                EV 12M
            </th>
        </tr>
    </thead>
    <tbody>{rows}</tbody>
</table>

{"<h2>⚠️ Kill Conditions</h2><table style='width:100%;border-collapse:collapse;'><thead><tr style='background:#dc3545;color:white;'><th style='padding:10px;text-align:left;'>Ticker</th><th style='padding:10px;text-align:left;'>Weight</th><th style='padding:10px;text-align:left;'>Kill Condition</th></tr></thead><tbody>" + kills + "</tbody></table>" if kills else ""}

{"<h2>🎯 Thesis generadas</h2>" + thesis_html if thesis_html else ""}

<hr style="margin:30px 0;">
<p style="color:#999;font-size:12px;">
    Not advice, just how I'm sizing my own book.
</p>

</body>
</html>"""

    Path("data").mkdir(parents=True, exist_ok=True)
    with open(
        "data/email_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            {"subject": subject, "body": body},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("✓ Email report guardado")


# ── Guardar resultados ────────────────────────────────────────────────────────

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
    with open(
        f"data/rebalances/{today}_rebalance.json", "w"
    ) as f:
        json.dump(rebalance, f, indent=2)

    print("✓ Resultados guardados")
    return positions


# ── Main ──────────────────────────────────────────────────────────────────────

def run_rebalance():
    global FREE_MODELS_CACHE

    print(f"\n{'='*60}")
    print(
        f"PORTFOLIO REBALANCE — "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    print(f"{'='*60}\n")

    # 0. Descubrir modelos disponibles ahora mismo
    print("🔍 Descubriendo modelos gratuitos...")
    FREE_MODELS_CACHE = fetch_free_models()

    if not FREE_MODELS_CACHE:
        raise Exception(
            "No hay modelos de texto gratuitos disponibles. "
            "Verifica tu API key en openrouter.ai"
        )
    print(
        f"  Usando {len(FREE_MODELS_CACHE)} modelos "
        f"en rotación\n"
    )

    # 1. Cargar config y estado
    config            = load_config()
    current_positions = load_current_positions()
    universe          = load_universe()

    print(f"Universe:   {len(universe)} tickers")
    print(f"Posiciones: {len(current_positions)} nombres\n")

    # 2. Macro
    print("📊 Macro context...")
    macro = get_macro_context()
    print(f"✓ Macro listo\n")

    # 3. Scoring
    print("🔍 Scoring universe...")
    scores = score_universe(universe, macro, top_n=40)
    if scores:
        print(
            f"✓ Top: {scores[0]['ticker']} "
            f"({scores[0]['composite_score']:.1f})\n"
        )

    # 4. Scenarios
    print("📐 Building scenarios...")
    scenarios = {}
    for i, score in enumerate(scores):
        ticker = score["ticker"]
        print(f"  [{i+1}/{len(scores)}] {ticker}...")
        scenario = build_scenario(
            ticker, score["data_snapshot"], macro
        )
        scenarios[ticker] = scenario
        time.sleep(2)
    print(f"✓ {len(scenarios)} scenarios\n")

    # 5. Optimize
    print("⚙️  Optimizing...")
    current_weights = {
        t: p["weight"] for t, p in current_positions.items()
    }
    result = optimize_portfolio(
        list(scenarios.values()), current_weights, config
    )
    print(
        f"✓ {len(result['weights'])} nombres | "
        f"Turnover {result['turnover_used']:.1%} | "
        f"EV {result['expected_return']:.1%}\n"
    )

    # 6. Thesis (solo para cambios reales)
    print("📝 Generating thesis...")
    all_thesis = []
    min_change = config["turnover"]["min_position_change"]

    for ticker, weight in result["weights"].items():
        old_w  = current_weights.get(ticker, 0)
        diff   = weight - old_w

        if ticker in result["added_names"]:
            action = "OPEN"
        elif abs(diff) >= min_change:
            action = "ADD" if diff > 0 else "TRIM"
        else:
            action = "HOLD"

        if action != "HOLD" and ticker in scenarios:
            thesis = generate_thesis(
                ticker, scenarios[ticker],
                weight, action, macro
            )
            all_thesis.append(thesis)
            print(f"  ✓ {ticker} [{action}]")
            time.sleep(2)

    for ticker in result["dropped_names"]:
        if ticker in scenarios:
            thesis = generate_thesis(
                ticker, scenarios[ticker],
                0.0, "CLOSE", macro
            )
            all_thesis.append(thesis)
            print(f"  ✓ {ticker} [CLOSE]")
            time.sleep(2)

    # 7. Commentary
    print("\n📣 Commentary...")
    summary              = generate_rebalance_summary(
        result, all_thesis, macro
    )
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

    # 10. Resumen uso API
    print(f"\n📊 API calls por modelo:")
    for model, count in sorted(
        request_counts.items(), key=lambda x: -x[1]
    ):
        print(f"   {model}: {count} calls")

    print(f"\n{'='*60}")
    print("✅ Rebalance completo")
    print(f"{'='*60}\n")
    print(summary)

    return result


if __name__ == "__main__":
    run_rebalance()
