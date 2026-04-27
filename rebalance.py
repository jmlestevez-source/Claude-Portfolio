# rebalance.py
"""
Orquestador principal del sistema de rebalanceo.
Pipeline completo optimizado para Russell 1000.
"""

import os
import json
import yaml
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Imports del sistema
from src.llm         import call_llm, request_counts
from src.universe    import load_universe, update_universe
from src.data_fetcher import (
    fetch_fundamentals_parallel,
    fetch_price_history,
    fetch_macro_data,
    cache,
)
from src.screener    import prescreening
from src.scorer      import score_batch
from src.scenarios   import build_scenario
from src.optimizer   import optimize_portfolio
from src.thesis      import generate_thesis
from src.performance import (
    compute_performance_metrics,
    update_performance,
    record_trades,
)
from src.email_report import generate_email_report


# ── Config y estado ───────────────────────────────────────────────────────────

def load_config() -> dict:
    with open("config/portfolio_config.yaml") as f:
        return yaml.safe_load(f)


def load_current_positions() -> dict:
    f = Path("data/positions/current.json")
    return json.load(open(f)) if f.exists() else {}


def save_results(
    result: dict, scenarios: dict
) -> dict:
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
    rb_path = Path(f"data/rebalances/{today}_rebalance.json")
    with open(rb_path, "w") as f:
        json.dump(rebalance, f, indent=2)

    print("✓ Resultados guardados")
    return positions


def get_macro_context(macro_data: dict) -> str:
    """Genera texto de contexto macro usando datos reales."""
    market_lines = []
    for label, d in macro_data.items():
        price  = d.get("price",  0)
        r5d    = d.get("ret_5d", 0)
        market_lines.append(
            f"- {label}: {price:.2f} ({r5d:+.1f}% 5d)"
        )
    market_str = "\n".join(market_lines)

    system = (
        "Eres un analista macro experto. "
        "Responde siempre en español. "
        "Sé conciso, específico y cuantitativo."
    )
    prompt = (
        f"Datos de mercado reales:\n{market_str}\n\n"
        "Describe el contexto macro actual para un "
        "portfolio long-only de renta variable "
        "en máximo 120 palabras.\n"
        "Incluye:\n"
        "1. Postura de la Fed y tipos (cuantificado)\n"
        "2. Fase del ciclo económico\n"
        "3. Apetito por riesgo (referencia VIX)\n"
        "4. Sectores con viento de cola vs en contra\n"
        "5. Top 2 riesgos macro próximos 3 meses\n"
        "Sin frases genéricas."
    )

    return call_llm(
        prompt, task="macro", system=system, max_tokens=300
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run_rebalance(force_universe_update: bool = False):
    start_time = time.time()

    print(f"\n{'='*60}")
    print(
        "REBALANCEO PORTFOLIO — "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    print(f"{'='*60}\n")

    # Verificar credenciales
    groq_key   = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not groq_key and not gemini_key:
        raise Exception(
            "Necesitas al menos una API key:\n"
            "  GROQ_API_KEY   → console.groq.com\n"
            "  GEMINI_API_KEY → aistudio.google.com"
        )
    if groq_key:   print("✓ Groq disponible")
    if gemini_key: print("✓ Gemini disponible (backup)")

    # Limpiar caché vieja
    cache.cleanup_old(keep_days=2)

    config            = load_config()
    current_positions = load_current_positions()
    print(
        f"Posiciones actuales: "
        f"{len(current_positions)}\n"
    )

    # ── PASO 1: Universo Russell 1000 ─────────────────────
    print("📋 PASO 1: Universo Russell 1000")
    if force_universe_update:
        universe_tickers = update_universe()
    else:
        universe_tickers = load_universe()
    print(f"  Total: {len(universe_tickers)} tickers\n")

    # ── PASO 2: Datos macro ───────────────────────────────
    print("📊 PASO 2: Contexto macro")
    print("  Descargando datos de mercado reales...")
    macro_data = fetch_macro_data()
    print(
        f"  ✓ {len(macro_data)} indicadores: "
        f"{', '.join(macro_data.keys())}"
    )
    macro_context = get_macro_context(macro_data)
    print("  ✓ Contexto macro generado\n")

    # ── PASO 3: Histórico de precios (batch) ──────────────
    print("📈 PASO 3: Histórico de precios")
    price_history = fetch_price_history(
        universe_tickers,
        period="1y",
        chunk_size=100,
    )
    print()

    # ── PASO 4: Fundamentales en paralelo ─────────────────
    print("🔎 PASO 4: Fundamentales (paralelo)")
    fundamentals = fetch_fundamentals_parallel(
        universe_tickers, max_workers=8
    )
    print()

    # ── PASO 5: Pre-filtro cuantitativo ───────────────────
    print("🔬 PASO 5: Pre-filtro cuantitativo")
    candidates, no_data_tickers = prescreening(
        fundamentals,
        price_history,
        top_n=config.get("screening", {}).get(
            "prescreen_top_n", 100
        ),
    )
    print(
        f"  {len(candidates)} candidatos | "
        f"{len(no_data_tickers)} sin datos\n"
    )

    # ── PASO 6: Scoring LLM en batches ────────────────────
    print("🧠 PASO 6: Scoring LLM (batches)")
    stocks_to_score = [
        {**fundamentals[t], "ticker": t}
        for t in candidates
        if t in fundamentals
    ]
    scored = score_batch(
        stocks_to_score,
        macro_context,
        batch_size=config.get("screening", {}).get(
            "llm_batch_size", 8
        ),
    )
    scored.sort(
        key=lambda x: x.get("composite_score", 0),
        reverse=True,
    )
    top_n   = config["portfolio"]["max_positions"] * 2
    top_scored = scored[:top_n]
    print(
        f"  ✓ Top {len(top_scored)} candidatos finales\n"
    )

    # ── PASO 7: Escenarios detallados ─────────────────────
    print("📐 PASO 7: Construyendo escenarios")
    scenarios: dict = {}
    for i, s in enumerate(top_scored):
        ticker = s["ticker"]
        print(f"  [{i+1}/{len(top_scored)}] {ticker}...")
        scenarios[ticker] = build_scenario(
            ticker,
            {**fundamentals.get(ticker, {}), **s},
            macro_context,
        )
    print(f"  ✓ {len(scenarios)} escenarios\n")

    # ── PASO 8: Optimización ──────────────────────────────
    print("⚙️  PASO 8: Optimizando portfolio")
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
        f"  ✓ {len(result['weights'])} posiciones | "
        f"Turnover {result['turnover_used']:.1%} | "
        f"EV {result['expected_return']:.1%}\n"
    )

    # En rebalance.py, después del PASO 8 (optimización),
    # añadir este bloque:

    # ── PASO 8b: Enriquecimiento PortfolioLabs ────────────
    use_portfoliolabs = os.getenv(
    "USE_PORTFOLIOLABS", "false"
    ).lower() == "true"

    if use_portfoliolabs:
    print("🔍 PASO 8b: Enriquecimiento PortfolioLabs")
    from src.portfoliolabs import enrich_with_portfoliolabs

    portfolio_tickers = list(result["weights"].keys())
    fundamentals = enrich_with_portfoliolabs(
    portfolio_tickers,
    fundamentals,
    )

    # Mostrar divergencias en consola
    for t in portfolio_tickers:
    divs = fundamentals.get(t, {}).get(
    "_pl_divergences", {}
    )
    if divs:
    print(f"  ⚠ {t}: {list(divs.keys())}")
    else:
    print(
    "  (PortfolioLabs desactivado. "
    "Activar con USE_PORTFOLIOLABS=true)"
    )
    print()

    # ── PASO 9: Operaciones y P&L ─────────────────────────
    print("💰 PASO 9: Registrando operaciones")
    new_trades = record_trades(
        result, scenarios, current_positions
    )
    print(f"  ✓ {len(new_trades)} operaciones registradas\n")

    # ── PASO 10: Guardar posiciones ───────────────────────
    positions = save_results(result, scenarios)

    # ── PASO 11: Performance metrics ─────────────────────
    print("📊 PASO 11: Calculando performance vs SPY")
    perf_metrics = compute_performance_metrics(
        positions, scenarios
    )
    update_performance(result, positions, scenarios)

    port_ret = perf_metrics.get("portfolio_return_pct", 0)
    spy_ret  = perf_metrics.get("spy_return_pct",       0)
    alpha    = perf_metrics.get("alpha_pct",             0)
    sign_p   = "+" if port_ret >= 0 else ""
    sign_s   = "+" if spy_ret  >= 0 else ""
    sign_a   = "+" if alpha    >= 0 else ""
    print(
        f"  Portfolio: {sign_p}{port_ret:.2f}% | "
        f"SPY: {sign_s}{spy_ret:.2f}% | "
        f"Alpha: {sign_a}{alpha:.2f}%\n"
    )

    # ── PASO 12: Tesis ────────────────────────────────────
    print("📝 PASO 12: Generando tesis")
    all_thesis   = []
    min_change   = config["turnover"]["min_position_change"]

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
                ticker, scenarios[ticker],
                weight, action, macro_context,
            )
            all_thesis.append(thesis)
            accion = {
                "OPEN": "ABRIR", "ADD": "AÑADIR",
                "TRIM": "REDUCIR",
            }.get(action, action)
            print(f"  ✓ {ticker} [{accion}]")

    for ticker in result["dropped_names"]:
        if ticker in scenarios:
            thesis = generate_thesis(
                ticker, scenarios[ticker],
                0.0, "CLOSE", macro_context,
            )
            all_thesis.append(thesis)
            print(f"  ✓ {ticker} [CERRAR]")

    # ── PASO 13: Commentary ───────────────────────────────
    print("\n📣 PASO 13: Generando commentary")
    summary = _generate_commentary(
        result, macro_context, perf_metrics
    )
    result["commentary"] = summary

    # Actualizar fichero rebalanceo con commentary
    today   = datetime.now().strftime("%Y-%m-%d")
    rb_file = Path(f"data/rebalances/{today}_rebalance.json")
    if rb_file.exists():
        rb = json.load(open(rb_file))
        rb["commentary"] = summary
        with open(rb_file, "w") as f:
            json.dump(rb, f, indent=2)

    # ── PASO 14: Email report ─────────────────────────────
    print("📧 PASO 14: Generando email report")
    generate_email_report(
        result          = result,
        all_thesis      = all_thesis,
        summary         = summary,
        positions       = positions,
        perf_metrics    = perf_metrics,
        no_data_tickers = no_data_tickers,
        new_trades      = new_trades,
    )

    # ── Resumen final ─────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Rebalanceo completado en {elapsed:.0f}s")
    print(f"{'='*60}")

    print("\n📊 Llamadas API:")
    for model, count in sorted(
        request_counts.items(), key=lambda x: -x[1]
    ):
        print(f"   {model}: {count}")

    print(f"\n{summary}\n")
    return result


def _generate_commentary(
    result:       dict,
    macro_context: str,
    perf_metrics:  dict,
) -> str:
    weights  = result["weights"]
    added    = result["added_names"]
    dropped  = result["dropped_names"]
    port_ret = perf_metrics.get("portfolio_return_pct", 0)
    spy_ret  = perf_metrics.get("spy_return_pct",       0)
    alpha    = perf_metrics.get("alpha_pct",             0)

    lines = "\n".join(
        f"  {t}: {w:.1%}"
        for t, w in sorted(
            weights.items(), key=lambda x: -x[1]
        )
    )

    system = (
        "Eres un gestor de portfolio profesional. "
        "Escribe en primera persona. "
        "Directo, cuantitativo, sin lenguaje vago. "
        "Responde en español."
    )

    prompt = (
        "Commentary del rebalanceo semanal. "
        "Máximo 300 palabras. Primera persona.\n\n"
        f"Portfolio:\n{lines}\n\n"
        f"Cambios: +{added} -{dropped}\n"
        f"Turnover: {result['turnover_used']:.1%}\n"
        f"EV 12M: {result['expected_return']:.1%}\n\n"
        f"Performance real:\n"
        f"  Portfolio: {port_ret:+.2f}%\n"
        f"  SPY mismo periodo: {spy_ret:+.2f}%\n"
        f"  Alpha: {alpha:+.2f}%\n\n"
        f"Macro: {macro_context[:200]}\n\n"
        "Estructura:\n"
        "1. Cambio macro y efecto en portfolio\n"
        "2. Qué se compró/vendió y por qué\n"
        "3. Qué se mantiene y por qué\n"
        "4. Performance real vs SPY\n"
        "5. Qué vigilar hasta el próximo rebalanceo\n\n"
        "Termina con: 'No es consejo de inversión, "
        "es como estoy gestionando mi propio capital.'"
    )

    try:
        return call_llm(
            prompt, task="commentary",
            system=system, max_tokens=500,
        )
    except Exception as e:
        return f"Error generando commentary: {e}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-universe",
        action="store_true",
        help="Forzar actualización del universo Russell 1000",
    )
    args = parser.parse_args()
    run_rebalance(
        force_universe_update=args.update_universe
    )
