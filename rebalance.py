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

from src.llm import call_llm, request_counts
from src.universe import load_universe, update_universe
from src.data_fetcher import (
    fetch_fundamentals_parallel,
    fetch_price_history,
    fetch_macro_data,
    cache,
)
from src.screener  import prescreening
from src.scorer    import score_batch
from src.scenarios import build_scenario
from src.optimizer import optimize_portfolio
from src.thesis    import generate_thesis
from src.performance import (
    compute_performance_metrics,
    update_performance,
    record_trades,
)
from src.email_report import (
    generate_email_report,
    send_email_report,
)


# ── Config y estado ───────────────────────────────────────────────────────────

def load_config() -> dict:
    with open("config/portfolio_config.yaml") as f:
        return yaml.safe_load(f)


def load_current_positions() -> dict:
    f = Path("data/positions/current.json")
    return json.load(open(f)) if f.exists() else {}


def save_results(
    result:    dict,
    scenarios: dict,
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

    print("  ✓ Resultados guardados")
    return positions


def get_macro_context(macro_data: dict) -> str:
    """Genera contexto macro con datos reales."""
    market_lines = []
    for label, d in macro_data.items():
        price = d.get("price",  0)
        r5d   = d.get("ret_5d", 0)
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
        prompt,
        task="macro",
        system=system,
        max_tokens=300,
    )


def _generate_commentary(
    result:        dict,
    macro_context: str,
    perf_metrics:  dict,
) -> str:
    weights  = result["weights"]
    added    = result["added_names"]
    dropped  = result["dropped_names"]
    port_ret = perf_metrics.get(
        "portfolio_return_pct", 0.0
    )
    spy_ret  = perf_metrics.get("spy_return_pct", 0.0)
    alpha    = perf_metrics.get("alpha_pct",       0.0)

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

    added_str   = str(added)
    dropped_str = str(dropped)
    to_val      = result["turnover_used"]
    ev_val      = result["expected_return"]
    macro_s     = macro_context[:200]
    port_s      = f"{port_ret:+.2f}%"
    spy_s       = f"{spy_ret:+.2f}%"
    alpha_s     = f"{alpha:+.2f}%"

    prompt = (
        "Commentary del rebalanceo semanal. "
        "Máximo 300 palabras. Primera persona.\n\n"
        f"Portfolio:\n{lines}\n\n"
        f"Cambios: +{added_str} -{dropped_str}\n"
        f"Turnover: {to_val:.1%}\n"
        f"EV 12M: {ev_val:.1%}\n\n"
        f"Performance real:\n"
        f"  Portfolio: {port_s}\n"
        f"  SPY mismo periodo: {spy_s}\n"
        f"  Alpha: {alpha_s}\n\n"
        f"Macro: {macro_s}\n\n"
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
            prompt,
            task="commentary",
            system=system,
            max_tokens=500,
        )
    except Exception as e:
        return f"Error generando commentary: {e}"


# ── Main ──────────────────────────────────────────────────────────────────────

def run_rebalance(
    force_universe_update: bool = False,
) -> dict:
    start_time = time.time()

    print(f"\n{'='*60}")
    print(
        "REBALANCEO PORTFOLIO — "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    print(f"{'='*60}\n")

    # Verificar credenciales LLM
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

    # Limpiar caché antigua
    cache.cleanup_old(keep_days=2)

    config            = load_config()
    current_positions = load_current_positions()
    print(
        f"Posiciones actuales: "
        f"{len(current_positions)}\n"
    )

    # ── PASO 1: Universo Russell 1000 ─────────────────────
    print("=" * 50)
    print("PASO 1: Universo Russell 1000")
    print("=" * 50)

    if force_universe_update:
        universe_tickers = update_universe()
    else:
        universe_tickers = load_universe()

    print(
        f"  Total: {len(universe_tickers)} tickers\n"
    )

    # ── PASO 2: Datos macro ───────────────────────────────
    print("=" * 50)
    print("PASO 2: Contexto macro")
    print("=" * 50)

    print("  Descargando datos de mercado reales...")
    macro_data = fetch_macro_data()
    indicators = ", ".join(macro_data.keys())
    print(
        f"  ✓ {len(macro_data)} indicadores: "
        f"{indicators}"
    )
    macro_context = get_macro_context(macro_data)
    print("  ✓ Contexto macro generado\n")

    # ── PASO 3: Histórico de precios ──────────────────────
    print("=" * 50)
    print("PASO 3: Histórico de precios (batch)")
    print("=" * 50)

    price_history = fetch_price_history(
        universe_tickers,
        period="1y",
        chunk_size=100,
    )
    print()

    # ── PASO 4: Fundamentales en paralelo ─────────────────
    print("=" * 50)
    print("PASO 4: Fundamentales (paralelo)")
    print("=" * 50)

    fundamentals = fetch_fundamentals_parallel(
        universe_tickers,
        max_workers=8,
    )
    print()

    # ── PASO 5: Pre-filtro cuantitativo ───────────────────
    print("=" * 50)
    print("PASO 5: Pre-filtro cuantitativo")
    print("=" * 50)

    prescreen_n = config.get(
        "screening", {}
    ).get("prescreen_top_n", 100)

    candidates, no_data_tickers = prescreening(
        fundamentals,
        price_history,
        top_n=prescreen_n,
    )

    n_cand   = len(candidates)
    n_nodata = len(no_data_tickers)
    print(
        f"  {n_cand} candidatos | "
        f"{n_nodata} sin datos\n"
    )

    # ── PASO 6: Scoring LLM en batches ────────────────────
    print("=" * 50)
    print("PASO 6: Scoring LLM (batches)")
    print("=" * 50)

    batch_size = config.get(
        "screening", {}
    ).get("llm_batch_size", 8)

    stocks_to_score = [
        {**fundamentals[t], "ticker": t}
        for t in candidates
        if t in fundamentals
    ]

    scored = score_batch(
        stocks_to_score,
        macro_context,
        batch_size=batch_size,
    )
    scored.sort(
        key=lambda x: x.get("composite_score", 0),
        reverse=True,
    )

    max_pos    = config["portfolio"]["max_positions"]
    top_n      = max_pos * 2
    top_scored = scored[:top_n]

    print(
        f"  ✓ Top {len(top_scored)} candidatos "
        f"finales\n"
    )

    # ── PASO 7: Escenarios detallados ─────────────────────
    print("=" * 50)
    print("PASO 7: Construyendo escenarios")
    print("=" * 50)

    scenarios:  dict = {}
    total_sc         = len(top_scored)

    for i, s in enumerate(top_scored):
        ticker    = s["ticker"]
        fund_data = fundamentals.get(ticker, {})
        merged    = {**fund_data, **s}
        print(f"  [{i+1}/{total_sc}] {ticker}...")
        scenarios[ticker] = build_scenario(
            ticker,
            merged,
            macro_context,
        )

    print(f"  ✓ {len(scenarios)} escenarios\n")

    # ── PASO 8: Optimización ──────────────────────────────
    print("=" * 50)
    print("PASO 8: Optimizando portfolio")
    print("=" * 50)

    current_weights = {
        t: p["weight"]
        for t, p in current_positions.items()
    }

    result = optimize_portfolio(
        list(scenarios.values()),
        current_weights,
        config,
    )

    n_pos    = len(result["weights"])
    turnover = result["turnover_used"]
    ev_port  = result["expected_return"]

    print(
        f"  ✓ {n_pos} posiciones | "
        f"Turnover {turnover:.1%} | "
        f"EV {ev_port:.1%}\n"
    )

    # ── PASO 8b: Enriquecimiento PortfolioLabs ────────────
    use_pl = (
        os.getenv("USE_PORTFOLIOLABS", "false")
        .lower() == "true"
    )

    if use_pl:
        print("=" * 50)
        print("PASO 8b: Enriquecimiento PortfolioLabs")
        print("=" * 50)
        try:
            from src.portfoliolabs import (
                enrich_with_portfoliolabs,
            )
            portfolio_tickers = list(
                result["weights"].keys()
            )
            fundamentals = enrich_with_portfoliolabs(
                portfolio_tickers,
                fundamentals,
            )
            for t in portfolio_tickers:
                divs = fundamentals.get(t, {}).get(
                    "_pl_divergences", {}
                )
                if divs:
                    print(
                        f"  ⚠ {t} divergencias: "
                        f"{list(divs.keys())}"
                    )
        except Exception as e:
            print(
                f"  Error PortfolioLabs: {e}. "
                f"Continuando sin enriquecimiento."
            )
        print()
    else:
        print(
            "  (PortfolioLabs desactivado. "
            "Activar con USE_PORTFOLIOLABS=true)\n"
        )

    # ── PASO 9: Registrar operaciones y P&L ───────────────
    print("=" * 50)
    print("PASO 9: Registrando operaciones")
    print("=" * 50)

    new_trades = record_trades(
        result,
        scenarios,
        current_positions,
    )
    print(
        f"  ✓ {len(new_trades)} operaciones "
        f"registradas\n"
    )

    # ── PASO 10: Guardar posiciones ───────────────────────
    print("=" * 50)
    print("PASO 10: Guardando posiciones")
    print("=" * 50)

    positions = save_results(result, scenarios)
    print()

    # ── PASO 11: Performance vs SPY ───────────────────────
    print("=" * 50)
    print("PASO 11: Performance vs SPY")
    print("=" * 50)

    perf_metrics = compute_performance_metrics(
        positions,
        scenarios,
    )
    update_performance(result, positions, scenarios)

    port_ret   = perf_metrics.get(
        "portfolio_return_pct", 0.0
    )
    spy_ret    = perf_metrics.get("spy_return_pct", 0.0)
    alpha      = perf_metrics.get("alpha_pct",       0.0)
    port_sign  = "+" if port_ret >= 0 else ""
    spy_sign   = "+" if spy_ret  >= 0 else ""
    alpha_sign = "+" if alpha    >= 0 else ""

    print(
        f"  Portfolio: {port_sign}{port_ret:.2f}% | "
        f"SPY: {spy_sign}{spy_ret:.2f}% | "
        f"Alpha: {alpha_sign}{alpha:.2f}%\n"
    )

    # ── PASO 12: Tesis ────────────────────────────────────
    print("=" * 50)
    print("PASO 12: Generando tesis")
    print("=" * 50)

    all_thesis = []
    min_change = config["turnover"]["min_position_change"]

    for ticker, weight in result["weights"].items():
        old_w  = current_weights.get(ticker, 0.0)
        diff   = weight - old_w

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
                macro_context,
            )
            all_thesis.append(thesis)
            accion_map = {
                "OPEN": "ABRIR",
                "ADD":  "AÑADIR",
                "TRIM": "REDUCIR",
            }
            accion = accion_map.get(action, action)
            print(f"  ✓ {ticker} [{accion}]")

    for ticker in result["dropped_names"]:
        if ticker in scenarios:
            thesis = generate_thesis(
                ticker,
                scenarios[ticker],
                0.0,
                "CLOSE",
                macro_context,
            )
            all_thesis.append(thesis)
            print(f"  ✓ {ticker} [CERRAR]")

    print()

    # ── PASO 13: Commentary ───────────────────────────────
    print("=" * 50)
    print("PASO 13: Generando commentary")
    print("=" * 50)

    summary = _generate_commentary(
        result,
        macro_context,
        perf_metrics,
    )
    result["commentary"] = summary

    today   = datetime.now().strftime("%Y-%m-%d")
    rb_file = Path(
        f"data/rebalances/{today}_rebalance.json"
    )
    if rb_file.exists():
        try:
            rb = json.load(open(rb_file))
            rb["commentary"] = summary
            with open(rb_file, "w") as f:
                json.dump(rb, f, indent=2)
        except Exception as e:
            print(
                f"  ⚠ No se pudo actualizar "
                f"rebalance.json: {e}"
            )

    print("  ✓ Commentary generado\n")

    # ── PASO 14: Email report ─────────────────────────────
    print("=" * 50)
    print("PASO 14: Generando y enviando email report")
    print("=" * 50)

    generate_email_report(
        result          = result,
        all_thesis      = all_thesis,
        summary         = summary,
        positions       = positions,
        perf_metrics    = perf_metrics,
        no_data_tickers = no_data_tickers,
        new_trades      = new_trades,
    )

    email_sent = send_email_report()
    if not email_sent:
        print(
            "  El report está guardado en "
            "data/email_report.json"
        )
    print()

    # ── Resumen final ─────────────────────────────────────
    elapsed = time.time() - start_time
    mins    = int(elapsed // 60)
    secs    = int(elapsed  % 60)

    print(f"\n{'='*60}")
    print(
        f"✅ Rebalanceo completado en "
        f"{mins}m {secs}s"
    )
    print(f"{'='*60}")

    print("\n📊 Llamadas API:")
    for model, count in sorted(
        request_counts.items(),
        key=lambda x: -x[1],
    ):
        print(f"   {model}: {count}")

    print(f"\n{summary}\n")
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sistema de rebalanceo de portfolio"
    )
    parser.add_argument(
        "--update-universe",
        action="store_true",
        help=(
            "Forzar actualización del universo "
            "Russell 1000 desde Wikipedia"
        ),
    )
    args = parser.parse_args()

    run_rebalance(
        force_universe_update=args.update_universe,
    )
