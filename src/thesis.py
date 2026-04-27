# src/thesis.py
"""
Generación de tesis de posición (sin cambios lógicos).
"""

import json
from pathlib import Path
from datetime import datetime
from src.llm import call_llm


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
    pb       = scenario.get("prob_bull", 0)
    pba      = scenario.get("prob_base", 0)
    pbe      = scenario.get("prob_bear", 0)
    t12      = scenario.get("targets_12m", {})
    kill     = scenario.get("kill_condition", "")
    cat      = scenario.get("key_catalyst", "")

    accion_es = {
        "OPEN": "ABRIR", "ADD": "AÑADIR",
        "TRIM": "REDUCIR", "CLOSE": "CERRAR",
        "HOLD": "MANTENER",
    }.get(action, action)

    system = (
        "Eres un gestor de portfolio profesional. "
        "Escribe en primera persona, en español. "
        "Directo, cuantitativo, sin lenguaje vago. "
        "Cada afirmación debe ser falsable. "
        "No inventes fechas ni datos."
    )

    prompt = (
        f"Tesis {ticker} | {accion_es} | {weight:.1%}\n\n"
        f"Precio: ${price:.2f} | "
        f"VE 12M: ${ev_12m:.2f} ({ev_pct:+.1f}%) | "
        f"Bajista: ${bear_12m:.2f} ({bear_pct:.1f}%) | "
        f"Ratio U/D: {ratio:.2f}x\n\n"
        f"Bull ({pb:.0%}): {scenario.get('bull_thesis','')}\n"
        f"Base ({pba:.0%}): {scenario.get('base_thesis','')}\n"
        f"Bear ({pbe:.0%}): {scenario.get('bear_thesis','')}\n\n"
        f"Kill condition: {kill}\n"
        f"Catalizador: {cat}\n"
        f"Macro: {macro_summary[:150]}\n\n"
        "Formato:\n"
        "---\n"
        f"TESIS: {ticker} | {accion_es} | "
        f"{weight:.1%} | {ts}\n"
        "---\n"
        "**Oportunidad:** [2 frases con datos]\n\n"
        f"**Bull ({pb:.0%}):** "
        f"[drivers. Objetivo ${t12.get('bull',0):.2f}]\n\n"
        f"**Base ({pba:.0%}):** "
        f"[ejecución. Objetivo ${t12.get('base',0):.2f}]\n\n"
        f"**Bear ({pbe:.0%}):** "
        f"[riesgos. Objetivo ${bear_12m:.2f}]\n\n"
        f"**VE:** ${ev_12m:.2f} ({ev_pct:+.1f}%) vs "
        f"bajista {bear_pct:.1f}%. Ratio {ratio:.2f}x\n\n"
        f"**Sizing {weight:.1%}:** "
        f"[relación ratio U/D={ratio:.2f}x y "
        f"bear=${bear_12m:.2f}]\n\n"
        f"**Kill condition:** {kill}\n\n"
        "**Checkpoint:** [evento real próximo]\n"
        "---"
    )

    try:
        thesis_text = call_llm(
            prompt, task="thesis", system=system,
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
    fname = (
        f"{ts[:10]}_{ticker}_{action}.json"
    )
    with open(
        f"data/thesis/{fname}", "w", encoding="utf-8"
    ) as f:
        json.dump(thesis, f, indent=2, ensure_ascii=False)

    return thesis
