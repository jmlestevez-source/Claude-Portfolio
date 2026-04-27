# src/optimizer.py
"""
Optimización de portfolio con scipy SLSQP.
Sin cambios lógicos, mejor estructurado.
"""

import numpy as np
from scipy.optimize import minimize


def optimize_portfolio(
    scenarios: list[dict],
    current_weights: dict[str, float],
    config: dict,
) -> dict:
    max_pos = config["portfolio"]["max_positions"]
    max_w   = config["portfolio"]["max_position_size"]
    min_w   = config["portfolio"]["min_position_size"]
    max_to  = config["turnover"]["max_one_sided_turnover"]

    candidates = [
        s for s in scenarios
        if s.get("ev_12m", 0) > s.get("current_price", 0)
        and s.get("upside_downside_ratio", 0) > 0
        and s.get("current_price", 0) > 0
    ]
    candidates.sort(
        key=lambda s: s["upside_downside_ratio"],
        reverse=True,
    )
    candidates = candidates[: max_pos * 2]

    if not candidates:
        print("  Sin candidatos válidos, manteniendo posiciones")
        return {
            "weights":              current_weights,
            "expected_return":      0,
            "risk_score":           0,
            "risk_adjusted_return": 0,
            "turnover_used":        0,
            "added_names":          [],
            "dropped_names":        [],
        }

    tickers  = [s["ticker"] for s in candidates]
    n        = len(tickers)

    ev_ret = np.array([
        (s["ev_12m"] - s["current_price"])
        / s["current_price"]
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

    def turnover_constraint(w):
        buys   = np.sum(np.maximum(w - current_w, 0))
        closed = sum(
            v for t, v in current_weights.items()
            if t not in tickers
        )
        return max_to - buys - closed

    x0 = np.full(n, 1.0 / min(max_pos, n))
    x0 = np.clip(x0, 0, max_w)
    x0 /= x0.sum()

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0, max_w)] * n,
        constraints=[
            {"type": "eq",   "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": turnover_constraint},
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
        w * (sm[t]["ev_12m"] - sm[t]["current_price"])
        / sm[t]["current_price"]
        for t, w in fw.items()
        if sm.get(t) and sm[t]["current_price"] > 0
    )
    port_risk = sum(
        w * abs(sm[t]["bear_case_downside_12m"])
        for t, w in fw.items()
        if sm.get(t)
    )
    all_t    = set(list(fw.keys()) + list(current_weights.keys()))
    turnover = sum(
        max(fw.get(t, 0) - current_weights.get(t, 0), 0)
        for t in all_t
    )

    return {
        "weights":              fw,
        "expected_return":      port_ev,
        "risk_score":           port_risk,
        "risk_adjusted_return": port_ev / (port_risk + 0.001),
        "turnover_used":        turnover,
        "added_names":  [t for t in fw if t not in current_weights],
        "dropped_names": [t for t in current_weights if t not in fw],
    }
