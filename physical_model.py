"""
physical_model.py — parametric, vol-conditioned probability model.

Replaces the sparse 2D empirical table (strategy_v2._EMPIRICAL_TABLE) with a
smooth, monotone, continuous-in-time estimate:

    P(Up) = logistic( gamma + beta * z ),   z = delta / (sigma * sqrt(tau))

where
    delta = (price_now - window_open) / window_open   (signed fraction)
    tau   = seconds remaining in the window
    sigma = ambient per-second realized volatility (see vol_horizon)

`z` is the standardized distance of the current move from the open, scaled by
the volatility expected over the remaining time. This single statistic encodes
both the size of the move and the time/vol context, so two coefficients
(gamma, beta) generalize across every horizon — fixing the table's coarse
4-row time grid, its sub-180s blind spot, and its noisy extreme cells.

Volatility input
----------------
The move is standardized by an *ambient* realized vol (default a trailing
~15-min window), NOT a 30-second micro-vol. The ambient vol is stable, matches
the multi-minute horizon being predicted, and is computable consistently from
both 1-min kline history (fitting) and the live 1-second price feed (serving).
The horizon used is recorded in the coeffs file as `vol_horizon`.

This module is intentionally dependency-light at import time (numpy only for
fitting). `prob_up` / `z_score` are pure-Python so the strategy can call them
without importing the fitter.
"""

import json
import math

DEFAULT_CLAMP = (0.02, 0.98)

# Maps a coeffs `vol_horizon` label to the live calib-log field that carries
# the matching ambient vol, and to the price_feed lookback (seconds) used live.
VOL_HORIZONS = {
    "5m": {"log_field": "vol_5m", "lookback_s": 300, "lookback_min": 5},
    "15m": {"log_field": "vol_15m", "lookback_s": 900, "lookback_min": 15},
    "60m": {"log_field": "vol_60m", "lookback_s": 3600, "lookback_min": 60},
}


def z_score(delta, secs_remaining, vol):
    """
    Standardized distance to the barrier. Returns None when it is undefined
    (non-positive vol or time), so callers can fall back to 0.5.

    `delta` is a signed fraction; `vol` is per-second return std.
    """
    if vol is None or vol <= 0 or secs_remaining is None or secs_remaining <= 0:
        return None
    return delta / (vol * math.sqrt(secs_remaining))


def _logistic(eta):
    # numerically stable logistic
    if eta >= 0:
        return 1.0 / (1.0 + math.exp(-eta))
    e = math.exp(eta)
    return e / (1.0 + e)


def prob_up(delta, secs_remaining, vol, coeffs):
    """
    P(Up) from the fitted model. Falls back to 0.5 when z is undefined.
    `coeffs` is the dict produced by `fit` / loaded by `load_coeffs`.
    """
    z = z_score(delta, secs_remaining, vol)
    if z is None:
        return 0.5
    gamma, beta = coeffs["beta"][0], coeffs["beta"][1]
    p = _logistic(gamma + beta * z)
    lo, hi = coeffs.get("clamp", DEFAULT_CLAMP)
    return min(hi, max(lo, p))


def fit(z_values, outcomes, clamp=DEFAULT_CLAMP, iters=100, ridge=1e-6):
    """
    Fit logistic(gamma + beta*z) by Newton-Raphson / IRLS (numpy).

    z_values : sequence of floats (the standardized moves)
    outcomes : sequence of 0/1 (1 == Up won)
    Returns a coeffs dict (without metadata; caller adds vol_horizon etc.).
    """
    import numpy as np

    z = np.asarray(z_values, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    X = np.column_stack([np.ones_like(z), z])  # [intercept, z]

    beta = np.zeros(2)
    for _ in range(iters):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        grad = X.T @ (y - p)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(2)
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break

    return {
        "model": "logistic_z",
        "beta": [float(beta[0]), float(beta[1])],
        "features": ["intercept", "z"],
        "clamp": list(clamp),
    }


def save_coeffs(coeffs, path):
    with open(path, "w") as f:
        json.dump(coeffs, f, indent=2)


def load_coeffs(path):
    """Load a coeffs dict, or None if missing/malformed."""
    try:
        with open(path) as f:
            c = json.load(f)
        if "beta" not in c or len(c["beta"]) < 2:
            return None
        return c
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return None
