"""
calib_log.py — per-tick calibration logger: model P(win) vs the live book.

Records, on a regular cadence within each 15-min window, the model's estimated
P(Up) alongside the book-implied P(Up) (de-vigged from the order book), then
stamps the realized outcome at resolution and appends one JSON line per tick.

The resulting `calibration_log.jsonl` is consumed by `brier_eval.py`, which
scores the model against the book (Brier / log loss / skill score). This is a
read-only research harness: it never places, cancels, or alters any order.

Book-implied probability:
    Both legs of a binary market carry the market's own probability estimate,
    but each is inflated by the spread/vig. We de-vig by normalising the pair:
        p_book_up = price_up / (price_up + price_down)
    Preferred source is the CLOB order-book mid for each leg; we fall back to
    the Gamma `outcomePrices` when a CLOB book is unavailable.
"""

import json
import time
import logging
from datetime import datetime, timezone

from strategy_v2 import _estimate_prob_from_delta
from executor import get_clob_mid

log = logging.getLogger("calib_log")

VOL_LOOKBACK = 30  # match LateScalpStrategy / EarlyMomentumStrategy vol lookback


def _devig(p_up, p_down):
    """Normalise a two-leg price pair into an implied P(Up). None if unusable."""
    if p_up is None or p_down is None:
        return None
    total = p_up + p_down
    if total <= 0:
        return None
    return p_up / total


def build_tick(window, price_feed, market, client, capture_clob=True):
    """
    Build one calibration record from the current market snapshot.

    Captures model P(Up) and book-implied P(Up) plus the context needed to
    slice calibration later (time remaining, delta, realised vol, regime).
    Returns a dict, or None if neither a CLOB nor a Gamma book reference is
    available (without a book there is nothing to score against).
    """
    up = market.get("Up")
    down = market.get("Down")
    if not up or not down:
        return None

    # Real Gamma `outcomePrices`. Note: bot._fetch_market_safe overwrites `price`
    # with the best ask and preserves the original under `gamma_price`, so prefer
    # that key when present (raw fetch_market has only `price`).
    gamma_up = up.get("gamma_price")
    if gamma_up is None:
        gamma_up = up.get("price")
    gamma_down = down.get("gamma_price")
    if gamma_down is None:
        gamma_down = down.get("price")

    delta = price_feed.get_window_delta()  # fraction, e.g. 0.0012 == 0.12%
    vol = price_feed.get_volatility(lookback=VOL_LOOKBACK)
    # Ambient realized vols over longer horizons — these are the inputs the
    # parametric model (physical_model.py) standardizes by. Computed locally
    # from the price-history deque, so no extra API calls. Long horizons read
    # 0.0 until enough feed uptime has accumulated.
    vm = price_feed.get_vol_multi()  # {"5min","15min","60min"} per-second std
    secs_remaining = window.window_end - int(time.time())

    # Model estimate — uses strategy_v2's live (recalibrated) empirical table.
    p_model_up = _estimate_prob_from_delta(delta, secs_remaining, vol)

    clob_mid_up = clob_mid_down = None
    if capture_clob and client is not None:
        clob_mid_up = get_clob_mid(client, up.get("token_id"))
        clob_mid_down = get_clob_mid(client, down.get("token_id"))

    p_book_up = _devig(clob_mid_up, clob_mid_down)
    book_src = "clob"
    if p_book_up is None:
        p_book_up = _devig(gamma_up, gamma_down)
        book_src = "gamma"
    if p_book_up is None:
        return None

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "window": window.window_start,
        "secs_remaining": secs_remaining,
        "delta": round(delta, 6),
        "delta_pct": round(delta * 100, 4),
        "vol": round(vol, 8),
        "vol_5m": round(vm.get("5min", 0.0), 8),
        "vol_15m": round(vm.get("15min", 0.0), 8),
        "vol_60m": round(vm.get("60min", 0.0), 8),
        "vol_regime": price_feed.get_vol_regime(),
        "btc_price": price_feed.current_price,
        "btc_open": window.btc_open_price,
        "p_model_up": round(p_model_up, 4),
        "p_book_up": round(p_book_up, 4),
        "book_src": book_src,
        "gamma_up": gamma_up,
        "gamma_down": gamma_down,
        "clob_mid_up": clob_mid_up,
        "clob_mid_down": clob_mid_down,
    }


def flush_window(window, winning_side, path):
    """
    Stamp every buffered tick of `window` with the realized outcome and append
    them to `path` as JSON lines. Returns the number of rows written.

    Caller is responsible for skipping flat windows (no clear winner).
    """
    ticks = getattr(window, "calib_ticks", None)
    if not ticks:
        return 0

    outcome_up = winning_side == "Up"
    lines = []
    for tick in ticks:
        row = dict(tick)
        row["winning_side"] = winning_side
        row["outcome_up"] = outcome_up
        lines.append(json.dumps(row))

    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")

    return len(lines)
