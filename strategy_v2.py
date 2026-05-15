# strategy_v2.py — Polymarket BTC 15-min strategy (MM removed)
#
# STRATEGIES:
# ─────────────────────────────────────────────────────────
# A) Early Momentum (T-780 to T-480):
#    Directional GTC maker bet when BTC has moved >0.3% from the
#    window open in the first ~5 min of Phase 2. Uses volatility-
#    normalised z-score to estimate P(win) and eighth-Kelly sizing.
#    Fires at most once per window.
#
# B) Fade Extreme Odds (T-780 to T-30):
#    Buy the cheap side when market price is extreme (>0.85) AND
#    the move looks like a spike (5s vol >> 60s vol). GTC maker order.
#    Phase 4 (T-180 to T-30): primary strategy when Δ has run hard.
#
# C) Mid-Window Scalp (T-780 to T-30):
#    Single-shot GTC maker bet after checking order book depth.
#    Phase 2 (T-780 to T-180): main directional entries.
#    Phase 4 (T-180 to T-30): tighter thresholds, skips low-Δ entries.
#    Fires at most once per window.
#
# Phase timeline (per 20260511_strategy.md):
#   Phase 1 (T-900→T-780): fetch market, skip noise, confirm liquidity
#   Phase 2 (T-780→T-180): main entry window — GTC maker-first
#   Phase 3 (after fill):  manage positions, exit on inverted edge
#   Phase 4 (T-180→T-30):  tighter thresholds, FADE focus
#   No new entries after T-30
#
# WHY MM WAS REMOVED:
# ─────────────────────────────────────────────────────────
# In live (non dry-run) mode, Polymarket's CLOB filled maker bids
# instantly as taker orders, spending USDC with no way to cancel
# ("matched orders can't be canceled"). The cancel-after-fill
# approach that looked profitable in dry-run is not replicable
# live because the CLOB immediately matches aggressive bids.
# ─────────────────────────────────────────────────────────

import logging
from dataclasses import dataclass
import time as _time

log = logging.getLogger("strategy_v2")

MAX_BUY_PRICE = 0.85  # strategy doc: "don't buy above e.g. 0.85"

# Phase boundaries in seconds remaining
_PHASE1_END = 780   # T-13:00 — end of setup-only phase
_PHASE2_END = 180   # T-3:00  — switch to closing-window rules
_PHASE4_END = 30    # T-0:30  — no new entries after this


@dataclass
class OpenPosition:
    """Tracks a filled position for Phase 3 management."""
    side: str
    token_id: str
    entry_price: float
    shares: float
    prob_at_entry: float
    entry_time: float   # unix timestamp
    strategy: str


# ═══════════════════════════════════════════════════════════
# STRATEGY A: Early Momentum
# ═══════════════════════════════════════════════════════════
#
# When BTC moves strongly early in Phase 2 (T-780 to T-480),
# the Polymarket price often lags. We place a GTC maker bid in
# the direction of the move once delta clears a 0.3% threshold.
# The probability model normalises the delta by remaining
# volatility (z-score → normal CDF), so it automatically
# becomes less aggressive when there is still a lot of time left.

class EarlyMomentumStrategy:
    """
    Directional GTC maker bet when delta is large during T-780 to T-480.
    """

    def __init__(
        self,
        min_delta_pct: float = 0.30,      # need ≥0.30% BTC move
        kelly_fraction: float = 0.125,     # eighth-Kelly
        min_edge: float = 0.05,            # need 5% net edge after fees
        min_bet: float = 2.50,             # GTC min: 5 shares × $0.50 = $2.50
        min_shares: int = 5,               # Polymarket CLOB GTC minimum
        max_bet_pct: float = 0.10,         # max 10% of bankroll
        entry_start: int = 780,            # start evaluating at T-780
        entry_end: int = 480,              # stop at T-480 (scalp/fade take over)
    ):
        self.min_delta_pct = min_delta_pct
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.min_bet = min_bet
        self.min_shares = min_shares
        self.max_bet_pct = max_bet_pct
        self.entry_start = entry_start
        self.entry_end = entry_end

    def evaluate(self, market, bankroll, price_feed, seconds_remaining):
        """Returns a trade dict or None."""
        if seconds_remaining > self.entry_start or seconds_remaining < self.entry_end:
            return None

        delta = price_feed.get_window_delta()
        delta_pct = abs(delta) * 100

        if delta_pct < self.min_delta_pct:
            return None

        vol = price_feed.get_volatility(lookback=30)
        prob_up = _estimate_prob_from_delta(delta, seconds_remaining, vol)

        if delta > 0:
            side = "Up"
            prob_win = prob_up
            market_price = market["Up"]["price"]
        else:
            side = "Down"
            prob_win = 1.0 - prob_up
            market_price = market["Down"]["price"]

        if market_price > MAX_BUY_PRICE:
            return None

        edge = prob_win - market_price
        taker_fee = 4 * market_price * (1 - market_price) * 0.0156
        net_edge = edge - taker_fee

        if net_edge < self.min_edge:
            return None

        b = (1.0 - market_price) / market_price
        p = prob_win
        q = 1.0 - p
        f_star = (b * p - q) / b
        if f_star <= 0:
            return None

        kelly_bet = f_star * self.kelly_fraction
        bet_amount = kelly_bet * bankroll
        bet_amount = min(bet_amount, bankroll * self.max_bet_pct)
        bet_amount = max(self.min_bet, bet_amount)

        # Enforce 5-share minimum for GTC orders
        shares = max(self.min_shares, round(bet_amount / market_price, 1))
        bet_amount = round(shares * market_price, 2)

        # Max price we'll accept — keeps at least min_edge positive EV
        price_buffer = 0.04 if edge >= self.min_edge * 2 else 0.03
        max_price = round(min(prob_win - self.min_edge + price_buffer, 0.95), 2)

        log.info(
            f"MOMENTUM | {side} @ ${market_price:.2f} | "
            f"Delta: {delta*100:+.3f}% | Vol: {vol*100:.4f}% | "
            f"P(win): {prob_win:.2f} | Edge: {net_edge:.3f} | "
            f"Shares: {shares} | Bet: ${bet_amount:.2f}"
        )

        return {
            "side": side,
            "token_id": market[side]["token_id"],
            "outcome_index": market[side]["outcome_index"],
            "price": market_price,
            "maker_price": market_price,   # GTC limit bid at current ask
            "max_price": max_price,
            "bet_amount": bet_amount,
            "shares": shares,
            "edge": round(net_edge, 4),
            "kelly_pct": round(kelly_bet, 4),
            "estimated_prob": round(prob_win, 4),
            "use_maker": True,
            "strategy": "momentum",
        }


# ═══════════════════════════════════════════════════════════
# STRATEGY B: Fade Extreme Odds (opportunistic)
# ═══════════════════════════════════════════════════════════
#
# Sometimes the market overshoots — a big BTC spike pushes
# Up to 0.90+ but the spike is from a single large trade
# that's likely to mean-revert. Betting the opposite side at
# extreme odds has favorable risk/reward:
#   Buy Down @ $0.08 → 12.5:1 payout if it reverts
#   Need only ~10% reversion rate to break even
#
# In Phase 4 (T-180 to T-30), FADE becomes the primary strategy:
# when Δ has run hard with <2 min left, overshoots tend to
# mean-revert into resolution.
#
# Filter: only fade when the delta is driven by a spike (high
# instantaneous volatility) rather than a steady drift.

class FadeExtremeStrategy:
    """
    Buy the cheap side when market odds are extreme (>0.85)
    and the move looks like a spike rather than a drift.
    Active from T-780 to T-30 (primary strategy in Phase 4).
    """

    def __init__(
        self,
        extreme_threshold: float = 0.85,  # market price > this = extreme
        max_bet_pct: float = 0.03,         # tiny bets — these are longshots
        min_bet: float = 2.50,             # GTC min: 5 shares × $0.50 = $2.50
        min_shares: int = 5,               # Polymarket CLOB GTC minimum
        spike_vol_ratio: float = 3.0,      # current vol must be 3x avg
    ):
        self.extreme_threshold = extreme_threshold
        self.max_bet_pct = max_bet_pct
        self.min_bet = min_bet
        self.min_shares = min_shares
        self.spike_vol_ratio = spike_vol_ratio

    def evaluate(self, market, bankroll, price_feed, seconds_remaining):
        """Bet against spikes when odds are extreme. Active T-780 to T-30."""
        if seconds_remaining < 30 or seconds_remaining > 780:
            return None

        up_price = market["Up"]["price"]
        down_price = market["Down"]["price"]

        # Detect extreme
        if up_price > self.extreme_threshold:
            cheap_side = "Down"
            cheap_price = down_price
        elif down_price > self.extreme_threshold:
            cheap_side = "Up"
            cheap_price = up_price
        else:
            return None

        if cheap_price <= 0.01:
            log.warning(
                f"FADE | {cheap_side} price ${cheap_price:.4f} ≤ $0.01 — "
                f"market data corrupt, skipping"
            )
            return None

        # Check if this is a spike (high recent vol vs avg)
        vol_5s = price_feed.get_volatility(lookback=5)
        vol_60s = price_feed.get_volatility(lookback=60)

        if vol_60s <= 0:
            return None

        spike_ratio = vol_5s / vol_60s
        if spike_ratio < self.spike_vol_ratio:
            # Not a spike — steady drift, don't fade it
            return None

        # Small fixed-size bet (not Kelly — this is a speculative play)
        bet_amount = min(bankroll * self.max_bet_pct, 3.0)
        bet_amount = max(self.min_bet, bet_amount)

        if bet_amount > bankroll * 0.10:
            return None  # don't risk too much on longshots

        # Enforce 5-share minimum for GTC orders
        shares = max(self.min_shares, round(bet_amount / cheap_price, 1))
        bet_amount = round(shares * cheap_price, 2)

        log.info(
            f"FADE | {cheap_side} @ ${cheap_price:.2f} | "
            f"Spike ratio: {spike_ratio:.1f}x | "
            f"Shares: {shares} | Bet: ${bet_amount:.2f}"
        )

        return {
            "side": cheap_side,
            "token_id": market[cheap_side]["token_id"],
            "outcome_index": market[cheap_side]["outcome_index"],
            "price": cheap_price,
            "maker_price": cheap_price,    # GTC limit bid at current ask
            "bet_amount": bet_amount,
            "shares": shares,
            "edge": round(1.0 / cheap_price * 0.10 - 1.0, 4),  # rough EV
            "kelly_pct": 0.0,
            "estimated_prob": round(1.0 - self.extreme_threshold + 0.05, 4),
            "use_maker": True,
            "strategy": "fade",
        }


# ═══════════════════════════════════════════════════════════
# STRATEGY C: Mid-Window Scalp
# ═══════════════════════════════════════════════════════════
#
# Directional GTC maker bet when delta is significant.
# Phase 2 (T-780 to T-180): main use case.
# Phase 4 (T-180 to T-30): allowed with tighter edge gates
# applied by the orchestrator (≥0.07 edge, ≥0.25% delta).
# Fires at most once per window.

class LateScalpStrategy:
    """
    Directional GTC maker bet when delta is large.
    Caps the fill price at prob_win - min_edge to preserve positive EV.
    Skips if the order book has no asks (illiquid).
    """

    def __init__(
        self,
        min_delta_pct: float = 0.20,         # need ≥0.20% BTC move
        kelly_fraction: float = 0.125,        # eighth-Kelly
        min_edge: float = 0.05,               # need 5% edge minimum
        high_edge_threshold: float = 0.10,    # edge above this → 2% max_price buffer
        min_bet: float = 2.50,                # GTC min: 5 shares × $0.50 = $2.50
        min_shares: int = 5,                  # Polymarket CLOB minimum
        max_bet_pct: float = 0.10,            # max 10% of bankroll
        entry_window_seconds: int = 780,      # Phase 2 starts at T-780
    ):
        self.min_delta_pct = min_delta_pct
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.high_edge_threshold = high_edge_threshold
        self.min_bet = min_bet
        self.min_shares = min_shares
        self.max_bet_pct = max_bet_pct
        self.entry_window_seconds = entry_window_seconds

    def evaluate(self, market, bankroll, price_feed, seconds_remaining):
        """
        Returns a trade dict or None.
        Only triggers within `entry_window_seconds` and up to T-30.
        """
        if seconds_remaining > self.entry_window_seconds:
            return None

        if seconds_remaining < 30:
            return None  # too late, order won't fill

        delta = price_feed.get_window_delta()
        delta_pct = abs(delta) * 100

        if delta_pct < self.min_delta_pct:
            return None

        vol = price_feed.get_volatility(lookback=30)
        prob_up = _estimate_prob_from_delta(delta, seconds_remaining, vol)

        if delta > 0:
            side = "Up"
            prob_win = prob_up
            market_price = market["Up"]["price"]
        else:
            side = "Down"
            prob_win = 1.0 - prob_up
            market_price = market["Down"]["price"]

        if market_price > MAX_BUY_PRICE:
            return None

        edge = prob_win - market_price
        taker_fee = 4 * market_price * (1 - market_price) * 0.0156
        net_edge = edge - taker_fee

        if net_edge < self.min_edge:
            return None

        b = (1.0 - market_price) / market_price
        p = prob_win
        q = 1.0 - p
        f_star = (b * p - q) / b
        if f_star <= 0:
            return None

        kelly_bet = f_star * self.kelly_fraction
        bet_amount = kelly_bet * bankroll
        bet_amount = min(bet_amount, bankroll * self.max_bet_pct)
        bet_amount = max(self.min_bet, bet_amount)

        # Enforce 5-share minimum for GTC orders
        shares = max(self.min_shares, round(bet_amount / market_price, 1))
        bet_amount = round(shares * market_price, 2)

        # Max price we'll accept — keeps at least min_edge positive EV.
        # Buffer absorbs the ~1s submission latency where the best ask can move
        # 1-3 ticks above the cap before the GTC reaches the matching engine.
        price_buffer = 0.04 if edge >= self.high_edge_threshold else 0.03
        max_price = round(min(prob_win - self.min_edge + price_buffer, 0.95), 2)

        eval_token_id = market[side]["token_id"]
        log.info(
            f"SCALP | {side} @ ${market_price:.2f} (max ${max_price:.2f}) | "
            f"Delta: {delta*100:+.3f}% | Vol: {vol*100:.4f}% | "
            f"P(win): {prob_win:.2f} | Edge: {net_edge:.3f} | "
            f"Buffer: +{int(price_buffer*100)}% | "
            f"Shares: {shares} | Bet: ${bet_amount:.2f} | "
            f"token={eval_token_id}"
        )

        return {
            "side": side,
            "token_id": market[side]["token_id"],
            "outcome_index": market[side]["outcome_index"],
            "price": market_price,
            "max_price": max_price,
            "bet_amount": bet_amount,
            "shares": shares,
            "edge": round(net_edge, 4),
            "kelly_pct": round(kelly_bet, 4),
            "estimated_prob": round(prob_win, 4),
            "use_maker": True,
            "strategy": "scalp",
        }


# ═══════════════════════════════════════════════════════════
# Combined Orchestrator
# ═══════════════════════════════════════════════════════════

class CombinedStrategy:
    """
    Orchestrates all three strategies across the 15-min window lifecycle.

    Phase structure (per 20260511_strategy.md):
      Phase 1 (T-900→T-780): skip — no trades, liquidity confirmation window
      Phase 2 (T-780→T-180): main entry window (MOMENTUM → FADE → SCALP)
      Phase 3 (after any fill): position management via evaluate_exits()
      Phase 4 (T-180→T-30):   tighter thresholds, FADE is primary strategy
      No new entries after T-30

    The bot loop calls evaluate_phase() on every 5s tick and
    evaluate_exits() on every tick whenever open_positions is non-empty.
    Call track_fill(trade) immediately after confirming an order fill.
    """

    def __init__(self, dry_run: bool = True):  # dry_run kept for API compatibility
        self.momentum = EarlyMomentumStrategy()
        self.fade = FadeExtremeStrategy()
        self.scalp = LateScalpStrategy()
        self._momentum_fired = False
        self._scalp_fired = False
        self.open_positions: list[OpenPosition] = []

    def on_new_window(self):
        """Reset per-window state at the start of each 15-min window."""
        self._momentum_fired = False
        self._scalp_fired = False
        self.open_positions = []

    def is_liquid(self, market) -> bool:
        """
        Phase 1 liquidity check: confirm best bid/ask depth ≥ $50
        on at least one side before entering Phase 2.
        """
        for side in ("Up", "Down"):
            side_data = market.get(side, {})
            if (
                side_data.get("best_bid_size", 0) >= 50
                or side_data.get("best_ask_size", 0) >= 50
            ):
                return True
        return False

    def track_fill(self, trade: dict):
        """
        Phase 3: record a newly filled position for ongoing management.
        Call this immediately after the bot confirms an order fill.
        """
        self.open_positions.append(OpenPosition(
            side=trade["side"],
            token_id=trade["token_id"],
            entry_price=trade["price"],
            shares=trade["shares"],
            prob_at_entry=trade["estimated_prob"],
            entry_time=_time.time(),
            strategy=trade["strategy"],
        ))

    def evaluate_exits(self, market, price_feed, seconds_remaining) -> list[dict]:
        """
        Phase 3: re-evaluate all open positions on every tick.

        Returns a list of exit signals:
          {"token_id", "side", "shares", "reason", "current_price"}

        Triggers:
          "lock_profit"   — token trading at ≥0.85 (marginal EV from holding is tiny)
          "edge_inverted" — model P(win) < market price by >5% (likely wrong, cut loss)
        """
        if not self.open_positions:
            return []

        delta = price_feed.get_window_delta()
        vol = price_feed.get_volatility(lookback=30)
        prob_up = _estimate_prob_from_delta(delta, seconds_remaining, vol)

        exits = []
        for pos in self.open_positions:
            current_price = market[pos.side]["price"]

            if current_price >= 0.85:
                exits.append({
                    "token_id": pos.token_id,
                    "side": pos.side,
                    "shares": pos.shares,
                    "reason": "lock_profit",
                    "current_price": current_price,
                })
                continue

            prob_win = prob_up if pos.side == "Up" else 1.0 - prob_up
            if prob_win - current_price < -0.05:
                exits.append({
                    "token_id": pos.token_id,
                    "side": pos.side,
                    "shares": pos.shares,
                    "reason": "edge_inverted",
                    "current_price": current_price,
                })

        exit_ids = {e["token_id"] for e in exits}
        self.open_positions = [p for p in self.open_positions if p.token_id not in exit_ids]
        return exits

    def evaluate_phase(self, market, bankroll, price_feed, seconds_remaining):
        """
        Called on every 5s tick. Returns:
          ("momentum", trade)  — Phase 2 early directional bet
          ("fade", trade)      — opportunistic fade (Phase 2 or 4)
          ("scalp", trade)     — directional GTC bet (Phase 2 or 4)
          ("skip", None)       — do nothing this tick
        """
        # Phase 1 (T-900 → T-780): setup only, no entries
        if seconds_remaining > _PHASE1_END:
            return ("skip", None)

        # Past T-30: no new entries
        if seconds_remaining <= _PHASE4_END:
            return ("skip", None)

        # Phase 4 (T-180 → T-30): tighter thresholds, FADE focus
        if seconds_remaining <= _PHASE2_END:
            delta = price_feed.get_window_delta()
            delta_pct = abs(delta) * 100

            # Skip coinflips: small delta with <90s remaining
            if delta_pct < 0.10 and seconds_remaining < 90:
                return ("skip", None)

            # FADE is primary in Phase 4 — overshoots mean-revert into resolution
            trade = self.fade.evaluate(market, bankroll, price_feed, seconds_remaining)
            if trade and trade["edge"] >= 0.07:
                return ("fade", trade)

            # SCALP still allowed in Phase 4 on significant delta with tight edge
            if not self._scalp_fired and delta_pct >= 0.25:
                trade = self.scalp.evaluate(market, bankroll, price_feed, seconds_remaining)
                if trade and trade["edge"] >= 0.07:
                    self._scalp_fired = True
                    return ("scalp", trade)

            return ("skip", None)

        # Phase 2 (T-780 → T-180): main entry window
        # Priority: MOMENTUM (early) → FADE (opportunistic) → SCALP
        if not self._momentum_fired:
            trade = self.momentum.evaluate(market, bankroll, price_feed, seconds_remaining)
            if trade:
                self._momentum_fired = True
                return ("momentum", trade)

        trade = self.fade.evaluate(market, bankroll, price_feed, seconds_remaining)
        if trade:
            return ("fade", trade)

        if not self._scalp_fired:
            trade = self.scalp.evaluate(market, bankroll, price_feed, seconds_remaining)
            if trade:
                self._scalp_fired = True
                return ("scalp", trade)

        return ("skip", None)


# ═══════════════════════════════════════════════════════════
# Shared utility: probability from delta
# ═══════════════════════════════════════════════════════════

# Empirical P(Up wins) by (seconds_remaining, delta%).
# Originally built from 5-min BTC data; time keys scaled 3x for 15-min windows.
# NOTE: recalibrate with 15-min BTC klines for best accuracy.
# Four distinct time rows (180s elapsed, 360s, 540s, 720s into the window).
# Refresh by re-running: venv/bin/python3 calibrate_model.py --days 60 --table
_EMPIRICAL_TABLE = {
    720: {-0.35: 0.2174, -0.30: 0.1455, -0.25: 0.2000, -0.20: 0.2412, -0.15: 0.2694, -0.10: 0.2852, -0.05: 0.3504, 0.00: 0.5015, 0.05: 0.6389, 0.10: 0.7245, 0.15: 0.7861, 0.20: 0.8182, 0.25: 0.8172, 0.30: 0.8696, 0.35: 0.8750, 0.40: 0.8571},
    540: {-0.45: 0.0571, -0.40: 0.0769, -0.35: 0.0667, -0.30: 0.0656, -0.25: 0.1375, -0.20: 0.1731, -0.15: 0.1822, -0.10: 0.2392, -0.05: 0.3466, 0.00: 0.5009, 0.05: 0.6539, 0.10: 0.7591, 0.15: 0.8412, 0.20: 0.8432, 0.25: 0.9018, 0.30: 0.9326, 0.35: 0.9038, 0.40: 0.9348, 0.45: 1.0000, 0.50: 0.9545},
    360: {-0.60: 0.0000, -0.55: 0.0000, -0.50: 0.0455, -0.45: 0.0392, -0.40: 0.0274, -0.35: 0.0297, -0.30: 0.0405, -0.25: 0.0448, -0.20: 0.0865, -0.15: 0.1152, -0.10: 0.1646, -0.05: 0.2799, 0.00: 0.5065, 0.05: 0.7087, 0.10: 0.8287, 0.15: 0.9177, 0.20: 0.9088, 0.25: 0.9431, 0.30: 0.9672, 0.35: 0.9444, 0.40: 0.9692, 0.45: 1.0000, 0.50: 1.0000, 0.55: 0.9643},
    180: {-0.65: 0.0000, -0.55: 0.0000, -0.50: 0.0000, -0.45: 0.0000, -0.40: 0.0000, -0.35: 0.0000, -0.30: 0.0053, -0.25: 0.0246, -0.20: 0.0114, -0.15: 0.0487, -0.10: 0.0904, -0.05: 0.1993, 0.00: 0.5162, 0.05: 0.7737, 0.10: 0.9172, 0.15: 0.9501, 0.20: 0.9743, 0.25: 0.9886, 0.30: 1.0000, 0.35: 1.0000, 0.40: 1.0000, 0.45: 1.0000, 0.50: 1.0000, 0.55: 1.0000, 0.60: 1.0000, 0.65: 1.0000},
}
_EMPIRICAL_SECS = sorted(_EMPIRICAL_TABLE.keys(), reverse=True)  # [720, 540, 360, 180]


def update_empirical_table(table: dict):
    """
    Replace the live calibration table. Called by the bot's calibration loop
    (Phase 0) after each Binance kline fetch so the strategy always uses the
    most recently calibrated probabilities.
    Falls back to the hardcoded table above if called with an empty dict.
    """
    if not table:
        return
    global _EMPIRICAL_TABLE, _EMPIRICAL_SECS
    _EMPIRICAL_TABLE = table
    _EMPIRICAL_SECS = sorted(_EMPIRICAL_TABLE.keys(), reverse=True)
    log.info(
        f"Empirical table updated — "
        f"{len(_EMPIRICAL_SECS)} time rows, "
        f"{sum(len(v) for v in _EMPIRICAL_TABLE.values())} cells"
    )


def _interp_delta(row: dict[float, float], delta_pct: float) -> float:
    """Linear interpolation across delta buckets in one time row."""
    deltas = sorted(row.keys())
    if delta_pct <= deltas[0]:
        return row[deltas[0]]
    if delta_pct >= deltas[-1]:
        return row[deltas[-1]]
    for i in range(len(deltas) - 1):
        lo, hi = deltas[i], deltas[i + 1]
        if lo <= delta_pct <= hi:
            t = (delta_pct - lo) / (hi - lo)
            return row[lo] * (1.0 - t) + row[hi] * t
    return 0.50


def _estimate_prob_from_delta(delta, seconds_remaining, volatility):
    """
    Estimate P(Up wins) via bilinear interpolation on the empirical table.

    `delta` is a fraction (e.g. 0.001 for a 0.1% BTC move).
    `volatility` is accepted for API compatibility but not used — the empirical
    table already encodes regime-average volatility scaling.
    Clamped to [0.05, 0.95].
    """
    delta_pct = delta * 100.0

    secs = _EMPIRICAL_SECS  # [720, 540, 360, 180]

    if seconds_remaining >= secs[0]:
        # Earlier than our earliest row — very weak signal, use 720s row
        return max(0.05, min(0.95, _interp_delta(_EMPIRICAL_TABLE[secs[0]], delta_pct)))

    if seconds_remaining <= secs[-1]:
        # Later than our latest row — signal is even stronger, use 180s row
        return max(0.05, min(0.95, _interp_delta(_EMPIRICAL_TABLE[secs[-1]], delta_pct)))

    # Interpolate between the two bracketing time rows
    for i in range(len(secs) - 1):
        upper, lower = secs[i], secs[i + 1]
        if lower <= seconds_remaining <= upper:
            t = (seconds_remaining - lower) / (upper - lower)  # 1 = upper, 0 = lower
            p_upper = _interp_delta(_EMPIRICAL_TABLE[upper], delta_pct)
            p_lower = _interp_delta(_EMPIRICAL_TABLE[lower], delta_pct)
            prob_up = t * p_upper + (1.0 - t) * p_lower
            return max(0.05, min(0.95, prob_up))

    return 0.50  # unreachable
