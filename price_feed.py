# price_feed.py
import json
import asyncio
import time
import logging
import websockets
from collections import deque
from dotenv import load_dotenv
import os

load_dotenv()
log = logging.getLogger("price_feed")


class BinancePriceFeed:
    """
    Real-time BTC price feed via Binance aggTrade WebSocket.

    Uses @aggTrade (not @kline_1m) for trade-level updates — typically
    10-50 messages/second, giving sub-second price resolution instead
    of the 1-second kline updates.

    Features:
    - Auto-reconnect with exponential backoff on disconnect
    - Readiness gating: is_ready property blocks trading until first price arrives
    - Stale price detection: is_stale flags if no update in 10+ seconds
    - Window open price auto-capture from the 15-min boundary
    - Tracks last_update_time so the bot can pause on feed failure
    """

    RECONNECT_BASE_DELAY = 1.0   # seconds
    RECONNECT_MAX_DELAY = 30.0   # seconds
    STALE_THRESHOLD = 10.0       # seconds without update = stale
    # 3601 samples = 3600 one-sec returns, enough for the 60-min vol lookback.
    HISTORY_SIZE = 3601

    # EMA smoothing factors (applied on every aggTrade tick)
    _EMA_FAST_ALPHA = 2 / (10 + 1)   # ~10s fast EMA
    _EMA_SLOW_ALPHA = 2 / (60 + 1)   # ~60s slow EMA

    def __init__(self):
        self.ws_url = os.getenv(
            "BINANCE_WS",
            "wss://stream.binance.com:9443/ws/btcusdt@aggTrade",
        )
        self.current_price: float | None = None
        self.last_update_time: float = 0.0
        self.window_open_price: float | None = None
        self._current_window_start: int | None = None

        # Downsampled 1-per-second history for volatility/momentum calculations
        self.price_history: deque[float] = deque(maxlen=self.HISTORY_SIZE)
        self._last_history_second: int = 0

        # EMA state — updated on every aggTrade tick for high-resolution trend signal
        self._ema_fast: float | None = None
        self._ema_slow: float | None = None

        self._ws = None
        self._running = False
        self._reconnect_delay = self.RECONNECT_BASE_DELAY
        self._connect_task: asyncio.Task | None = None

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self):
        """Start the WebSocket listener. Call once at bot startup."""
        self._running = True
        self._connect_task = asyncio.create_task(self._connection_loop())

    async def stop(self):
        """Gracefully shut down the WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._connect_task:
            self._connect_task.cancel()

    # ── Connection loop with auto-reconnect ────────────────────

    async def _connection_loop(self):
        """Outer loop: connects, listens, reconnects on failure."""
        while self._running:
            try:
                log.info(f"Connecting to Binance WS: {self.ws_url}")
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,    # Binance expects pings
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._reconnect_delay = self.RECONNECT_BASE_DELAY  # reset on success
                    log.info("Binance WS connected")
                    await self._listen(ws)

            except (
                websockets.ConnectionClosed,
                websockets.InvalidStatusCode,
                ConnectionRefusedError,
                OSError,
            ) as e:
                log.warning(f"Binance WS disconnected: {e}")

            except asyncio.CancelledError:
                log.info("Binance WS task cancelled")
                return

            except Exception as e:
                log.error(f"Unexpected Binance WS error: {e}")

            finally:
                self._ws = None

            if self._running:
                log.info(f"Reconnecting in {self._reconnect_delay:.1f}s...")
                await asyncio.sleep(self._reconnect_delay)
                # Exponential backoff, capped
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self.RECONNECT_MAX_DELAY
                )

    async def _listen(self, ws):
        """Inner loop: process each aggTrade message."""
        async for raw_msg in ws:
            try:
                msg = json.loads(raw_msg)
                # aggTrade payload: {"e":"aggTrade","p":"87654.32",...}
                price = float(msg["p"])
                self.current_price = price
                self.last_update_time = time.monotonic()

                # Update EMAs on every tick for smooth trend signal
                if self._ema_fast is None:
                    self._ema_fast = price
                    self._ema_slow = price
                else:
                    self._ema_fast = self._EMA_FAST_ALPHA * price + (1 - self._EMA_FAST_ALPHA) * self._ema_fast
                    self._ema_slow = self._EMA_SLOW_ALPHA * price + (1 - self._EMA_SLOW_ALPHA) * self._ema_slow

                # Downsample to 1 entry per second for history
                now_sec = int(time.time())
                if now_sec != self._last_history_second:
                    self.price_history.append(price)
                    self._last_history_second = now_sec

                # Auto-capture window open price at 15-min boundaries
                self._maybe_capture_window_open(price)

            except (KeyError, ValueError) as e:
                log.debug(f"Skipping malformed aggTrade message: {e}")

    # ── Window open price auto-capture ─────────────────────────

    def _maybe_capture_window_open(self, price: float):
        """
        Automatically set window_open_price at each 15-min boundary.
        Detects when the window has changed and captures the first price.
        """
        now = int(time.time())
        current_window = now - (now % 900)
        if current_window != self._current_window_start:
            self._current_window_start = current_window
            self.window_open_price = price
            log.info(
                f"New 15-min window {current_window} | "
                f"Open price: ${price:,.2f}"
            )

    # ── Readiness checks ───────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True once we have received at least one price."""
        return self.current_price is not None

    @property
    def is_stale(self) -> bool:
        """True if no price update received in STALE_THRESHOLD seconds."""
        if self.last_update_time == 0:
            return True
        return (time.monotonic() - self.last_update_time) > self.STALE_THRESHOLD

    @property
    def is_connected(self) -> bool:
        """True if WebSocket connection is currently open."""
        return self._ws is not None and self._ws.open

    async def wait_until_ready(self, timeout: float = 30.0):
        """
        Block until the first price arrives or timeout.
        Raises TimeoutError if no price within timeout.
        """
        start = time.monotonic()
        while not self.is_ready:
            if time.monotonic() - start > timeout:
                raise TimeoutError(
                    f"Binance price feed not ready after {timeout}s"
                )
            await asyncio.sleep(0.1)
        log.info(f"Price feed ready | BTC = ${self.current_price:,.2f}")

    # ── Price signals ──────────────────────────────────────────

    def get_window_delta(self) -> float:
        """
        Percentage change from window open to current price.
        Returns 0.0 if either price is unavailable.
        """
        if self.window_open_price and self.current_price:
            return (
                (self.current_price - self.window_open_price)
                / self.window_open_price
            )
        return 0.0

    def get_momentum(self, lookback: int = 10) -> float:
        """
        Simple momentum: price change over last `lookback` seconds.
        Uses the downsampled 1-per-second price history.
        Returns 0.0 if insufficient data.
        """
        if len(self.price_history) < lookback + 1:
            return 0.0
        recent = list(self.price_history)
        start_price = recent[-(lookback + 1)]
        end_price = recent[-1]
        if start_price == 0:
            return 0.0
        return (end_price - start_price) / start_price

    def get_volatility(self, lookback: int = 30) -> float:
        """
        Rolling standard deviation of 1-second returns.
        Useful for gauging whether a window is unusually volatile.
        """
        if len(self.price_history) < lookback + 1:
            return 0.0
        prices = list(self.price_history)[-(lookback + 1):]
        returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1]
            for i in range(1, len(prices))
            if prices[i - 1] != 0
        ]
        if not returns:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return variance ** 0.5

    # ── Phase 0: volatility regime tracking ───────────────────

    def get_vol_multi(self) -> dict[str, float]:
        """
        Realized vol over 5 / 15 / 60 minute windows.
        Returns zeroes for any window with insufficient history.
        """
        return {
            "5min":  self.get_volatility(lookback=300),
            "15min": self.get_volatility(lookback=900),
            "60min": self.get_volatility(lookback=3600),
        }

    def get_ema_trend(self) -> float:
        """
        Normalized EMA crossover: (fast_ema - slow_ema) / price.
        Positive = bullish momentum, negative = bearish.
        Returns 0.0 until enough ticks have arrived.
        """
        if self._ema_fast is None or self._ema_slow is None or not self.current_price:
            return 0.0
        return (self._ema_fast - self._ema_slow) / self.current_price

    def get_atr(self, lookback: int = 300) -> float:
        """
        Pseudo-ATR: (rolling_high - rolling_low) / mid over `lookback` seconds.
        Approximates ATR from tick prices (no OHLCV candles available).
        """
        if len(self.price_history) < lookback:
            return 0.0
        prices = list(self.price_history)[-lookback:]
        hi, lo = max(prices), min(prices)
        mid = (hi + lo) / 2
        return (hi - lo) / mid if mid > 0 else 0.0

    # Per-second return std cutoffs, annualized via sqrt(31_557_600).
    #   2.0e-5 ≈ 11% annualized
    #   5.0e-5 ≈ 28% annualized
    # Derived from the realized vol_15m distribution in calibration_log.jsonl
    # (Apr-May 2026): p33 ≈ 2.1e-5, p95 ≈ 7.9e-5. The prior 2e-4 / 5e-4 cutoffs
    # corresponded to ~112% / ~280% annualized, which BTC almost never touches,
    # so the regime was pinned to "low" 99.8% of the time.
    _VOL_LOW_MAX = 2.0e-5
    _VOL_MED_MAX = 5.0e-5

    def get_vol_regime(self) -> str:
        """
        Classify current volatility as 'low', 'medium', or 'high'
        based on realized vol. Prefers 60-min history; cascades to
        15-min then 5-min when the feed is young. Returns 'unknown'
        only if fewer than 5 minutes of history have accumulated.

        Used as a conditioning key for the calibration table.
        """
        for lookback in (3600, 900, 300):
            vol = self.get_volatility(lookback=lookback)
            if vol > 0.0:
                if vol < self._VOL_LOW_MAX:
                    return "low"
                if vol < self._VOL_MED_MAX:
                    return "medium"
                return "high"
        return "unknown"