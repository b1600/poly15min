# Poly15min — Polymarket BTC 15-min trading bot

![Poly15min trading bot](assets/bot.png)

Directional bot that trades the BTC Up/Down binary markets on Polymarket. Each market resolves every 15 minutes based on whether BTC closes above or below the window open price. The bot reads live BTC prices from Binance, estimates win probability using an empirically calibrated model, and places GTC or IOC orders when it finds positive expected value.

---

## Strategy overview

The bot runs three entry strategies per window, evaluated in priority order:

**Momentum** (T−13:00 → T−8:00)
When BTC moves ≥ 0.30% from the window open price early in the window, the Polymarket price often lags. The bot places a resting GTC maker bid in the direction of the move. Half-Kelly sizing, max 10% of bankroll.

**Scalp** (T−13:00 → T−0:30)
A single-shot IOC taker order when BTC has moved ≥ 0.20% (≥ 0.25% in the final 3 minutes). Checks order book depth first — skips if no asks or if the best ask exceeds the price cap. Eighth-Kelly sizing.

**Velocity** (T−13:00 → T−4:00)
Fires when BTC has been drifting consistently in one direction over the prior 3 minutes (≥ 0.12% delta velocity, monotone in 3 of 3 consecutive snapshots) but the total delta is too small to trip the scalp gate. Shares the scalp slot — only one of Scalp or Velocity fires per window. Eighth-Kelly sizing, tighter price cap (≤ 0.80 vs ≤ 0.85).

All entries require:
- Net edge ≥ 5% after estimated taker fee
- Token price ≤ 0.85 (paying above this makes the payoff ratio unfavorable)
- No new entries after T−0:30

**Position management:** open positions are re-evaluated every 5 seconds. The bot exits early if the token reaches ≥ 0.85 (lock profit) or if the model edge inverts by more than 5% (cut loss).

**Risk controls:**
- Hard cap: 3 trades per window, max 10% of bankroll per trade
- Daily loss limit: if the session bankroll drops 25%, trading pauses for 30 minutes
- Kelly throttle: if the realized win rate lags the model by >10% over the last 20 trades, all bet sizes are halved until the gap closes

---

## Setup

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

Copy `.env.example` to `.env` (or create `.env`) and fill in:

```
# Polymarket CLOB API credentials
POLY_API_KEY=...
POLY_API_SECRET=...
POLY_API_PASSPHRASE=...
POLY_FUNDER_ADDRESS=...   # your proxy wallet address

# Optional: Telegram alerts
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Dry-run starting bankroll (only used when --dry-run flag is set)
STARTING_BANKROLL=100
```

Polymarket credentials are generated in the Polymarket UI under Settings → API Keys.

### 3. Run in dry-run mode first

```bash
source venv/bin/activate
python bot.py --dry-run
```

Dry-run mode simulates trades without placing real orders. Watch `bot.log` or the console to confirm signals are firing as expected.

### 4. Run live

```bash
source venv/bin/activate
python bot.py
```

The bot runs indefinitely. Stop it with `Ctrl+C` — it will print session stats and save `trade_log.json` on exit.

### 5. Recalibrate the probability model (optional)

The bot recalibrates automatically every 24 hours at runtime. To regenerate the table manually:

```bash
source venv/bin/activate
python calibrate_model.py --days 90 --table
```

This fetches 90 days of Binance 1-min klines, computes empirical win rates by (time remaining, delta%) bucket, and saves `calibration_table.json`. The bot loads this file on startup.

---

## Key files

| File | Purpose |
|---|---|
| `bot.py` | Main trading loop, window orchestration, order execution |
| `strategy_v2.py` | Entry logic — Momentum, Scalp, Velocity strategies and probability model |
| `executor.py` | Polymarket CLOB API wrappers (place, cancel, redeem) |
| `price_feed.py` | Binance WebSocket price feed with auto-reconnect |
| `calibrate_model.py` | Builds empirical probability table from Binance kline history |
| `market_discovery.py` | Resolves current window slug and fetches Polymarket market data |
| `trade_log.json` | Full trade history written at shutdown |
| `bot.log` | Rolling log of all bot activity |
| `calibration_table.json` | Cached probability table (gitignored, regenerated at runtime) |

---

## FAQ

**Why are some windows skipped entirely?**
The bot skips if: the Binance price feed is stale (no update in 10+ seconds), there is no window open price yet, the Polymarket order book has less than $50 depth on both sides, or the daily loss limit has been hit.

**Why does Momentum use GTC and Scalp use IOC?**
Momentum entries are placed early (T−13:00 to T−8:00) when there is time for the order to rest and fill naturally at a good price. IOC is used for Scalp and Velocity because they fire closer to resolution when waiting for a fill is impractical.

**What happened to market-making?**
Removed. In live mode, Polymarket's CLOB fills aggressive maker bids instantly as taker orders. The cancel-after-fill approach that appeared profitable in dry-run is not replicable live because matched orders cannot be cancelled.

**What happened to the Fade strategy?**
Removed (`00ddf02`). Fade entries (betting against an overshooted move) were losing positions in live testing.

**Why does the bot use Binance prices for resolution instead of Polymarket's oracle?**
Polymarket's Chainlink oracle can lag the actual window close by 30–180 seconds, which caused false "Resolution unclear" results in the trade log. Comparing live Binance BTC prices at window open vs close is more reliable.

**What is the Kelly throttle?**
A real-time safety check. If the bot's rolling win rate over the last 20 resolved trades is more than 10% below what the probability model predicted, it halves all bet sizes. This protects the bankroll during periods when the model is miscalibrated. Bet sizes return to normal once the win rate recovers.

**How do winning positions get redeemed on Polymarket?**
Redemption is handled by a separate process. The trading bot logs `condition_id` and `outcome_index` per trade. The redemption logic in `executor.py` (`redeem_positions`) calls the NegRisk on-chain contract but is currently disabled in the main bot loop.

**Can I run multiple windows at once?**
The bot processes one window at a time. Resolution of the previous window runs as a background async task so the main loop re-enters the next window immediately without waiting for the oracle.
