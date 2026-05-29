"""
brier_eval.py — score probability models against the live book.

Consumes `calibration_log.jsonl` (written by calib_log via the bot) and scores,
against the realized outcome, any of:

  - book   : de-vigged book-implied P(Up)   (the baseline to beat)
  - table  : the current 2D empirical model  (logged live as p_model_up)
  - param  : the parametric logistic(z) model (replayed from physical_model.json)
  - refit  : param coefficients re-fit on THIS log (closes the train/serve loop)

Metrics per predictor: Brier score, log loss, and Brier skill score vs the book
(>0 means it beats the market). The book is the baseline — a model that cannot
achieve positive skill out-of-sample adds no information over the market.

This is the Path-1 validation step: the parametric model is fit on Binance
history (calibrate_model.py --fit-physical) and proven here on live data via
replay, without ever being wired into live trading until it earns positive skill.

Usage:
    venv/bin/python3 brier_eval.py                      # book vs current table
    venv/bin/python3 brier_eval.py --model param        # + parametric replay
    venv/bin/python3 brier_eval.py --model param --refit # + live-refit upper bound
    venv/bin/python3 brier_eval.py --model param --min-secs 30 --max-secs 780
    venv/bin/python3 brier_eval.py --model param --reliability
"""

import argparse
import json
import math

import physical_model

EPS = 1e-6  # log-loss clip


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("outcome_up") is None or r.get("p_book_up") is None:
                continue
            r["_y"] = 1.0 if r["outcome_up"] else 0.0
            rows.append(r)
    return rows


def apply_filters(rows, min_secs, max_secs, regime, book_src):
    out = []
    for r in rows:
        s = r.get("secs_remaining")
        if min_secs is not None and (s is None or s < min_secs):
            continue
        if max_secs is not None and (s is None or s > max_secs):
            continue
        if regime is not None and r.get("vol_regime") != regime:
            continue
        if book_src is not None and r.get("book_src") != book_src:
            continue
        out.append(r)
    return out


def add_parametric(rows, coeffs, out_key):
    """Replay a parametric model onto each row. Returns (vol_field, n_usable)."""
    field = coeffs.get("log_field") or "vol_15m"
    n = 0
    for r in rows:
        vol = r.get(field)
        delta = r.get("delta")
        secs = r.get("secs_remaining")
        z = physical_model.z_score(delta, secs, vol) if (vol is not None and delta is not None) else None
        if z is None:
            r[out_key] = None
            continue
        r[out_key] = physical_model.prob_up(delta, secs, vol, coeffs)
        n += 1
    return field, n


def refit_on_log(rows, vol_field):
    """Fit logistic(gamma+beta*z) on the live log itself. Returns coeffs or None."""
    zs, ys = [], []
    for r in rows:
        vol = r.get(vol_field)
        delta = r.get("delta")
        secs = r.get("secs_remaining")
        z = physical_model.z_score(delta, secs, vol) if (vol is not None and delta is not None) else None
        if z is None:
            continue
        zs.append(z)
        ys.append(r["_y"])
    if len(zs) < 50:
        return None
    coeffs = physical_model.fit(zs, ys)
    coeffs["log_field"] = vol_field
    return coeffs


# ── Scoring (each predictor scored only on rows where it is defined) ─────────

def _pairs(rows, key):
    return [(r[key], r["_y"]) for r in rows if r.get(key) is not None]


def brier(rows, key):
    pairs = _pairs(rows, key)
    if not pairs:
        return None
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def log_loss(rows, key):
    pairs = _pairs(rows, key)
    if not pairs:
        return None
    total = 0.0
    for p, y in pairs:
        p = min(1.0 - EPS, max(EPS, p))
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return total / len(pairs)


def skill_vs_book(rows, key):
    """Brier skill of `key` vs book, on the row subset where BOTH are defined."""
    sub = [r for r in rows if r.get(key) is not None and r.get("p_book_up") is not None]
    if not sub:
        return None
    bs_m = sum((r[key] - r["_y"]) ** 2 for r in sub) / len(sub)
    bs_b = sum((r["p_book_up"] - r["_y"]) ** 2 for r in sub) / len(sub)
    if bs_b <= 0:
        return None
    return 1.0 - bs_m / bs_b


def reliability(rows, key, bins):
    buckets = [{"n": 0, "ps": 0.0, "w": 0} for _ in range(bins)]
    for p, y in _pairs(rows, key):
        idx = min(bins - 1, max(0, int(p * bins)))
        buckets[idx]["n"] += 1
        buckets[idx]["ps"] += p
        buckets[idx]["w"] += y
    out = []
    for i, b in enumerate(buckets):
        if b["n"]:
            out.append((i / bins, (i + 1) / bins, b["n"], b["ps"] / b["n"], b["w"] / b["n"]))
    return out


# ── Reporting ────────────────────────────────────────────────────────────────

def print_headline(rows, predictors):
    n = len(rows)
    base = sum(r["_y"] for r in rows) / n
    print("=" * 70)
    print("MODELS vs LIVE BOOK — calibration scoring")
    print("=" * 70)
    print(f"  Samples            : {n:,}")
    print(f"  Actual Up rate     : {base:.4f}")
    print(f"  {'predictor':<10}{'N':>8}{'Brier':>10}{'LogLoss':>10}{'skill vs book':>16}")
    print(f"  {'-'*10}{'-'*8}{'-'*10}{'-'*10}{'-'*16}")
    for label, key in predictors:
        bs = brier(rows, key)
        if bs is None:
            continue
        ll = log_loss(rows, key)
        nn = len(_pairs(rows, key))
        if key == "p_book_up":
            sk = "  (baseline)"
        else:
            s = skill_vs_book(rows, key)
            sk = f"{s:+.4f}" if s is not None else "  n/a"
        print(f"  {label:<10}{nn:>8}{bs:>10.5f}{ll:>10.5f}{sk:>16}")


def secs_bucket(secs):
    if secs is None:
        return None
    if secs > 780:
        return "P1 >780"
    if secs > 480:
        return "P2 780-480"
    if secs > 180:
        return "P2 480-180"
    if secs > 30:
        return "P4 180-30"
    return "P4 <30"


def print_breakdown(rows, attr, label, predictors, ordered=None):
    groups = {}
    for r in rows:
        groups.setdefault(r.get(attr), []).append(r)
    keys = ordered if ordered is not None else sorted(groups, key=lambda k: (k is None, k))

    model_preds = [(lab, k) for lab, k in predictors if k != "p_book_up"]
    head = f"  {label:>11}  {'N':>6}" + "".join(f"  {lab+' skill':>13}" for lab, _ in model_preds)
    print(f"\n  Skill vs book by {label}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for k in keys:
        g = groups.get(k)
        if not g:
            continue
        cells = ""
        for _, key in model_preds:
            s = skill_vs_book(g, key)
            cells += f"  {s:+13.4f}" if s is not None else f"  {'n/a':>13}"
        print(f"  {str(k):>11}  {len(g):6d}{cells}")


def print_reliability(rows, predictors, bins):
    for label, key in predictors:
        tbl = reliability(rows, key, bins)
        if not tbl:
            continue
        print(f"\n  Reliability — {label}")
        print(f"  {'bin':>11}  {'N':>6}  {'pred':>6}  {'actual':>7}  {'gap':>7}")
        print(f"  {'-'*11}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}")
        for lo, hi, nn, pred, act in tbl:
            print(f"  {lo:.2f}-{hi:.2f}  {nn:6d}  {pred:6.3f}  {act:7.3f}  {act-pred:+7.3f}")


def main():
    ap = argparse.ArgumentParser(description="Score probability models against the live book")
    ap.add_argument("--log", default="calibration_log.jsonl")
    ap.add_argument("--model", choices=["table", "param", "both"], default="table",
                    help="Which model(s) to score alongside the book (default table)")
    ap.add_argument("--coeffs", default="physical_model.json", help="Parametric coeffs file")
    ap.add_argument("--refit", action="store_true", help="Also re-fit param on this log (upper bound)")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--min-secs", type=int, default=None)
    ap.add_argument("--max-secs", type=int, default=None)
    ap.add_argument("--regime", default=None)
    ap.add_argument("--book-src", default=None)
    ap.add_argument("--reliability", action="store_true")
    args = ap.parse_args()

    try:
        rows = load_rows(args.log)
    except FileNotFoundError:
        print(f"No log found at {args.log}. Run the bot to accumulate calibration ticks.")
        return

    rows = apply_filters(rows, args.min_secs, args.max_secs, args.regime, args.book_src)
    if not rows:
        print("No rows match the given filters.")
        return

    # Predictors: book baseline + current table are always available from the log.
    predictors = [("book", "p_book_up")]
    if any(r.get("p_model_up") is not None for r in rows):
        predictors.append(("table", "p_model_up"))

    if args.model in ("param", "both"):
        coeffs = physical_model.load_coeffs(args.coeffs)
        if coeffs is None:
            print(f"Could not load coeffs from {args.coeffs} — run "
                  f"`calibrate_model.py --fit-physical` first.")
        else:
            field, n_used = add_parametric(rows, coeffs, "p_param_up")
            predictors.append(("param", "p_param_up"))
            print(f"# param: vol_horizon={coeffs.get('vol_horizon')} field={field} "
                  f"usable_rows={n_used}/{len(rows)} beta={coeffs['beta']}\n")
            if args.refit:
                rc = refit_on_log(rows, field)
                if rc is not None:
                    add_parametric(rows, rc, "p_refit_up")
                    predictors.append(("refit", "p_refit_up"))
                    print(f"# refit on log: beta={[round(b,4) for b in rc['beta']]}\n")

    print_headline(rows, predictors)

    for r in rows:
        r["_phase"] = secs_bucket(r.get("secs_remaining"))
    print_breakdown(rows, "_phase", "phase", predictors,
                    ordered=["P1 >780", "P2 780-480", "P2 480-180", "P4 180-30", "P4 <30"])
    print_breakdown(rows, "vol_regime", "regime", predictors,
                    ordered=["low", "medium", "high", "unknown"])

    if args.reliability:
        print_reliability(rows, predictors, args.bins)

    print()


if __name__ == "__main__":
    main()
