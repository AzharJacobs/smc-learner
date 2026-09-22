"""Respect rule + trailing stop.

The setup, in words:
  1. A liquidity sweep happens (wick past an old swing high/low, close back inside).
  2. A CHoCH follows within 10 bars, in the sweep's direction.
  3. The CHoCH closes inside a killzone (London or NY session).
  4. RESPECT: the very next candle must NOT trade back through the broken CHoCH level.
     (long: its low stays above the level. short: its high stays below.)
  5. Entry = close of that respect candle.
  6. Stop = the sweep extreme. No take profit.
  7. The stop trails to each newly confirmed swing low (long) / swing high (short),
     but only ever in the profitable direction. The trade ends when the stop is hit,
     or at market after MAX_HOLD bars.

Results when this was written (15M, USTECm / US30m / XAUUSDm, Exness):
    2021-23   n=309   38.8% win   +0.092R per trade   beats 90% of random
    2024      n=118   37.3% win   +0.268R per trade   beats 97% of random
    2025-26   n=203   38.4% win   +0.085R per trade   beats 81% of random
  Not yet proven on markets it has never seen. Treat as unconfirmed.

Usage:
    python -m model.respect_trail                      # all three periods
    python -m model.respect_trail 2025-01-01 2027-01-01
"""
from __future__ import annotations

import bisect
import sys

import numpy as np
import pandas as pd

from events.build import load_pinned
from events.bos import find_swings
from events.build import CHOCH_WINDOW_15M
from events.sweep import atr
from model.combo_scan import (
    MAX_BARS_FORWARD,
    SYMBOLS,
    SymbolCtx,
    find_chain,
    in_killzone,
    run_test,
)

DATA_START = "2021-07-02"      # how far back the context is built
MAX_HOLD = 400                 # bars (~4 days on 15M) before closing at market
RANDOM_REPS = 200              # random-entry runs used as the comparison


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------
def build_contexts(end=None):
    """SymbolCtx per symbol, plus the swing index and ATR the trailing stop needs."""
    ctxs = {}
    for sym in SYMBOLS:
        ctx = SymbolCtx(sym, DATA_START, end)
        df = load_pinned(sym, "15M", DATA_START, end)
        ctx.times = df["close_time_utc"].tolist()
        ctx.atr_series = atr(df)
        swings = find_swings(df)
        ctx.swings = {t: [s for s in swings if s["type"] == t] for t in ("low", "high")}
        ctx.swing_idx = {
            t: [ctx.time_to_idx[s["knowable_at"]] for s in ctx.swings[t]]
            for t in ctx.swings
        }
        ctxs[sym] = ctx
    return ctxs


# --------------------------------------------------------------------------
# signal
# --------------------------------------------------------------------------
def find_signals(ctxs):
    """Return [(symbol, direction, entry_idx, stop_price), ...]."""
    out = []
    for sym, ctx in ctxs.items():
        for sweep in ctx.sweeps:
            is_long = sweep["direction"] == "down"
            sweep_idx = ctx.time_to_idx[sweep["knowable_at"]]
            stop = ctx.sweep_stop(sweep, is_long)

            # 2. CHoCH in the sweep's direction, within the window
            chain = find_chain(
                ctx.ctx_by_dir["up" if is_long else "down"],
                sweep_idx,
                "demand" if is_long else "supply",
            )
            if not chain["choch"]:
                continue
            choch_idx = ctx.time_to_idx[chain["choch"]["knowable_at"]]
            level = chain["choch"]["price"]
            if choch_idx - sweep_idx > CHOCH_WINDOW_15M:
                continue

            # 3. killzone, judged on the CHoCH candle
            if not in_killzone(ctx.times[choch_idx]):
                continue

            # 4. respect: the next candle must hold the broken level
            nxt = choch_idx + 1
            if nxt >= ctx.n:
                continue
            held = ctx.lows[nxt] > level if is_long else ctx.highs[nxt] < level
            if not held:
                continue

            # 5/6. entry at that candle's close, original stop still valid
            entry = float(ctx.closes[nxt]) + (ctx.spread if is_long else 0.0)
            if (is_long and entry <= stop) or (not is_long and entry >= stop):
                continue

            out.append((sym, "long" if is_long else "short", nxt, stop))

    # one trade per symbol/direction/bar
    seen, deduped = set(), []
    for t in sorted(out, key=lambda x: (x[0], x[2])):
        key = (t[0], t[1], t[2])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    return deduped


# --------------------------------------------------------------------------
# exit: trailing stop, no target
# --------------------------------------------------------------------------
def trail_exit(ctx, is_long, entry_idx, stop, risk):
    """Walk the trade forward. Returns (r_multiple, bars_held) or None if unresolved.

    Prices are bid OHLC. A long enters at the ask (close + spread) and exits at the
    bid; a short enters at the bid and exits at the ask (high + spread).
    """
    entry = float(ctx.closes[entry_idx]) + (ctx.spread if is_long else 0.0)
    kind = "low" if is_long else "high"
    idxs, swings = ctx.swing_idx[kind], ctx.swings[kind]
    k = bisect.bisect_right(idxs, entry_idx)
    end = min(entry_idx + 1 + MAX_HOLD, ctx.n)

    for i in range(entry_idx + 1, end):
        # pull the stop up to any swing confirmed before this bar
        while k < len(idxs) and idxs[k] < i:
            price = swings[k]["price"]
            confirm_close = ctx.closes[idxs[k]]
            if is_long and price > stop and price < confirm_close:
                stop = price
            elif not is_long and price < stop and price > confirm_close:
                stop = price
            k += 1

        if is_long and ctx.lows[i] <= stop:
            return (stop - entry) / risk, i - entry_idx
        if not is_long and ctx.highs[i] + ctx.spread >= stop:
            return (entry - stop) / risk, i - entry_idx

    if end - 1 <= entry_idx:
        return None
    last = float(ctx.closes[end - 1])
    r = (last - entry) if is_long else (entry - last - ctx.spread)
    return r / risk, end - 1 - entry_idx


def run_trades(ctxs, trades):
    rows = []
    for sym, direction, idx, stop in trades:
        ctx = ctxs[sym]
        is_long = direction == "long"
        entry = float(ctx.closes[idx]) + (ctx.spread if is_long else 0.0)
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        res = trail_exit(ctx, is_long, idx, stop, risk)
        if res is None:
            continue
        r, bars = res
        rows.append(
            dict(
                symbol=sym,
                direction=direction,
                entry_time=ctx.times[idx],
                entry=entry,
                stop=stop,
                risk=risk,
                r_multiple=r,
                bars_held=bars,
                hours_held=bars * 0.25,
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# random comparison: same direction and stop size, random entry bar
# --------------------------------------------------------------------------
def random_baseline(ctxs, trades, start, end, reps=RANDOM_REPS, seed=0):
    rng = np.random.default_rng(seed)
    bounds = {}
    for sym, ctx in ctxs.items():
        times = pd.DatetimeIndex(ctx.times)
        lo = int(np.searchsorted(times, start))
        hi = min(int(np.searchsorted(times, end)), ctx.n - MAX_HOLD - 2)
        bounds[sym] = (lo, max(lo + 1, hi))
    means = []
    for _ in range(reps):
        out = []
        for sym, direction, idx, stop in trades:
            ctx = ctxs[sym]
            is_long = direction == "long"
            entry = float(ctx.closes[idx]) + (ctx.spread if is_long else 0.0)
            risk = abs(entry - stop)
            j = int(rng.integers(*bounds[sym]))
            fake_entry = float(ctx.closes[j]) + (ctx.spread if is_long else 0.0)
            fake_stop = fake_entry - risk if is_long else fake_entry + risk
            res = trail_exit(ctx, is_long, j, fake_stop, risk)
            if res:
                out.append(res[0])
        if out:
            means.append(float(np.mean(out)))
    return np.array(means)


# --------------------------------------------------------------------------
def report(ctxs, trades, label, start, end):
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    subset = [t for t in trades if start <= ctxs[t[0]].times[t[2]] < end]
    df = run_trades(ctxs, subset)
    if df.empty:
        print(f"{label}: no trades")
        return df
    rnd = random_baseline(ctxs, subset, start, end)
    beats = float(np.mean(rnd < df.r_multiple.mean())) if len(rnd) else float("nan")
    print(
        f"{label}: n={len(df)}  win {df.r_multiple.gt(0).mean():.1%}  "
        f"avg {df.r_multiple.mean():+.3f}R  best {df.r_multiple.max():+.1f}R  "
        f"median hold {df.hours_held.median():.1f}h  "
        f"| random {rnd.mean():+.3f}R, beats {beats:.0%} of random"
    )
    return df


def main():
    if len(sys.argv) == 3:
        periods = [("custom", sys.argv[1], sys.argv[2])]
        end = sys.argv[2]
    else:
        periods = [
            ("2021-23", "2021-07-02", "2024-01-01"),
            ("2024   ", "2024-01-01", "2025-01-01"),
            ("2025-26", "2025-01-01", "2027-01-01"),
        ]
        end = None

    ctxs = build_contexts(end)
    trades = find_signals(ctxs)
    print(f"signals found: {len(trades)}\n")

    frames = []
    for label, a, b in periods:
        frames.append(report(ctxs, trades, label, a, b))

    all_df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if not all_df.empty:
        print("\nby symbol (all periods):")
        print(
            all_df.groupby("symbol")
            .r_multiple.agg(["mean", "count"])
            .round(3)
            .to_string()
        )
        all_df.to_csv("respect_trail_trades.csv", index=False)
        print("\ntrades written to respect_trail_trades.csv")


if __name__ == "__main__":
    main()
