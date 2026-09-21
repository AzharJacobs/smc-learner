"""Key-liquidity hypothesis: SMC sweeps only matter at obvious liquidity
(previous day H/L, Asian range H/L, equal highs/lows), not at every
3-candle swing — the earlier any-swing sweep detector likely drowned any
real signal in noise. Reuses events/sweep.py's generalized
detect_from_levels() and model/combo_scan.py's outcome/random-baseline
machinery.

Test 1 (drift, no stops): for each level type, mean ATR-normalized
drift in the reversal direction at 1/2/4/8/12/24h after the sweep,
against 200 direction-matched random entries from the same period.
Hours are converted to 15M bar counts (4 bars/hour) rather than
wall-clock offsets, so a sweep near a weekend doesn't get a horizon that
silently spans a 60-hour gap.

Test 2 (trades): for each level type, two entries — (a) the sweep's own
confirming close, (b) the next matching-direction CHoCH within
CHOCH_WINDOW_15M candles (no window was specified for this "next CHoCH"
search, so the same default used everywhere else in this project is
reused for consistency) — at 1R and 2R. Stop is the extreme across the
sweep's own bars, same as make_labels.py. 3 levels x 2 entries x 2 R =
12 tests, using the exact same evaluate_trades/random_baseline/run_test
as the combo scanner (same spread, gap, and dedup handling).

Everything here — the three level definitions, the sweep rule, the two
entry styles, the R targets, and the pass rule — is fixed before any
result is seen.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from events.bos import compute_structure  # noqa: E402
from events.build import CHOCH_WINDOW_15M, _first_after, load_pinned  # noqa: E402
from events.liquidity_levels import asian_range_levels, equal_highs_lows, previous_day_levels  # noqa: E402
from events.sweep import detect as detect_old_sweeps  # noqa: E402
from events.sweep import detect_from_levels  # noqa: E402
from model.combo_scan import N_RANDOM_REPS, RANDOM_SEED  # noqa: E402
from model.combo_scan import _dedup, run_test  # noqa: E402

_CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
SYMBOLS = ["USTECm", "US30m", "XAUUSDm"]
TRAIN_START, TRAIN_END = _CONFIG["dates"]["train"]
VALID_START, VALID_END = _CONFIG["dates"]["validation"]
SPREADS = _CONFIG.get("spread", {})

LEVEL_GENERATORS = {
    "prev_day": previous_day_levels,
    "asian_range": asian_range_levels,
    "equal_hl": equal_highs_lows,
}
BARS_PER_HOUR_15M = 4
DRIFT_HORIZONS_HOURS = [1, 2, 4, 8, 12, 24]


class KLSymbolCtx:
    def __init__(self, symbol, start, end):
        self.symbol = symbol
        df_15m = load_pinned(symbol, "15M", start, end)
        bos_events, choch_events = compute_structure(df_15m)

        self.time_to_idx = {t: i for i, t in enumerate(df_15m["close_time_utc"])}
        self.opens = df_15m["open"].to_numpy()
        self.highs = df_15m["high"].to_numpy()
        self.lows = df_15m["low"].to_numpy()
        self.closes = df_15m["close"].to_numpy()
        self.gaps = df_15m["gap_bar"].to_numpy()
        self.n = len(df_15m)
        self.spread = SPREADS.get(symbol, 0.0)

        self.sweeps_by_level = {name: detect_from_levels(df_15m, gen(df_15m)) for name, gen in LEVEL_GENERATORS.items()}
        self.old_sweeps = detect_old_sweeps(df_15m)

        self.choch_by_dir = {d: [c for c in choch_events if c["direction"] == d] for d in ("up", "down")}
        self.choch_idx_by_dir = {
            d: [self.time_to_idx[c["knowable_at"]] for c in cs] for d, cs in self.choch_by_dir.items()
        }

    def sweep_stop(self, sw, is_long):
        wick_idx = self.time_to_idx.get(sw["started_at"])
        sweep_idx = self.time_to_idx.get(sw["knowable_at"])
        if wick_idx is None or sweep_idx is None:
            return None
        if is_long:
            return float(self.lows[wick_idx : sweep_idx + 1].min())
        return float(self.highs[wick_idx : sweep_idx + 1].max())


def _atr(ctx):
    from events.sweep import ATR_PERIOD, atr as _atr_fn

    df = pd.DataFrame({"high": ctx.highs, "low": ctx.lows, "close": ctx.closes})
    return _atr_fn(df, ATR_PERIOD)


# --------------------------------------------------------------------------
# Test 1: direction-only drift, no stops
# --------------------------------------------------------------------------


def sweep_list(ctx, level_name):
    return ctx.sweeps_by_level[level_name] if level_name != "old" else ctx.old_sweeps


def drift_analysis(ctxs, level_name, n_reps=N_RANDOM_REPS, seed=RANDOM_SEED):
    """Returns a DataFrame indexed by horizon (hours) with columns
    real_drift, random_mean, random_std, percentile."""
    horizon_bars = {h: h * BARS_PER_HOUR_15M for h in DRIFT_HORIZONS_HOURS}

    prepped = []  # (ctx, entry_idx, sign, atr_i)
    for ctx in ctxs.values():
        atr_values = _atr(ctx)
        for sw in sweep_list(ctx, level_name):
            is_long = sw["direction"] == "down"  # low swept -> bullish reversal expected
            entry_idx = ctx.time_to_idx.get(sw["knowable_at"])
            if entry_idx is None:
                continue
            atr_i = atr_values[entry_idx]
            if np.isnan(atr_i) or atr_i <= 0:
                continue
            prepped.append((ctx, entry_idx, 1.0 if is_long else -1.0, atr_i))

    rows = []
    rng = np.random.default_rng(seed)
    for h in DRIFT_HORIZONS_HOURS:
        b = horizon_bars[h]
        real_vals = []
        for ctx, entry_idx, sign, atr_i in prepped:
            future_idx = entry_idx + b
            if future_idx >= ctx.n:
                continue
            drift = sign * (ctx.closes[future_idx] - ctx.closes[entry_idx]) / atr_i
            real_vals.append(drift)
        real_drift = float(np.mean(real_vals)) if real_vals else float("nan")

        rep_means = np.full(n_reps, np.nan)
        for rep in range(n_reps):
            vals = []
            for ctx, entry_idx, sign, atr_i in prepped:
                lo, hi = 1, ctx.n - b - 1
                if hi <= lo:
                    continue
                random_idx = int(rng.integers(lo, hi))
                future_idx = random_idx + b
                if future_idx >= ctx.n:
                    continue
                drift = sign * (ctx.closes[future_idx] - ctx.closes[random_idx]) / atr_i
                vals.append(drift)
            if vals:
                rep_means[rep] = np.mean(vals)

        valid = ~np.isnan(rep_means)
        percentile = (rep_means[valid] < real_drift).mean() if valid.any() else float("nan")
        rows.append(
            {
                "horizon_h": h,
                "n": len(real_vals),
                "real_drift_atr": real_drift,
                "random_mean_atr": np.nanmean(rep_means),
                "random_std_atr": np.nanstd(rep_means),
                "percentile": percentile,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Test 2: trades — (a) sweep close, (b) next CHoCH
# --------------------------------------------------------------------------


def build_trades(ctxs, level_name, entry_style):
    trades = []
    for ctx in ctxs.values():
        for sw in sweep_list(ctx, level_name):
            is_long = sw["direction"] == "down"
            sweep_idx = ctx.time_to_idx.get(sw["knowable_at"])
            if sweep_idx is None:
                continue
            stop = ctx.sweep_stop(sw, is_long)
            if stop is None:
                continue

            if entry_style == "sweep_close":
                entry_idx = sweep_idx
            else:
                choch_dir = "up" if is_long else "down"
                choch_match, choch_idx = _first_after(
                    ctx.choch_idx_by_dir[choch_dir], ctx.choch_by_dir[choch_dir], sweep_idx, CHOCH_WINDOW_15M
                )
                if choch_match is None:
                    continue
                entry_idx = choch_idx

            trades.append((ctx.symbol, "long" if is_long else "short", entry_idx, stop))
    return _dedup(trades)


def run_all(ctxs):
    results = []
    trades_by_key = {}
    for level_name in LEVEL_GENERATORS:
        for entry_style in ("sweep_close", "next_choch"):
            trades = build_trades(ctxs, level_name, entry_style)
            trades_by_key[(level_name, entry_style)] = trades
            for target_r in (1.0, 2.0):
                result = run_test(f"{level_name}_{entry_style}", trades, ctxs, target_r)
                results.append(result)
    return pd.DataFrame(results), trades_by_key


if __name__ == "__main__":
    ctxs = {s: KLSymbolCtx(s, TRAIN_START, TRAIN_END) for s in SYMBOLS}

    old_total = sum(len(ctx.old_sweeps) for ctx in ctxs.values())
    new_totals = {name: sum(len(ctx.sweeps_by_level[name]) for ctx in ctxs.values()) for name in LEVEL_GENERATORS}
    print("sweep counts, pooled, train range:")
    print(f"  old any-swing sweeps: {old_total}")
    for name, count in new_totals.items():
        print(f"  {name}: {count}")
    print(f"  key-liquidity total: {sum(new_totals.values())}  ({sum(new_totals.values())/old_total:.1%} of old)")
    print()

    print("=== Test 1: drift (ATR units), by level type ===")
    for level_name in list(LEVEL_GENERATORS) + ["old"]:
        print(f"-- {level_name} --")
        df = drift_analysis(ctxs, level_name)
        print(df.to_string(index=False))
        print()

    print("=== Test 2: 12 pre-registered trade tests ===")
    results, trades_by_key = run_all(ctxs)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(results.to_string(index=False))
    n_pass = results["passes"].sum()
    print()
    print(f"{n_pass} of {len(results)} tests passed")
    print(f"expected passes from luck alone at a 5% false-positive rate per test: {0.05 * len(results):.1f}")

    print()
    print("=== Reference only (not pre-registered): old any-swing sweeps through the same Test 2 ===")
    ref_rows = []
    for entry_style in ("sweep_close", "next_choch"):
        trades = build_trades(ctxs, "old", entry_style)
        for target_r in (1.0, 2.0):
            ref_rows.append(run_test(f"old_{entry_style}", trades, ctxs, target_r))
    print(pd.DataFrame(ref_rows).to_string(index=False))
