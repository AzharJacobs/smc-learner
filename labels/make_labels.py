"""Label sweeps at three different possible entry points — right at the
sweep's own confirming close, at the following CHoCH's confirmation, or
at the following BOS's confirmation. Each is a SEPARATE candidate set
with its own features and label, because a feature usable at one entry
point (e.g. "a CHoCH followed") would be looking into the future
relative to an earlier one. Every feature is evaluated using only
information knowable at or before that candidate's own entry time —
nothing here is forward-searched. See tests/test_labels_lookahead.py.

Costs: a representative spread (config.yaml `spread:` — NOT pulled from
real broker history) is applied on whichever leg of each trade is the
buy side. Our OHLC is bid-side data (the standard MT5 rates convention),
so a long buys at ask (bid + spread) and later sells at bid — the whole
spread cost lands on entry; a short sells at bid (no adjustment) and
later buys back at ask — the whole cost lands on the exit checks
instead. Candidates whose spread-adjusted risk is under half an ATR are
dropped as too noise-dominated to be a meaningful trade.

Multiple sweeps can chain into the identical CHoCH or BOS bar; only the
first (chronologically nearest) sweep's row is kept per (direction,
entry bar), since a duplicate row would just be double-counting the same
trade under a different origin story.

The label itself is allowed to look into the future — that's what a
label is — but is scored honestly: gap bars are not skipped when
checking for a stop/target touch, and if a bar's open has already moved
past a level, the fill is recorded at that open price (slippage), not
at the untouched theoretical level. No model is fit here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from events.bos import compute_structure  # noqa: E402
from events.build import ChainContext, find_chain, load_pinned  # noqa: E402
from events.build import _trend_lookup  # noqa: E402
from events.sweep import ATR_PERIOD, atr  # noqa: E402
from events.sweep import detect as detect_sweeps  # noqa: E402
from events.zones import detect as detect_zones  # noqa: E402

_CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
TARGET_R = _CONFIG["horizons"]["target_r"]
STOP_R = _CONFIG["horizons"]["stop_r"]
MAX_BARS_FORWARD = _CONFIG["horizons"]["max_bars_forward"]
SPREADS = _CONFIG.get("spread", {})
MIN_RISK_ATR_MULT = 0.5

CANDIDATE_SETS = ("sweep", "choch", "bos")

# Columns allowed to look into the future relative to a candidate's own
# entry time — the outcome, not a feature. Excluded from the look-ahead
# comparison in tests/test_labels_lookahead.py for the same reason a
# zone's died_at is excluded in tests/test_lookahead.py.
OUTCOME_COLUMNS = ("label", "bars_to_outcome", "exit_price")


def _label_outcome(opens, highs, lows, entry_idx, is_long_like, entry_price, stop_price, spread, n):
    """+1R before -1R, checked bar by bar from just after entry.

    Our OHLC is bid-side data. Closing a long is a sell — fills at bid,
    i.e. the raw series, unadjusted. Closing a short is a buy — fills at
    ask (raw + spread), so every bar checked against a short's stop or
    target is shifted up by the spread first, not just the stop.

    Gap bars are NOT skipped: if a bar's open has already moved past a
    level, the fill is recorded at that (spread-adjusted, for a short)
    open price — slippage, not the untouched theoretical level. If a
    bar's range touches both target and stop without gapping through
    either, it's scored as a loss — conservative, since there's no tick
    data to know which was touched first.
    """
    risk = abs(entry_price - stop_price)
    if is_long_like:
        target_price = entry_price + TARGET_R * risk
        stop_level = entry_price - STOP_R * risk
    else:
        target_price = entry_price - TARGET_R * risk
        stop_level = entry_price + STOP_R * risk

    for j in range(entry_idx + 1, min(entry_idx + 1 + MAX_BARS_FORWARD, n)):
        if is_long_like:
            exit_open, exit_high, exit_low = opens[j], highs[j], lows[j]
        else:
            exit_open, exit_high, exit_low = opens[j] + spread, highs[j] + spread, lows[j] + spread

        gapped_stop = (exit_open <= stop_level) if is_long_like else (exit_open >= stop_level)
        if gapped_stop:
            return 0, j - entry_idx, float(exit_open)
        gapped_target = (exit_open >= target_price) if is_long_like else (exit_open <= target_price)
        if gapped_target:
            return 1, j - entry_idx, float(exit_open)

        hit_stop = exit_low <= stop_level if is_long_like else exit_high >= stop_level
        if hit_stop:
            return 0, j - entry_idx, float(stop_level)
        hit_target = exit_high >= target_price if is_long_like else exit_low <= target_price
        if hit_target:
            return 1, j - entry_idx, float(target_price)

    return None, None, None


def make_labels_from_frames(df_1h, df_15m, symbol):
    """Core logic over in-memory frames — used directly by the look-ahead
    test (which needs arbitrarily truncated frames), and wrapped by
    make_labels() for the normal by-symbol/date-range entry point.
    Returns ({candidate_set: DataFrame}, dropped_low_risk_counts)."""
    spread = SPREADS.get(symbol, 0.0)

    bos_1h, _ = compute_structure(df_1h)
    trend_at = _trend_lookup(bos_1h)

    sweeps = detect_sweeps(df_15m)
    zones = detect_zones(df_15m)
    bos_events, choch_events = compute_structure(df_15m)

    time_to_idx = {t: i for i, t in enumerate(df_15m["close_time_utc"])}
    close_times = df_15m["close_time_utc"].tolist()
    opens = df_15m["open"].to_numpy()
    highs = df_15m["high"].to_numpy()
    lows = df_15m["low"].to_numpy()
    closes = df_15m["close"].to_numpy()
    n = len(df_15m)
    atr_values = atr(df_15m, ATR_PERIOD)

    ctx_by_dir = {
        "up": ChainContext(df_15m, choch_events, bos_events, zones, "up"),
        "down": ChainContext(df_15m, choch_events, bos_events, zones, "down"),
    }
    zone_index = ctx_by_dir["up"].zone_index  # same zone index regardless of direction context

    rows = {name: [] for name in CANDIDATE_SETS}
    dropped_low_risk = {name: 0 for name in CANDIDATE_SETS}
    dedup_removed = {name: 0 for name in CANDIDATE_SETS}
    seen = {name: set() for name in CANDIDATE_SETS}  # (direction, entry_idx)

    for sw in sweeps:
        is_long_like = sw["direction"] == "down"  # a low-sweep sits in the same context as the long setup
        expected_trend = "up" if is_long_like else "down"
        expected_zone_type = "demand" if is_long_like else "supply"
        choch_bos_dir = "up" if is_long_like else "down"
        ctx = ctx_by_dir[choch_bos_dir]

        sweep_idx = time_to_idx.get(sw["knowable_at"])
        wick_idx = time_to_idx.get(sw["started_at"])
        if sweep_idx is None or wick_idx is None:
            continue

        # Stop: the extreme across every bar of the sweep, from its first
        # (triggering) bar through its confirming close — the return leg
        # can wick further than the trigger bar did before closing back in.
        if is_long_like:
            stop_price = float(lows[wick_idx : sweep_idx + 1].min())
        else:
            stop_price = float(highs[wick_idx : sweep_idx + 1].max())

        zone_touched = (
            zone_index.alive_and_touching(
                expected_zone_type, wick_idx, sw["started_at"], lows[wick_idx], highs[wick_idx]
            )
            is not None
        )

        chain = find_chain(ctx, sweep_idx, expected_zone_type)
        choch_match, bos_match = chain["choch"], chain["bos"]
        choch_idx = time_to_idx[choch_match["knowable_at"]] if choch_match is not None else None
        bos_idx = time_to_idx[bos_match["knowable_at"]] if bos_match is not None else None

        entry_points = [("sweep", sweep_idx, {})]
        if choch_match is not None:
            entry_points.append(("choch", choch_idx, {"choch_bars_after": choch_idx - sweep_idx}))
            if bos_match is not None:
                entry_points.append(
                    (
                        "bos",
                        bos_idx,
                        {"choch_bars_after": choch_idx - sweep_idx, "bos_bars_after": bos_idx - choch_idx},
                    )
                )

        direction_str = "long" if is_long_like else "short"

        for name, entry_idx, extra in entry_points:
            dedup_key = (direction_str, entry_idx)
            if dedup_key in seen[name]:
                dedup_removed[name] += 1
                continue
            seen[name].add(dedup_key)

            atr_i = atr_values[entry_idx]
            if np.isnan(atr_i) or atr_i <= 0:
                continue

            raw_entry_price = float(closes[entry_idx])
            # Bid-side data: a long buys at ask (bid + spread); a short
            # sells at bid — unadjusted — and pays the spread on exit instead.
            entry_price = raw_entry_price + spread if is_long_like else raw_entry_price
            risk = abs(entry_price - stop_price)
            if risk <= 0:
                continue
            if risk < MIN_RISK_ATR_MULT * atr_i:
                dropped_low_risk[name] += 1
                continue

            entry_time = close_times[entry_idx]
            trend_event = trend_at(entry_time)
            trend_agrees = trend_event is not None and trend_event["direction"] == expected_trend

            label, bars_to_outcome, exit_price = _label_outcome(
                opens, highs, lows, entry_idx, is_long_like, entry_price, stop_price, spread, n
            )

            row = {
                "symbol": symbol,
                "direction": direction_str,
                "sweep_started_at": sw["started_at"],
                "sweep_knowable_at": sw["knowable_at"],
                "entry_knowable_at": entry_time,
                "zone_touched": zone_touched,
                "trend_agrees": trend_agrees,
                "atr": float(atr_i),
                "hour_of_day": entry_time.hour,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "risk": risk,
                "label": label,
                "bars_to_outcome": bars_to_outcome,
                "exit_price": exit_price,
            }
            row.update(extra)
            rows[name].append(row)

    return {name: pd.DataFrame(r) for name, r in rows.items()}, dropped_low_risk, dedup_removed


def make_labels(symbol, timeframe="15M", train_start=None, train_end=None):
    train_start = train_start or _CONFIG["dates"]["train"][0]
    train_end = train_end or _CONFIG["dates"]["train"][1]

    df_1h = load_pinned(symbol, "1H", train_start, train_end)
    df_15m = load_pinned(symbol, timeframe, train_start, train_end)

    return make_labels_from_frames(df_1h, df_15m, symbol)


if __name__ == "__main__":
    for symbol in ["USTECm", "US30m", "XAUUSDm"]:
        frames, dropped, dedup = make_labels(symbol)
        for name in CANDIDATE_SETS:
            df = frames[name]
            for direction in ("long", "short"):
                sub = df[df["direction"] == direction]
                wins = (sub["label"] == 1).sum()
                losses = (sub["label"] == 0).sum()
                censored = sub["label"].isna().sum()
                win_rate = wins / (wins + losses) if (wins + losses) else float("nan")
                print(
                    f"{symbol:9s} {name:6s} {direction:5s}: {len(sub):5d} candidates | "
                    f"wins={wins:4d} losses={losses:4d} censored={censored:3d} | win_rate={win_rate:.3f}"
                )
            print(
                f"{symbol:9s} {name:6s}  dropped (risk < 0.5xATR): {dropped[name]}  "
                f"deduplicated: {dedup[name]}"
            )
        print()
