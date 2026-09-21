"""Combo scanner: a fixed, pre-registered list of 12 entry combinations,
each tested at 1R and 2R (24 tests total), on the train range only.
Passing combos (100+ trades AND beating 95% of a matched random-entry
baseline) are then re-run once on 2024 (validation). The test set is
never opened.

Costs and mechanics follow labels/make_labels.py: bid-side OHLC, spread
applied on whichever leg is the buy side, gap bars are not skipped when
scoring outcomes (a bar that opens past a level fills at that open,
i.e. with slippage), and duplicate trades sharing the same
(symbol, direction, entry bar) are deduplicated to the chronologically
first one. The 0.5xATR minimum-risk filter from make_labels.py is
deliberately NOT applied here — it isn't one of the three things this
task named, and each combo's own conditions are already its filter.

The combo list, and the entry/stop rule for each, is fixed before any
result is seen:

 1. sweep_zone              - sweep touching an alive same-side zone. Entry = sweep.
 2. sweep_choch              - sweep -> CHoCH (chained). Entry = CHoCH.
 3. sweep_choch_bos          - sweep -> CHoCH -> BOS (chained). Entry = BOS.
 4. sweep_fvg                - sweep + a same-direction FVG within CHOCH_WINDOW_15M candles. Entry = FVG.
 5. choch_fvg_retrace        - sweep -> CHoCH; a same-direction FVG within BOS_WINDOW_15M of the CHoCH;
                               first retrace touch of that FVG within PULLBACK_WINDOW_15M. Entry = retrace bar.
 6. sweep_discount_premium   - sweep with price in discount (long) / premium (short) at the sweep. Entry = sweep.
 7. sweep_killzone           - sweep confirmed inside a killzone. Entry = sweep.
 8. choch_trend              - sweep -> CHoCH with 1H trend agreeing at the CHoCH. Entry = CHoCH.
 9. sweep_choch_zone_fvg     - sweep -> CHoCH; sweep touched a zone; a same-direction FVG confirmed
                               between the sweep and the CHoCH. Entry = CHoCH.
10. sweep_discount_zone_fvg_choch - #9 plus discount/premium at the sweep. Entry = CHoCH.
11. sweep_choch_killzone     - sweep -> CHoCH with the CHoCH itself inside a killzone. Entry = CHoCH.
12. first_touch_zone_choch   - the first-ever touch of a fresh zone (no sweep involved), followed by a
                               same-direction CHoCH within CHOCH_WINDOW_15M candles. Entry = CHoCH.
                               Stop = the zone's far edge, since there is no sweep to set it.

Stop, for every combo except #12: the extreme across the originating
sweep's own bars (started_at -> knowable_at) — same rule as
make_labels.py. For #12: the zone's far edge (low for demand, high for
supply).
"""

import bisect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from events.bos import compute_structure  # noqa: E402
from events.build import BOS_WINDOW_15M, CHOCH_WINDOW_15M, PULLBACK_WINDOW_15M  # noqa: E402
from events.build import ChainContext, _first_after, _trend_lookup, find_chain, load_pinned  # noqa: E402
from events.fvg import detect as detect_fvg  # noqa: E402
from events.killzone import in_killzone  # noqa: E402
from events.premium_discount import build_lookup as build_pd_lookup  # noqa: E402
from events.sweep import detect as detect_sweeps  # noqa: E402
from events.zones import detect as detect_zones  # noqa: E402

_CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
SYMBOLS = ["USTECm", "US30m", "XAUUSDm"]
TRAIN_START, TRAIN_END = _CONFIG["dates"]["train"]
VALID_START, VALID_END = _CONFIG["dates"]["validation"]
SPREADS = _CONFIG.get("spread", {})
STOP_R = _CONFIG["horizons"]["stop_r"]
MAX_BARS_FORWARD = _CONFIG["horizons"]["max_bars_forward"]

N_RANDOM_REPS = 200
RANDOM_SEED = 0
MIN_TRADES = 100
PASS_PERCENTILE = 0.95


# --------------------------------------------------------------------------
# Per-symbol precomputation
# --------------------------------------------------------------------------


class SymbolCtx:
    def __init__(self, symbol, start, end):
        self.symbol = symbol
        df_1h = load_pinned(symbol, "1H", start, end)
        df_15m = load_pinned(symbol, "15M", start, end)

        bos_1h, _ = compute_structure(df_1h)
        self.trend_at = _trend_lookup(bos_1h)
        self.position_at = build_pd_lookup(df_1h)

        self.sweeps = detect_sweeps(df_15m)
        self.zones = detect_zones(df_15m)
        bos_events, choch_events = compute_structure(df_15m)
        self.fvgs = detect_fvg(df_15m)

        self.time_to_idx = {t: i for i, t in enumerate(df_15m["close_time_utc"])}
        self.opens = df_15m["open"].to_numpy()
        self.highs = df_15m["high"].to_numpy()
        self.lows = df_15m["low"].to_numpy()
        self.closes = df_15m["close"].to_numpy()
        self.gaps = df_15m["gap_bar"].to_numpy()
        self.n = len(df_15m)
        self.spread = SPREADS.get(symbol, 0.0)

        self.ctx_by_dir = {
            "up": ChainContext(df_15m, choch_events, bos_events, self.zones, "up"),
            "down": ChainContext(df_15m, choch_events, bos_events, self.zones, "down"),
        }
        self.zone_index = self.ctx_by_dir["up"].zone_index

        bullish = [f for f in self.fvgs if f["fvg_type"] == "bullish"]
        bearish = [f for f in self.fvgs if f["fvg_type"] == "bearish"]
        self.fvg_by_dir = {
            "bullish": (bullish, [self.time_to_idx[f["knowable_at"]] for f in bullish]),
            "bearish": (bearish, [self.time_to_idx[f["knowable_at"]] for f in bearish]),
        }

    def sweep_stop(self, sw, is_long):
        wick_idx = self.time_to_idx[sw["started_at"]]
        sweep_idx = self.time_to_idx[sw["knowable_at"]]
        if is_long:
            return float(self.lows[wick_idx : sweep_idx + 1].min())
        return float(self.highs[wick_idx : sweep_idx + 1].max())

    def first_fvg_after(self, is_long, after_idx, window):
        fvg_type = "bullish" if is_long else "bearish"
        candidates, idx_list = self.fvg_by_dir[fvg_type]
        return _first_after(idx_list, candidates, after_idx, window)

    def fvg_exists_in_range(self, is_long, lo_exclusive, hi_inclusive):
        fvg_type = "bullish" if is_long else "bearish"
        _, idx_list = self.fvg_by_dir[fvg_type]
        pos = bisect.bisect_right(idx_list, lo_exclusive)
        return pos < len(idx_list) and idx_list[pos] <= hi_inclusive

    def first_touch_index(self, zone):
        start_idx = self.time_to_idx[zone["knowable_at"]] + 1
        death_idx = self.time_to_idx.get(zone["died_at"]) if zone["died_at"] is not None else None
        end_idx = death_idx if death_idx is not None else self.n - 1
        for i in range(start_idx, end_idx + 1):
            if self.gaps[i]:
                continue
            if self.lows[i] <= zone["high"] and self.highs[i] >= zone["low"]:
                return i
        return None


# --------------------------------------------------------------------------
# Trades: (symbol, direction, entry_idx, stop_price)
# --------------------------------------------------------------------------


def _dedup(trades):
    seen = set()
    out = []
    for t in trades:
        key = (t[0], t[1], t[2])
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def combo_sweep_zone(ctxs):
    trades = []
    for ctx in ctxs.values():
        for sw in ctx.sweeps:
            is_long = sw["direction"] == "down"
            zone_type = "demand" if is_long else "supply"
            wick_idx = ctx.time_to_idx.get(sw["started_at"])
            if wick_idx is None:
                continue
            if ctx.zone_index.alive_and_touching(zone_type, wick_idx, sw["started_at"], ctx.lows[wick_idx], ctx.highs[wick_idx]) is None:
                continue
            entry_idx = ctx.time_to_idx[sw["knowable_at"]]
            trades.append((ctx.symbol, "long" if is_long else "short", entry_idx, ctx.sweep_stop(sw, is_long)))
    return _dedup(trades)


def combo_sweep_choch(ctxs):
    trades = []
    for ctx in ctxs.values():
        for sw in ctx.sweeps:
            is_long = sw["direction"] == "down"
            choch_bos_dir = "up" if is_long else "down"
            zone_type = "demand" if is_long else "supply"
            sweep_idx = ctx.time_to_idx.get(sw["knowable_at"])
            if sweep_idx is None:
                continue
            chain = find_chain(ctx.ctx_by_dir[choch_bos_dir], sweep_idx, zone_type)
            if chain["choch"] is None:
                continue
            entry_idx = ctx.time_to_idx[chain["choch"]["knowable_at"]]
            trades.append((ctx.symbol, "long" if is_long else "short", entry_idx, ctx.sweep_stop(sw, is_long)))
    return _dedup(trades)


def combo_sweep_choch_bos(ctxs):
    trades = []
    for ctx in ctxs.values():
        for sw in ctx.sweeps:
            is_long = sw["direction"] == "down"
            choch_bos_dir = "up" if is_long else "down"
            zone_type = "demand" if is_long else "supply"
            sweep_idx = ctx.time_to_idx.get(sw["knowable_at"])
            if sweep_idx is None:
                continue
            chain = find_chain(ctx.ctx_by_dir[choch_bos_dir], sweep_idx, zone_type)
            if chain["choch"] is None or chain["bos"] is None:
                continue
            entry_idx = ctx.time_to_idx[chain["bos"]["knowable_at"]]
            trades.append((ctx.symbol, "long" if is_long else "short", entry_idx, ctx.sweep_stop(sw, is_long)))
    return _dedup(trades)


def combo_sweep_fvg(ctxs):
    trades = []
    for ctx in ctxs.values():
        for sw in ctx.sweeps:
            is_long = sw["direction"] == "down"
            sweep_idx = ctx.time_to_idx.get(sw["knowable_at"])
            if sweep_idx is None:
                continue
            fvg_match, fvg_idx = ctx.first_fvg_after(is_long, sweep_idx, CHOCH_WINDOW_15M)
            if fvg_match is None:
                continue
            trades.append((ctx.symbol, "long" if is_long else "short", fvg_idx, ctx.sweep_stop(sw, is_long)))
    return _dedup(trades)


def combo_choch_fvg_retrace(ctxs):
    trades = []
    for ctx in ctxs.values():
        for sw in ctx.sweeps:
            is_long = sw["direction"] == "down"
            choch_bos_dir = "up" if is_long else "down"
            zone_type = "demand" if is_long else "supply"
            sweep_idx = ctx.time_to_idx.get(sw["knowable_at"])
            if sweep_idx is None:
                continue
            chain = find_chain(ctx.ctx_by_dir[choch_bos_dir], sweep_idx, zone_type)
            choch_match = chain["choch"]
            if choch_match is None:
                continue
            choch_idx = ctx.time_to_idx[choch_match["knowable_at"]]

            fvg_match, fvg_idx = ctx.first_fvg_after(is_long, choch_idx, BOS_WINDOW_15M)
            if fvg_match is None:
                continue

            retrace_idx = None
            for i in range(fvg_idx + 1, min(fvg_idx + PULLBACK_WINDOW_15M + 1, ctx.n)):
                if fvg_match["died_at"] is not None:
                    from_time = ctx.time_to_idx.get(fvg_match["died_at"])
                    if from_time is not None and i > from_time:
                        break
                if ctx.gaps[i]:
                    continue
                if ctx.lows[i] <= fvg_match["high"] and ctx.highs[i] >= fvg_match["low"]:
                    retrace_idx = i
                    break
            if retrace_idx is None:
                continue

            trades.append((ctx.symbol, "long" if is_long else "short", retrace_idx, ctx.sweep_stop(sw, is_long)))
    return _dedup(trades)


def combo_sweep_discount_premium(ctxs):
    trades = []
    for ctx in ctxs.values():
        for sw in ctx.sweeps:
            is_long = sw["direction"] == "down"
            sweep_idx = ctx.time_to_idx.get(sw["knowable_at"])
            if sweep_idx is None:
                continue
            zone, _pos = ctx.position_at(sw["knowable_at"], ctx.closes[sweep_idx])
            wanted = "discount" if is_long else "premium"
            if zone != wanted:
                continue
            trades.append((ctx.symbol, "long" if is_long else "short", sweep_idx, ctx.sweep_stop(sw, is_long)))
    return _dedup(trades)


def combo_sweep_killzone(ctxs):
    trades = []
    for ctx in ctxs.values():
        for sw in ctx.sweeps:
            is_long = sw["direction"] == "down"
            if not in_killzone(sw["knowable_at"]):
                continue
            sweep_idx = ctx.time_to_idx.get(sw["knowable_at"])
            if sweep_idx is None:
                continue
            trades.append((ctx.symbol, "long" if is_long else "short", sweep_idx, ctx.sweep_stop(sw, is_long)))
    return _dedup(trades)


def combo_choch_trend(ctxs):
    trades = []
    for ctx in ctxs.values():
        for sw in ctx.sweeps:
            is_long = sw["direction"] == "down"
            choch_bos_dir = "up" if is_long else "down"
            zone_type = "demand" if is_long else "supply"
            sweep_idx = ctx.time_to_idx.get(sw["knowable_at"])
            if sweep_idx is None:
                continue
            chain = find_chain(ctx.ctx_by_dir[choch_bos_dir], sweep_idx, zone_type)
            choch_match = chain["choch"]
            if choch_match is None:
                continue
            trend_event = ctx.trend_at(choch_match["knowable_at"])
            expected_trend = "up" if is_long else "down"
            if trend_event is None or trend_event["direction"] != expected_trend:
                continue
            entry_idx = ctx.time_to_idx[choch_match["knowable_at"]]
            trades.append((ctx.symbol, "long" if is_long else "short", entry_idx, ctx.sweep_stop(sw, is_long)))
    return _dedup(trades)


def combo_sweep_choch_zone_fvg(ctxs):
    trades = []
    for ctx in ctxs.values():
        for sw in ctx.sweeps:
            is_long = sw["direction"] == "down"
            choch_bos_dir = "up" if is_long else "down"
            zone_type = "demand" if is_long else "supply"
            sweep_idx = ctx.time_to_idx.get(sw["knowable_at"])
            wick_idx = ctx.time_to_idx.get(sw["started_at"])
            if sweep_idx is None or wick_idx is None:
                continue
            if ctx.zone_index.alive_and_touching(zone_type, wick_idx, sw["started_at"], ctx.lows[wick_idx], ctx.highs[wick_idx]) is None:
                continue
            chain = find_chain(ctx.ctx_by_dir[choch_bos_dir], sweep_idx, zone_type)
            choch_match = chain["choch"]
            if choch_match is None:
                continue
            choch_idx = ctx.time_to_idx[choch_match["knowable_at"]]
            if not ctx.fvg_exists_in_range(is_long, sweep_idx, choch_idx):
                continue
            trades.append((ctx.symbol, "long" if is_long else "short", choch_idx, ctx.sweep_stop(sw, is_long)))
    return _dedup(trades)


def combo_sweep_discount_zone_fvg_choch(ctxs):
    trades = []
    for ctx in ctxs.values():
        for sw in ctx.sweeps:
            is_long = sw["direction"] == "down"
            choch_bos_dir = "up" if is_long else "down"
            zone_type = "demand" if is_long else "supply"
            sweep_idx = ctx.time_to_idx.get(sw["knowable_at"])
            wick_idx = ctx.time_to_idx.get(sw["started_at"])
            if sweep_idx is None or wick_idx is None:
                continue

            zone, _pos = ctx.position_at(sw["knowable_at"], ctx.closes[sweep_idx])
            wanted = "discount" if is_long else "premium"
            if zone != wanted:
                continue
            if ctx.zone_index.alive_and_touching(zone_type, wick_idx, sw["started_at"], ctx.lows[wick_idx], ctx.highs[wick_idx]) is None:
                continue

            chain = find_chain(ctx.ctx_by_dir[choch_bos_dir], sweep_idx, zone_type)
            choch_match = chain["choch"]
            if choch_match is None:
                continue
            choch_idx = ctx.time_to_idx[choch_match["knowable_at"]]
            if not ctx.fvg_exists_in_range(is_long, sweep_idx, choch_idx):
                continue
            trades.append((ctx.symbol, "long" if is_long else "short", choch_idx, ctx.sweep_stop(sw, is_long)))
    return _dedup(trades)


def combo_sweep_choch_killzone(ctxs):
    trades = []
    for ctx in ctxs.values():
        for sw in ctx.sweeps:
            is_long = sw["direction"] == "down"
            choch_bos_dir = "up" if is_long else "down"
            zone_type = "demand" if is_long else "supply"
            sweep_idx = ctx.time_to_idx.get(sw["knowable_at"])
            if sweep_idx is None:
                continue
            chain = find_chain(ctx.ctx_by_dir[choch_bos_dir], sweep_idx, zone_type)
            choch_match = chain["choch"]
            if choch_match is None or not in_killzone(choch_match["knowable_at"]):
                continue
            entry_idx = ctx.time_to_idx[choch_match["knowable_at"]]
            trades.append((ctx.symbol, "long" if is_long else "short", entry_idx, ctx.sweep_stop(sw, is_long)))
    return _dedup(trades)


def combo_first_touch_zone_choch(ctxs):
    trades = []
    for ctx in ctxs.values():
        for zone in ctx.zones:
            is_long = zone["zone_type"] == "demand"
            choch_bos_dir = "up" if is_long else "down"
            touch_idx = ctx.first_touch_index(zone)
            if touch_idx is None:
                continue
            choch_candidates = ctx.ctx_by_dir[choch_bos_dir].choch_candidates
            choch_idx_list = ctx.ctx_by_dir[choch_bos_dir].choch_idx_list
            choch_match, choch_idx = _first_after(choch_idx_list, choch_candidates, touch_idx, CHOCH_WINDOW_15M)
            if choch_match is None:
                continue
            stop_price = zone["low"] if is_long else zone["high"]
            trades.append((ctx.symbol, "long" if is_long else "short", choch_idx, stop_price))
    return _dedup(trades)


COMBOS = {
    "sweep_zone": combo_sweep_zone,
    "sweep_choch": combo_sweep_choch,
    "sweep_choch_bos": combo_sweep_choch_bos,
    "sweep_fvg": combo_sweep_fvg,
    "choch_fvg_retrace": combo_choch_fvg_retrace,
    "sweep_discount_premium": combo_sweep_discount_premium,
    "sweep_killzone": combo_sweep_killzone,
    "choch_trend": combo_choch_trend,
    "sweep_choch_zone_fvg": combo_sweep_choch_zone_fvg,
    "sweep_discount_zone_fvg_choch": combo_sweep_discount_zone_fvg_choch,
    "sweep_choch_killzone": combo_sweep_choch_killzone,
    "first_touch_zone_choch": combo_first_touch_zone_choch,
}


# --------------------------------------------------------------------------
# Outcome evaluation (vectorized — this gets called for every trade, every
# random rep, every combo, every R target: needs to be fast)
# --------------------------------------------------------------------------


def label_outcome_vec(opens, highs, lows, entry_idx, is_long, entry_price, stop_price, spread, target_r, n):
    risk = abs(entry_price - stop_price)
    if is_long:
        target_price = entry_price + target_r * risk
        stop_level = entry_price - STOP_R * risk
    else:
        target_price = entry_price - target_r * risk
        stop_level = entry_price + STOP_R * risk

    end = min(entry_idx + 1 + MAX_BARS_FORWARD, n)
    if end <= entry_idx + 1:
        return None, None

    seg_o = opens[entry_idx + 1 : end]
    seg_hi = highs[entry_idx + 1 : end]
    seg_lo = lows[entry_idx + 1 : end]
    if not is_long:
        seg_o = seg_o + spread
        seg_hi = seg_hi + spread
        seg_lo = seg_lo + spread

    if is_long:
        gapped_stop = seg_o <= stop_level
        gapped_target = seg_o >= target_price
        hit_stop = seg_lo <= stop_level
        hit_target = seg_hi >= target_price
    else:
        gapped_stop = seg_o >= stop_level
        gapped_target = seg_o <= target_price
        hit_stop = seg_hi >= stop_level
        hit_target = seg_lo <= target_price

    resolved_stop = gapped_stop | (~gapped_target & hit_stop)
    resolved_target = (~gapped_stop & gapped_target) | (~gapped_stop & ~gapped_target & ~hit_stop & hit_target)
    resolved = resolved_stop | resolved_target
    if not resolved.any():
        return None, None

    i = int(np.argmax(resolved))
    if resolved_stop[i]:
        exit_price = float(seg_o[i]) if gapped_stop[i] else float(stop_level)
        return 0, exit_price
    exit_price = float(seg_o[i]) if gapped_target[i] else float(target_price)
    return 1, exit_price


def _entry_price(ctx, entry_idx, is_long):
    raw = float(ctx.closes[entry_idx])
    return raw + ctx.spread if is_long else raw


def evaluate_trades(trades, ctxs, target_r):
    """Returns a DataFrame: one row per resolved trade, with label and
    R-multiple (accounting for gap slippage), plus each trade's own risk
    (spread-adjusted) for reuse by the random baseline."""
    rows = []
    for symbol, direction, entry_idx, stop_price in trades:
        ctx = ctxs[symbol]
        is_long = direction == "long"
        entry_price = _entry_price(ctx, entry_idx, is_long)
        risk = abs(entry_price - stop_price)
        if risk <= 0:
            continue
        label, exit_price = label_outcome_vec(
            ctx.opens, ctx.highs, ctx.lows, entry_idx, is_long, entry_price, stop_price, ctx.spread, target_r, ctx.n
        )
        r_multiple = None
        if label is not None:
            r_multiple = (exit_price - entry_price) / risk if is_long else (entry_price - exit_price) / risk
        rows.append(
            {
                "symbol": symbol,
                "direction": direction,
                "entry_idx": entry_idx,
                "risk": risk,
                "label": label,
                "r_multiple": r_multiple,
            }
        )
    return pd.DataFrame(rows)


def random_baseline(trades, ctxs, target_r, n_reps=N_RANDOM_REPS, seed=RANDOM_SEED, bounds_by_symbol=None):
    """200 reps: every real trade's entry is replaced by a uniformly
    random bar of the same symbol, keeping direction and risk (stop
    distance in price units) fixed. Returns arrays of per-rep win rate
    and expectancy.

    `bounds_by_symbol`, if given, restricts the random draw to a
    (lo, hi) index range per symbol — used to keep a validation-period
    comparison confined to the validation period itself, rather than
    drawing random entries from the training years too."""
    rng = np.random.default_rng(seed)

    # Precompute each trade's own (symbol, direction, risk) once.
    prepped = []
    for symbol, direction, entry_idx, stop_price in trades:
        ctx = ctxs[symbol]
        is_long = direction == "long"
        entry_price = _entry_price(ctx, entry_idx, is_long)
        risk = abs(entry_price - stop_price)
        if risk <= 0:
            continue
        if bounds_by_symbol is not None:
            lo_bound, hi_bound = bounds_by_symbol[symbol]
        else:
            lo_bound, hi_bound = 1, ctx.n - MAX_BARS_FORWARD - 2
        if hi_bound <= lo_bound:
            continue
        prepped.append((ctx, is_long, risk, lo_bound, hi_bound))

    win_rates = np.full(n_reps, np.nan)
    expectancies = np.full(n_reps, np.nan)

    for rep in range(n_reps):
        wins = losses = 0
        r_sum = 0.0
        for ctx, is_long, risk, lo_bound, hi_bound in prepped:
            random_idx = int(rng.integers(lo_bound, hi_bound))
            raw = float(ctx.closes[random_idx])
            entry_price = raw + ctx.spread if is_long else raw
            stop_price = entry_price - risk if is_long else entry_price + risk
            label, exit_price = label_outcome_vec(
                ctx.opens, ctx.highs, ctx.lows, random_idx, is_long, entry_price, stop_price, ctx.spread, target_r, ctx.n
            )
            if label is None:
                continue
            r_multiple = (exit_price - entry_price) / risk if is_long else (entry_price - exit_price) / risk
            if label == 1:
                wins += 1
            else:
                losses += 1
            r_sum += r_multiple
        total = wins + losses
        if total:
            win_rates[rep] = wins / total
            expectancies[rep] = r_sum / total

    return win_rates, expectancies


def run_test(name, trades, ctxs, target_r, bounds_by_symbol=None):
    df = evaluate_trades(trades, ctxs, target_r)
    resolved = df.dropna(subset=["label"])
    n_trades = len(resolved)
    wins = (resolved["label"] == 1).sum()
    losses = (resolved["label"] == 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) else float("nan")
    expectancy = resolved["r_multiple"].mean() if n_trades else float("nan")

    if n_trades == 0:
        return {
            "combo": name, "target_r": target_r, "n_trades": 0, "win_rate": float("nan"),
            "expectancy": float("nan"), "win_rate_pctile": float("nan"), "passes": False,
        }

    random_win_rates, random_expectancies = random_baseline(trades, ctxs, target_r, bounds_by_symbol=bounds_by_symbol)
    valid = ~np.isnan(random_win_rates)
    win_pctile = (random_win_rates[valid] < win_rate).mean() if valid.any() else float("nan")
    exp_pctile = (random_expectancies[valid] < expectancy).mean() if valid.any() else float("nan")

    passes = n_trades >= MIN_TRADES and win_pctile >= PASS_PERCENTILE

    return {
        "combo": name,
        "target_r": target_r,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "random_win_rate_mean": np.nanmean(random_win_rates),
        "win_rate_pctile": win_pctile,
        "random_expectancy_mean": np.nanmean(random_expectancies),
        "expectancy_pctile": exp_pctile,
        "passes": passes,
    }


def run_all(ctxs):
    results = []
    all_trades = {}
    for name, fn in COMBOS.items():
        trades = fn(ctxs)
        all_trades[name] = trades
        for target_r in (1.0, 2.0):
            results.append(run_test(name, trades, ctxs, target_r))
    return pd.DataFrame(results), all_trades


if __name__ == "__main__":
    ctxs = {symbol: SymbolCtx(symbol, TRAIN_START, TRAIN_END) for symbol in SYMBOLS}
    results, all_trades = run_all(ctxs)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(results.to_string(index=False))

    n_pass = results["passes"].sum()
    print()
    print(f"{n_pass} of {len(results)} tests passed (100+ trades AND beat 95% of random draws)")
    print(f"expected passes from luck alone at a 5% false-positive rate per test: {0.05 * len(results):.1f}")
