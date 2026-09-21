"""Full multi-timeframe setup: 1H trend, a 15M sweep into a live zone,
15M CHoCH, 15M BOS, a pullback into the zone that BOS's own displacement
formed, and a 5M entry. LONG is implemented directly; SHORT mirrors it
by swapping direction, sweep side, and zone type.

Every step's timestamp is that step's own knowable_at — the moment its
defining bar closed, never earlier. A setup's overall knowable_at is its
entry's, since that's the last piece to be confirmed. No win/loss/R
evaluation happens here: stop and target are recorded as price levels
only, per the request that outcomes not be computed yet.
"""

import bisect
from pathlib import Path

import pandas as pd
import yaml

from .bos import compute_structure
from .sweep import detect as detect_sweeps
from .zones import MAX_AGE_BARS as ZONE_MAX_AGE_BARS
from .zones import detect as detect_zones

ROOT = Path(__file__).resolve().parent.parent
PINNED_DIR = ROOT / "data" / "pinned"
_CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

CHOCH_WINDOW_15M = 10
BOS_WINDOW_15M = 10
PULLBACK_WINDOW_15M = 20
ENTRY_SEARCH_CAP_5M = 500  # generous safety cap; zone-bound exit ends the search well before this in practice
TARGET_R_MULT = 2.0

DIRECTIONS = {
    "long": {
        "trend": "up",
        "sweep": "down",   # a sweep of a swing LOW
        "zone_type": "demand",
        "choch_bos": "up",
    },
    "short": {
        "trend": "down",
        "sweep": "up",     # a sweep of a swing HIGH
        "zone_type": "supply",
        "choch_bos": "down",
    },
}


def load_pinned(symbol, timeframe, start=None, end=None):
    df = pd.read_csv(
        PINNED_DIR / f"{symbol}_{timeframe}.csv",
        parse_dates=["close_time_utc", "open_time_utc"],
    )
    if start is not None:
        df = df[df["close_time_utc"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["close_time_utc"] <= pd.Timestamp(end, tz="UTC")]
    return df.reset_index(drop=True)


def _trend_lookup(bos_events_1h):
    events = sorted(bos_events_1h, key=lambda e: e["knowable_at"])
    times = [e["knowable_at"] for e in events]

    def trend_at(t):
        idx = bisect.bisect_right(times, t) - 1
        if idx < 0:
            return None
        return events[idx]

    return trend_at


class _ZoneIndex:
    """Zones sorted by confirmation bar index, per type, so an "alive and
    touching" query only has to scan the handful of zones confirmed in
    the last MAX_AGE_BARS candles — anything older is guaranteed dead —
    instead of the full multi-thousand-zone list.

    Pruning must use bar index, not wall-clock time: a zone's 50-candle
    lifespan is 50 *bars*, and a weekend (or any session gap) between two
    consecutive bars means 50 bars can span far more than 50 bar-widths
    of calendar time. Pruning by `at_time - 50*bar_width` would wrongly
    drop zones that are still alive across a gap."""

    def __init__(self, zones, time_to_idx):
        self.by_type = {}
        for zt in ("demand", "supply"):
            zs = sorted((z for z in zones if z["zone_type"] == zt), key=lambda z: z["knowable_at"])
            self.by_type[zt] = (zs, [time_to_idx[z["knowable_at"]] for z in zs])

    def alive_and_touching(self, zone_type, at_idx, at_time, low, high):
        zs, idxs = self.by_type[zone_type]
        lo = bisect.bisect_left(idxs, at_idx - ZONE_MAX_AGE_BARS)
        hi = bisect.bisect_right(idxs, at_idx)
        for z in zs[lo:hi]:
            if z["died_at"] is not None and z["died_at"] <= at_time:
                continue
            if low <= z["high"] and high >= z["low"]:
                return z
        return None

    def in_range(self, zone_type, lo_idx_exclusive, hi_idx_inclusive):
        """Zones of `zone_type` confirmed strictly after `lo_idx_exclusive`
        and up to and including `hi_idx_inclusive`."""
        zs, idxs = self.by_type[zone_type]
        lo = bisect.bisect_right(idxs, lo_idx_exclusive)
        hi = bisect.bisect_right(idxs, hi_idx_inclusive)
        return zs[lo:hi], idxs[lo:hi]


def _first_after(idx_list, candidates, after_idx, window):
    pos = bisect.bisect_right(idx_list, after_idx)
    if pos < len(idx_list) and idx_list[pos] - after_idx <= window:
        return candidates[pos], idx_list[pos]
    return None, None


class ChainContext:
    """Precomputed per-timeframe structure (CHoCH/BOS candidates, zone
    index) for one zone/CHoCH/BOS direction, shared across every sweep
    chain lookup so it's built once, not once per sweep."""

    def __init__(self, df_15m, choch_15m, bos_15m, zones_15m, choch_bos_dir):
        self.time_to_idx = {t: i for i, t in enumerate(df_15m["close_time_utc"])}
        self.highs = df_15m["high"].to_numpy()
        self.lows = df_15m["low"].to_numpy()
        self.gaps = df_15m["gap_bar"].to_numpy()
        self.close_times = df_15m["close_time_utc"].tolist()
        self.n = len(df_15m)

        self.zone_index = _ZoneIndex(zones_15m, self.time_to_idx)

        self.choch_candidates = [c for c in choch_15m if c["direction"] == choch_bos_dir]
        self.choch_idx_list = [self.time_to_idx[c["knowable_at"]] for c in self.choch_candidates]
        self.bos_candidates = [b for b in bos_15m if b["direction"] == choch_bos_dir]
        self.bos_idx_list = [self.time_to_idx[b["knowable_at"]] for b in self.bos_candidates]


def find_chain(ctx, sweep_idx, zone_type):
    """CHoCH (<=CHOCH_WINDOW_15M after the sweep) -> BOS (<=BOS_WINDOW_15M
    after the CHoCH) -> the zone formed by the displacement leg that
    produced that BOS (confirmed anywhere between the CHoCH and the BOS,
    preferring the latest-confirmed / most recent-origin candidate) ->
    a pullback into it (<=PULLBACK_WINDOW_15M after the BOS). Stops at
    the first stage that doesn't resolve; every found stage is returned
    regardless of how far the chain got."""
    result = {"choch": None, "bos": None, "zone": None, "pullback_idx": None}

    choch_match, choch_idx = _first_after(ctx.choch_idx_list, ctx.choch_candidates, sweep_idx, CHOCH_WINDOW_15M)
    if choch_match is None:
        return result
    result["choch"] = choch_match

    bos_match, bos_idx = _first_after(ctx.bos_idx_list, ctx.bos_candidates, choch_idx, BOS_WINDOW_15M)
    if bos_match is None:
        return result
    result["bos"] = bos_match

    # the zone formed by the displacement leg that produced this BOS: any
    # zone of the right type confirmed between the CHoCH and the BOS,
    # preferring the one confirmed closest to the BOS (and, among ties,
    # the one with the most recent origin candle)
    candidates, cand_idxs = ctx.zone_index.in_range(zone_type, choch_idx, bos_idx)
    if not candidates:
        return result
    bos_zone = max(zip(cand_idxs, candidates), key=lambda pair: (pair[0], pair[1]["started_at"]))[1]
    result["zone"] = bos_zone

    pullback_idx = None
    for i in range(bos_idx + 1, min(bos_idx + PULLBACK_WINDOW_15M + 1, ctx.n)):
        if bos_zone["died_at"] is not None and ctx.close_times[i] > bos_zone["died_at"]:
            break
        if ctx.gaps[i]:
            continue
        if ctx.lows[i] <= bos_zone["high"] and ctx.highs[i] >= bos_zone["low"]:
            pullback_idx = i
            break
    result["pullback_idx"] = pullback_idx
    return result


def find_setups(df_1h, df_15m, df_5m, direction):
    """Detect completed setups for one symbol/direction. Returns
    (setups, funnel) where funnel counts how many candidates survived
    each stage: sweeps -> choch -> bos -> pullback -> entry."""
    cfg = DIRECTIONS[direction]
    is_long = direction == "long"

    bos_1h, _ = compute_structure(df_1h)
    trend_at = _trend_lookup(bos_1h)

    sweeps_15m = detect_sweeps(df_15m)
    zones_15m = detect_zones(df_15m)
    bos_15m, choch_15m = compute_structure(df_15m)

    ctx = ChainContext(df_15m, choch_15m, bos_15m, zones_15m, cfg["choch_bos"])
    time_to_idx = ctx.time_to_idx
    highs, lows = ctx.highs, ctx.lows

    funnel = {"sweeps": 0, "choch": 0, "bos": 0, "pullback": 0, "entry": 0}
    setups = []
    seen_bos = set()  # multiple sweeps can chain into the identical CHoCH/BOS;
                       # everything downstream of a given BOS is deterministic,
                       # so re-processing it would just emit the same trade twice

    for sw in sweeps_15m:
        if sw["direction"] != cfg["sweep"]:
            continue

        trend_event = trend_at(sw["knowable_at"])
        if trend_event is None or trend_event["direction"] != cfg["trend"]:
            continue

        wick_idx = time_to_idx.get(sw["started_at"])
        if wick_idx is None:
            continue
        touched_zone = ctx.zone_index.alive_and_touching(
            cfg["zone_type"], wick_idx, sw["started_at"], lows[wick_idx], highs[wick_idx]
        )
        if touched_zone is None:
            continue

        funnel["sweeps"] += 1
        sweep_extreme = lows[wick_idx] if is_long else highs[wick_idx]
        sweep_idx = time_to_idx[sw["knowable_at"]]

        chain = find_chain(ctx, sweep_idx, cfg["zone_type"])
        choch_match, bos_match, bos_zone, pullback_idx = (
            chain["choch"], chain["bos"], chain["zone"], chain["pullback_idx"]
        )

        if choch_match is None:
            continue
        funnel["choch"] += 1

        if bos_match is None:
            continue
        if bos_match["knowable_at"] in seen_bos:
            continue
        seen_bos.add(bos_match["knowable_at"])
        funnel["bos"] += 1

        if bos_zone is None:
            continue

        if pullback_idx is None:
            continue
        funnel["pullback"] += 1

        pullback_confirmed_at = ctx.close_times[pullback_idx]

        # entry on 5M: only look after the pullback bar has actually closed —
        # its touch isn't knowable until then, so starting earlier would be a look-ahead leak
        candidates_5m = df_5m[df_5m["close_time_utc"] > pullback_confirmed_at].head(ENTRY_SEARCH_CAP_5M)
        entry = None
        for bar5 in candidates_5m.itertuples():
            close = bar5.close
            if is_long and (close < bos_zone["low"] or close > bos_zone["high"]):
                break
            if not is_long and (close > bos_zone["high"] or close < bos_zone["low"]):
                break
            if bar5.gap_bar:
                continue
            directional = (close > bar5.open) if is_long else (close < bar5.open)
            if directional:
                entry = bar5
                break
        if entry is None:
            continue
        funnel["entry"] += 1

        entry_price = float(entry.close)
        stop_price = float(sweep_extreme)
        risk = abs(entry_price - stop_price)
        target_price = entry_price + TARGET_R_MULT * risk if is_long else entry_price - TARGET_R_MULT * risk

        setups.append(
            {
                "direction": direction,
                "steps": {
                    "trend_confirmed_at": trend_event["knowable_at"],
                    "sweep_started_at": sw["started_at"],
                    "sweep_knowable_at": sw["knowable_at"],
                    "choch_knowable_at": choch_match["knowable_at"],
                    "bos_knowable_at": bos_match["knowable_at"],
                    "pullback_knowable_at": pullback_confirmed_at,
                    "entry_knowable_at": entry.close_time_utc,
                },
                "zone": bos_zone,
                "sweep": sw,
                "choch": choch_match,
                "bos": bos_match,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "risk": risk,
                "knowable_at": entry.close_time_utc,
            }
        )

    return setups, funnel


def find_all_setups(symbol, train_start=None, train_end=None):
    train_start = train_start or _CONFIG["dates"]["train"][0]
    train_end = train_end or _CONFIG["dates"]["train"][1]

    df_1h = load_pinned(symbol, "1H", train_start, train_end)
    df_15m = load_pinned(symbol, "15M", train_start, train_end)
    df_5m = load_pinned(symbol, "5M", train_start, train_end)

    long_setups, long_funnel = find_setups(df_1h, df_15m, df_5m, "long")
    short_setups, short_funnel = find_setups(df_1h, df_15m, df_5m, "short")

    setups = sorted(long_setups + short_setups, key=lambda s: s["knowable_at"])
    return setups, {"long": long_funnel, "short": short_funnel}


if __name__ == "__main__":
    for symbol in ["USTECm", "US30m", "XAUUSDm"]:
        setups, funnel = find_all_setups(symbol)
        print(f"{symbol}: {len(setups)} setups  long={funnel['long']}  short={funnel['short']}")
