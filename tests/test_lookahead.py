"""Look-ahead guard: any event knowable as of bar N must not change when
the series is extended by more bars. If it does, some detector is
peeking at bars it shouldn't."""

import random
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from events.bos import compute_structure, find_swings  # noqa: E402
from events.build import find_setups  # noqa: E402
from events.fvg import detect as detect_fvg  # noqa: E402
from events.liquidity_levels import asian_range_levels, equal_highs_lows, previous_day_levels  # noqa: E402
from events.premium_discount import build_lookup as build_pd_lookup  # noqa: E402
from events.sweep import detect as detect_sweeps  # noqa: E402
from events.sweep import detect_from_levels  # noqa: E402
from events.zones import detect as detect_zones  # noqa: E402

SYMBOLS = ["USTECm", "US30m", "XAUUSDm"]
TIMEFRAME = "15M"
TRAIN_START, TRAIN_END = yaml.safe_load((ROOT / "config.yaml").read_text())["dates"]["train"]


def _load(symbol, timeframe=TIMEFRAME):
    df = pd.read_csv(
        ROOT / "data" / "pinned" / f"{symbol}_{timeframe}.csv",
        parse_dates=["close_time_utc", "open_time_utc"],
    )
    mask = (df["close_time_utc"] >= pd.Timestamp(TRAIN_START, tz="UTC")) & (
        df["close_time_utc"] <= pd.Timestamp(TRAIN_END, tz="UTC")
    )
    return df.loc[mask].reset_index(drop=True)


def _known(events, cutoff):
    return [e for e in events if e["knowable_at"] <= cutoff]


def _known_zones(zones, cutoff):
    """A zone's creation is knowable at `knowable_at`, but its death is a
    separate fact only knowable at `died_at` — which can arrive well
    after creation, or not at all yet. Comparing raw dicts across two
    truncations would flag a legitimate new discovery (one truncation
    happened to include the bar the zone died on, the other didn't) as a
    look-ahead failure. So death info gets clipped to what `cutoff` could
    actually have seen before comparing."""
    result = []
    for z in zones:
        if z["knowable_at"] > cutoff:
            continue
        z = dict(z)
        if z["died_at"] is None or z["died_at"] > cutoff:
            z["died_at"] = None
            z["died_reason"] = None
        result.append(z)
    return result


def _known_fvgs(fvgs, cutoff):
    """Same died_at-clipping issue as zones: an FVG's fill is a separate,
    later-knowable fact from its own confirmation."""
    result = []
    for f in fvgs:
        if f["knowable_at"] > cutoff:
            continue
        f = dict(f)
        if f["died_at"] is not None and f["died_at"] > cutoff:
            f["died_at"] = None
        result.append(f)
    return result


def _snapshot(df, n, cutoff):
    truncated = df.iloc[:n].reset_index(drop=True)
    swings = _known(find_swings(truncated), cutoff)
    bos_events, choch_events = compute_structure(truncated)
    sweeps = _known(detect_sweeps(truncated), cutoff)
    zones = _known_zones(detect_zones(truncated), cutoff)
    fvgs = _known_fvgs(detect_fvg(truncated), cutoff)
    return swings, _known(bos_events, cutoff), _known(choch_events, cutoff), sweeps, zones, fvgs


def _sample_points(n_bars, n_samples=50, min_bars=500, seed=0):
    """50 random cut points rather than an even stride, so the test isn't
    only exercising a handful of structurally-similar positions."""
    population = range(min_bars, n_bars - 1)
    n_samples = min(n_samples, len(population))
    return sorted(random.Random(seed).sample(population, n_samples))


def _reduce_setup(s):
    """Drop the embedded sweep/choch/bos/zone sub-dicts before comparing.
    Those carry their own knowable_at (and, for the zone, a died_at that
    can land after the setup's own entry) — exactly the same "future fact
    baked into a nested field" issue _known_zones works around above. The
    steps/prices already capture everything knowable by entry time, so
    the sub-dicts are dropped rather than re-clipped."""
    return {
        "direction": s["direction"],
        "steps": s["steps"],
        "entry_price": s["entry_price"],
        "stop_price": s["stop_price"],
        "target_price": s["target_price"],
        "risk": s["risk"],
        "knowable_at": s["knowable_at"],
    }


def _known_setups(setups, cutoff):
    return [_reduce_setup(s) for s in setups if s["knowable_at"] <= cutoff]


def _setup_snapshot(dfs, truncate_cutoff, filter_cutoff):
    truncated = {tf: df[df["close_time_utc"] <= truncate_cutoff].reset_index(drop=True) for tf, df in dfs.items()}
    long_setups, _ = find_setups(truncated["1H"], truncated["15M"], truncated["5M"], "long")
    short_setups, _ = find_setups(truncated["1H"], truncated["15M"], truncated["5M"], "short")
    return _known_setups(long_setups + short_setups, filter_cutoff)


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_no_lookahead(symbol):
    df = _load(symbol)

    for n in _sample_points(len(df)):
        cutoff = df["close_time_utc"].iloc[n - 1]

        swings_n, bos_n, choch_n, sweeps_n, zones_n, fvgs_n = _snapshot(df, n, cutoff)
        swings_n1, bos_n1, choch_n1, sweeps_n1, zones_n1, fvgs_n1 = _snapshot(df, n + 1, cutoff)

        assert swings_n == swings_n1, f"{symbol}: swings knowable at bar {n - 1} changed after adding bar {n}"
        assert bos_n == bos_n1, f"{symbol}: BOS knowable at bar {n - 1} changed after adding bar {n}"
        assert choch_n == choch_n1, f"{symbol}: CHoCH knowable at bar {n - 1} changed after adding bar {n}"
        assert sweeps_n == sweeps_n1, f"{symbol}: sweeps knowable at bar {n - 1} changed after adding bar {n}"
        assert zones_n == zones_n1, f"{symbol}: zones knowable at bar {n - 1} changed after adding bar {n}"
        assert fvgs_n == fvgs_n1, f"{symbol}: FVGs knowable at bar {n - 1} changed after adding bar {n}"


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_no_lookahead_premium_discount(symbol):
    """position_at() is built on find_swings(), already proven stable
    above — this confirms that guarantee actually carries through the
    lookup wrapper, using 1H data (the timeframe this detector runs on)."""
    df_1h = _load(symbol, "1H")

    for n in _sample_points(len(df_1h), n_samples=20, min_bars=200):
        cutoff = df_1h["close_time_utc"].iloc[n - 1]
        cutoff_next = df_1h["close_time_utc"].iloc[n]

        lookup_a = build_pd_lookup(df_1h[df_1h["close_time_utc"] <= cutoff].reset_index(drop=True))
        lookup_b = build_pd_lookup(df_1h[df_1h["close_time_utc"] <= cutoff_next].reset_index(drop=True))

        # query at a handful of prices around this point — the mechanism
        # being tested doesn't depend on which price, only on the swing
        # lookup, so a few fixed offsets from the bar's own close suffice
        price = df_1h["close"].iloc[n - 1]
        for probe_price in (price * 0.98, price, price * 1.02):
            assert lookup_a(cutoff, probe_price) == lookup_b(cutoff, probe_price), (
                f"{symbol}: premium/discount at bar {n - 1} changed after adding bar {n}"
            )


def _known_levels(levels, cutoff):
    """Same died_at/valid_until-clipping issue as zones and FVGs — a
    level's own knowable_at can be a fact confirmed well before cutoff,
    but its expiry is a separate, later fact."""
    result = []
    for lvl in levels:
        if lvl["knowable_at"] > cutoff:
            continue
        lvl = dict(lvl)
        if lvl.get("valid_until") is not None and lvl["valid_until"] > cutoff:
            lvl["valid_until"] = None
        result.append(lvl)
    return result


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_no_lookahead_liquidity_levels(symbol):
    """Each of the three new level generators, plus the sweeps detected
    against them, must be stable under truncation the same way every
    other detector in this project is."""
    df = _load(symbol)

    for n in _sample_points(len(df), n_samples=15, min_bars=2000):
        cutoff = df["close_time_utc"].iloc[n - 1]
        truncated_a = df.iloc[:n].reset_index(drop=True)
        truncated_b = df.iloc[: n + 1].reset_index(drop=True)

        for gen in (previous_day_levels, asian_range_levels, equal_highs_lows):
            levels_a = _known_levels(gen(truncated_a), cutoff)
            levels_b = _known_levels(gen(truncated_b), cutoff)
            assert levels_a == levels_b, (
                f"{symbol}: {gen.__name__} knowable at bar {n - 1} changed after adding bar {n}"
            )

            sweeps_a = _known(detect_from_levels(truncated_a, gen(truncated_a)), cutoff)
            sweeps_b = _known(detect_from_levels(truncated_b, gen(truncated_b)), cutoff)
            assert sweeps_a == sweeps_b, (
                f"{symbol}: {gen.__name__} sweeps knowable at bar {n - 1} changed after adding bar {n}"
            )


SETUP_SAMPLES = 10  # the full multi-timeframe pipeline costs much more per
                    # call than a single detector, so fewer random cutoffs


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_no_lookahead_setup(symbol):
    dfs = {"1H": _load(symbol, "1H"), "15M": _load(symbol, "15M"), "5M": _load(symbol, "5M")}
    df15 = dfs["15M"]

    for n in _sample_points(len(df15), n_samples=SETUP_SAMPLES, min_bars=2000):
        cutoff = df15["close_time_utc"].iloc[n - 1]
        cutoff_next = df15["close_time_utc"].iloc[n]

        setups_n = _setup_snapshot(dfs, cutoff, cutoff)
        setups_n1 = _setup_snapshot(dfs, cutoff_next, cutoff)

        assert setups_n == setups_n1, f"{symbol}: setups knowable at 15M bar {n - 1} changed after adding bar {n}"


if __name__ == "__main__":
    for symbol in SYMBOLS:
        test_no_lookahead(symbol)
        test_no_lookahead_setup(symbol)
        print(f"{symbol}: no look-ahead detected")
