"""Three 'obvious' liquidity levels, as an alternative sweep target to
every 3-candle swing pivot: the previous UTC day's high/low, the Asian
session's high/low, and equal highs/lows (two confirmed swings within
0.1xATR of each other). Each returns level dicts compatible with
events/sweep.py's detect_from_levels().

Equal highs/lows checks CONSECUTIVE same-type swings only (swing i vs
swing i+1 in confirmation order), not every pair — a design choice, not
a shortcut: "equal highs" as a liquidity concept means a visible
double-top/double-bottom on the chart, i.e. two temporally adjacent
swings, not two swings from unrelated points in history that happen to
share a price by coincidence. All-pairs would also be O(n^2) over
several thousand swings per symbol.
"""

import numpy as np
import pandas as pd

from .bos import LEFT, RIGHT, find_swings
from .sweep import ATR_PERIOD, atr

EQUAL_ATR_MULT = 0.1


def previous_day_levels(df):
    """Previous UTC day's high/low, knowable at 00:00 UTC of the next day
    with data, valid until the following such level takes over."""
    df = df.reset_index(drop=True)
    dates = df["close_time_utc"].dt.date
    daily = df.groupby(dates).agg(high=("high", "max"), low=("low", "min")).sort_index()
    unique_dates = list(daily.index)

    levels = []
    for i in range(len(unique_dates) - 1):
        d, d_next = unique_dates[i], unique_dates[i + 1]
        knowable_at = pd.Timestamp(d_next, tz="UTC")
        valid_until = pd.Timestamp(unique_dates[i + 2], tz="UTC") if i + 2 < len(unique_dates) else None
        levels.append(
            {
                "type": "high", "level_type": "prev_day_high",
                "price": float(daily.loc[d, "high"]), "knowable_at": knowable_at, "valid_until": valid_until,
            }
        )
        levels.append(
            {
                "type": "low", "level_type": "prev_day_low",
                "price": float(daily.loc[d, "low"]), "knowable_at": knowable_at, "valid_until": valid_until,
            }
        )
    return levels


def asian_range_levels(df):
    """Asian session (00:00-06:00 UTC) high/low, knowable at 06:00 UTC,
    valid only through the end of that same UTC day."""
    df = df.reset_index(drop=True)
    hours = df["close_time_utc"].dt.hour
    dates = df["close_time_utc"].dt.date
    asian = df[hours < 6]
    grouped = asian.groupby(dates[hours < 6]).agg(high=("high", "max"), low=("low", "min"))

    levels = []
    for d, row in grouped.iterrows():
        knowable_at = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=6)
        valid_until = pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=1)
        levels.append(
            {"type": "high", "level_type": "asian_high", "price": float(row["high"]), "knowable_at": knowable_at, "valid_until": valid_until}
        )
        levels.append(
            {"type": "low", "level_type": "asian_low", "price": float(row["low"]), "knowable_at": knowable_at, "valid_until": valid_until}
        )
    return levels


def equal_highs_lows(df, left=None, right=None):
    """Two confirmed same-type swings within 0.1xATR(14) of each other.
    The level price is the more extreme of the two (the higher of two
    equal highs, the lower of two equal lows), since sweeping that price
    sweeps both. No expiry — like a plain swing pivot, it's watched
    indefinitely."""
    left = LEFT if left is None else left
    right = RIGHT if right is None else right

    df = df.reset_index(drop=True)
    swings = find_swings(df, left, right)
    atr_values = atr(df, ATR_PERIOD)
    time_to_idx = {t: i for i, t in enumerate(df["close_time_utc"])}

    highs = sorted((s for s in swings if s["type"] == "high"), key=lambda s: s["knowable_at"])
    lows = sorted((s for s in swings if s["type"] == "low"), key=lambda s: s["knowable_at"])

    levels = []
    for group, out_type, level_type in ((highs, "high", "equal_high"), (lows, "low", "equal_low")):
        for i in range(len(group) - 1):
            s1, s2 = group[i], group[i + 1]
            idx2 = time_to_idx.get(s2["knowable_at"])
            if idx2 is None:
                continue
            atr2 = atr_values[idx2]
            if np.isnan(atr2) or atr2 <= 0:
                continue
            if abs(s1["price"] - s2["price"]) <= EQUAL_ATR_MULT * atr2:
                price = max(s1["price"], s2["price"]) if out_type == "high" else min(s1["price"], s2["price"])
                levels.append(
                    {
                        "type": out_type,
                        "level_type": level_type,
                        "price": price,
                        "knowable_at": s2["knowable_at"],
                        "valid_until": None,
                    }
                )

    levels.sort(key=lambda l: l["knowable_at"])
    return levels
