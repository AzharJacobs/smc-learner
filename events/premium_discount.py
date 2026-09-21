"""Premium/discount: where price sits between the last confirmed 1H
swing high and low. Below the 50% midpoint is discount (favours longs);
above is premium (favours shorts). This isn't a discrete event with its
own knowable_at — it's a continuous state, queried at another event's
time, built on the same lookahead-safe find_swings() used everywhere
else in this project."""

import bisect

from .bos import find_swings


def build_lookup(df_1h):
    swings = find_swings(df_1h)
    highs = sorted((s for s in swings if s["type"] == "high"), key=lambda s: s["knowable_at"])
    lows = sorted((s for s in swings if s["type"] == "low"), key=lambda s: s["knowable_at"])
    high_times = [s["knowable_at"] for s in highs]
    low_times = [s["knowable_at"] for s in lows]

    def position_at(query_time, price):
        """Returns (zone, position) where zone is "discount"/"premium" and
        position is price's fraction of the way from the last confirmed
        swing low to the last confirmed swing high (0 = at the low, 1 =
        at the high). (None, None) if no swing of either type is
        confirmed yet, or the range is degenerate."""
        hi_idx = bisect.bisect_right(high_times, query_time) - 1
        lo_idx = bisect.bisect_right(low_times, query_time) - 1
        if hi_idx < 0 or lo_idx < 0:
            return None, None

        swing_high = highs[hi_idx]["price"]
        swing_low = lows[lo_idx]["price"]
        if swing_high <= swing_low:
            return None, None

        position = (price - swing_low) / (swing_high - swing_low)
        zone = "discount" if position < 0.5 else "premium"
        return zone, position

    return position_at
