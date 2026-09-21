"""Supply/demand zone detector (order blocks): the last opposite-colour
candle before a displacement move away from it. Demand = last bearish
candle before a bullish displacement; supply = last bullish candle
before a bearish displacement. A zone dies on a close through its far
side, a third touch, or reaching 50 candles old — whichever comes
first — and that outcome is recorded, not just the zone's creation."""

import numpy as np

from .sweep import ATR_PERIOD, atr

DISPLACEMENT_ATR_MULT = 1.0
DISPLACEMENT_WINDOW = 3
MAX_TOUCHES = 3
MAX_AGE_BARS = 50


def detect(df):
    df = df.reset_index(drop=True)
    atr_values = atr(df, ATR_PERIOD)

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    gaps = df["gap_bar"].to_numpy()
    close_times = df["close_time_utc"].tolist()
    n = len(df)

    zones = []

    for k in range(n - 1):
        if gaps[k]:
            continue

        is_bearish = closes[k] < opens[k]
        is_bullish = closes[k] > opens[k]
        if not (is_bearish or is_bullish):
            continue  # doji — no directional candle to anchor a zone

        origin_close = closes[k]

        # A bearish origin looks for a bullish displacement (-> demand);
        # a bullish origin looks for a bearish displacement (-> supply).
        confirm_idx = None
        for j in range(1, DISPLACEMENT_WINDOW + 1):
            idx = k + j
            if idx >= n:
                break
            if gaps[idx]:
                continue
            atr_idx = atr_values[idx]
            if np.isnan(atr_idx) or atr_idx <= 0:
                continue

            move = closes[idx] - origin_close
            if is_bearish and move >= DISPLACEMENT_ATR_MULT * atr_idx:
                confirm_idx = idx
                break
            if is_bullish and -move >= DISPLACEMENT_ATR_MULT * atr_idx:
                confirm_idx = idx
                break

        if confirm_idx is None:
            continue

        zone_type = "demand" if is_bearish else "supply"
        zone_high = float(highs[k])
        zone_low = float(lows[k])

        died_at = None
        died_reason = None
        touches = 0
        inside = False  # a touch is one visit: consecutive overlapping
                        # bars count once, and it only counts again once
                        # price has fully left and come back
        for i in range(confirm_idx + 1, n):
            age = i - confirm_idx
            if age >= MAX_AGE_BARS:
                died_at = close_times[i]
                died_reason = "expired"
                break
            if gaps[i]:
                continue

            if zone_type == "demand" and closes[i] < zone_low:
                died_at = close_times[i]
                died_reason = "close_through"
                break
            if zone_type == "supply" and closes[i] > zone_high:
                died_at = close_times[i]
                died_reason = "close_through"
                break

            touched = lows[i] <= zone_high and highs[i] >= zone_low
            if touched and not inside:
                touches += 1
                if touches >= MAX_TOUCHES:
                    died_at = close_times[i]
                    died_reason = "third_touch"
                    break
            inside = touched

        zones.append(
            {
                "event": "zone",
                "zone_type": zone_type,
                "high": zone_high,
                "low": zone_low,
                "started_at": close_times[k],
                "knowable_at": close_times[confirm_idx],
                "died_at": died_at,
                "died_reason": died_reason,
            }
        )

    zones.sort(key=lambda z: z["knowable_at"])
    return zones
