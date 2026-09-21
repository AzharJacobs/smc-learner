"""Fair value gap detector: a 3-candle imbalance where candle 1's high
sits below candle 3's low (bullish) or candle 1's low sits above candle
3's high (bearish) — the range candle 2 displaced through without any
candle 1/3 overlap. Confirmed once candle 3 closes (knowable_at). Dies
the first time price fully trades back through the gap. Gap bars are
excluded from all three candle roles and from filling."""

from pathlib import Path

import yaml

_CONFIG = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())


def detect(df):
    df = df.reset_index(drop=True)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    gaps = df["gap_bar"].to_numpy()
    close_times = df["close_time_utc"].tolist()
    n = len(df)

    fvgs = []
    for c1 in range(n - 2):
        c2, c3 = c1 + 1, c1 + 2
        if gaps[c1] or gaps[c2] or gaps[c3]:
            continue

        if highs[c1] < lows[c3]:
            fvg_type = "bullish"
            gap_low, gap_high = float(highs[c1]), float(lows[c3])
        elif lows[c1] > highs[c3]:
            fvg_type = "bearish"
            gap_low, gap_high = float(highs[c3]), float(lows[c1])
        else:
            continue

        died_at = None
        for j in range(c3 + 1, n):
            if gaps[j]:
                continue
            if fvg_type == "bullish" and lows[j] <= gap_low:
                died_at = close_times[j]
                break
            if fvg_type == "bearish" and highs[j] >= gap_high:
                died_at = close_times[j]
                break

        fvgs.append(
            {
                "event": "fvg",
                "fvg_type": fvg_type,
                "low": gap_low,
                "high": gap_high,
                "started_at": close_times[c1],
                "knowable_at": close_times[c3],
                "died_at": died_at,
            }
        )

    fvgs.sort(key=lambda f: f["knowable_at"])
    return fvgs
