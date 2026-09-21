"""Liquidity sweep detector: a wick pierces a target level by a small,
ATR-scaled amount and closes back inside within a few bars — a stop-run,
not a genuine break of structure. Each level gets one attempt: its first
qualifying wick either resolves into a sweep or, if price never closes
back inside within the window (or blows straight through), the level is
done being watched — or its own validity window ends first.

detect_from_levels() is the general engine, working against any list of
target levels (a price + type + knowable_at + optional expiry). detect()
is the original, specific case: sweeps of ordinary confirmed swing
pivots, kept as the default so nothing that already depends on it
changes behaviour."""

import bisect
from pathlib import Path

import numpy as np
import yaml

from .bos import LEFT, RIGHT, find_swings

_CONFIG = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())

ATR_PERIOD = 14
MIN_ATR_MULT = 0.10
MAX_ATR_MULT = 0.50
MAX_RETURN_BARS = 3
SCAN_CHUNK = 2000  # most levels resolve within a chunk or two; scanning
                   # in vectorized chunks instead of bar-by-bar Python
                   # loops is what keeps this tractable over years of
                   # 15-minute bars.


def atr(df, period=ATR_PERIOD):
    """Wilder's ATR: seeded with a simple average of the first `period`
    true ranges, then smoothed."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    values = np.full(len(df), np.nan)
    if len(df) >= period:
        values[period - 1] = tr[:period].mean()
        for i in range(period, len(df)):
            values[i] = (values[i - 1] * (period - 1) + tr[i]) / period
    return values


def detect_from_levels(df, levels):
    """levels: iterable of dicts with:
      - "type": "high" or "low"
      - "price": float
      - "knowable_at": Timestamp — first bar tested is the first one
        strictly after this (found via bisect, not an exact-match
        lookup, since a level like "00:00 UTC" may or may not line up
        with an actual bar close)
      - "valid_until": Timestamp or None — scanning stops here if given
      - "level_type": optional label carried through to the output event
    Returns sweep events sorted by knowable_at, same shape as detect()'s,
    plus level_type/level_knowable_at for traceability."""
    df = df.reset_index(drop=True)
    atr_values = atr(df)

    close_times = df["close_time_utc"].tolist()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    gaps = df["gap_bar"].to_numpy()
    n = len(df)

    events = []

    for lvl in levels:
        start_idx = bisect.bisect_right(close_times, lvl["knowable_at"])
        if start_idx >= n:
            continue

        end_idx = n
        valid_until = lvl.get("valid_until")
        if valid_until is not None:
            end_idx = min(end_idx, bisect.bisect_left(close_times, valid_until))
        if end_idx <= start_idx:
            continue

        price = lvl["price"]
        is_high = lvl["type"] == "high"

        resolved_i = None
        is_overshoot = False
        pos = start_idx
        while pos < end_idx:
            chunk_end = min(pos + SCAN_CHUNK, end_idx)
            seg_atr = atr_values[pos:chunk_end]
            seg_wick = (highs[pos:chunk_end] - price) if is_high else (price - lows[pos:chunk_end])
            valid = (~gaps[pos:chunk_end]) & ~np.isnan(seg_atr) & (seg_atr > 0)

            qualify = valid & (seg_wick >= MIN_ATR_MULT * seg_atr) & (seg_wick <= MAX_ATR_MULT * seg_atr)
            overshoot = valid & (seg_wick > MAX_ATR_MULT * seg_atr)
            hit = qualify | overshoot
            if hit.any():
                rel = int(np.argmax(hit))
                resolved_i = pos + rel
                is_overshoot = bool(overshoot[rel])
                break
            pos = chunk_end

        if resolved_i is None or is_overshoot:
            continue  # never resolved, blew straight through, or ran past valid_until — no event

        i = resolved_i
        wick_beyond = highs[i] - price if is_high else price - lows[i]
        atr_i = atr_values[i]

        confirmed_at = None
        for j in range(i, min(i + MAX_RETURN_BARS + 1, end_idx)):
            if gaps[j]:
                continue
            back_inside = closes[j] < price if is_high else closes[j] > price
            if back_inside:
                confirmed_at = j
                break

        if confirmed_at is not None:
            events.append(
                {
                    "event": "sweep",
                    "direction": "up" if is_high else "down",
                    "swing_price": price,
                    "level_type": lvl.get("level_type", lvl["type"]),
                    "level_knowable_at": lvl["knowable_at"],
                    "wick_atr_mult": float(wick_beyond / atr_i),
                    "started_at": close_times[i],
                    "knowable_at": close_times[confirmed_at],
                }
            )

    events.sort(key=lambda e: e["knowable_at"])
    return events


def detect(df, left=LEFT, right=RIGHT):
    """Original behaviour: sweeps of ordinary confirmed swing pivots."""
    swings = find_swings(df, left, right)
    levels = [
        {
            "type": s["type"],
            "price": s["price"],
            "knowable_at": s["knowable_at"],
            "valid_until": None,
            "level_type": s["type"],
        }
        for s in swings
    ]
    return detect_from_levels(df, levels)
