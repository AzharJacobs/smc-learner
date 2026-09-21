"""Break of structure detector. Owns the shared swing/trend engine that
choch.py also reads from, since a BOS and a CHoCH are two readings of the
same market-structure state machine, not independent detectors.

Trend rule: a break in the direction that already matches the current
trend (or the very first break, before any trend exists) is a BOS and
sets/confirms that trend. A break against the current trend is a CHoCH
and does NOT change trend — but if the *next* confirmed swing in that
same opposing direction is also broken before the original trend
reasserts itself, that second break is a BOS and flips the trend. This
is what "CHoCH doesn't change trend, only the next BOS does" means in
practice: one opposing break is a warning, two in a row is a flip.
"""

from pathlib import Path

import yaml

_CONFIG = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
LEFT = RIGHT = _CONFIG["swing"]["candles_each_side"]


def find_swings(df, left=LEFT, right=RIGHT):
    """Confirmed swing highs/lows: `left` bars before and `right` bars
    after must all be less extreme than the pivot. A swing is only
    knowable once its `right` confirming bars have closed."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    close_times = df["close_time_utc"].tolist()  # plain-list indexing avoids
                                                  # Series.iloc's per-call overhead,
                                                  # which dominates runtime on large frames

    swings = []
    for i in range(left, len(df) - right):
        h = highs[i]
        if all(h > highs[i - k] for k in range(1, left + 1)) and all(
            h > highs[i + k] for k in range(1, right + 1)
        ):
            swings.append(
                {
                    "type": "high",
                    "index": i,
                    "price": float(h),
                    "started_at": close_times[i],
                    "knowable_at": close_times[i + right],
                }
            )
        l = lows[i]
        if all(l < lows[i - k] for k in range(1, left + 1)) and all(
            l < lows[i + k] for k in range(1, right + 1)
        ):
            swings.append(
                {
                    "type": "low",
                    "index": i,
                    "price": float(l),
                    "started_at": close_times[i],
                    "knowable_at": close_times[i + right],
                }
            )

    swings.sort(key=lambda s: s["knowable_at"])
    return swings


def compute_structure(df, left=LEFT, right=RIGHT):
    """Walk the bars in order, tracking swings and trend. Returns
    (bos_events, choch_events). Gap bars are skipped as break triggers —
    their close can't confirm a break — but they still participate in
    swing detection like any other bar."""
    df = df.reset_index(drop=True)
    swings = find_swings(df, left, right)

    close_times = df["close_time_utc"].tolist()
    closes = df["close"].to_numpy()
    gaps = df["gap_bar"].to_numpy()

    pending_high = None
    pending_low = None
    trend = None
    choched_up_since_bos = False
    choched_down_since_bos = False

    bos_events = []
    choch_events = []

    swing_ptr = 0
    n_swings = len(swings)

    for i in range(len(df)):
        bar_time = close_times[i]

        while swing_ptr < n_swings and swings[swing_ptr]["knowable_at"] <= bar_time:
            s = swings[swing_ptr]
            if s["type"] == "high":
                pending_high = s
            else:
                pending_low = s
            swing_ptr += 1

        if gaps[i]:
            continue

        close = closes[i]

        if pending_high is not None and close > pending_high["price"]:
            broken = pending_high
            if trend is None or trend == "up":
                event_type = "bos"
                trend = "up"
                choched_down_since_bos = False
            elif choched_up_since_bos:
                event_type = "bos"
                trend = "up"
                choched_up_since_bos = False
                choched_down_since_bos = False
            else:
                event_type = "choch"
                choched_up_since_bos = True

            event = {
                "event": event_type,
                "direction": "up",
                "price": broken["price"],
                "broken_swing_started_at": broken["started_at"],
                "started_at": bar_time,
                "knowable_at": bar_time,
            }
            (bos_events if event_type == "bos" else choch_events).append(event)
            pending_high = None

        if pending_low is not None and close < pending_low["price"]:
            broken = pending_low
            if trend is None or trend == "down":
                event_type = "bos"
                trend = "down"
                choched_up_since_bos = False
            elif choched_down_since_bos:
                event_type = "bos"
                trend = "down"
                choched_up_since_bos = False
                choched_down_since_bos = False
            else:
                event_type = "choch"
                choched_down_since_bos = True

            event = {
                "event": event_type,
                "direction": "down",
                "price": broken["price"],
                "broken_swing_started_at": broken["started_at"],
                "started_at": bar_time,
                "knowable_at": bar_time,
            }
            (bos_events if event_type == "bos" else choch_events).append(event)
            pending_low = None

    return bos_events, choch_events


def detect(df, left=LEFT, right=RIGHT):
    bos_events, _ = compute_structure(df, left, right)
    return bos_events
