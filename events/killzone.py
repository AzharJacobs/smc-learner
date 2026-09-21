"""Killzone: fixed UTC session windows associated with the highest
liquidity/volatility. London 07:00-10:00 UTC, New York 12:00-15:00 UTC.
A pure function of an already-known timestamp — no look-ahead is
possible here, but it's tested alongside the other new detectors for
consistency."""

LONDON = (7, 10)
NEW_YORK = (12, 15)


def in_killzone(timestamp):
    hour = timestamp.hour
    return LONDON[0] <= hour < LONDON[1] or NEW_YORK[0] <= hour < NEW_YORK[1]
