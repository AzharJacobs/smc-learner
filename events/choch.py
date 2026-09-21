"""Change of character detector. Reads the same swing/trend engine as
bos.py — a CHoCH is a break against the current trend that does not
flip it (see events/bos.py for the full trend rule)."""

from .bos import LEFT, RIGHT, compute_structure


def detect(df, left=LEFT, right=RIGHT):
    _, choch_events = compute_structure(df, left, right)
    return choch_events
