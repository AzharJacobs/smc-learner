"""Look-ahead guard for labels/make_labels.py: every FEATURE computed for
a candidate must be knowable at or before that candidate's own entry
time, AND the set of candidates that exist must not change once their
entry time is in the past. Verified the standard way used throughout
this project — truncate the data at two points and confirm nothing
about an already-decided candidate changes when more future bars are
appended — but split into two explicit checks rather than one blended
comparison:

1. row existence — the *set* of (direction, sweep, entry-bar) keys
   knowable by `cutoff` must be identical between truncations. This
   catches a row appearing or vanishing (e.g. a dedup or chain-matching
   bug that behaves differently depending on how much future data is
   present), which a naive list-equality check could mask if it also
   happened to fail on ordering or on an unrelated feature diff.
2. feature stability — for every key present in both truncations, its
   feature columns must be identical.

label/bars_to_outcome/exit_price are excluded from check 2 on purpose:
they are the outcome, not a feature, and are allowed (expected) to look
into the future relative to entry — resolving only once enough future
bars exist. Comparing them raw would flag a legitimate new discovery
(one truncation happened to include the resolving bar, the other
didn't) as a false look-ahead failure, the same reasoning already
applied to a zone's died_at in tests/test_lookahead.py.
"""

import random
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from events.build import load_pinned  # noqa: E402
from labels.make_labels import CANDIDATE_SETS, OUTCOME_COLUMNS, make_labels_from_frames  # noqa: E402

SYMBOLS = ["USTECm", "US30m", "XAUUSDm"]
TRAIN_START, TRAIN_END = yaml.safe_load((ROOT / "config.yaml").read_text())["dates"]["train"]
SAMPLES = 8  # the full label pipeline (all 3 candidate sets) is expensive per call

KEY_COLUMNS = ("direction", "sweep_started_at", "entry_knowable_at")


def _sample_points(n_bars, n_samples=SAMPLES, min_bars=2000, seed=0):
    population = range(min_bars, n_bars - 1)
    n_samples = min(n_samples, len(population))
    return sorted(random.Random(seed).sample(population, n_samples))


def _known_rows(frames, cutoff):
    """Per candidate set: {key -> feature dict}, restricted to candidates
    whose own entry is already knowable by `cutoff`."""
    out = {}
    for name in CANDIDATE_SETS:
        df = frames[name]
        known = df[df["entry_knowable_at"] <= cutoff]
        feature_cols = [c for c in known.columns if c not in OUTCOME_COLUMNS]
        keyed = {}
        for record in known[feature_cols].to_dict("records"):
            key = tuple(record[c] for c in KEY_COLUMNS)
            keyed[key] = record
        out[name] = keyed
    return out


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_no_lookahead_labels(symbol):
    df_1h_full = load_pinned(symbol, "1H", TRAIN_START, TRAIN_END)
    df_15m_full = load_pinned(symbol, "15M", TRAIN_START, TRAIN_END)

    for n in _sample_points(len(df_15m_full)):
        cutoff = df_15m_full["close_time_utc"].iloc[n - 1]
        cutoff_next = df_15m_full["close_time_utc"].iloc[n]

        df_1h_a = df_1h_full[df_1h_full["close_time_utc"] <= cutoff].reset_index(drop=True)
        df_15m_a = df_15m_full[df_15m_full["close_time_utc"] <= cutoff].reset_index(drop=True)
        df_1h_b = df_1h_full[df_1h_full["close_time_utc"] <= cutoff_next].reset_index(drop=True)
        df_15m_b = df_15m_full[df_15m_full["close_time_utc"] <= cutoff_next].reset_index(drop=True)

        frames_a, _, _ = make_labels_from_frames(df_1h_a, df_15m_a, symbol)
        frames_b, _, _ = make_labels_from_frames(df_1h_b, df_15m_b, symbol)

        known_a = _known_rows(frames_a, cutoff)
        known_b = _known_rows(frames_b, cutoff)

        for name in CANDIDATE_SETS:
            keys_a, keys_b = set(known_a[name]), set(known_b[name])
            only_a = keys_a - keys_b
            only_b = keys_b - keys_a
            assert not only_a and not only_b, (
                f"{symbol}/{name}: candidate rows knowable at 15M bar {n - 1} changed after adding bar {n} "
                f"(only in smaller truncation: {only_a}; only after adding bar {n}: {only_b})"
            )

            for key in keys_a:
                assert known_a[name][key] == known_b[name][key], (
                    f"{symbol}/{name}: features for {key} changed after adding bar {n}"
                )


if __name__ == "__main__":
    for symbol in SYMBOLS:
        test_no_lookahead_labels(symbol)
        print(f"{symbol}: no look-ahead detected in labels")
