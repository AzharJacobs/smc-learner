"""Date-bounded access to pinned bars, per config.yaml's train/validation/
test split. The `test` split is refused unless the call originates from
model/evaluate.py — nothing else may read it."""

import inspect
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
PINNED_DIR = ROOT / "data" / "pinned"
FINAL_EVAL_SCRIPT = "evaluate.py"


def _load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _caller_is_final_eval():
    return any(Path(frame.filename).name == FINAL_EVAL_SCRIPT for frame in inspect.stack())


def load_split(symbol, timeframe, split):
    """Bars for `symbol`/`timeframe` within the date range configured for
    `split` ('train', 'validation', or 'test')."""
    if split == "test" and not _caller_is_final_eval():
        raise PermissionError(
            f"the test split is restricted to {FINAL_EVAL_SCRIPT}; refusing to load it here"
        )

    start, end = _load_config()["dates"][split]
    df = pd.read_csv(
        PINNED_DIR / f"{symbol}_{timeframe}.csv",
        parse_dates=["close_time_utc", "open_time_utc"],
    )

    mask = df["close_time_utc"] >= pd.Timestamp(start, tz="UTC")
    if end is not None:
        mask &= df["close_time_utc"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return df.loc[mask].reset_index(drop=True)
