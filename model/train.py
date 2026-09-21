"""First model: logistic regression on the CHoCH candidate set, pooled
across all three symbols. Trained on the train range, evaluated on
validation (2024) only — the test set is never opened here.

Structure (trend/swings/zones/sweeps) is computed over one continuous
timeline from train start through validation end, so 2024's candidates
have real market history behind them rather than a cold start at the
calendar boundary; candidates are then split into train/validate purely
by each one's own entry date. This never reads past validation_end, so
the test range is untouched.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from events.build import load_pinned  # noqa: E402
from labels.make_labels import make_labels_from_frames  # noqa: E402

_CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
SYMBOLS = ["USTECm", "US30m", "XAUUSDm"]
TRAIN_START, TRAIN_END = _CONFIG["dates"]["train"]
VALID_START, VALID_END = _CONFIG["dates"]["validation"]

NUMERIC_FEATURES = ["choch_bars_after", "hour_of_day", "risk_atr"]
BOOLEAN_FEATURES = ["zone_touched", "trend_agrees"]
CATEGORICAL_FEATURES = ["direction", "symbol"]
ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

TOP_FRACTION = 0.30
N_RANDOM_REPS = 200
RANDOM_SEED = 0


def _pooled_choch_candidates():
    """CHoCH-set candidates for every symbol, split into (train, validate)
    DataFrames by each candidate's own entry date. Censored candidates
    (label is NaN — outcome never resolved within the horizon) are
    dropped, since there's nothing to score them against."""
    train_parts, valid_parts = [], []
    for symbol in SYMBOLS:
        df_1h = load_pinned(symbol, "1H", TRAIN_START, VALID_END)
        df_15m = load_pinned(symbol, "15M", TRAIN_START, VALID_END)
        frames, _, _ = make_labels_from_frames(df_1h, df_15m, symbol)

        df = frames["choch"].dropna(subset=["label"]).copy()
        df["risk_atr"] = df["risk"] / df["atr"]

        train_end_bound = pd.Timestamp(TRAIN_END, tz="UTC") + pd.Timedelta(days=1)
        valid_start_bound = pd.Timestamp(VALID_START, tz="UTC")
        valid_end_bound = pd.Timestamp(VALID_END, tz="UTC") + pd.Timedelta(days=1)

        train_parts.append(df[df["entry_knowable_at"] < train_end_bound])
        valid_parts.append(
            df[(df["entry_knowable_at"] >= valid_start_bound) & (df["entry_knowable_at"] < valid_end_bound)]
        )

    train_df = pd.concat(train_parts, ignore_index=True)
    valid_df = pd.concat(valid_parts, ignore_index=True)
    return train_df, valid_df


def _build_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
            ("cat", OneHotEncoder(drop="first"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


def _coefficients(pipeline):
    names = pipeline.named_steps["pre"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_[0]
    return list(zip(names, coefs))


def _evaluate(pipeline, valid_df, top_frac=TOP_FRACTION, n_reps=N_RANDOM_REPS, seed=RANDOM_SEED):
    scores = pipeline.predict_proba(valid_df[ALL_FEATURES])[:, 1]
    labels = valid_df["label"].to_numpy()
    n_total = len(valid_df)
    n_top = max(1, round(n_total * top_frac))

    base_win_rate = labels.mean()

    top_order = np.argsort(-scores)[:n_top]
    top_win_rate = labels[top_order].mean()

    rng = np.random.default_rng(seed)
    random_rates = np.array([labels[rng.choice(n_total, size=n_top, replace=False)].mean() for _ in range(n_reps)])

    return {
        "n_total": n_total,
        "n_top": n_top,
        "base_win_rate": base_win_rate,
        "top_win_rate": top_win_rate,
        "random_mean": random_rates.mean(),
        "random_std": random_rates.std(),
        "random_rates": random_rates,
    }


def train():
    train_df, valid_df = _pooled_choch_candidates()
    for df in (train_df, valid_df):
        df["zone_touched"] = df["zone_touched"].astype(int)
        df["trend_agrees"] = df["trend_agrees"].astype(int)

    pipeline = _build_pipeline()
    pipeline.fit(train_df[ALL_FEATURES], train_df["label"])

    result = _evaluate(pipeline, valid_df)
    coefs = _coefficients(pipeline)
    return pipeline, train_df, valid_df, result, coefs


if __name__ == "__main__":
    pipeline, train_df, valid_df, result, coefs = train()

    print(f"train candidates: {len(train_df)}   validation (2024) candidates: {result['n_total']}")
    print()
    print(f"base win rate (all 2024 CHoCH candidates): {result['base_win_rate']:.3f}")
    print(f"top {TOP_FRACTION:.0%} by model score (n={result['n_top']}): {result['top_win_rate']:.3f}")
    print(
        f"random {TOP_FRACTION:.0%} baseline, {N_RANDOM_REPS} reps: "
        f"mean={result['random_mean']:.3f}  std={result['random_std']:.3f}  "
        f"range=[{result['random_rates'].min():.3f}, {result['random_rates'].max():.3f}]"
    )
    beats_random = result["top_win_rate"] > result["random_mean"]
    print()
    print(
        f"-> top {TOP_FRACTION:.0%} {'BEATS' if beats_random else 'DOES NOT beat'} the random baseline "
        f"({result['top_win_rate']:.3f} vs {result['random_mean']:.3f})"
    )
    print()
    print("coefficients (standardized for numeric features):")
    print(f"  {'intercept':30s} {pipeline.named_steps['clf'].intercept_[0]:+.4f}")
    for name, coef in coefs:
        sign = "+" if coef >= 0 else "-"
        print(f"  {name:30s} {coef:+.4f}  ({sign})")
