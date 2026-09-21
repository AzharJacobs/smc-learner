"""Pull bars from MT5, report each series' true intraday start, and pin
frozen snapshots to data/pinned/. Once pinned, nothing downstream reads
MT5 again."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PINNED_DIR = Path(__file__).resolve().parent / "pinned"

SYMBOLS = ["USTECm", "US30m", "XAUUSDm"]
TIMEFRAMES = {
    "5M": mt5.TIMEFRAME_M5,
    "15M": mt5.TIMEFRAME_M15,
    "1H": mt5.TIMEFRAME_H1,
}
TIMEFRAME_SECONDS = {"5M": 300, "15M": 900, "1H": 3600}

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def connect():
    load_dotenv(ROOT / ".env")
    login = int(os.environ["MT5_LOGIN"])
    password = os.environ["MT5_PASSWORD"]
    server = os.environ["MT5_SERVER"]
    if not mt5.initialize(login=login, password=password, server=server):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")


def claimed_start_time(symbol, timeframe):
    """Earliest bar the server reports for this symbol/timeframe."""
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from(symbol, timeframe, EPOCH, 1)
    if rates is None or len(rates) == 0:
        return None
    return datetime.fromtimestamp(int(rates[0]["time"]), tz=timezone.utc)


def fetch_rates(symbol, timeframe, start):
    mt5.symbol_select(symbol, True)
    now = datetime.now(timezone.utc) + timedelta(days=1)
    rates = mt5.copy_rates_range(symbol, timeframe, start, now)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data returned for {symbol}")
    return pd.DataFrame(rates)


def stamp_close_and_flag_gaps(raw, timeframe_seconds):
    df = raw.copy()
    df["open_time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["close_time_utc"] = df["open_time_utc"] + pd.Timedelta(seconds=timeframe_seconds)

    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    df["gap_bar"] = (df["open"] > prev_high) | (df["open"] < prev_low)
    df.loc[df.index[:1], "gap_bar"] = False

    df = df.drop(columns=["time"])
    ordered = [
        "close_time_utc",
        "open_time_utc",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "real_volume",
        "gap_bar",
    ]
    return df[[c for c in ordered if c in df.columns]]


MIN_FAKE_RUN_DAYS = 5  # a genuine "daily bar relabeled as intraday" era
                       # spans years; isolated 1-2 day anomalies (DST
                       # shifts, holiday half-sessions, feed hiccups) are
                       # noise and shouldn't invalidate real history.


def detect_true_start(df, timeframe_seconds, min_fake_run_days=MIN_FAKE_RUN_DAYS):
    """First date from which bars are genuinely spaced at the timeframe's
    interval, rather than a daily bar relabeled and stamped at 00:00.

    A calendar day is flagged "fake" if it holds one bar (a daily bar
    wearing an intraday label) or if none of its bar-to-bar gaps equal
    the timeframe's own spacing (no real intraday activity at all).
    Only a *run* of at least `min_fake_run_days` consecutive fake trading
    days counts as a real break — a lone anomalous day is noise. The true
    start is the day after the last qualifying run.
    """
    d = df[["open_time_utc"]].copy()
    d["date"] = d["open_time_utc"].dt.date
    d["delta_s"] = d["open_time_utc"].diff().dt.total_seconds()
    d["is_match"] = d["delta_s"] == timeframe_seconds

    daily = d.groupby("date").agg(bars=("open_time_utc", "size"), matches=("is_match", "sum"))
    daily["fake"] = (daily["bars"] <= 1) | (daily["matches"] == 0)

    # Group consecutive identical `fake` values into runs.
    run_id = (daily["fake"] != daily["fake"].shift()).cumsum()
    run_lengths = daily.groupby(run_id)["fake"].transform("size")
    daily["real_break"] = daily["fake"] & (run_lengths >= min_fake_run_days)

    break_dates = daily.index[daily["real_break"]]
    if len(break_dates) == 0:
        true_start_date = daily.index.min()
    else:
        later = daily.index[daily.index > break_dates.max()]
        true_start_date = later.min() if len(later) else None

    return true_start_date, daily


def analyze_series(symbol, label, timeframe):
    claimed_start = claimed_start_time(symbol, timeframe)
    if claimed_start is None:
        return None
    raw = fetch_rates(symbol, timeframe, claimed_start)
    df = stamp_close_and_flag_gaps(raw, TIMEFRAME_SECONDS[label])
    true_start_date, daily = detect_true_start(df, TIMEFRAME_SECONDS[label])
    return {
        "claimed_start": claimed_start,
        "true_start_date": true_start_date,
        "df": df,
        "daily": daily,
    }


def report_true_start():
    """Analyze every symbol/timeframe and return (table, analyses), where
    analyses carries the fetched data forward so `pin` doesn't have to
    query MT5 a second time."""
    rows = []
    analyses = {}
    for symbol in SYMBOLS:
        for label, timeframe in TIMEFRAMES.items():
            result = analyze_series(symbol, label, timeframe)
            if result is None:
                continue
            analyses[(symbol, label)] = result
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": label,
                    "claimed_start_utc": result["claimed_start"].date().isoformat(),
                    "true_start_utc": (
                        result["true_start_date"].isoformat()
                        if result["true_start_date"] is not None
                        else "not found"
                    ),
                }
            )
    return pd.DataFrame(rows), analyses


def pin_all(analyses):
    """Pin only the validated intraday portion of each series (from its
    true start onward), reusing the data already fetched for the report
    so MT5 isn't queried twice."""
    mt5_version = mt5.version()
    manifest = {
        "fetch_date_utc": datetime.now(timezone.utc).isoformat(),
        "mt5_build": {
            "terminal_version": mt5_version[0],
            "build": mt5_version[1],
            "build_date": mt5_version[2],
        },
        "files": {},
    }
    PINNED_DIR.mkdir(parents=True, exist_ok=True)

    for (symbol, label), result in analyses.items():
        true_start_date = result["true_start_date"]
        if true_start_date is None:
            continue
        df = result["df"]
        trimmed = df[df["open_time_utc"].dt.date >= true_start_date].reset_index(drop=True)
        if trimmed.empty:
            continue

        filename = f"{symbol}_{label}.csv"
        path = PINNED_DIR / filename
        trimmed.to_csv(path, index=False)
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        manifest["files"][filename] = {
            "symbol": symbol,
            "timeframe": label,
            "rows": len(trimmed),
            "claimed_start_utc": result["claimed_start"].date().isoformat(),
            "true_start_utc": true_start_date.isoformat(),
            "first_close_utc": trimmed["close_time_utc"].iloc[0].isoformat(),
            "last_close_utc": trimmed["close_time_utc"].iloc[-1].isoformat(),
            "gap_bars": int(trimmed["gap_bar"].sum()),
            "sha256": content_hash,
        }

    (PINNED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["report", "pin"])
    args = parser.parse_args()

    connect()
    try:
        table, analyses = report_true_start()
        print(table.to_string(index=False))
        if args.command == "pin":
            manifest = pin_all(analyses)
            print(f"\nPinned {len(manifest['files'])} files to {PINNED_DIR}")
            for filename, info in manifest["files"].items():
                print(
                    f"  {filename}: {info['rows']} rows, "
                    f"true_start={info['true_start_utc']}, "
                    f"{info['gap_bars']} gap bars"
                )
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
