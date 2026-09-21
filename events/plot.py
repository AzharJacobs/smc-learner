"""Plot bars with swings, BOS, and CHoCH marked, saved to results/.
Structure is computed over the full train range so trend state going
into the plotted window is correct, then only that window is drawn."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from matplotlib.patches import Rectangle

from .bos import compute_structure, find_swings
from .build import find_all_setups
from .sweep import detect as detect_sweeps
from .zones import detect as detect_zones

ROOT = Path(__file__).resolve().parent.parent
PINNED_DIR = ROOT / "data" / "pinned"
RESULTS_DIR = ROOT / "results"

INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
COLOR_SWING = "#1baf7a"
COLOR_BOS = "#2a78d6"
COLOR_CHOCH = "#eb6834"
COLOR_SWEEP = "#4a3aa7"
COLOR_DEMAND = "#008300"
COLOR_SUPPLY = "#e34948"

LEGEND_HANDLES = [
    Line2D([0], [0], marker="v", color="none", markerfacecolor=COLOR_SWING, markersize=9, label="Swing high"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor=COLOR_SWING, markersize=9, label="Swing low"),
    Line2D([0], [0], color=COLOR_BOS, linewidth=1.5, label="BOS"),
    Line2D([0], [0], color=COLOR_CHOCH, linewidth=1.5, linestyle="--", label="CHoCH"),
]

SWEEP_LEGEND_HANDLES = [
    Line2D([0], [0], marker="v", color="none", markerfacecolor=COLOR_SWING, markersize=9, label="Swing high"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor=COLOR_SWING, markersize=9, label="Swing low"),
    Line2D([0], [0], marker="x", color=COLOR_SWEEP, markersize=9, linewidth=0, markeredgewidth=2, label="Sweep wick"),
    Line2D([0], [0], color=COLOR_SWEEP, linewidth=1.5, linestyle=":", label="Sweep return window"),
]

# Red/green alone sit in the CVD warn band (deutan/protan readers can
# lose the pair), so hatching carries the demand/supply distinction too,
# not just fill colour.
ZONE_LEGEND_HANDLES = [
    Line2D([0], [0], marker="v", color="none", markerfacecolor=COLOR_SWING, markersize=9, label="Swing high"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor=COLOR_SWING, markersize=9, label="Swing low"),
    Rectangle((0, 0), 1, 1, facecolor=COLOR_DEMAND, alpha=0.18, hatch="//", edgecolor=COLOR_DEMAND, label="Demand zone"),
    Rectangle((0, 0), 1, 1, facecolor=COLOR_SUPPLY, alpha=0.18, hatch="\\\\", edgecolor=COLOR_SUPPLY, label="Supply zone"),
]

COLOR_ENTRY = "#eda100"

SETUP_LEGEND_HANDLES = [
    Line2D([0], [0], marker="x", color=COLOR_SWEEP, markersize=9, linewidth=0, markeredgewidth=2, label="Sweep wick"),
    Line2D([0], [0], color=COLOR_CHOCH, linewidth=1.5, linestyle="--", label="CHoCH"),
    Line2D([0], [0], color=COLOR_BOS, linewidth=1.5, label="BOS"),
    Rectangle((0, 0), 1, 1, facecolor=COLOR_DEMAND, alpha=0.18, hatch="//", edgecolor=COLOR_DEMAND, label="Zone (demand shown)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=9, label="Pullback"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor=COLOR_ENTRY, markeredgecolor=INK, markersize=14, label="Entry"),
    Line2D([0], [0], color=INK, linewidth=1, linestyle=":", label="Stop / target"),
]


def _as_utc(t):
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def load_pinned(symbol, timeframe, start=None, end=None):
    df = pd.read_csv(
        PINNED_DIR / f"{symbol}_{timeframe}.csv",
        parse_dates=["close_time_utc", "open_time_utc"],
    )
    if start is not None:
        df = df[df["close_time_utc"] >= _as_utc(start)]
    if end is not None:
        df = df[df["close_time_utc"] <= _as_utc(end)]
    return df.reset_index(drop=True)


def pick_busiest_week(bos_events, choch_events):
    times = pd.Series([e["started_at"] for e in bos_events + choch_events])
    weeks = times.dt.tz_localize(None).dt.to_period("W-SUN")
    return weeks.value_counts().idxmax()


def pick_typical_day(bos_events, choch_events, percentile=0.6):
    """A day with a representative event count — not the busiest, not the
    quietest — so lookback settings are compared on a fair sample."""
    times = pd.Series([e["started_at"] for e in bos_events + choch_events])
    counts = times.dt.date.value_counts().rename_axis("day").reset_index(name="count")
    counts = counts.sort_values(["count", "day"]).reset_index(drop=True)
    idx = min(int(len(counts) * percentile), len(counts) - 1)
    return counts.loc[idx, "day"]


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)


def draw_ohlc(ax, df, tick_frac=0.2):
    if len(df) < 2:
        return
    step = (df["close_time_utc"].iloc[1] - df["close_time_utc"].iloc[0]) * tick_frac
    for _, bar in df.iterrows():
        t = bar["close_time_utc"]
        ax.plot([t, t], [bar["low"], bar["high"]], color=INK, linewidth=1, zorder=2)
        ax.plot([t - step, t], [bar["open"], bar["open"]], color=INK, linewidth=1, zorder=2)
        ax.plot([t, t + step], [bar["close"], bar["close"]], color=INK, linewidth=1, zorder=2)


def plot_week(symbol="USTECm", timeframe="15M", train_end="2023-12-31"):
    df = load_pinned(symbol, timeframe, end=train_end)
    swings = find_swings(df)
    bos_events, choch_events = compute_structure(df)

    week = pick_busiest_week(bos_events, choch_events)
    week_start = week.start_time.tz_localize("UTC")
    week_end = week.end_time.tz_localize("UTC")

    week_df = df[(df["close_time_utc"] >= week_start) & (df["close_time_utc"] <= week_end)].reset_index(drop=True)
    week_swings = [s for s in swings if week_start <= s["started_at"] <= week_end]
    week_bos = [e for e in bos_events if week_start <= e["started_at"] <= week_end]
    week_choch = [e for e in choch_events if week_start <= e["started_at"] <= week_end]

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)
    draw_ohlc(ax, week_df)

    for s in week_swings:
        marker = "v" if s["type"] == "high" else "^"
        offset = 1.0015 if s["type"] == "high" else 0.9985
        ax.scatter(
            s["started_at"], s["price"] * offset,
            color=COLOR_SWING, marker=marker, s=70, zorder=5, edgecolors=SURFACE, linewidths=0.5,
        )

    for e in week_bos:
        arrow = "↑" if e["direction"] == "up" else "↓"
        ax.axvline(e["started_at"], color=COLOR_BOS, linestyle="-", linewidth=1.2, alpha=0.8, zorder=3)
        ax.annotate(
            f"BOS {arrow}", (e["started_at"], ax.get_ylim()[1]),
            color=COLOR_BOS, fontsize=8, fontweight="bold", ha="left", va="top",
            xytext=(3, -3), textcoords="offset points",
        )

    for e in week_choch:
        arrow = "↑" if e["direction"] == "up" else "↓"
        ax.axvline(e["started_at"], color=COLOR_CHOCH, linestyle="--", linewidth=1.2, alpha=0.8, zorder=3)
        ax.annotate(
            f"CHoCH {arrow}", (e["started_at"], ax.get_ylim()[1]),
            color=COLOR_CHOCH, fontsize=8, fontweight="bold", ha="left", va="top",
            xytext=(3, -15), textcoords="offset points",
        )

    ax.legend(handles=LEGEND_HANDLES, loc="upper left", frameon=False, labelcolor=INK)
    ax.set_title(f"{symbol} {timeframe} — {week_start.date()} to {week_end.date()}", color=INK, fontsize=13, loc="left")
    ax.set_ylabel("price", color=MUTED)
    fig.autofmt_xdate()

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{symbol}_{timeframe}_week_structure.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_day_swing_comparison(symbol="USTECm", timeframe="15M", train_end="2023-12-31", lookbacks=(2, 3, 5)):
    """Same trading day, rendered once per swing lookback (`lookbacks`
    candles each side), so the lookback can be chosen by eye rather than
    by outcome. Returns (image_path, counts_df) where counts_df holds
    swing/BOS/CHoCH totals over the full train range per lookback."""
    df = load_pinned(symbol, timeframe, end=train_end)

    mid = lookbacks[len(lookbacks) // 2]
    ref_bos, ref_choch = compute_structure(df, left=mid, right=mid)
    day = pick_typical_day(ref_bos, ref_choch)
    day_start = pd.Timestamp(day, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)

    fig, axes = plt.subplots(1, len(lookbacks), figsize=(7.5 * len(lookbacks), 7), sharey=False)
    fig.patch.set_facecolor(SURFACE)
    counts_rows = []

    for ax, n in zip(axes, lookbacks):
        swings = find_swings(df, left=n, right=n)
        bos_events, choch_events = compute_structure(df, left=n, right=n)

        counts_rows.append(
            {
                "candles_each_side": n,
                "swings_total_train": len(swings),
                "bos_total_train": len(bos_events),
                "choch_total_train": len(choch_events),
            }
        )

        day_df = df[(df["close_time_utc"] >= day_start) & (df["close_time_utc"] < day_end)].reset_index(drop=True)
        day_swings = [s for s in swings if day_start <= s["started_at"] < day_end]
        day_events = sorted(
            [(e, True) for e in bos_events if day_start <= e["started_at"] < day_end]
            + [(e, False) for e in choch_events if day_start <= e["started_at"] < day_end],
            key=lambda pair: pair[0]["started_at"],
        )

        style_axis(ax)
        draw_ohlc(ax, day_df)

        if len(day_df):
            y0, y1 = day_df["low"].min(), day_df["high"].max()
        else:
            y0, y1 = 0, 1
        pad = max((y1 - y0) * 0.05, 1e-6)
        # Marker offset must stay well inside `pad`, or a swing sitting at
        # the day's actual high/low gets pushed past the axis limit set
        # below and silently clipped — it happened to the day's low here,
        # since a fixed 0.1%-of-price offset can exceed a 5%-of-range pad
        # whenever the day's range is narrow relative to the price level.
        marker_offset = pad * 0.3
        n_slots = 8
        ax.set_ylim(y0 - pad, y1 + pad * (1.5 * n_slots))

        for s in day_swings:
            marker = "v" if s["type"] == "high" else "^"
            y = s["price"] + marker_offset if s["type"] == "high" else s["price"] - marker_offset
            ax.scatter(
                s["started_at"], y,
                color=COLOR_SWING, marker=marker, s=90, zorder=5, edgecolors=SURFACE, linewidths=0.6,
            )

        for i, (e, is_bos) in enumerate(day_events):
            color = COLOR_BOS if is_bos else COLOR_CHOCH
            style = "-" if is_bos else "--"
            label = "BOS" if is_bos else "CHoCH"
            arrow = "↑" if e["direction"] == "up" else "↓"
            ax.axvline(e["started_at"], color=color, linestyle=style, linewidth=1.1, alpha=0.75, zorder=3)
            slot = i % n_slots
            y = (y1 + pad) + slot * (pad * 1.4)
            ax.annotate(
                f"{label} {arrow}", (e["started_at"], y),
                color=color, fontsize=7.5, fontweight="bold", ha="center", va="bottom",
            )

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_title(
            f"{n} candles each side\n"
            f"day: swings={len(day_swings)} bos={sum(1 for _, b in day_events if b)} "
            f"choch={sum(1 for _, b in day_events if not b)}",
            color=INK, fontsize=11,
        )

    axes[0].set_ylabel("price", color=MUTED)
    fig.autofmt_xdate()
    fig.legend(handles=LEGEND_HANDLES, loc="lower center", ncol=4, frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{symbol} {timeframe} — {day} — swing lookback comparison", color=INK, fontsize=14)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{symbol}_{timeframe}_day_swing_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    return out_path, pd.DataFrame(counts_rows)


def plot_day_sweeps(symbol="USTECm", timeframe="15M", day="2021-08-12", train_end="2023-12-31", left=None, right=None):
    """Same single day as the swing-lookback comparison, with swings and
    liquidity sweeps marked. Structure/sweeps are computed over the full
    train range so nothing before `day` is missing context."""
    from .bos import LEFT, RIGHT

    left = LEFT if left is None else left
    right = RIGHT if right is None else right

    df = load_pinned(symbol, timeframe, end=train_end)
    swings = find_swings(df, left, right)
    sweeps = detect_sweeps(df, left, right)

    day_start = pd.Timestamp(day, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)

    day_df = df[(df["close_time_utc"] >= day_start) & (df["close_time_utc"] < day_end)].reset_index(drop=True)
    day_swings = [s for s in swings if day_start <= s["started_at"] < day_end]
    day_sweeps = [e for e in sweeps if day_start <= e["started_at"] < day_end]

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)
    draw_ohlc(ax, day_df)

    if len(day_df):
        y0, y1 = day_df["low"].min(), day_df["high"].max()
    else:
        y0, y1 = 0, 1
    pad = max((y1 - y0) * 0.05, 1e-6)
    marker_offset = pad * 0.3
    ax.set_ylim(y0 - pad, y1 + pad)

    for s in day_swings:
        marker = "v" if s["type"] == "high" else "^"
        y = s["price"] + marker_offset if s["type"] == "high" else s["price"] - marker_offset
        ax.scatter(
            s["started_at"], y,
            color=COLOR_SWING, marker=marker, s=90, zorder=5, edgecolors=SURFACE, linewidths=0.6,
        )

    for e in day_sweeps:
        wick_time = e["started_at"]
        wick_row = day_df[day_df["close_time_utc"] == wick_time]
        if wick_row.empty:
            continue
        extreme = wick_row["high"].iloc[0] if e["direction"] == "up" else wick_row["low"].iloc[0]

        ax.scatter(wick_time, extreme, color=COLOR_SWEEP, marker="x", s=90, zorder=6, linewidths=2)
        ax.plot(
            [e["started_at"], e["knowable_at"]], [e["swing_price"], e["swing_price"]],
            color=COLOR_SWEEP, linestyle=":", linewidth=1.4, zorder=4,
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(handles=SWEEP_LEGEND_HANDLES, loc="upper left", frameon=False, labelcolor=INK)
    ax.set_title(
        f"{symbol} {timeframe} — {day} — swing={left} each side, sweeps={len(day_sweeps)}",
        color=INK, fontsize=13, loc="left",
    )
    ax.set_ylabel("price", color=MUTED)
    fig.autofmt_xdate()

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{symbol}_{timeframe}_day_sweeps.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    return out_path, sweeps


def plot_day_zones(symbol="USTECm", timeframe="15M", day="2021-08-12", train_end="2023-12-31", left=None, right=None):
    """Same single day, with swings and supply/demand zones marked as
    shaded boxes spanning each zone's origin candle through its death
    (or through the plotted day, if it's still alive)."""
    from .bos import LEFT, RIGHT

    left = LEFT if left is None else left
    right = RIGHT if right is None else right

    df = load_pinned(symbol, timeframe, end=train_end)
    swings = find_swings(df, left, right)
    zones = detect_zones(df)

    day_start = pd.Timestamp(day, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)
    last_ts = df["close_time_utc"].iloc[-1]

    day_df = df[(df["close_time_utc"] >= day_start) & (df["close_time_utc"] < day_end)].reset_index(drop=True)
    day_swings = [s for s in swings if day_start <= s["started_at"] < day_end]
    day_zones = [
        z for z in zones
        if z["started_at"] <= day_end and (z["died_at"] or last_ts) >= day_start
    ]

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)
    draw_ohlc(ax, day_df)

    if len(day_df):
        y0, y1 = day_df["low"].min(), day_df["high"].max()
    else:
        y0, y1 = 0, 1
    pad = max((y1 - y0) * 0.05, 1e-6)
    marker_offset = pad * 0.3
    ax.set_ylim(y0 - pad, y1 + pad)

    for z in day_zones:
        x0 = max(z["started_at"], day_start)
        x1 = min(z["died_at"] or day_end, day_end)
        color = COLOR_DEMAND if z["zone_type"] == "demand" else COLOR_SUPPLY
        hatch = "//" if z["zone_type"] == "demand" else "\\\\"
        rect = Rectangle(
            (mdates.date2num(x0), z["low"]), mdates.date2num(x1) - mdates.date2num(x0), z["high"] - z["low"],
            facecolor=color, edgecolor=color, alpha=0.18, hatch=hatch, linewidth=1, zorder=1,
        )
        ax.add_patch(rect)

    for s in day_swings:
        marker = "v" if s["type"] == "high" else "^"
        y = s["price"] + marker_offset if s["type"] == "high" else s["price"] - marker_offset
        ax.scatter(
            s["started_at"], y,
            color=COLOR_SWING, marker=marker, s=90, zorder=5, edgecolors=SURFACE, linewidths=0.6,
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(handles=ZONE_LEGEND_HANDLES, loc="upper left", frameon=False, labelcolor=INK)
    ax.set_title(
        f"{symbol} {timeframe} — {day} — zones overlapping this day: {len(day_zones)}",
        color=INK, fontsize=13, loc="left",
    )
    ax.set_ylabel("price", color=MUTED)
    fig.autofmt_xdate()

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{symbol}_{timeframe}_day_zones.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    return out_path, zones


def plot_setup(symbol, setup, ax):
    """Draw one setup's full chain — sweep, CHoCH, BOS, its zone, pullback,
    entry, stop and target — on a 15M window from just before the trend
    confirmed through just after entry."""
    steps = setup["steps"]
    is_long = setup["direction"] == "long"

    window_start = steps["trend_confirmed_at"] - pd.Timedelta(hours=6)
    window_end = steps["entry_knowable_at"] + pd.Timedelta(hours=3)
    df = load_pinned(symbol, "15M", start=window_start, end=window_end)

    style_axis(ax)
    draw_ohlc(ax, df)

    if len(df):
        y0, y1 = df["low"].min(), df["high"].max()
    else:
        y0, y1 = 0, 1
    pad = max((y1 - y0) * 0.05, 1e-6)
    ax.set_ylim(y0 - pad, y1 + pad)

    zone = setup["zone"]
    zone_color = COLOR_DEMAND if zone["zone_type"] == "demand" else COLOR_SUPPLY
    hatch = "//" if zone["zone_type"] == "demand" else "\\\\"
    zone_x0 = zone["started_at"]
    zone_x1 = min(zone["died_at"] or window_end, window_end)
    rect = Rectangle(
        (mdates.date2num(zone_x0), zone["low"]), mdates.date2num(zone_x1) - mdates.date2num(zone_x0),
        zone["high"] - zone["low"], facecolor=zone_color, edgecolor=zone_color, alpha=0.18, hatch=hatch, zorder=1,
    )
    ax.add_patch(rect)

    sweep = setup["sweep"]
    wick_row = df[df["close_time_utc"] == sweep["started_at"]]
    if not wick_row.empty:
        extreme = wick_row["low"].iloc[0] if is_long else wick_row["high"].iloc[0]
        ax.scatter(sweep["started_at"], extreme, color=COLOR_SWEEP, marker="x", s=110, zorder=6, linewidths=2.2)

    ax.axvline(steps["choch_knowable_at"], color=COLOR_CHOCH, linestyle="--", linewidth=1.3, alpha=0.8, zorder=3)
    ax.axvline(steps["bos_knowable_at"], color=COLOR_BOS, linestyle="-", linewidth=1.3, alpha=0.8, zorder=3)

    pullback_row = df[df["close_time_utc"] == steps["pullback_knowable_at"]]
    if not pullback_row.empty:
        pb_price = (pullback_row["high"].iloc[0] + pullback_row["low"].iloc[0]) / 2
        ax.scatter(
            steps["pullback_knowable_at"], pb_price, marker="o", s=90, zorder=6,
            facecolor=SURFACE, edgecolor=INK, linewidths=1.3,
        )

    ax.scatter(
        steps["entry_knowable_at"], setup["entry_price"], marker="*", s=260, zorder=7,
        facecolor=COLOR_ENTRY, edgecolor=INK, linewidths=0.8,
    )
    ax.axhline(setup["stop_price"], color=INK, linestyle=":", linewidth=1.1, alpha=0.7, zorder=2)
    ax.axhline(setup["target_price"], color=INK, linestyle=":", linewidth=1.1, alpha=0.7, zorder=2)
    ax.annotate("stop", (df["close_time_utc"].iloc[0], setup["stop_price"]), color=MUTED, fontsize=8, va="bottom")
    ax.annotate("target", (df["close_time_utc"].iloc[0], setup["target_price"]), color=MUTED, fontsize=8, va="bottom")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    ax.set_title(
        f"{symbol} {setup['direction'].upper()} — entry {steps['entry_knowable_at']:%Y-%m-%d %H:%M} — "
        f"R={setup['risk']:.2f}",
        color=INK, fontsize=11, loc="left",
    )


def plot_setup_examples(setups_by_symbol, n=3):
    """`setups_by_symbol`: {symbol: [setups...]}. Picks up to `n` examples
    (one per symbol where possible) and renders each on its own panel."""
    chosen = []
    for symbol, setups in setups_by_symbol.items():
        if setups:
            chosen.append((symbol, setups[0]))
        if len(chosen) >= n:
            break

    fig, axes = plt.subplots(len(chosen), 1, figsize=(13, 6 * len(chosen)))
    fig.patch.set_facecolor(SURFACE)
    if len(chosen) == 1:
        axes = [axes]

    for ax, (symbol, setup) in zip(axes, chosen):
        plot_setup(symbol, setup, ax)

    fig.legend(handles=SETUP_LEGEND_HANDLES, loc="lower center", ncol=4, frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, -0.02 / len(chosen)))
    fig.suptitle("Example setups: sweep -> CHoCH -> BOS -> pullback -> entry", color=INK, fontsize=14)
    fig.autofmt_xdate()

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "setup_examples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path, counts = plot_day_swing_comparison()
    print(f"Saved plot to {path}")
    print(counts.to_string(index=False))
