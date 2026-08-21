"""
Presentation-quality EDA figures.

Matplotlib only -- no seaborn or plotly, because every chart here is a static
report figure and a third plotting library would be a dependency with no job.

Design rules, applied uniformly:
  * One idea per figure.
  * Direct value labels instead of gridlines where the count is the point.
  * Consistent platform colours across every figure, so a reader learns the
    legend once.
  * Captions state the caveat on the figure itself. A chart that leaves the
    dataset's sampling bias in the surrounding prose will eventually be
    screenshotted without it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed, deterministic output
import matplotlib.pyplot as plt
import pandas as pd

# Brand-adjacent, and distinguishable in greyscale as well as colour.
PLATFORM_COLOURS: dict[str, str] = {
    "blinkit": "#E8B21E",
    "zepto": "#7B3FE4",
    "jiomart": "#0A5FB4",
}
NEUTRAL = "#4A5568"
ACCENT_NEGATIVE = "#C53030"
ACCENT_POSITIVE = "#2F855A"

FIGSIZE = (10, 5.5)
DPI = 160


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E0",
            "axes.labelcolor": "#2D3748",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#4A5568",
            "ytick.color": "#4A5568",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
        }
    )


def _save(fig: plt.Figure, path: Path, caption: str | None = None) -> Path:
    """Lay out, add the caveat text, and write the figure.

    The caption goes on the figure itself rather than only in the surrounding
    prose, because a chart that carries its caveat separately will eventually be
    screenshotted without it. Layout reserves space first so the caption cannot
    collide with rotated tick labels.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if caption:
        fig.tight_layout(rect=(0, 0.10, 1, 1))
        fig.text(
            0.01, 0.015, caption,
            fontsize=7.5, color="#718096", ha="left", va="bottom", wrap=True,
        )
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def collapse_sparse_tail(monthly: pd.DataFrame, keep_from: str) -> pd.DataFrame:
    """Fold the long thin pre-``keep_from`` tail into a single aggregate column.

    The corpus spans 54 months but 96% of it sits in the last six. Plotting all
    54 devotes most of the axis to near-empty cells and squashes the part that
    carries the information. Aggregating the tail keeps it visible and honest
    without letting it dominate.
    """
    tail = monthly.loc[monthly.index < keep_from]
    head = monthly.loc[monthly.index >= keep_from]
    if tail.empty:
        return head
    label = f"before\n{keep_from}"
    aggregated = pd.DataFrame([tail.sum()], index=[label])
    return pd.concat([aggregated, head])


def _colours_for(columns) -> list[str]:
    return [PLATFORM_COLOURS.get(str(c), NEUTRAL) for c in columns]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_rating_distribution(frame: pd.DataFrame, out: Path) -> Path:
    _style()
    counts = frame["rating"].value_counts().sort_index()
    total = counts.sum()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    colours = [ACCENT_NEGATIVE if r <= 2 else NEUTRAL if r == 3 else ACCENT_POSITIVE for r in counts.index]
    bars = ax.bar([f"{r}★" for r in counts.index], counts.values, color=colours, width=0.62)

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + total * 0.012,
            f"{value:,}\n{value / total * 100:.1f}%",
            ha="center", va="bottom", fontsize=9, color="#2D3748",
        )

    ax.set_title("Rating distribution — heavily skewed negative")
    ax.set_ylabel("Reviews")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    return _save(
        fig, out,
        f"n={total:,}. {(frame['rating'] <= 2).mean() * 100:.1f}% negative (1–2★). "
        "This is a complaint-biased sample of app-store reviews, not a representative "
        "customer survey — suitable for finding problems, not for measuring satisfaction.",
    )


def plot_platform_distribution(frame: pd.DataFrame, out: Path) -> Path:
    _style()
    counts = frame["platform"].value_counts()
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(9, 4))
    left = 0.0
    for platform, value in counts.items():
        ax.barh(
            0, value, left=left, height=0.5,
            color=PLATFORM_COLOURS.get(str(platform), NEUTRAL),
            edgecolor="white", linewidth=2,
        )
        ax.text(
            left + value / 2, 0,
            f"{platform}\n{value:,} ({value / total * 100:.1f}%)",
            ha="center", va="center", color="white", fontsize=10, fontweight="bold",
        )
        left += value

    ax.set_xlim(0, total)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    ax.set_title("Reviews by platform", pad=16)
    return _save(fig, out, f"n={total:,}. Volume is reasonably balanced; date coverage is not — see the coverage figure.")


def plot_monthly_volume(monthly: pd.DataFrame, out: Path) -> Path:
    _style()
    data = monthly.drop(columns=["total"], errors="ignore")
    months = [str(m) for m in data.index]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = pd.Series(0.0, index=data.index)
    for platform in data.columns:
        ax.bar(
            months, data[platform], bottom=bottom, width=0.68,
            label=str(platform), color=PLATFORM_COLOURS.get(str(platform), NEUTRAL),
        )
        bottom += data[platform]

    # Only annotate months substantial enough that the number is legible.
    threshold = bottom.max() * 0.02
    for x, value in enumerate(bottom.values):
        if value > threshold:
            ax.text(x, value + bottom.max() * 0.015, f"{int(value):,}", ha="center", fontsize=8, color="#4A5568")

    ax.set_title("Monthly review volume — a scraping pattern, not a demand curve")
    ax.set_ylabel("Reviews")
    ax.legend(ncol=len(data.columns), loc="upper left")
    ax.set_ylim(0, bottom.max() * 1.16)

    # 54 months of labels is unreadable; label every third and keep the last.
    step = max(1, len(months) // 18)
    ticks = list(range(0, len(months), step))
    if len(months) - 1 not in ticks:
        ticks.append(len(months) - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([months[i] for i in ticks], rotation=90, fontsize=8)

    return _save(
        fig, out,
        "Volume reflects when reviews were collected, NOT customer demand. Do not read the rise "
        "into December as growth; it is the shape of the scrape. The corpus spans 54 months but "
        "96% of it falls in the last six.",
    )


def plot_platform_coverage(monthly: pd.DataFrame, window_months: list[str], out: Path) -> Path:
    """Coverage heatmap: shows exactly where cross-platform comparison is valid."""
    _style()
    import numpy as np

    raw = monthly.drop(columns=["total"], errors="ignore")
    keep_from = min(window_months) if window_months else "2024-07"
    # Step back a few months from the window so the run-up stays visible.
    all_months = [str(m) for m in raw.index]
    earlier = [m for m in all_months if m < keep_from]
    context_from = earlier[-3] if len(earlier) >= 3 else keep_from
    data = collapse_sparse_tail(raw, context_from)
    labels = [str(m) for m in data.index]

    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    values = np.log10(data.T.values + 1)
    ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=values.max())

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(data.columns)))
    ax.set_yticklabels([str(c) for c in data.columns], fontsize=10)
    ax.tick_params(length=0)

    for row in range(data.shape[1]):
        for col in range(data.shape[0]):
            count = int(data.iloc[col, row])
            ax.text(
                col, row, f"{count:,}" if count else "0",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if values[row, col] > values.max() * 0.62 else "#2D3748",
            )

    if window_months:
        indices = [labels.index(m) for m in window_months if m in labels]
        if indices:
            ax.add_patch(
                plt.Rectangle(
                    (min(indices) - 0.5, -0.5), len(indices), data.shape[1],
                    fill=False, edgecolor=ACCENT_NEGATIVE, linewidth=2.5, zorder=5,
                )
            )
            ax.annotate(
                "valid comparison window",
                xy=(min(indices) - 0.5 + len(indices) / 2, -0.5),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, color=ACCENT_NEGATIVE, fontweight="bold",
            )

    ax.set_title("Platform coverage by month — reviews collected", pad=28)
    return _save(
        fig, out,
        "The first column aggregates every month before it. Summed across ALL months preceding "
        "the window, Blinkit contributes 12 reviews and Zepto 1 — outside the boxed window the "
        "corpus is effectively JioMart-only, so any cross-platform difference measured there "
        "would reflect scrape coverage rather than customers.",
    )


def plot_rating_by_platform(frame: pd.DataFrame, out: Path) -> Path:
    _style()
    shares = (
        pd.crosstab(frame["platform"], frame["rating"], normalize="index").mul(100).sort_index()
    )
    means = frame.groupby("platform", observed=True)["rating"].mean()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    palette = {1: "#9B2C2C", 2: "#C53030", 3: "#A0AEC0", 4: "#48BB78", 5: "#2F855A"}
    left = pd.Series(0.0, index=shares.index)
    for rating in shares.columns:
        ax.barh(
            [str(p) for p in shares.index], shares[rating], left=left, height=0.55,
            color=palette.get(int(rating), NEUTRAL), label=f"{rating}★",
        )
        for y, (value, base) in enumerate(zip(shares[rating], left)):
            if value >= 6:
                ax.text(base + value / 2, y, f"{value:.0f}%", ha="center", va="center",
                        color="white", fontsize=8.5, fontweight="bold")
        left += shares[rating]

    for y, platform in enumerate(shares.index):
        ax.text(101, y, f"mean {means[platform]:.2f}★", va="center", fontsize=9, color="#2D3748")

    ax.set_xlim(0, 118)
    ax.set_xlabel("Share of that platform's reviews (%)")
    ax.set_title("Rating mix by platform — within this dataset only")
    ax.legend(ncol=5, loc="lower right", bbox_to_anchor=(1.0, -0.28))
    return _save(
        fig, out,
        "Within this dataset Blinkit has a higher share of positive reviews. This is NOT evidence "
        "that Blinkit customers are more satisfied — the platforms may have been scraped under "
        "different sort orders or filters.",
    )


def plot_review_length(frame: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})

    ax_left.hist(frame["char_len"], bins=50, color=NEUTRAL, alpha=0.85)
    cap = frame.loc[frame["is_truncated"], "char_len"].min() if frame["is_truncated"].any() else None
    if cap is not None:
        ax_left.axvline(cap, color=ACCENT_NEGATIVE, linestyle="--", linewidth=1.5)
        ax_left.text(
            cap - 8, ax_left.get_ylim()[1] * 0.92,
            f"scraper cap\n{frame['is_truncated'].sum():,} reviews "
            f"({frame['is_truncated'].mean() * 100:.1f}%)",
            ha="right", fontsize=8, color=ACCENT_NEGATIVE,
        )
    ax_left.set_title("Review length distribution")
    ax_left.set_xlabel("Characters (whitespace-normalised)")
    ax_left.set_ylabel("Reviews")

    data = [frame.loc[frame["platform"] == p, "char_len"].values for p in PLATFORM_COLOURS if (frame["platform"] == p).any()]
    labels = [p for p in PLATFORM_COLOURS if (frame["platform"] == p).any()]
    # matplotlib >= 3.9 renamed the boxplot `labels` kwarg to `tick_labels`.
    box = ax_right.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5, showfliers=False)
    for patch, platform in zip(box["boxes"], labels):
        patch.set_facecolor(PLATFORM_COLOURS[platform])
        patch.set_alpha(0.85)
    for median in box["medians"]:
        median.set_color("white")
        median.set_linewidth(2)
    ax_right.set_title("Length by platform")
    ax_right.set_ylabel("Characters")

    return _save(
        fig, out,
        "Reviews at the cap are cut before any resolution is described, which biases their "
        "sentiment negative — they average 1.36★ against 1.87★ for intact reviews.",
    )


def plot_rating_over_time(over_time: pd.DataFrame, monthly: pd.DataFrame, out: Path) -> Path:
    """Mean rating per platform per month, suppressing thin months."""
    _style()
    counts = monthly.drop(columns=["total"], errors="ignore")
    fig, ax = plt.subplots(figsize=(11, 5))

    months = [str(m) for m in over_time.index]
    for platform in over_time.columns:
        series = over_time[platform].copy()
        # Blank out months too thin to mean anything, rather than drawing a
        # confident line through a single review.
        thin = counts[platform] < 30 if platform in counts else None
        if thin is not None:
            series[thin.reindex(series.index, fill_value=True)] = float("nan")
        ax.plot(
            months, series, marker="o", markersize=5, linewidth=2,
            color=PLATFORM_COLOURS.get(str(platform), NEUTRAL), label=str(platform),
        )

    ax.set_ylim(1, 5)
    ax.set_ylabel("Mean rating")
    ax.set_title("Mean rating over time — months with under 30 reviews omitted")
    ax.legend(ncol=3)
    step = max(1, len(months) // 18)
    ticks = list(range(0, len(months), step))
    if len(months) - 1 not in ticks:
        ticks.append(len(months) - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([months[i] for i in ticks], rotation=90, fontsize=8)
    ax.grid(axis="y", color="#EDF2F7", linewidth=1)
    ax.set_axisbelow(True)
    return _save(
        fig, out,
        "Gaps are months where a platform had too few reviews to average meaningfully. "
        "Only the final three months support a genuine cross-platform read.",
    )


def plot_theme_comparison(share_window: pd.DataFrame, out: Path) -> Path:
    _style()
    platforms = [c for c in share_window.columns if c != "all"]
    data = share_window.sort_values("all", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    import numpy as np

    y = np.arange(len(data))
    height = 0.8 / len(platforms)
    for offset, platform in enumerate(platforms):
        ax.barh(
            y + offset * height, data[platform], height=height,
            color=PLATFORM_COLOURS.get(str(platform), NEUTRAL), label=str(platform),
        )

    ax.set_yticks(y + height * (len(platforms) - 1) / 2)
    ax.set_yticklabels([str(i) for i in data.index], fontsize=9)
    ax.set_xlabel("Share of that platform's reviews mentioning the theme (%)")
    ax.set_title("Exploratory theme shares by platform — comparison window only")
    ax.legend(ncol=len(platforms), loc="lower right")
    ax.grid(axis="x", color="#EDF2F7", linewidth=1)
    ax.set_axisbelow(True)
    return _save(
        fig, out,
        "KEYWORD-PROBE ESTIMATES, not classifications — high precision, mediocre recall, so every "
        "share is a lower bound. Restricted to the comparison window so platforms are compared "
        "over the same period. Final labels come from Phase 3.",
    )


def plot_negative_share_over_time(neg_share: pd.DataFrame, out: Path) -> Path:
    """The confound that makes raw complaint counts unusable."""
    _style()
    fig, ax_left = plt.subplots(figsize=(11, 5))
    months = [str(m) for m in neg_share.index]

    ax_left.bar(months, neg_share["reviews"], color="#CBD5E0", width=0.68, label="reviews collected")
    ax_left.set_ylabel("Reviews collected", color="#718096")
    step = max(1, len(months) // 18)
    ticks = list(range(0, len(months), step))
    if len(months) - 1 not in ticks:
        ticks.append(len(months) - 1)
    ax_left.set_xticks(ticks)
    ax_left.set_xticklabels([months[i] for i in ticks], rotation=90, fontsize=8)

    ax_right = ax_left.twinx()
    ax_right.plot(
        months, neg_share["negative_share_pct"], marker="o", markersize=5,
        linewidth=2.2, color=ACCENT_NEGATIVE, label="% negative",
    )
    ax_right.set_ylabel("Negative share (%)", color=ACCENT_NEGATIVE)
    ax_right.set_ylim(0, 105)
    ax_right.spines["right"].set_visible(True)

    ax_left.set_title("Sampling composition changes over time")
    return _save(
        fig, out,
        "The negative share of the scrape drifts month to month, so raw complaint COUNTS measure "
        "scrape composition rather than customer experience. Trends must be reported as shares.",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_all(frame: pd.DataFrame, result, figures_dir: Path) -> list[Path]:
    """Render every EDA figure. Returns the paths written, in report order."""
    frame = frame.copy()
    frame["platform"] = frame["platform"].astype(str)

    return [
        plot_rating_distribution(frame, figures_dir / "01_rating_distribution.png"),
        plot_platform_distribution(frame, figures_dir / "02_platform_distribution.png"),
        plot_monthly_volume(result.temporal.monthly_volume, figures_dir / "03_monthly_volume.png"),
        plot_platform_coverage(
            result.temporal.monthly_volume,
            result.temporal.months_meeting_threshold,
            figures_dir / "04_platform_coverage.png",
        ),
        plot_rating_by_platform(frame, figures_dir / "05_rating_by_platform.png"),
        plot_review_length(frame, figures_dir / "06_review_length.png"),
        plot_rating_over_time(
            result.ratings.over_time, result.temporal.monthly_volume,
            figures_dir / "07_rating_over_time.png",
        ),
        plot_negative_share_over_time(
            result.temporal.monthly_negative_share, figures_dir / "08_sampling_composition.png"
        ),
        plot_theme_comparison(result.themes.share_window, figures_dir / "09_theme_comparison.png"),
    ]
