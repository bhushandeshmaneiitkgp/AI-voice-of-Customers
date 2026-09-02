"""
Competitive metrics across platforms, and a trend analysis if the data allows one.

This is the Phase 5 deliverable. It answers "is it worse on us than on them"
with tested differences, and reports honestly on whether "is it getting worse"
is answerable at all.

    # full analysis over the comparable window
    python scripts/07_analyse_trends.py

    # include pre-window months (NOT recommended -- see the report caveats)
    python scripts/07_analyse_trends.py --all-months

    # stricter or looser significance
    python scripts/07_analyse_trends.py --alpha 0.01

Reads  : data/processed/reviews_enriched.parquet
         data/processed/review_labels.parquet
Writes : data/processed/platform_metrics.parquet
         data/processed/platform_comparisons.parquet
         data/processed/area_rates_by_platform.parquet
         data/processed/monthly_rates.parquet
         reports/COMPETITIVE.md
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- must precede project imports
import argparse
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

from config.settings import Paths, get_settings
from voc.trends import (
    ALPHA,
    MIN_MONTHS_FOR_TREND,
    METRICS,
    TrendVerdict,
    area_rates_by_platform,
    assess_trend_feasibility,
    compare_platforms,
    month_over_month_change,
    monthly_rates,
    platform_metrics,
)


def build_report(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    areas: pd.DataFrame,
    monthly: pd.DataFrame,
    changes: pd.DataFrame,
    verdict: TrendVerdict,
    reviews: pd.DataFrame,
    alpha: float,
) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    platforms = sorted(reviews["platform"].unique())
    lines: list[str] = []
    add = lines.append

    add("# Competitive Metrics — Quick-Commerce Voice of Customer")
    add("")
    add(f"**{generated}** · {len(reviews):,} reviews in the comparable window · "
        f"{len(platforms)} platforms · α = {alpha}")
    add("")
    add("Phase 4 ranked what hurts. This asks whether it hurts *more here than "
        "elsewhere*, and whether it is getting worse.")
    add("")
    add("---")
    add("")

    # --- Method, stated before any number ----------------------------------
    add("## How to read this")
    add("")
    add("**Every figure is a rate, never a count.** December holds three times "
        "October's reviews — that is scraping intensity, not customer behaviour. "
        "A count-based table would rank December worst on every measure for "
        "every platform, automatically.")
    add("")
    add("**Every difference is tested.** With ~900–1,700 reviews per platform, a "
        "few points of gap is well inside noise. Rates carry Wilson 95% "
        "intervals; comparisons carry p-values corrected for multiplicity with "
        "Benjamini–Hochberg. The table below runs dozens of tests, and at "
        "α=0.05 a couple would clear the bar on noise alone — so **read the "
        "`significant` column, not the raw p-value**.")
    add("")
    add("Reviews per platform in the window:")
    add("")
    add("| Platform | Reviews |")
    add("|---|---:|")
    for platform in platforms:
        add(f"| `{platform}` | {int((reviews['platform'] == platform).sum()):,} |")
    add("")
    add("---")
    add("")

    # --- Headline rates ----------------------------------------------------
    add("## Platform rates")
    add("")
    add("Wilson 95% intervals. Overlapping intervals mean the difference is not "
        "established, whatever the point estimates suggest.")
    add("")
    for spec in METRICS:
        subset = metrics[metrics["metric"] == spec.key].sort_values(
            "rate", ascending=not spec.higher_is_worse
        )
        if subset.empty:
            continue
        direction = "higher is worse" if spec.higher_is_worse else "higher is better"
        add(f"**{spec.label}** ({direction})")
        add("")
        add("| Platform | Rate | 95% CI |")
        add("|---|---:|---|")
        for row in subset.itertuples():
            add(f"| `{row.platform}` | {row.rate * 100:.1f}% | "
                f"{row.ci_low * 100:.1f}–{row.ci_high * 100:.1f}% |")
        add("")

    # --- What actually differs --------------------------------------------
    add("### Differences that survive correction")
    add("")
    established = comparisons[comparisons["significant"]]
    if established.empty:
        add("**None.** No pairwise platform difference on these metrics survives "
            "FDR correction. On this corpus the platforms are not "
            "distinguishable on sentiment, severity, escalation or churn — which "
            "is itself the competitive finding, and the opposite of what a table "
            "of raw percentages would have implied.")
    else:
        add(f"{len(established)} of {len(comparisons)} comparisons survive.")
        add("")
        add("| Metric | A | B | Rate A | Rate B | Difference | p |")
        add("|---|---|---|---:|---:|---:|---:|")
        for row in established.itertuples():
            add(f"| {row.label} | `{row.platform_a}` | `{row.platform_b}` | "
                f"{row.rate_a * 100:.1f}% | {row.rate_b * 100:.1f}% | "
                f"{row.difference * 100:+.1f}pp | {row.p_value:.4f} |")
        add("")
        not_established = comparisons[~comparisons["significant"]]
        add(f"The remaining {len(not_established)} comparisons are **not "
            "established**. Their point estimates differ; the evidence does not "
            "support saying so.")
    add("")
    add("---")
    add("")

    # --- Where each platform over-indexes ----------------------------------
    add("## Where each platform over-indexes")
    add("")
    add("Share of a platform's own reviews raising each area, against the same "
        "rate across the other platforms pooled. `lift` above 1.0 means the area "
        "is raised more often here than corpus-wide.")
    add("")
    flagged = areas[areas["significant"]].sort_values("lift", ascending=False)
    if flagged.empty:
        add("No area differs significantly by platform after correction.")
    else:
        add("| Area | Platform | Rate | 95% CI | Corpus | Lift | p |")
        add("|---|---|---:|---|---:|---:|---:|")
        for row in flagged.itertuples():
            add(f"| `{row.product_area}` | `{row.platform}` | {row.rate * 100:.1f}% | "
                f"{row.ci_low * 100:.1f}–{row.ci_high * 100:.1f}% | "
                f"{row.corpus_rate * 100:.1f}% | {row.lift:.2f}× | {row.p_value:.4f} |")
    add("")
    add("---")
    add("")

    # --- Trend, or the reason there is none --------------------------------
    add("## Trend")
    add("")
    if verdict.computable:
        add(f"Series length: {len(verdict.months)} months "
            f"({verdict.months[0]} → {verdict.months[-1]}).")
        add("")
        if not changes.empty:
            add("| Platform | Metric | From | To | Change | Intervals disjoint |")
            add("|---|---|---:|---:|---:|---|")
            for row in changes.itertuples():
                add(f"| `{row.platform}` | {row.label} | {row.rate_from * 100:.1f}% | "
                    f"{row.rate_to * 100:.1f}% | {row.change * 100:+.1f}pp | "
                    f"{'yes' if row.intervals_disjoint else 'no'} |")
            add("")
            add("`intervals disjoint = no` means the change is not established.")
    else:
        add(f"**Not computed — {verdict.reason}**")
        add("")
        add(f"This module requires **{MIN_MONTHS_FOR_TREND} monthly observations** "
            "before it will describe a direction. Three points can be joined by a "
            "line, and that is exactly the danger: with one collection artefact "
            "anywhere in the series, the line *is* the artefact.")
        add("")
        add("Phase 4 learned this the expensive way. An unguarded recent-vs-prior "
            "ratio reported growth of up to **197×**, which was entirely an "
            "artefact of when reviews were scraped — the corpus spans 50 months, "
            "but 42 of the earliest 47 hold fewer than ten reviews each.")
        add("")
        add("**What would fix it:** roughly six months of collection at the "
            "current rate inside a window where all three platforms are present. "
            "That is a data-collection task, not an analysis one, and no amount "
            "of modelling substitutes for it.")
    add("")

    # --- Monthly rates, always safe to show --------------------------------
    if not monthly.empty:
        add("### Monthly rates (description, not direction)")
        add("")
        add("Safe to show at any series length because each value is a share of "
            "its own platform-month. Reading a direction into three points is "
            "what the guard above prevents.")
        add("")
        headline = monthly[monthly["metric"] == "negative_share"]
        if not headline.empty:
            months = sorted(headline["year_month"].unique())
            add("**Negative sentiment**")
            add("")
            add("| Platform | " + " | ".join(months) + " |")
            add("|---" * (len(months) + 1) + "|")
            for platform in platforms:
                cells = []
                for month in months:
                    match = headline[
                        (headline["platform"] == platform)
                        & (headline["year_month"] == month)
                    ]
                    cells.append(f"{match.iloc[0]['rate'] * 100:.1f}%" if not match.empty else "—")
                add(f"| `{platform}` | " + " | ".join(cells) + " |")
            add("")
    add("---")
    add("")

    # --- Limits ------------------------------------------------------------
    add("## Caveats")
    add("")
    add("**Reviews are not users.** App-store reviews are written by people "
        "motivated enough to write one, which skews negative everywhere. These "
        "rates compare platforms against each other, not against reality.")
    add("")
    add("**Labels are model output.** Grounding was verified at 98.4%, but no "
        "hand-labelled gold set exists until Phase 9. A systematic model bias "
        "would move every platform's rate together — which is partly why "
        "*differences* are the unit here rather than absolute levels.")
    add("")
    add("**Review volume differs threefold across platforms.** Wilson intervals "
        "account for that; the narrower interval simply belongs to the platform "
        "with more reviews.")
    add("")
    add("**Significance is not importance.** A difference can be real and too "
        "small to act on. The `difference` column is in percentage points for "
        "exactly that reason — judge the size, not just the asterisk.")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-months", action="store_true",
                        help="Include pre-window months. Not recommended.")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="Two-sided significance level before correction.")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format="%(levelname)-8s %(name)s | %(message)s")
    log = logging.getLogger("trends")
    Paths.ensure_output_dirs()

    for required in (Paths.enriched_reviews, Paths.enriched_labels):
        if not required.exists():
            log.error("%s not found. Run: python scripts/04_run_enrichment.py --all",
                      required.name)
            return 1

    reviews = pd.read_parquet(Paths.enriched_reviews)
    labels = pd.read_parquet(Paths.enriched_labels)

    verdict = assess_trend_feasibility(reviews)

    scoped = reviews
    if not args.all_months:
        scoped = reviews[reviews["in_comparable_window"]]
    scoped_labels = labels[labels["review_id"].isin(scoped["review_id"])]

    print()
    print("=" * 78)
    print("  COMPETITIVE METRICS")
    print("=" * 78)
    print(f"  Reviews      : {len(scoped):,} of {len(reviews):,} "
          + ("(all months)" if args.all_months else "(comparable window only)"))
    print(f"  Platforms    : {', '.join(sorted(scoped['platform'].unique()))}")
    print(f"  Alpha        : {args.alpha}  (Benjamini-Hochberg corrected)")
    print("-" * 78)

    metrics = platform_metrics(scoped, args.alpha)
    comparisons = compare_platforms(scoped, args.alpha)
    areas = area_rates_by_platform(scoped_labels, scoped, alpha=args.alpha)
    monthly = monthly_rates(scoped)
    changes = month_over_month_change(monthly, verdict)

    established = int(comparisons["significant"].sum()) if not comparisons.empty else 0
    flagged = int(areas["significant"].sum()) if not areas.empty else 0

    print(f"  Comparisons  : {established} of {len(comparisons)} survive correction")
    print(f"  Area effects : {flagged} of {len(areas)} survive correction")
    print(f"  Trend        : " + ("computable" if verdict.computable
                                  else f"REFUSED — {verdict.reason.split('.')[0]}"))

    for frame, path in (
        (metrics, Paths.platform_metrics),
        (comparisons, Paths.platform_comparisons),
        (areas, Paths.area_rates),
        (monthly, Paths.monthly_rates),
    ):
        if not frame.empty:
            frame.to_parquet(path, index=False)

    Paths.competitive_report.write_text(
        build_report(metrics, comparisons, areas, monthly, changes, verdict,
                     scoped, args.alpha),
        encoding="utf-8",
    )

    print("-" * 78)
    print(f"  Report  : {Paths.competitive_report}")
    print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
