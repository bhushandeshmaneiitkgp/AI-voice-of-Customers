"""
Layer 7 -- competitive metrics and trend analysis.

Phase 4 ranked what hurts. This asks the two follow-up questions a PM actually
has: **is it worse on us than on them**, and **is it getting worse**.

The first question this corpus can answer. The second it largely cannot, and
saying so is part of the deliverable rather than a gap in it.

Two rules govern everything here, both forced by how the data was collected.

**Rates, never counts.** December holds three times October's reviews -- that
is scraping intensity, not customer behaviour. Any metric expressed as a count
would rank December worst for every platform on every measure, automatically.
Every figure below is a share of that platform-month's own reviews.

**Differences are tested, not eyeballed.** With ~900-1,700 reviews per platform
a three-point gap in some rate is well inside noise. An untested percentage
table invites exactly the confident wrong conclusion this pipeline exists to
prevent, so every comparison carries a confidence interval and a p-value
corrected for the number of comparisons made.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

#: Monthly observations needed before this module will describe a direction.
#: Three points can be joined by a line, which is precisely the problem: with
#: one collection artefact anywhere in the series the line is the artefact.
#: Six is still few, but it survives a single bad month.
MIN_MONTHS_FOR_TREND = 6

#: Two-sided significance level, before multiplicity correction.
ALPHA = 0.05

#: A platform-month below this many reviews is not a measurement. Rates over
#: tiny denominators swing wildly and would dominate any ranking by variance.
MIN_PLATFORM_MONTH_REVIEWS = 50


# ---------------------------------------------------------------------------
# Proportion statistics
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, total: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Wilson rather than the textbook normal approximation, which misbehaves
    exactly where these rates live: near 0 or 1, and on small denominators it
    happily returns bounds below 0% or above 100%. A churn rate whose interval
    starts at -0.4% is not a number anyone should put in front of a PM.
    """
    if total <= 0:
        return (0.0, 0.0)

    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / total
    denominator = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def two_proportion_test(
    successes_a: int, total_a: int, successes_b: int, total_b: int
) -> tuple[float, float]:
    """Two-sided z-test for a difference in proportions. Returns (diff, p).

    ``diff`` is A minus B, so a positive value means the first group is higher.
    Returns p=1.0 when either group is empty: no data is not evidence of no
    difference, and 1.0 keeps it out of the significant set.
    """
    if total_a <= 0 or total_b <= 0:
        return (0.0, 1.0)

    p_a, p_b = successes_a / total_a, successes_b / total_b
    pooled = (successes_a + successes_b) / (total_a + total_b)
    standard_error = np.sqrt(pooled * (1 - pooled) * (1 / total_a + 1 / total_b))
    if standard_error == 0:
        return (p_a - p_b, 1.0)

    z = (p_a - p_b) / standard_error
    return (p_a - p_b, float(2 * (1 - stats.norm.cdf(abs(z)))))


def benjamini_hochberg(p_values: list[float], alpha: float = ALPHA) -> list[bool]:
    """Which p-values survive an FDR correction. Returns a mask.

    Comparing three platforms across a dozen product areas is ~36 tests. At
    alpha=0.05 roughly two will look significant on pure noise, and those two
    are exactly the ones that would get written up as findings. Benjamini-
    Hochberg controls the expected proportion of false discoveries while
    keeping far more power than Bonferroni, which at this many tests would
    reject almost everything real.
    """
    n = len(p_values)
    if n == 0:
        return []

    order = np.argsort(p_values)
    ranked = np.asarray(p_values, dtype=float)[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passing = ranked <= thresholds

    keep = np.zeros(n, dtype=bool)
    if passing.any():
        # Everything up to the largest passing rank is significant, including
        # any p-value that individually failed its own threshold.
        cutoff = np.flatnonzero(passing)[-1]
        keep[order[: cutoff + 1]] = True
    return keep.tolist()


# ---------------------------------------------------------------------------
# Competitive metrics
# ---------------------------------------------------------------------------


@dataclass
class MetricSpec:
    """One comparable rate: how to compute it and how to say it."""

    key: str
    label: str
    #: Given the per-review frame, return a boolean Series of "successes".
    predicate: object
    #: True when a higher rate is worse for the platform.
    higher_is_worse: bool = True


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("negative_share", "Negative sentiment",
               lambda f: f["sentiment"] == "negative"),
    MetricSpec("severe_share", "High or critical severity",
               lambda f: f["severity"].isin(["high", "critical"])),
    MetricSpec("escalation_rate", "Drove a support contact",
               lambda f: f["support_escalation"].astype(bool)),
    MetricSpec("churn_share", "Stated intent to leave",
               lambda f: f["customer_intent"] == "churn_warning"),
    MetricSpec("praise_share", "Praise",
               lambda f: f["customer_intent"] == "praise", higher_is_worse=False),
)


def platform_metrics(reviews: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Per-platform rate for each metric, with a Wilson interval.

    Restricted to the comparable window by the caller. Every row is a share of
    that platform's own reviews, so platforms with different review counts stay
    comparable.
    """
    rows = []
    for platform, group in reviews.groupby("platform", observed=True):
        for spec in METRICS:
            hits = int(spec.predicate(group).sum())
            total = len(group)
            low, high = wilson_interval(hits, total, alpha)
            rows.append(
                {
                    "platform": platform,
                    "metric": spec.key,
                    "label": spec.label,
                    "reviews": total,
                    "hits": hits,
                    "rate": hits / total if total else 0.0,
                    "ci_low": low,
                    "ci_high": high,
                    "higher_is_worse": spec.higher_is_worse,
                }
            )
    return pd.DataFrame(rows)


def compare_platforms(reviews: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Every pairwise platform comparison on every metric, FDR-corrected.

    ``significant`` is the column to read. A raw p-value below 0.05 here means
    little: the table runs to dozens of tests, and some will clear that bar by
    chance alone.
    """
    platforms = sorted(reviews["platform"].unique())
    rows = []
    for spec in METRICS:
        for i, left in enumerate(platforms):
            for right in platforms[i + 1:]:
                a = reviews[reviews["platform"] == left]
                b = reviews[reviews["platform"] == right]
                diff, p_value = two_proportion_test(
                    int(spec.predicate(a).sum()), len(a),
                    int(spec.predicate(b).sum()), len(b),
                )
                rows.append(
                    {
                        "metric": spec.key,
                        "label": spec.label,
                        "platform_a": left,
                        "platform_b": right,
                        "rate_a": spec.predicate(a).mean() if len(a) else 0.0,
                        "rate_b": spec.predicate(b).mean() if len(b) else 0.0,
                        "difference": diff,
                        "p_value": p_value,
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["significant"] = benjamini_hochberg(frame["p_value"].tolist(), alpha)
    return frame.sort_values("p_value").reset_index(drop=True)


def area_rates_by_platform(
    labels: pd.DataFrame,
    reviews: pd.DataFrame,
    min_mentions: int = 30,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Which platform over-indexes on which product area, tested.

    The denominator is the platform's review count, not its label count: the
    question is "what share of this platform's customers raised this", which a
    label-count share would distort for any platform whose reviews mention more
    areas each.
    """
    issues = labels[labels["polarity"] == "issue"]
    if issues.empty:
        return pd.DataFrame()

    totals = reviews.groupby("platform", observed=True).size()
    mentions = (
        issues.groupby(["product_area", "platform"], observed=True)["review_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    mentions = mentions.reindex(columns=totals.index, fill_value=0)
    mentions = mentions[mentions.sum(axis=1) >= min_mentions]

    rows = []
    for area, counts in mentions.iterrows():
        overall = counts.sum() / totals.sum()
        for platform in totals.index:
            hits, total = int(counts[platform]), int(totals[platform])
            # Each platform against the pooled remainder: "is this worse here
            # than elsewhere", which is the competitive question.
            others_hits = int(counts.sum() - hits)
            others_total = int(totals.sum() - total)
            diff, p_value = two_proportion_test(hits, total, others_hits, others_total)
            low, high = wilson_interval(hits, total, alpha)
            rows.append(
                {
                    "product_area": area,
                    "platform": platform,
                    "reviews": total,
                    "mentions": hits,
                    "rate": hits / total if total else 0.0,
                    "ci_low": low,
                    "ci_high": high,
                    "corpus_rate": overall,
                    "lift": (hits / total) / overall if total and overall else np.nan,
                    "p_value": p_value,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["significant"] = benjamini_hochberg(frame["p_value"].tolist(), alpha)
    return frame.sort_values(["product_area", "rate"], ascending=[True, False]).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


@dataclass
class TrendVerdict:
    """Whether a direction may be described at all, and why not."""

    months: list[str]
    computable: bool
    reason: str


def assess_trend_feasibility(
    reviews: pd.DataFrame, min_months: int = MIN_MONTHS_FOR_TREND
) -> TrendVerdict:
    """Decide up front whether this corpus supports a trend claim.

    Called before anything is computed, so the answer cannot be reverse-
    engineered from a chart that happens to slope. Phase 4 already learned this
    the expensive way: an unguarded ratio produced 197x growth that was entirely
    an artefact of when reviews were scraped.
    """
    window = reviews[reviews["in_comparable_window"]] if "in_comparable_window" in reviews else reviews
    months = sorted(window["year_month"].dropna().unique().tolist())

    if len(months) < min_months:
        return TrendVerdict(
            months=months,
            computable=False,
            reason=(
                f"{len(months)} comparable month(s) available, {min_months} needed. "
                "Outside the comparable window review volume tracks collection "
                "rather than customer behaviour, so extending the series would "
                "measure the scraper."
            ),
        )
    return TrendVerdict(months=months, computable=True, reason="")


def monthly_rates(reviews: pd.DataFrame, min_reviews: int = MIN_PLATFORM_MONTH_REVIEWS) -> pd.DataFrame:
    """Per platform-month metric rates. A description, not a trend.

    Safe to report at any series length because every value is a share of its
    own platform-month. Reading a direction into it is what
    ``assess_trend_feasibility`` gates.
    """
    rows = []
    grouped = reviews.groupby(["platform", "year_month"], observed=True)
    for (platform, month), group in grouped:
        if len(group) < min_reviews:
            logger.info(
                "Skipping %s %s: %d reviews is below the floor of %d",
                platform, month, len(group), min_reviews,
            )
            continue
        for spec in METRICS:
            hits = int(spec.predicate(group).sum())
            low, high = wilson_interval(hits, len(group))
            rows.append(
                {
                    "platform": platform,
                    "year_month": month,
                    "metric": spec.key,
                    "label": spec.label,
                    "reviews": len(group),
                    "rate": hits / len(group),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["metric", "platform", "year_month"]).reset_index(drop=True)


def month_over_month_change(monthly: pd.DataFrame, verdict: TrendVerdict) -> pd.DataFrame:
    """First-to-last change per platform-metric, only if trend is permitted.

    Returns empty when it is not. A caller that wants the number anyway has to
    change the guard, which is a visible decision rather than an accident.
    """
    if not verdict.computable or monthly.empty:
        return pd.DataFrame()

    rows = []
    for (platform, metric), group in monthly.groupby(["platform", "metric"], observed=True):
        ordered = group.sort_values("year_month")
        first, last = ordered.iloc[0], ordered.iloc[-1]
        # Non-overlapping Wilson intervals is a deliberately conservative test
        # for "actually moved" -- it is stricter than a formal two-proportion
        # test, which is the right side to err on for a headline claim.
        moved = last["ci_low"] > first["ci_high"] or last["ci_high"] < first["ci_low"]
        rows.append(
            {
                "platform": platform,
                "metric": metric,
                "label": first["label"],
                "from_month": first["year_month"],
                "to_month": last["year_month"],
                "rate_from": first["rate"],
                "rate_to": last["rate"],
                "change": last["rate"] - first["rate"],
                "intervals_disjoint": bool(moved),
            }
        )
    return pd.DataFrame(rows).sort_values("change", ascending=False).reset_index(drop=True)
