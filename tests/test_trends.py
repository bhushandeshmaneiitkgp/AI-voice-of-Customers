"""Tests for Phase 5: competitive metrics and trend analysis.

The risk in this layer is not a crash, it is a confident wrong claim. A table
of percentages always looks authoritative, so most of what follows checks that
the statistics refuse to overstate: intervals stay inside [0, 1], differences
that are noise are not reported as findings, counts never masquerade as rates,
and a three-month series cannot become a direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voc.trends import (
    ALPHA,
    MIN_MONTHS_FOR_TREND,
    MIN_PLATFORM_MONTH_REVIEWS,
    area_rates_by_platform,
    assess_trend_feasibility,
    benjamini_hochberg,
    compare_platforms,
    month_over_month_change,
    monthly_rates,
    platform_metrics,
    two_proportion_test,
    wilson_interval,
)


def _reviews(spec: list[tuple]) -> pd.DataFrame:
    """spec: (platform, year_month, sentiment, severity, intent, escalated)."""
    frame = pd.DataFrame(
        spec,
        columns=["platform", "year_month", "sentiment", "severity",
                 "customer_intent", "support_escalation"],
    )
    frame["review_id"] = [f"r{i}" for i in range(len(frame))]
    frame["in_comparable_window"] = True
    return frame


def _uniform(platform: str, month: str, n: int, negative: int) -> list[tuple]:
    return [
        (platform, month,
         "negative" if i < negative else "positive",
         "high" if i < negative else "low",
         "complaint" if i < negative else "praise",
         i < negative)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Proportion statistics
# ---------------------------------------------------------------------------


def test_wilson_interval_never_leaves_the_unit_range() -> None:
    """The normal approximation returns bounds below 0% on rates like these.

    A churn rate whose interval starts at -0.4% is not a number anyone should
    put in front of a PM.
    """
    for successes, total in [(0, 10), (1, 1000), (10, 10), (0, 1), (3, 5)]:
        low, high = wilson_interval(successes, total)
        assert 0.0 <= low <= high <= 1.0, (successes, total, low, high)


def test_wilson_interval_brackets_the_observed_rate() -> None:
    low, high = wilson_interval(50, 200)
    assert low < 0.25 < high


def test_wilson_interval_narrows_as_evidence_grows() -> None:
    """More reviews must buy more precision, or the interval means nothing."""
    small = wilson_interval(10, 100)
    large = wilson_interval(100, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_interval_of_an_empty_group_is_degenerate_not_a_crash() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_identical_proportions_are_not_a_difference() -> None:
    diff, p_value = two_proportion_test(50, 100, 100, 200)
    assert diff == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)


def test_a_large_clear_gap_is_detected() -> None:
    _, p_value = two_proportion_test(90, 100, 10, 100)
    assert p_value < 1e-10


def test_the_same_gap_on_tiny_samples_is_not_detected() -> None:
    """80% vs 40% on five each is not evidence. Sample size has to matter."""
    _, p_value = two_proportion_test(4, 5, 2, 5)
    assert p_value > ALPHA


def test_proportion_test_is_antisymmetric() -> None:
    forward, p_forward = two_proportion_test(30, 100, 10, 100)
    backward, p_backward = two_proportion_test(10, 100, 30, 100)
    assert forward == pytest.approx(-backward)
    assert p_forward == pytest.approx(p_backward)


def test_an_empty_group_yields_no_evidence_of_difference() -> None:
    """No data is not evidence of no difference; p=1 keeps it out of findings."""
    assert two_proportion_test(0, 0, 50, 100) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------


def test_bh_keeps_the_clearly_significant_and_drops_pure_noise() -> None:
    mask = benjamini_hochberg([0.0001, 0.0002, 0.60, 0.75, 0.90])
    assert mask[:2] == [True, True]
    assert mask[2:] == [False, False, False]


def test_bh_is_a_step_up_procedure() -> None:
    """A p-value failing its own threshold still passes below a larger rank.

    Testing each p-value independently against alpha*i/n is a common and wrong
    implementation; it drops true positives that the real procedure keeps.
    """
    # With n=3, thresholds are 0.0167, 0.0333, 0.05. p=0.02 fails its own rank-1
    # threshold, but p=0.03 passes at rank 2, so both are kept.
    assert benjamini_hochberg([0.02, 0.03, 0.90]) == [True, True, False]


def test_bh_is_never_more_permissive_than_uncorrected_alpha() -> None:
    """Correction must only ever remove findings, never invent them."""
    p_values = [0.001, 0.01, 0.02, 0.04, 0.049, 0.3, 0.8]
    corrected = benjamini_hochberg(p_values)
    for p_value, kept in zip(p_values, corrected):
        if kept:
            assert p_value <= ALPHA


def test_bh_on_all_noise_finds_nothing() -> None:
    assert not any(benjamini_hochberg([0.2, 0.4, 0.6, 0.8, 0.95]))


def test_bh_handles_an_empty_list() -> None:
    assert benjamini_hochberg([]) == []


# ---------------------------------------------------------------------------
# Platform metrics
# ---------------------------------------------------------------------------


def test_rates_are_shares_of_each_platforms_own_reviews() -> None:
    """Otherwise the platform with the most reviews looks worst automatically."""
    reviews = _reviews(
        _uniform("big", "2024-10", 400, 200) + _uniform("small", "2024-10", 100, 50)
    )
    metrics = platform_metrics(reviews)
    negative = metrics[metrics["metric"] == "negative_share"].set_index("platform")

    assert negative.loc["big", "rate"] == pytest.approx(0.5)
    assert negative.loc["small", "rate"] == pytest.approx(0.5)


def test_collection_volume_does_not_change_a_rate() -> None:
    """December holds 3x October's reviews. That must not move any metric."""
    october = _reviews(_uniform("p", "2024-10", 100, 60))
    december = _reviews(_uniform("p", "2024-12", 300, 180))

    a = platform_metrics(october)
    b = platform_metrics(december)
    rate_a = a[a["metric"] == "negative_share"].iloc[0]["rate"]
    rate_b = b[b["metric"] == "negative_share"].iloc[0]["rate"]

    assert rate_a == pytest.approx(rate_b)


def test_every_metric_is_reported_for_every_platform() -> None:
    reviews = _reviews(_uniform("a", "2024-10", 60, 30) + _uniform("b", "2024-10", 60, 20))
    metrics = platform_metrics(reviews)
    assert set(metrics["platform"]) == {"a", "b"}
    assert metrics.groupby("platform")["metric"].nunique().nunique() == 1


def test_a_real_gap_is_reported_and_a_noise_gap_is_not() -> None:
    """The whole point of the layer, in one test."""
    reviews = _reviews(
        _uniform("bad", "2024-10", 800, 700)     # 87.5% negative
        + _uniform("good", "2024-10", 800, 300)  # 37.5% negative
        + _uniform("similar", "2024-10", 800, 704)  # 88.0%, vs bad: noise
    )
    comparisons = compare_platforms(reviews)

    def verdict(a: str, b: str) -> bool:
        row = comparisons[
            (comparisons["metric"] == "negative_share")
            & (comparisons[["platform_a", "platform_b"]].apply(frozenset, axis=1)
               == frozenset({a, b}))
        ]
        return bool(row.iloc[0]["significant"])

    assert verdict("bad", "good") is True
    assert verdict("bad", "similar") is False


def test_comparisons_carry_a_corrected_verdict() -> None:
    reviews = _reviews(_uniform("a", "2024-10", 200, 100) + _uniform("b", "2024-10", 200, 100))
    comparisons = compare_platforms(reviews)
    assert "significant" in comparisons.columns
    assert not comparisons["significant"].any(), "identical platforms differ in nothing"


# ---------------------------------------------------------------------------
# Area rates
# ---------------------------------------------------------------------------


def test_area_rate_denominator_is_reviews_not_labels() -> None:
    """A platform whose reviews mention more areas each would otherwise look
    systematically worse on every area at once."""
    reviews = _reviews(_uniform("a", "2024-10", 100, 100) + _uniform("b", "2024-10", 100, 100))
    # Platform a: 50 reviews mention the area, but twice each.
    rows = []
    for i in range(50):
        rows += [(f"r{i}", "delivery", "issue"), (f"r{i}", "delivery", "issue")]
    for i in range(100, 150):
        rows.append((f"r{i}", "delivery", "issue"))
    labels = pd.DataFrame(rows, columns=["review_id", "product_area", "polarity"])
    labels["platform"] = labels["review_id"].map(reviews.set_index("review_id")["platform"])

    result = area_rates_by_platform(labels, reviews, min_mentions=1)
    rates = result.set_index("platform")["rate"]

    assert rates["a"] == pytest.approx(0.5)
    assert rates["b"] == pytest.approx(0.5)


def test_strength_labels_are_excluded_from_area_rates() -> None:
    reviews = _reviews(_uniform("a", "2024-10", 60, 30))
    labels = pd.DataFrame(
        [("r0", "speed", "strength"), ("r1", "speed", "strength")],
        columns=["review_id", "product_area", "polarity"],
    ).assign(platform="a")

    assert area_rates_by_platform(labels, reviews, min_mentions=1).empty


# ---------------------------------------------------------------------------
# Trend guard
# ---------------------------------------------------------------------------


def test_three_months_is_refused() -> None:
    """Three points can be joined by a line. That is the danger, not the fix."""
    reviews = _reviews(
        _uniform("a", "2024-10", 60, 30)
        + _uniform("a", "2024-11", 60, 30)
        + _uniform("a", "2024-12", 60, 30)
    )
    verdict = assess_trend_feasibility(reviews)

    assert verdict.computable is False
    assert "3 comparable month(s)" in verdict.reason


def test_enough_months_is_permitted() -> None:
    reviews = _reviews(
        sum((_uniform("a", f"2024-{m:02d}", 60, 30) for m in range(1, 8)), [])
    )
    assert assess_trend_feasibility(reviews).computable is True


def test_the_guard_counts_only_comparable_months() -> None:
    """Pre-window months hold a handful of reviews; counting them fakes history."""
    reviews = _reviews(
        sum((_uniform("a", f"2020-{m:02d}", 5, 3) for m in range(1, 9)), [])
        + _uniform("a", "2024-10", 60, 30)
        + _uniform("a", "2024-11", 60, 30)
    )
    reviews.loc[reviews["year_month"].str.startswith("2020"), "in_comparable_window"] = False

    verdict = assess_trend_feasibility(reviews)
    assert verdict.computable is False
    assert verdict.months == ["2024-10", "2024-11"]


def test_no_change_table_is_produced_when_trend_is_refused() -> None:
    """The refusal has to bind the output, not just the prose."""
    reviews = _reviews(
        _uniform("a", "2024-10", 60, 30)
        + _uniform("a", "2024-11", 60, 45)
        + _uniform("a", "2024-12", 60, 55)
    )
    verdict = assess_trend_feasibility(reviews)
    monthly = monthly_rates(reviews)

    assert not monthly.empty, "monthly rates are always safe to describe"
    assert month_over_month_change(monthly, verdict).empty


def test_change_is_produced_once_the_series_is_long_enough() -> None:
    reviews = _reviews(
        sum((_uniform("a", f"2024-{m:02d}", 200, 20) for m in range(1, 7)), [])
        + _uniform("a", "2024-07", 200, 180)
    )
    verdict = assess_trend_feasibility(reviews)
    changes = month_over_month_change(monthly_rates(reviews), verdict)

    negative = changes[changes["metric"] == "negative_share"].iloc[0]
    assert negative["change"] > 0.5
    # pandas stores this as numpy.bool_, so compare by value not identity.
    assert bool(negative["intervals_disjoint"])


def test_a_thin_platform_month_is_not_measured() -> None:
    """Rates over tiny denominators swing wildly and would dominate by variance."""
    reviews = _reviews(
        _uniform("a", "2024-10", MIN_PLATFORM_MONTH_REVIEWS + 10, 30)
        + _uniform("a", "2024-11", 5, 5)
    )
    months = set(monthly_rates(reviews)["year_month"])
    assert months == {"2024-10"}


def test_the_trend_threshold_is_documented_and_above_three() -> None:
    """A constant guarding a claim this easy to get wrong should not be 3."""
    assert MIN_MONTHS_FOR_TREND >= 6
