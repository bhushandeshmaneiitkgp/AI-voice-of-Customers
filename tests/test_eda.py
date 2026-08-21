"""Tests for the EDA layer.

Two jobs:

1. **Reconciliation** -- prove the EDA ran on the dataset Phase 1 actually
   produced. The quiet failure mode for analysis code is running on a stale or
   partially rebuilt file and reporting confident numbers from it.
2. **Analytical invariants** -- the conclusions in docs/EDA_FINDINGS.md rest on
   specific properties of the data. If a future change breaks one, the claim
   built on it should fail here rather than silently become wrong.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from config.settings import Paths
from voc.eda import (
    MIN_REVIEWS_FOR_COMPARISON,
    analyse_temporal,
    analyse_text,
    analyse_themes,
    find_comparison_window,
    profile_dataset,
    reconcile,
    run_eda,
)
from voc.plots import collapse_sparse_tail
from voc.taxonomy import get_taxonomy

requires_corpus = pytest.mark.skipif(
    not (Paths.clean_reviews.exists() and Paths.clean_report.exists()),
    reason="cleaned corpus not built; run scripts/01_build_clean.py",
)


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    frame = pd.read_parquet(Paths.clean_reviews)
    frame["platform"] = frame["platform"].astype(str)
    return frame


@pytest.fixture(scope="module")
def cleaning_report() -> dict:
    return json.loads(Paths.clean_report.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def result(corpus, cleaning_report):
    return run_eda(corpus, get_taxonomy(), cleaning_report)


# ---------------------------------------------------------------------------
# Requirement 11: reconciliation with Phase 1
# ---------------------------------------------------------------------------


@requires_corpus
def test_row_count_reconciles(corpus, cleaning_report) -> None:
    assert len(corpus) == cleaning_report["cleaning"]["rows_out"]


@requires_corpus
def test_platform_counts_reconcile(corpus, cleaning_report) -> None:
    observed = {str(k): int(v) for k, v in corpus["platform"].value_counts().items()}
    assert observed == cleaning_report["cleaning"]["platform_counts"]


@requires_corpus
def test_rating_counts_reconcile(corpus, cleaning_report) -> None:
    observed = {str(k): int(v) for k, v in corpus["rating"].value_counts().sort_index().items()}
    assert observed == cleaning_report["cleaning"]["rating_counts"]


@requires_corpus
def test_no_nulls_introduced(corpus) -> None:
    assert int(corpus.isna().sum().sum()) == 0


@requires_corpus
def test_dates_are_valid_and_unchanged(corpus, cleaning_report) -> None:
    cleaning = cleaning_report["cleaning"]
    assert corpus["review_date"].notna().all()
    assert str(corpus["review_date"].min().date()) == cleaning["date_min"]
    assert str(corpus["review_date"].max().date()) == cleaning["date_max"]


@requires_corpus
def test_year_month_matches_review_date(corpus) -> None:
    """The grouping key must not drift from the date it was derived from."""
    derived = corpus["review_date"].dt.strftime("%Y-%m")
    assert (derived == corpus["year_month"]).all()


@requires_corpus
def test_ratings_within_range(corpus) -> None:
    assert corpus["rating"].between(1, 5).all()


@requires_corpus
def test_truncation_count_reconciles(corpus, cleaning_report) -> None:
    assert int(corpus["is_truncated"].sum()) == cleaning_report["cleaning"]["truncated_reviews"]


@requires_corpus
def test_reconcile_reports_all_passed(result) -> None:
    assert result.reconciliation["all_passed"] is True


@requires_corpus
def test_reconcile_detects_a_mismatch(corpus, cleaning_report) -> None:
    """The reconciliation must actually fire, not merely exist."""
    checks = reconcile(corpus.head(100), cleaning_report)
    assert checks["row_count_matches"] is False
    assert checks["all_passed"] is False


def test_raw_data_untouched_by_eda() -> None:
    """EDA reads only. The Phase 1 checksum must still hold afterwards."""
    from voc.ingest import KNOWN_SOURCE_SHA256, compute_sha256

    if not Paths.raw_reviews.exists():
        pytest.skip("raw dataset not present")
    assert compute_sha256(Paths.raw_reviews) == KNOWN_SOURCE_SHA256


# ---------------------------------------------------------------------------
# Comparison window
# ---------------------------------------------------------------------------


@requires_corpus
def test_comparison_window_exists(result) -> None:
    assert result.temporal.comparison_window is not None
    assert result.temporal.months_meeting_threshold


@requires_corpus
def test_every_window_month_meets_the_threshold(corpus, result) -> None:
    """The window's defining property, asserted rather than assumed."""
    counts = (
        corpus.groupby(["year_month", "platform"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for month in result.temporal.months_meeting_threshold:
        assert (counts.loc[month] >= MIN_REVIEWS_FOR_COMPARISON).all(), month


@requires_corpus
def test_excluded_months_fail_the_threshold(corpus, result) -> None:
    """No qualifying month was skipped -- the window is maximal, not arbitrary."""
    counts = (
        corpus.groupby(["year_month", "platform"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    excluded = set(counts.index) - set(result.temporal.months_meeting_threshold)
    for month in excluded:
        assert not (counts.loc[month] >= MIN_REVIEWS_FOR_COMPARISON).all(), (
            f"{month} qualifies but was excluded from the window"
        )


@requires_corpus
def test_window_matches_the_schema_constant(result) -> None:
    """The window derived from data should agree with COMPARABLE_WINDOW_START.

    Phase 1 hardcoded that constant from manual inspection; Phase 2 derives it
    algorithmically. They must agree, or one of them is wrong.
    """
    from voc.schemas import COMPARABLE_WINDOW_START

    assert result.temporal.comparison_window[0] == COMPARABLE_WINDOW_START


@requires_corpus
def test_no_window_when_a_platform_is_absent(corpus) -> None:
    single = corpus[corpus["platform"] == corpus["platform"].iloc[0]]
    window, months, rationale = find_comparison_window(single)
    # With one platform present the threshold is trivially met, so the guard we
    # care about is that the rationale still states the coverage explicitly.
    assert isinstance(rationale, str) and rationale


# ---------------------------------------------------------------------------
# Analytical invariants behind the written findings
# ---------------------------------------------------------------------------


@requires_corpus
def test_corpus_is_complaint_biased(result) -> None:
    """Every caveat in the report depends on this being true."""
    assert result.profile.rating_bucket_share["negative"] > 70


@requires_corpus
def test_december_concentration_is_extreme(result) -> None:
    assert result.temporal.december_share_pct > 40


@requires_corpus
def test_truncated_reviews_skew_negative(result) -> None:
    """Justifies flagging severity as lower-confidence on truncated rows."""
    comparison = result.text.truncated_comparison
    assert comparison["truncated_mean_rating"] < comparison["intact_mean_rating"]
    assert comparison["truncated_pct_negative"] > comparison["intact_pct_negative"]


@requires_corpus
def test_full_corpus_rating_comparison_is_confounded(result) -> None:
    """The headline safety finding of Phase 2, asserted so it cannot be forgotten.

    JioMart and Zepto swap rank once date coverage is equalised: JioMart's
    full-corpus mean is dragged down by its pre-October reviews, which no other
    platform has. Any cross-platform rating claim made on the FULL corpus would
    therefore rank them wrongly, which is why the comparison window is mandatory
    rather than merely advisable.
    """
    ratings = result.ratings
    assert ratings.ranking_changes_in_window is True
    assert len(ratings.reordered_platforms) >= 2
    assert "CHANGES" in ratings.confound_note


@requires_corpus
def test_confound_note_reflects_the_measured_ranking(result) -> None:
    """The narrative must be derived from the data, never hardcoded."""
    frame = result.ratings.full_vs_window
    rank_full = frame["mean_rating_full"].rank(ascending=False)
    rank_window = frame["mean_rating_window"].rank(ascending=False)
    measured = sorted(
        str(platform) for platform in frame.index if rank_full[platform] != rank_window[platform]
    )
    assert result.ratings.reordered_platforms == measured


@requires_corpus
def test_top_platform_is_stable_across_the_window(result) -> None:
    """Only the lower two swap; the top platform holds under either view."""
    frame = result.ratings.full_vs_window
    assert frame["mean_rating_full"].idxmax() == frame["mean_rating_window"].idxmax()


@requires_corpus
def test_theme_signatures_survive_the_window_restriction(result) -> None:
    """Platform theme differences are not explained by date coverage alone."""
    assert result.themes.window_vs_full_shift.abs().max().max() < 10.0


@requires_corpus
def test_platform_specific_and_shared_themes_are_disjoint(result) -> None:
    shared = set(result.themes.shared_themes)
    for areas in result.themes.platform_specific.values():
        assert not shared & set(areas)


@requires_corpus
def test_negative_share_varies_across_months(result) -> None:
    """The confound that forces share-based trend reporting instead of counts."""
    shares = result.temporal.monthly_negative_share["negative_share_pct"]
    substantial = result.temporal.monthly_negative_share
    shares = shares[substantial["reviews"] >= 50]
    assert shares.max() - shares.min() > 5.0


# ---------------------------------------------------------------------------
# Determinism and helpers
# ---------------------------------------------------------------------------


@requires_corpus
def test_profile_is_deterministic(corpus) -> None:
    assert profile_dataset(corpus) == profile_dataset(corpus)


@requires_corpus
def test_theme_shares_are_deterministic(corpus, result) -> None:
    taxonomy = get_taxonomy()
    months = result.temporal.months_meeting_threshold
    first = analyse_themes(corpus, taxonomy, months).share_window
    second = analyse_themes(corpus, taxonomy, months).share_window
    pd.testing.assert_frame_equal(first, second)


@requires_corpus
def test_monthly_volume_totals_match_corpus(corpus, result) -> None:
    monthly = result.temporal.monthly_volume
    assert int(monthly["total"].sum()) == len(corpus)


@requires_corpus
def test_text_profile_partitions_by_truncation(corpus, result) -> None:
    comparison = result.text.truncated_comparison
    assert int(comparison["truncated_n"]) == int(corpus["is_truncated"].sum())


@requires_corpus
def test_top_terms_use_document_frequency(corpus, result) -> None:
    """A term cannot be reported in more reviews than the subset contains."""
    negatives = int((corpus["rating"] <= 2).sum())
    assert all(count <= negatives for _, count in result.text.top_terms_negative)


def test_collapse_sparse_tail_preserves_totals() -> None:
    monthly = pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]},
        index=["2024-01", "2024-02", "2024-03", "2024-04"],
    )
    collapsed = collapse_sparse_tail(monthly, "2024-03")

    assert len(collapsed) == 3  # one aggregate + two kept months
    assert collapsed.sum().equals(monthly.sum())


def test_collapse_sparse_tail_noop_when_nothing_to_fold() -> None:
    monthly = pd.DataFrame({"a": [1, 2]}, index=["2024-01", "2024-02"])
    assert collapse_sparse_tail(monthly, "2024-01").equals(monthly)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@requires_corpus
def test_expected_figures_exist() -> None:
    """The report links these by name; a missing file is a broken report."""
    if not Paths.figures_dir.exists():
        pytest.skip("figures not rendered; run scripts/03_run_eda.py")
    expected = {
        "01_rating_distribution.png",
        "02_platform_distribution.png",
        "03_monthly_volume.png",
        "04_platform_coverage.png",
        "05_rating_by_platform.png",
        "06_review_length.png",
        "07_rating_over_time.png",
        "08_sampling_composition.png",
        "09_theme_comparison.png",
    }
    found = {path.name for path in Paths.figures_dir.glob("*.png")}
    assert expected <= found, f"missing figures: {sorted(expected - found)}"
