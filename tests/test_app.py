"""Tests for Phase 8: the dashboard's data layer.

Streamlit itself is not exercised -- rendering is not where the bugs are. What
matters is everything the UI reads before it renders: which phases have run,
what a missing input should say, and whether a number arrives with the thing
that qualifies it.

The dashboard is the first place someone meets these figures without the report
around them, and a table strips context better than any other format. So the
tests here are mostly about context surviving the trip.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.loaders import (
    Artefact,
    build_state,
    corpus_summary,
    format_rate,
    load_report,
    load_table,
    scoring_mode,
    significance_label,
    split_packed,
)


class FakePaths:
    """Stands in for config.settings.Paths, pointed at a temp directory."""

    def __init__(self, root: Path) -> None:
        for name in (
            "clean_reviews", "enriched_reviews", "enriched_labels", "enrichment_report",
            "embeddings", "faiss_index", "pain_points", "cluster_summary",
            "platform_metrics", "platform_comparisons", "area_rates", "root_causes",
            "opportunities", "rice_scores", "experiment_plans",
        ):
            setattr(self, name, root / f"{name}.parquet")


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_a_fresh_checkout_reports_everything_missing(tmp_path) -> None:
    state = build_state(FakePaths(tmp_path))
    assert not state.available("reviews")
    assert state.missing("reviews", "pain_points")


def test_missing_inputs_name_the_script_that_produces_them(tmp_path) -> None:
    """An empty chart looks like a finding of zero; a command does not."""
    state = build_state(FakePaths(tmp_path))
    instructions = state.instructions_for("pain_points")
    assert instructions == ["python scripts/06_discover_painpoints.py"]


def test_one_script_producing_several_artefacts_is_named_once(tmp_path) -> None:
    """Telling someone to run the same command three times is noise."""
    state = build_state(FakePaths(tmp_path))
    instructions = state.instructions_for("opportunities", "rice", "experiments")
    assert instructions == ["python scripts/09_build_roadmap.py"]


def test_availability_flips_once_the_file_exists(tmp_path) -> None:
    paths = FakePaths(tmp_path)
    state = build_state(paths)
    assert not state.available("reviews")

    pd.DataFrame({"review_id": ["a"]}).to_parquet(paths.enriched_reviews)
    assert build_state(paths).available("reviews")


def test_asking_about_an_unknown_key_does_not_raise(tmp_path) -> None:
    """A page referring to an artefact that no longer exists must degrade."""
    state = build_state(FakePaths(tmp_path))
    assert state.available("not_a_real_artefact") is True
    assert state.missing("not_a_real_artefact") == []


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_a_missing_table_loads_as_empty_not_an_exception(tmp_path) -> None:
    """A dashboard that dies because Phase 7 has not run is less useful."""
    artefact = Artefact("x", tmp_path / "absent.parquet", "run something", "9")
    assert load_table(artefact).empty


def test_a_corrupt_table_loads_as_empty_not_an_exception(tmp_path) -> None:
    path = tmp_path / "broken.parquet"
    path.write_text("this is not parquet", encoding="utf-8")
    assert load_table(Artefact("x", path, "cmd", "9")).empty


def test_a_corrupt_report_loads_as_empty(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")
    assert load_report(Artefact("x", path, "cmd", "9")) == {}


def test_a_real_table_round_trips(tmp_path) -> None:
    path = tmp_path / "t.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(path)
    assert len(load_table(Artefact("x", path, "cmd", "9"))) == 2


# ---------------------------------------------------------------------------
# Packed list columns
# ---------------------------------------------------------------------------


def test_packed_lists_unpack_on_the_separator() -> None:
    """Phases 4 and 6 join lists for parquet; this is the inverse."""
    assert split_packed("first quote ||| second quote") == ["first quote", "second quote"]


def test_space_separated_ids_unpack_too() -> None:
    assert split_packed("a1 b2 c3") == ["a1", "b2", "c3"]


def test_an_empty_or_missing_packed_value_is_an_empty_list() -> None:
    for value in ("", "   ", None, float("nan")):
        assert split_packed(value) == []


def test_unpacking_does_not_split_a_quote_containing_spaces() -> None:
    """The separator exists precisely so quotes survive intact."""
    packed = "the app crashed twice ||| my wallet balance vanished"
    assert split_packed(packed) == ["the app crashed twice", "my wallet balance vanished"]


# ---------------------------------------------------------------------------
# Context that must survive into the UI
# ---------------------------------------------------------------------------


def test_an_unestablished_difference_is_labelled_in_words() -> None:
    """In a table, an insignificant row looks identical to a real finding.

    A word costs a column and prevents the most likely misreading in the
    product.
    """
    assert significance_label(False) == "not established"
    assert significance_label(True) == "established"


def test_a_rate_renders_with_its_interval() -> None:
    assert format_rate(0.851, 0.832, 0.866) == "85.1% (83.2–86.6)"


def test_a_rate_without_an_interval_still_renders() -> None:
    assert format_rate(0.851) == "85.1%"


def test_a_missing_rate_renders_as_a_dash_not_zero() -> None:
    """Zero is a measurement; missing is not, and they must not look alike."""
    assert format_rate(None) == "—"
    assert format_rate(float("nan")) == "—"


def test_scoring_mode_distinguishes_ric_from_rice() -> None:
    """A RIC table and a RICE table look alike; the difference is effort."""
    ric = pd.DataFrame({"scored": [False, False]})
    rice = pd.DataFrame({"scored": [True, False]})

    assert scoring_mode(ric) == "ric"
    assert scoring_mode(rice) == "rice"
    assert scoring_mode(pd.DataFrame()) == "none"


# ---------------------------------------------------------------------------
# Overview figures
# ---------------------------------------------------------------------------


def _reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(10)],
            "platform": ["zepto"] * 5 + ["blinkit"] * 5,
            "year_month": ["2024-10"] * 4 + ["2024-11"] * 3 + ["2020-01"] * 3,
            "sentiment": ["negative"] * 8 + ["positive"] * 2,
            "in_comparable_window": [True] * 7 + [False] * 3,
        }
    )


def test_summary_separates_comparable_months_from_all_months() -> None:
    """Conflating them is what produced the 197x trend artefact."""
    summary = corpus_summary(_reviews(), pd.DataFrame({"review_id": ["r0"]}), {})

    assert summary["months"] == 3
    assert summary["comparable_months"] == 2
    assert summary["comparable_reviews"] == 7


def test_summary_reports_labels_per_review() -> None:
    labels = pd.DataFrame({"review_id": ["r0"] * 20})
    summary = corpus_summary(_reviews(), labels, {})
    assert summary["areas_per_review"] == 2.0


def test_summary_carries_grounding_through_from_the_run_report() -> None:
    report = {"coverage_pct": 98.87, "grounding": {"mean_rate": 0.9836}}
    summary = corpus_summary(_reviews(), pd.DataFrame(), report)

    assert summary["coverage_pct"] == 98.87
    assert summary["grounding_pct"] == 98.4


def test_summary_of_an_empty_corpus_is_empty_not_a_crash() -> None:
    assert corpus_summary(pd.DataFrame(), pd.DataFrame(), {}) == {}


def test_summary_handles_a_report_without_grounding() -> None:
    """An aborted run writes no grounding; the overview must still render."""
    summary = corpus_summary(_reviews(), pd.DataFrame(), {"coverage_pct": 50.0})
    assert summary["grounding_pct"] is None
