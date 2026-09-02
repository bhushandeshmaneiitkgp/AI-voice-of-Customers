"""Tests for Phase 7: opportunities, RICE, and experiment plans.

The danger in this layer is arithmetic lending authority to a guess. A RICE
table looks quantitative whatever went into it, and an experiment plan without a
sample size reads like a plan.

So the tests concentrate on the refusals: effort is never invented, a zero
denominator never reaches the ranking, confidence decomposes into measurements,
an opportunity naming an unmeasurable metric is dropped, and an underpowered
experiment says so instead of implying a null result would mean anything.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from voc.experiments import (
    DEFAULT_MDE_PP,
    IMPRACTICAL_DURATION_MONTHS,
    baseline_for,
    build_experiment_plan,
    plan_sample_size,
    required_sample_per_arm,
)
from voc.opportunities import (
    MEASURABLE_METRICS,
    Opportunity,
    build_system_prompt,
    generate_opportunities,
    parse_response,
    validate_opportunities,
)
from voc.providers.base import CompletionResult
from voc.rice import (
    CONFIDENCE_WEIGHTS,
    IMPACT_SCALE,
    RiceInputs,
    build_rice_inputs,
    confidence_from_evidence,
    impact_from_signals,
    load_effort,
    reach_per_month,
    to_frame,
    write_effort_template,
)


def _pain_points() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("customer_support", "unhelpful_agent", 954, 3.22, 0.92, 0.011, 0.88, "a1 a2"),
            ("payments", "money_deducted_no_order", 95, 3.52, 0.68, 0.0, 0.90, "a3"),
        ],
        columns=["product_area", "issue_type", "volume", "mean_severity",
                 "escalation_rate", "churn_rate", "mean_confidence",
                 "evidence_review_ids"],
    )


def _reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["a1", "a2", "a3"],
            "grounding_rate": [1.0, 0.9, 1.0],
        }
    )


# ---------------------------------------------------------------------------
# Reach and impact
# ---------------------------------------------------------------------------


def test_reach_is_expressed_per_month() -> None:
    assert reach_per_month(300, 3) == pytest.approx(100.0)


def test_reach_needs_a_positive_observation_window() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        reach_per_month(300, 0)


def test_impact_uses_the_canonical_rice_scale() -> None:
    """A private scale sharing the name RICE is not comparable with anyone's."""
    assert IMPACT_SCALE == {"minimal": 0.25, "low": 0.5, "medium": 1.0,
                            "high": 2.0, "massive": 3.0}


def test_higher_severity_maps_to_higher_impact() -> None:
    low, _ = impact_from_signals(1.5, 0.0, 0.0)
    high, _ = impact_from_signals(3.6, 0.0, 0.0)
    assert high > low


def test_churn_intent_promotes_impact_one_step() -> None:
    """A customer saying they will leave is a different class of outcome."""
    without, label_without = impact_from_signals(3.1, 0.0, 0.0)
    with_churn, label_with = impact_from_signals(3.1, 0.05, 0.0)

    assert with_churn > without
    assert (label_without, label_with) == ("high", "massive")


def test_impact_cannot_exceed_the_top_of_the_scale() -> None:
    impact, label = impact_from_signals(4.0, 0.99, 0.99)
    assert impact == IMPACT_SCALE["massive"]
    assert label == "massive"


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_confidence_weights_sum_to_one() -> None:
    assert sum(CONFIDENCE_WEIGHTS.values()) == pytest.approx(1.0)


def test_confidence_decomposes_into_its_parts() -> None:
    """A confidence number nobody can decompose is a feeling with a decimal."""
    score, parts = confidence_from_evidence(0.98, 954, 0.88, True, True)
    assert set(parts) == set(CONFIDENCE_WEIGHTS)
    assert score == pytest.approx(
        sum(CONFIDENCE_WEIGHTS[k] * v for k, v in parts.items())
    )


def test_better_evidence_raises_confidence() -> None:
    weak, _ = confidence_from_evidence(0.5, 20, 0.5, False, False)
    strong, _ = confidence_from_evidence(1.0, 500, 1.0, True, True)
    assert 0.0 <= weak < strong <= 1.0


def test_a_missing_mechanism_costs_exactly_its_weight() -> None:
    with_mech, _ = confidence_from_evidence(1.0, 500, 1.0, True, True)
    without, _ = confidence_from_evidence(1.0, 500, 1.0, False, True)
    assert with_mech - without == pytest.approx(CONFIDENCE_WEIGHTS["mechanism"])


def test_sample_confidence_saturates_rather_than_growing_forever() -> None:
    """Past a point, more reviews stop buying certainty that a pattern is real."""
    _, at_cap = confidence_from_evidence(1.0, 300, 1.0, True, True)
    _, far_past = confidence_from_evidence(1.0, 50_000, 1.0, True, True)
    assert at_cap["sample"] == far_past["sample"] == 1.0


# ---------------------------------------------------------------------------
# Effort: the refusal
# ---------------------------------------------------------------------------


def test_effort_is_never_invented() -> None:
    """Customer reviews cannot contain how long a change takes to build."""
    inputs = build_rice_inputs(_pain_points(), _reviews(), months_observed=3)
    assert all(item.effort_person_weeks is None for item in inputs)
    assert all(item.rice is None for item in inputs)


def test_ric_is_still_computed_without_effort() -> None:
    """The part the data supports is reported; the part it does not, is not."""
    inputs = build_rice_inputs(_pain_points(), _reviews(), months_observed=3)
    assert all(item.ric > 0 for item in inputs)


def test_supplying_effort_produces_a_full_score() -> None:
    effort = {("customer_support", "unhelpful_agent"): 4.0}
    inputs = build_rice_inputs(_pain_points(), _reviews(), 3, effort=effort)
    scored = [i for i in inputs if i.has_effort]

    assert len(scored) == 1
    assert scored[0].rice == pytest.approx(scored[0].ric / 4.0)


def test_scored_and_unscored_items_are_not_interleaved() -> None:
    """Ranking them into one list would imply they are comparable."""
    effort = {("payments", "money_deducted_no_order"): 2.0}
    frame = to_frame(build_rice_inputs(_pain_points(), _reviews(), 3, effort=effort))

    assert frame.iloc[0]["scored"]
    assert not frame.iloc[-1]["scored"]


def test_a_zero_effort_estimate_is_skipped_not_treated_as_free(tmp_path) -> None:
    """A zero denominator sends an item straight to the top of the ranking."""
    path = tmp_path / "effort.csv"
    pd.DataFrame(
        [("customer_support", "unhelpful_agent", 0),
         ("payments", "money_deducted_no_order", 3)],
        columns=["product_area", "issue_type", "effort_person_weeks"],
    ).to_csv(path, index=False)

    loaded = load_effort(path)
    assert ("customer_support", "unhelpful_agent") not in loaded
    assert loaded[("payments", "money_deducted_no_order")] == 3.0


def test_a_blank_effort_estimate_is_skipped(tmp_path) -> None:
    path = tmp_path / "effort.csv"
    pd.DataFrame(
        [("a", "b", None)], columns=["product_area", "issue_type", "effort_person_weeks"]
    ).to_csv(path, index=False)
    assert load_effort(path) == {}


def test_an_effort_file_missing_a_column_is_rejected(tmp_path) -> None:
    path = tmp_path / "effort.csv"
    pd.DataFrame([("a", "b")], columns=["product_area", "issue_type"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing column"):
        load_effort(path)


def test_the_effort_template_shows_what_is_being_traded_against(tmp_path) -> None:
    """Estimating effort from a bare list of ids invites arbitrary numbers."""
    path = tmp_path / "template.csv"
    write_effort_template(build_rice_inputs(_pain_points(), _reviews(), 3), path)

    written = pd.read_csv(path)
    assert {"reach_per_month", "impact_label", "confidence", "ric"} <= set(written.columns)
    assert written["effort_person_weeks"].isna().all()


def test_grounding_falls_back_to_the_corpus_mean_not_to_perfection() -> None:
    """An optimistic default would inflate confidence for unverifiable rows."""
    pain_points = _pain_points().assign(evidence_review_ids=["unknown_id", "unknown_id"])
    reviews = _reviews().assign(grounding_rate=[0.5, 0.5, 0.5])

    inputs = build_rice_inputs(pain_points, reviews, 3)
    assert inputs[0].confidence_parts["grounding"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Experiment sample sizes
# ---------------------------------------------------------------------------


def test_smaller_effects_need_larger_samples() -> None:
    assert required_sample_per_arm(0.6, 1.0) > required_sample_per_arm(0.6, 5.0)


def test_more_power_needs_a_larger_sample() -> None:
    assert (required_sample_per_arm(0.6, 3.0, power=0.95)
            > required_sample_per_arm(0.6, 3.0, power=0.80))


def test_sample_size_matches_the_standard_result() -> None:
    """~4,200 per arm for 50% -> 47% at 80% power, two-sided alpha 0.05."""
    n = required_sample_per_arm(0.50, 3.0)
    assert 4000 <= n <= 4600, n


def test_a_zero_effect_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        required_sample_per_arm(0.5, 0.0)


def test_an_impossible_baseline_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        required_sample_per_arm(0.0, 3.0)


def test_an_effect_leaving_the_valid_range_is_rejected() -> None:
    """A 60pp move from 50% has nowhere to land: 50-60 is negative, 50+60 > 100."""
    with pytest.raises(ValueError, match="leaves the valid range"):
        required_sample_per_arm(0.50, 60.0)


def test_an_effect_that_would_go_negative_flips_direction() -> None:
    """A 3pp drop from 2% is impossible, so it is measured as a rise instead."""
    plan = plan_sample_size(0.02, monthly_volume=1000, mde_pp=3.0)
    assert plan.target_rate == pytest.approx(0.05)


def test_duration_is_computed_from_observed_volume() -> None:
    plan = plan_sample_size(0.6, monthly_volume=1000, mde_pp=3.0)
    assert plan.months_required == pytest.approx(plan.total / 1000)


def test_an_underpowered_experiment_says_so() -> None:
    """A null result from an underpowered test means 'we could not have seen it'."""
    plan = plan_sample_size(0.6, monthly_volume=10, mde_pp=1.0)
    assert plan.practical is False
    assert "too slow to act on" in plan.verdict


def test_a_practical_experiment_reports_a_duration() -> None:
    plan = plan_sample_size(0.6, monthly_volume=100_000, mde_pp=5.0)
    assert plan.practical is True
    assert plan.months_required < IMPRACTICAL_DURATION_MONTHS


def test_a_rare_baseline_is_flagged() -> None:
    plan = build_experiment_plan(
        "payments", "double_charge", "Title", "Hypothesis",
        baseline_rate=0.007, monthly_volume=1000,
        primary_metric="churn_share", guardrail_metric="negative_share",
    )
    assert any("very low" in note for note in plan.notes)


def test_every_plan_carries_a_guardrail() -> None:
    """Without one, an experiment can 'win' at the cost of something bigger."""
    plan = build_experiment_plan(
        "customer_support", "unhelpful_agent", "Route refunds to humans",
        "Skipping the bot reduces escalation", 0.6, 1000,
        "escalation_rate", "negative_share",
    )
    assert plan.guardrail_metric == "negative_share"
    assert "negative_share" in plan.rollback


def test_a_missing_baseline_returns_none_rather_than_a_default() -> None:
    """A substituted rate produces a sample size for an experiment nobody can run."""
    metrics = pd.DataFrame(
        [("zepto", "negative_share", 0.85, 1719)],
        columns=["platform", "metric", "rate", "reviews"],
    )
    assert baseline_for(metrics, "not_measured") is None


def test_a_pooled_baseline_is_weighted_by_review_count() -> None:
    """An unweighted mean lets a small platform drag the baseline."""
    metrics = pd.DataFrame(
        [("big", "negative_share", 0.90, 1000), ("small", "negative_share", 0.10, 10)],
        columns=["platform", "metric", "rate", "reviews"],
    )
    assert baseline_for(metrics, "negative_share") > 0.85


# ---------------------------------------------------------------------------
# Opportunities
# ---------------------------------------------------------------------------


def _opportunity(**overrides) -> Opportunity:
    payload = {
        "title": "Route refund tickets to a human queue",
        "change": "Detect refund keywords at intake and skip the bot entirely.",
        "addresses_hypothesis": "Bots cannot resolve refund cases.",
        "primary_metric": "escalation_rate",
        "expected_direction": "decrease",
        "risk_if_wrong": "Human queue backs up and response times rise for everyone.",
    }
    payload.update(overrides)
    return Opportunity(**payload)


def test_an_unmeasurable_metric_is_rejected() -> None:
    """'Improve satisfaction' has no success condition this pipeline can check."""
    kept, issues = validate_opportunities(
        [_opportunity(primary_metric="customer_happiness")], "a/b"
    )
    assert kept == []
    assert issues[0].kind == "unmeasurable_metric"


def test_every_allowed_metric_is_actually_measured_upstream() -> None:
    """The enum must not drift from what Phase 5 computes."""
    from voc.trends import METRICS

    measured = {spec.key for spec in METRICS} | {"pain_point_volume"}
    assert set(MEASURABLE_METRICS) <= measured


def test_an_opportunity_claiming_no_risk_is_rejected() -> None:
    """A change that cannot hurt anyone usually cannot help either."""
    kept, issues = validate_opportunities(
        [_opportunity(risk_if_wrong="No risk to customers at all.")], "a/b"
    )
    assert kept == []
    assert issues[0].kind == "no_stated_risk"


def test_a_well_formed_opportunity_survives() -> None:
    kept, issues = validate_opportunities([_opportunity()], "a/b")
    assert len(kept) == 1
    assert not issues


def test_opportunities_are_not_generated_without_a_hypothesis() -> None:
    """Without a mechanism these would be guesses dressed as analysis."""
    provider = MagicMock()
    result = generate_opportunities("a", "b", {}, [], MagicMock(), provider)

    assert result.issues[0].kind == "no_hypotheses"
    provider.complete.assert_not_called()


def test_the_prompt_forbids_the_model_from_estimating_effort() -> None:
    """It cannot see the codebase, and a guess would be read as an estimate."""
    prompt = build_system_prompt()
    assert "Do NOT estimate effort" in prompt


def test_generation_parses_a_valid_response() -> None:
    payload = json.dumps({"opportunities": [_opportunity().model_dump()]})
    provider = MagicMock()
    provider.complete.return_value = CompletionResult(
        text=payload, usage={"input_tokens": 100, "output_tokens": 50}
    )

    result = generate_opportunities(
        "customer_support", "unhelpful_agent", {"volume": 954},
        [{"hypothesis": "Bots cannot resolve refunds.", "mechanism": "..."}],
        MagicMock(), provider,
    )
    assert len(result.opportunities) == 1
    assert result.requests_made == 1


def test_an_api_failure_is_recorded_not_raised() -> None:
    provider = MagicMock()
    provider.complete.side_effect = RuntimeError("503")
    result = generate_opportunities(
        "a", "b", {}, [{"hypothesis": "h"}], MagicMock(), provider
    )
    assert result.issues[0].kind == "api_error"


def test_unparseable_output_is_recorded() -> None:
    result = parse_response("no json here", "a/b")
    assert result.opportunities == []
    assert result.issues[0].kind == "unparseable_response"
