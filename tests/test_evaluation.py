"""Tests for Phase 9: the evaluation framework.

The tests that matter most here are not the ones checking arithmetic. They are
the ones checking that the framework *cannot quietly become dishonest*: that the
annotation template never leaks the model's answer, that an unlabelled gold set
produces nothing rather than something, that a bad annotation is rejected instead
of being scored as a model error, and that agreement is never serialised in a
shape a downstream reader could mistake for accuracy.

An evaluation layer is the one place where a bug does not look like a bug. Every
number still renders, the report still reads well, and the only symptom is that
the system appears better than it is.
"""

from __future__ import annotations

import json
import random

import pandas as pd
import pytest

from voc.taxonomy import get_taxonomy

from evaluation import faults as faults_mod
from evaluation import integrity
from evaluation import retrieval_eval
from evaluation.agreement import area_label_sets, area_sets, compare_models, load_cache
from evaluation.goldset import (
    ANNOTATION_COLUMNS,
    MIN_GOLD_ITEMS,
    build_annotation_guide,
    build_gold_sample,
    build_provenance,
    build_template,
    disagreement_ids,
    format_label_tokens,
    load_gold,
    parse_label_tokens,
    score_gold,
)
from evaluation.metrics import (
    Rate,
    cohen_kappa,
    confusion_matrix,
    jaccard,
    per_label_scores,
    score_categorical,
    score_multilabel,
)


@pytest.fixture(scope="module")
def taxonomy():
    return get_taxonomy()


@pytest.fixture(scope="module")
def area_ids(taxonomy):
    """Two real area ids, resolved from the YAML rather than written down."""
    return taxonomy.area_ids[:3]


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


def test_a_rate_carries_its_interval() -> None:
    rate = Rate(90, 100)
    low, high = rate.interval
    assert rate.value == 0.9
    assert low < 0.9 < high


def test_an_empty_rate_is_none_rather_than_zero() -> None:
    """Zero is a measurement. Nothing measured is not, and they must differ."""
    rate = Rate(0, 0)
    assert rate.value is None
    assert rate.interval is None
    assert "n/a" in str(rate)


def test_a_smaller_sample_gives_a_wider_interval() -> None:
    """The whole reason intervals are mandatory in this phase."""
    narrow = Rate(900, 1000).interval
    wide = Rate(9, 10).interval
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


# ---------------------------------------------------------------------------
# Multi-label scoring
# ---------------------------------------------------------------------------


def test_two_empty_label_sets_agree_completely() -> None:
    """"No area applies" is a real answer, and getting it right is not a failure."""
    assert jaccard(set(), set()) == 1.0


def test_perfect_agreement_scores_one_across_the_board() -> None:
    reference = {"a": {"x", "y"}, "b": {"z"}}
    score = score_multilabel(reference, dict(reference))

    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.exact_match.value == 1.0
    assert score.mean_jaccard == 1.0


def test_partial_overlap_separates_precision_from_recall() -> None:
    score = score_multilabel({"a": {"x", "y"}}, {"a": {"x", "z"}})

    assert score.true_positives == 1
    assert score.false_positives == 1
    assert score.false_negatives == 1
    assert score.exact_match.value == 0.0
    assert score.mean_jaccard == pytest.approx(1 / 3)


def test_items_the_candidate_never_labelled_are_excluded_not_counted_as_misses() -> None:
    """Otherwise a model improves its recall by crashing on the hard reviews.

    A review that was never labelled is a coverage failure. Folding it in as a
    miss makes it look like a disagreement; folding it in as agreement rewards
    the omission. Both are wrong, so it is reported on its own.
    """
    score = score_multilabel({"a": {"x"}, "b": {"y"}}, {"a": {"x"}})

    assert score.compared == 1
    assert score.reference_only == 1
    assert score.recall == 1.0  # over what was actually compared
    assert score.false_negatives == 0


def test_a_score_with_no_predictions_is_undefined_not_zero() -> None:
    score = score_multilabel({"a": set()}, {"a": set()})
    assert score.precision is None
    assert score.f1 is None


def test_per_label_scores_are_ranked_by_reference_support() -> None:
    """The aggregate hides which area the model is unusable on."""
    reference = {"1": {"big"}, "2": {"big"}, "3": {"big", "small"}}
    candidate = {"1": {"big"}, "2": {"big"}, "3": {"big"}}

    frame = per_label_scores(reference, candidate)
    assert list(frame["label"]) == ["big", "small"]
    assert frame.iloc[0]["recall"] == 1.0
    assert frame.iloc[1]["recall"] == 0.0


# ---------------------------------------------------------------------------
# Categorical scoring
# ---------------------------------------------------------------------------


def test_accuracy_arrives_with_the_baseline_that_makes_it_readable() -> None:
    """On a 78%-negative corpus, 78% accuracy is a constant, not a classifier."""
    reference = {str(i): "negative" for i in range(78)}
    reference.update({str(i): "positive" for i in range(78, 100)})
    candidate = {key: "negative" for key in reference}

    score = score_categorical(reference, candidate)
    assert score.accuracy.value == pytest.approx(0.78)
    assert score.majority_baseline == pytest.approx(0.78)
    assert score.lift_over_baseline == pytest.approx(0.0)


def test_kappa_is_zero_for_a_classifier_that_only_guesses_the_majority() -> None:
    reference = {str(i): "negative" for i in range(80)}
    reference.update({str(i): "positive" for i in range(80, 100)})
    candidate = {key: "negative" for key in reference}

    assert cohen_kappa(list(reference.values()), list(candidate.values())) == pytest.approx(0.0)


def test_kappa_is_none_when_chance_agreement_is_total() -> None:
    """One value on both sides agrees perfectly for no informative reason.

    Returning 1.0 would assert the classifier is perfect; 0.0 would assert it is
    useless. The data supports neither.
    """
    assert cohen_kappa(["a", "a", "a"], ["a", "a", "a"]) is None


def test_kappa_refuses_unpaired_judgements() -> None:
    with pytest.raises(ValueError, match="paired"):
        cohen_kappa(["a", "b"], ["a"])


def test_the_confusion_matrix_shows_which_way_the_errors_go() -> None:
    """Two classifiers with the same accuracy can have opposite consequences."""
    reference = {"1": "neutral", "2": "neutral", "3": "negative"}
    candidate = {"1": "negative", "2": "negative", "3": "negative"}

    matrix = confusion_matrix(reference, candidate)
    assert matrix.loc["neutral", "negative"] == 2


# ---------------------------------------------------------------------------
# The gold-set label language
# ---------------------------------------------------------------------------


def test_a_valid_annotation_parses_into_both_views(taxonomy, area_ids) -> None:
    area = taxonomy.area(area_ids[0])
    token = f"{area.id}/{area.issue_types[0].id}"

    parsed = parse_label_tokens(token, taxonomy)
    assert parsed.valid
    assert parsed.areas == {area.id}
    assert parsed.area_labels == {token}


def test_a_strength_is_marked_and_parsed_separately(taxonomy) -> None:
    area = next(a for a in taxonomy.product_areas if a.strength_types)
    token = f"{area.id}/+{area.strength_types[0].id}"

    parsed = parse_label_tokens(token, taxonomy)
    assert parsed.valid
    assert parsed.area_labels == {token}


def test_an_annotator_typo_is_rejected_not_scored_as_a_model_error(taxonomy, area_ids) -> None:
    """The reference's mistakes must never be charged to the thing being measured."""
    parsed = parse_label_tokens(f"{area_ids[0]}/not_a_real_issue_type", taxonomy)

    assert not parsed.valid
    assert parsed.area_labels == set()


def test_an_invented_area_is_rejected(taxonomy) -> None:
    parsed = parse_label_tokens("checkout_vibes/slow", taxonomy)
    assert not parsed.valid
    assert parsed.areas == set()


def test_an_issue_filed_under_the_wrong_area_is_rejected(taxonomy) -> None:
    """The failure a spell-checker cannot see: both halves real, pairing wrong."""
    source = next(a for a in taxonomy.product_areas if a.issue_types)
    other = next(
        a for a in taxonomy.product_areas
        if a.id != source.id
        and source.issue_types[0].id not in {i.id for i in a.issue_types}
    )
    parsed = parse_label_tokens(f"{other.id}/{source.issue_types[0].id}", taxonomy)
    assert not parsed.valid


def test_an_empty_annotation_is_an_empty_set_not_an_error(taxonomy) -> None:
    """A review about nothing is a real answer the model can be scored against."""
    for value in ("", "   ", None, float("nan")):
        parsed = parse_label_tokens(value, taxonomy)
        assert parsed.valid
        assert parsed.areas == set()


def test_the_fallback_area_accepts_a_free_text_label(taxonomy) -> None:
    """It exists for reviews that fit nothing; demanding a real sub-label would
    force the annotator to invent one."""
    parsed = parse_label_tokens(f"{taxonomy.fallback_area.id}/generic", taxonomy)
    assert parsed.valid


def test_the_formatter_round_trips_through_the_parser(taxonomy) -> None:
    """The guide's worked examples are generated by the formatter, so a
    disagreement between the two would send an annotator down a path the
    validator rejects."""
    area = next(a for a in taxonomy.product_areas if a.issue_types and a.strength_types)
    rendered = format_label_tokens(
        [(area.id, area.issue_types[0].id, None), (area.id, None, area.strength_types[0].id)]
    )
    parsed = parse_label_tokens(rendered, taxonomy)
    assert parsed.valid
    assert len(parsed.area_labels) == 2


# ---------------------------------------------------------------------------
# Sampling and the template
# ---------------------------------------------------------------------------


def _reviews(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": [f"r{i:03d}" for i in range(n)],
            "platform": ["zepto", "blinkit", "jiomart"] * (n // 3),
            "rating": [1, 2, 5] * (n // 3),
            "rating_bucket": ["negative", "negative", "positive"] * (n // 3),
            "review_date": ["2024-11-01"] * n,
            "review_text": [f"review number {i}" for i in range(n)],
            "sentiment": ["negative"] * n,
            "severity": ["high"] * n,
            "customer_intent": ["complaint"] * n,
            "support_escalation": [True] * n,
        }
    )


def test_the_template_never_shows_the_model_prediction() -> None:
    """The single most important test in this phase.

    An annotator shown a plausible label agrees with it far more often than they
    would have chosen it. The gold set then drifts toward the system it exists to
    audit, and measured accuracy rises for a reason that has nothing to do with
    the model being right — while every number still renders and every report
    still reads correctly.
    """
    sample = build_gold_sample(_reviews(), n_random=10)
    template = build_template(sample)

    for leaked in ("sentiment", "severity", "customer_intent", "support_escalation"):
        assert leaked not in template.columns, f"template leaks the model's {leaked}"
    assert "product_area" not in template.columns
    assert "pain_point" not in template.columns


def test_the_template_does_not_tell_the_annotator_which_stratum_a_review_is_in() -> None:
    """Knowing a review was chosen because two models disagreed changes how it
    is read, and the value of the reference is that it was not read that way."""
    sample = build_gold_sample(_reviews(), n_random=10, hard_ids=["r001"], n_hard=1)
    assert "stratum" not in build_template(sample).columns
    assert "stratum" in build_provenance(sample, seed=42).columns


def test_the_template_has_an_empty_column_for_every_answer() -> None:
    template = build_template(build_gold_sample(_reviews(), n_random=10))
    for column in ANNOTATION_COLUMNS:
        assert column in template.columns
        assert (template[column] == "").all()


def test_a_review_cannot_land_in_both_strata() -> None:
    """Overlap would let one annotation be scored twice, in two populations
    that are explicitly never pooled."""
    reviews = _reviews()
    sample = build_gold_sample(
        reviews, n_random=30, hard_ids=list(reviews["review_id"]), n_hard=20
    )
    assert not sample["review_id"].duplicated().any()


def test_the_hard_stratum_is_capped_by_availability_not_padded() -> None:
    sample = build_gold_sample(_reviews(), n_random=5, hard_ids=["r001", "r002"], n_hard=20)
    assert (sample["stratum"] == "disagreement").sum() <= 2


def test_sampling_is_reproducible_under_a_seed() -> None:
    first = build_gold_sample(_reviews(), n_random=15, seed=7)
    second = build_gold_sample(_reviews(), n_random=15, seed=7)
    assert list(first["review_id"]) == list(second["review_id"])


def test_the_guide_lists_every_label_the_parser_accepts(taxonomy) -> None:
    """A guide documenting a vocabulary the validator rejects wastes the most
    expensive input in this phase, which is a person's afternoon."""
    guide = build_annotation_guide(taxonomy, 100, {"random": 100})
    for area in taxonomy.product_areas:
        assert f"`{area.id}`" in guide
        for item in area.issue_types:
            assert f"`{item.id}`" in guide


# ---------------------------------------------------------------------------
# Loading gold labels
# ---------------------------------------------------------------------------


def _write_gold(tmp_path, rows: list[dict]):
    path = tmp_path / "gold_labels.csv"
    columns = ["review_id", *ANNOTATION_COLUMNS]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    frame[columns].to_csv(path, index=False)
    return path


def test_a_missing_gold_file_is_unusable_rather_than_empty_and_wrong(tmp_path, taxonomy) -> None:
    gold = load_gold(tmp_path / "nothing.csv", taxonomy)
    assert not gold.usable
    assert gold.labelled == 0


def test_untouched_rows_are_reported_as_unlabelled_not_scored(tmp_path, taxonomy) -> None:
    """A blank row means nobody looked. Scoring it as "no areas" would charge
    the model for reviews that were never labelled."""
    path = _write_gold(tmp_path, [{"review_id": "r1"}, {"review_id": "r2"}])
    gold = load_gold(path, taxonomy)

    assert gold.labelled == 0
    assert gold.unlabelled == ["r1", "r2"]


def test_a_row_with_only_a_sentiment_counts_as_looked_at(tmp_path, taxonomy) -> None:
    """Empty areas with the other fields filled in means "I read it and found
    no product area" -- which is answerable, unlike a skipped row."""
    path = _write_gold(tmp_path, [{"review_id": "r1", "gold_sentiment": "negative"}])
    gold = load_gold(path, taxonomy)

    assert gold.labelled == 1
    assert gold.frame.iloc[0]["gold_areas"] == set()


def test_an_out_of_vocabulary_attribute_is_rejected_and_reported(tmp_path, taxonomy) -> None:
    path = _write_gold(tmp_path, [{"review_id": "r1", "gold_sentiment": "furious"}])
    gold = load_gold(path, taxonomy)

    assert any(issue.kind == "invalid_attribute" for issue in gold.issues)
    assert gold.frame.iloc[0]["gold_sentiment"] is None


def test_booleans_survive_however_the_annotator_wrote_them(tmp_path, taxonomy) -> None:
    path = _write_gold(
        tmp_path,
        [
            {"review_id": "r1", "gold_support_escalation": "TRUE"},
            {"review_id": "r2", "gold_support_escalation": "no"},
        ],
    )
    gold = load_gold(path, taxonomy)
    assert list(gold.frame["gold_support_escalation"]) == [True, False]


def test_a_review_labelled_twice_is_kept_once_and_flagged(tmp_path, taxonomy) -> None:
    path = _write_gold(
        tmp_path,
        [
            {"review_id": "r1", "gold_sentiment": "negative"},
            {"review_id": "r1", "gold_sentiment": "positive"},
        ],
    )
    gold = load_gold(path, taxonomy)

    assert gold.labelled == 1
    assert any(issue.kind == "duplicate_row" for issue in gold.issues)


def test_provenance_attaches_the_stratum_at_load_time(tmp_path, taxonomy) -> None:
    path = _write_gold(tmp_path, [{"review_id": "r1", "gold_sentiment": "negative"}])
    provenance = pd.DataFrame({"review_id": ["r1"], "stratum": ["disagreement"]})

    gold = load_gold(path, taxonomy, provenance)
    assert gold.frame.iloc[0]["stratum"] == "disagreement"


# ---------------------------------------------------------------------------
# Scoring against gold
# ---------------------------------------------------------------------------


def test_an_unlabelled_gold_set_produces_no_accuracy_at_all(tmp_path, taxonomy) -> None:
    """Refusal, not a default. An accuracy section describing zero reviews is
    worse than an absent one, because it looks like a result."""
    path = _write_gold(tmp_path, [{"review_id": "r1"}])
    gold = load_gold(path, taxonomy)

    assert score_gold(gold, _reviews(), pd.DataFrame()) == {}


def test_strata_are_scored_separately_and_never_pooled(tmp_path, taxonomy, area_ids) -> None:
    """The disagreement stratum is biased toward difficulty by construction.
    Averaging it with the random stratum produces a number that describes no
    population at all.
    """
    area = taxonomy.area(area_ids[0])
    token = f"{area.id}/{area.issue_types[0].id}"
    path = _write_gold(
        tmp_path,
        [
            {"review_id": "r000", "gold_labels": token, "gold_sentiment": "negative"},
            {"review_id": "r001", "gold_labels": token, "gold_sentiment": "negative"},
        ],
    )
    provenance = pd.DataFrame(
        {"review_id": ["r000", "r001"], "stratum": ["random", "disagreement"]}
    )
    gold = load_gold(path, taxonomy, provenance)

    labels = pd.DataFrame(
        {
            "review_id": ["r000", "r001"],
            "product_area": [area.id, area.id],
            "issue_type": [area.issue_types[0].id, None],
            "strength_type": [None, None],
        }
    )
    scores = score_gold(gold, _reviews(), labels)

    assert set(scores) == {"random", "disagreement"}
    assert scores["random"].n == 1
    assert scores["disagreement"].n == 1


def test_a_stratum_below_the_reporting_threshold_is_marked_insufficient(
    tmp_path, taxonomy
) -> None:
    path = _write_gold(tmp_path, [{"review_id": "r000", "gold_sentiment": "negative"}])
    gold = load_gold(path, taxonomy)
    scores = score_gold(gold, _reviews(), pd.DataFrame())

    assert scores["all"].n < MIN_GOLD_ITEMS
    assert scores["all"].sufficient is False


def test_a_review_the_model_labelled_with_nothing_is_scored_not_skipped(
    tmp_path, taxonomy, area_ids
) -> None:
    """"The model found no areas here" is a prediction, and a wrong one if the
    annotator found some. Skipping it would hide a whole class of miss."""
    area = taxonomy.area(area_ids[0])
    path = _write_gold(
        tmp_path,
        [{"review_id": "r000", "gold_labels": f"{area.id}/{area.issue_types[0].id}"}],
    )
    gold = load_gold(path, taxonomy)
    scores = score_gold(gold, _reviews(), pd.DataFrame())

    assert scores["all"].areas.false_negatives == 1
    assert scores["all"].areas.compared == 1


# ---------------------------------------------------------------------------
# Inter-model agreement
# ---------------------------------------------------------------------------


def _cache_file(tmp_path, name: str, payloads: list[dict]):
    path = tmp_path / name
    path.write_text(
        json.dumps({f"key{i}": p for i, p in enumerate(payloads)}), encoding="utf-8"
    )
    return path


def _payload(review_id: str, area: str, issue: str | None, sentiment: str = "negative") -> dict:
    return {
        "review_id": review_id,
        "areas": [
            {
                "product_area": area,
                "issue_type": issue,
                "strength_type": None,
                "evidence_span": "something",
                "confidence": 0.9,
            }
        ],
        "sentiment": sentiment,
        "severity": "high",
        "customer_intent": "complaint",
        "support_escalation": True,
        "overall_confidence": 0.9,
    }


def test_a_cache_is_rekeyed_on_review_id_so_two_models_can_be_joined(tmp_path, area_ids) -> None:
    """The stored key is a hash of review, model and prompt version -- right for
    lookup, useless for comparison."""
    path = _cache_file(tmp_path, "c.json", [_payload("r1", area_ids[0], None)])
    assert set(load_cache(path)) == {"r1"}


def test_a_corrupt_cache_loads_as_empty_rather_than_killing_the_run(tmp_path) -> None:
    path = tmp_path / "c.json"
    path.write_text("{ not json", encoding="utf-8")
    assert load_cache(path) == {}


def test_agreement_is_never_serialised_as_accuracy(tmp_path, area_ids) -> None:
    """A downstream reader must not be able to pick these numbers up and relabel
    them. The flag is in the artefact, not only in the prose around it."""
    left = {"r1": _payload("r1", area_ids[0], None)}
    right = {"r1": _payload("r1", area_ids[1], None)}

    payload = compare_models(left, right, "a", "b").as_dict()
    assert payload["is_accuracy"] is False
    assert "ground truth" in payload["note"]
    assert "accuracy" not in payload["product_area"]


def test_reviews_only_one_model_reached_are_coverage_not_disagreement(tmp_path, area_ids) -> None:
    left = {"r1": _payload("r1", area_ids[0], None), "r2": _payload("r2", area_ids[0], None)}
    right = {"r1": _payload("r1", area_ids[0], None)}

    result = compare_models(left, right, "a", "b")
    assert result.overlap == 1
    assert result.left_only == 1
    assert result.areas.exact_match.value == 1.0


def test_disagreement_ids_find_exactly_the_reviews_whose_area_sets_differ() -> None:
    left = {"r1": {"a"}, "r2": {"a"}, "r3": {"a", "b"}}
    right = {"r1": {"a"}, "r2": {"b"}, "r3": {"a"}}
    assert disagreement_ids(left, right) == ["r2", "r3"]


def test_area_label_sets_use_the_gold_sets_own_token_syntax(area_ids) -> None:
    """Otherwise the fine-grained comparison scores two spellings of the same
    judgement as a disagreement."""
    payloads = {"r1": _payload("r1", area_ids[0], "late_delivery")}
    assert area_label_sets(payloads) == {"r1": {f"{area_ids[0]}/late_delivery"}}
    assert area_sets(payloads) == {"r1": {area_ids[0]}}


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_enrichment(taxonomy):
    """One enrichment the pipeline's validators accept without complaint."""
    area = next(a for a in taxonomy.product_areas if a.issue_types and a.strength_types)
    text = "the delivery was extremely late and nobody answered my messages"
    payload = {
        "review_id": "r1",
        "areas": [
            {
                "product_area": area.id,
                "issue_type": area.issue_types[0].id,
                "strength_type": None,
                "evidence_span": "the delivery was extremely late",
                "confidence": 0.9,
            }
        ],
        "pain_point": "delivery was late",
        "sentiment": taxonomy.attribute_values("sentiment")[0],
        "severity": taxonomy.attribute_values("severity")[0],
        "customer_intent": taxonomy.attribute_values("customer_intent")[0],
        "support_escalation": True,
        "overall_confidence": 0.9,
    }
    return payload, text


def test_a_clean_enrichment_raises_nothing(valid_enrichment, taxonomy) -> None:
    """The control for everything below. If this fires, every capture rate is
    measuring the fixture rather than the fault."""
    payload, text = valid_enrichment
    assert faults_mod.detect(payload, "r1", text, taxonomy) == set()


@pytest.mark.parametrize("fault", faults_mod.FAULTS, ids=lambda f: f.kind)
def test_every_fault_is_caught_by_the_validator_that_owns_it(
    fault, valid_enrichment, taxonomy
) -> None:
    payload, text = valid_enrichment
    mutated = fault.mutate(payload, text, taxonomy, random.Random(0))
    if mutated is None:
        pytest.skip(f"{fault.kind} does not apply to this enrichment")

    found = faults_mod.detect(mutated, "r1", text, taxonomy)
    assert found, f"{fault.kind} went entirely undetected"
    assert fault.expected in found, f"{fault.kind} caught by {found}, not {fault.expected}"


def test_mutation_never_touches_the_original(valid_enrichment, taxonomy) -> None:
    """A study that corrupts its own subjects would report the first fault and
    then measure a progressively more damaged corpus."""
    payload, text = valid_enrichment
    before = json.dumps(payload, sort_keys=True)

    for fault in faults_mod.FAULTS:
        fault.mutate(payload, text, taxonomy, random.Random(0))

    assert json.dumps(payload, sort_keys=True) == before


def test_the_study_only_injects_into_enrichments_that_validate_cleanly(
    valid_enrichment, taxonomy
) -> None:
    """Damaging a label the validators were already flagging cannot change the
    verdict, and would be scored as a miss."""
    payload, text = valid_enrichment
    dirty = json.loads(json.dumps(payload))
    dirty["review_id"] = "r2"
    dirty["areas"][0]["evidence_span"] = "words that are not in the review"

    pool = faults_mod.clean_pool(
        {"r1": payload, "r2": dirty}, {"r1": text, "r2": text}, taxonomy
    )
    assert pool == ["r1"]


def test_the_study_reports_a_capture_rate_with_an_interval(valid_enrichment, taxonomy) -> None:
    payload, text = valid_enrichment
    study = faults_mod.run_fault_study(
        {"r1": payload}, {"r1": text}, taxonomy, per_kind=5, seed=1
    )

    assert not study.empty
    assert set(study["fault"]) == {f.kind for f in faults_mod.FAULTS}
    scored = study[study["injected"] > 0]
    assert (scored["capture_rate"] == 1.0).all()
    assert scored["capture_ci_low"].notna().all()


def test_a_no_op_mutation_counts_as_inapplicable_not_as_a_miss(taxonomy) -> None:
    """Clearing a polarity that was already absent is the mutator failing to
    create a fault, not the validator failing to catch one."""
    area = taxonomy.product_areas[0]
    payload = {
        "review_id": "r1",
        "areas": [
            {
                "product_area": area.id,
                "issue_type": None,
                "strength_type": None,
                "evidence_span": "some words here",
                "confidence": 0.5,
            }
        ],
        "sentiment": taxonomy.attribute_values("sentiment")[0],
        "severity": None,
        "customer_intent": taxonomy.attribute_values("customer_intent")[0],
        "support_escalation": False,
        "overall_confidence": 0.5,
    }
    drop = next(f for f in faults_mod.FAULTS if f.kind == "missing_polarity")
    assert drop.mutate(payload, "some words here", taxonomy, random.Random(0)) == payload


def test_the_study_is_reproducible_under_a_seed(valid_enrichment, taxonomy) -> None:
    payload, text = valid_enrichment
    args = ({"r1": payload}, {"r1": text}, taxonomy)
    first = faults_mod.run_fault_study(*args, per_kind=4, seed=3)
    second = faults_mod.run_fault_study(*args, per_kind=4, seed=3)
    assert first.equals(second)


def test_an_empty_corpus_produces_no_study_rather_than_a_perfect_score(taxonomy) -> None:
    assert faults_mod.run_fault_study({}, {}, taxonomy).empty


def test_the_summary_weights_by_injections_not_by_fault_kind() -> None:
    """An unweighted mean would let a fault injectable twice move the headline
    as much as one injected two hundred times."""
    study = pd.DataFrame(
        {
            "fault": ["a", "b"],
            "injected": [2, 200],
            "detected": [0, 200],
            "capture_rate": [0.0, 1.0],
            "correctly_attributed": [0, 200],
        }
    )
    summary = faults_mod.summarise(study)
    assert summary["capture_rate"]["value"] == pytest.approx(200 / 202)
    assert summary["weakest_fault"] == "a"


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------


def _labels(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in ("issue_type", "strength_type"):
        if column not in frame.columns:
            frame[column] = None
    return frame


def test_the_audit_counts_invalid_labels_that_reached_the_dataset(taxonomy, area_ids) -> None:
    """The validators report but do not gate, so "detected" and "excluded" are
    different questions and only one of them had ever been answered."""
    area = taxonomy.area(area_ids[0])
    labels = _labels(
        [
            {"review_id": "r1", "product_area": area.id, "issue_type": area.issue_types[0].id},
            {"review_id": "r2", "product_area": area.id, "issue_type": "not_a_real_type"},
            {"review_id": "r3", "product_area": "invented_area"},
        ]
    )
    audit = integrity.audit(pd.DataFrame(), labels, taxonomy)

    found = {f["key"]: f["affected_labels"] for f in audit["findings"]}
    assert found["invalid_issue_type"] == 1
    assert found["unknown_area"] == 1
    # One label trips two findings at once -- an invented area with no polarity
    # -- so the clean count must be a per-row mask, not total minus the sum.
    assert found["no_polarity"] == 1
    assert audit["clean_labels"] == 1


def test_a_label_with_no_polarity_is_counted_as_meaningless(taxonomy, area_ids) -> None:
    labels = _labels([{"review_id": "r1", "product_area": area_ids[0]}])
    audit = integrity.audit(pd.DataFrame(), labels, taxonomy)
    assert {f["key"] for f in audit["findings"]} == {"no_polarity"}


def test_the_taxonomys_own_monitoring_rule_is_read_rather_than_restated(taxonomy) -> None:
    """The threshold lives in config/taxonomy.yaml. Restating it here is exactly
    the drift the project forbids everywhere else."""
    threshold = integrity.fallback_threshold(taxonomy)
    assert 0 < threshold < 1
    assert f"{threshold * 100:.0f}%" in taxonomy.fallback_area.monitoring_rule


def test_the_fallback_rule_reports_a_breach_when_the_share_is_too_high(taxonomy) -> None:
    fallback = taxonomy.fallback_area.id
    reviews = pd.DataFrame({"review_id": ["r1", "r2"]})
    labels = _labels(
        [
            {"review_id": "r1", "product_area": fallback},
            {"review_id": "r2", "product_area": fallback},
        ]
    )
    audit = integrity.audit(reviews, labels, taxonomy)
    assert audit["fallback_area"]["breached"] is True


def test_the_fallback_area_is_not_itself_an_invalid_label(taxonomy) -> None:
    """It is offered to the model by the response schema, so counting it as a
    taxonomy violation reports the design as a defect -- which is exactly what a
    full run did, 71 times."""
    labels = _labels([{"review_id": "r1", "product_area": taxonomy.fallback_area.id}])
    audit = integrity.audit(pd.DataFrame(), labels, taxonomy)
    assert not any(f["key"] == "unknown_area" for f in audit["findings"])


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------


class FakeRetriever:
    """Returns whatever it is told to, so retrieval scoring can be tested
    without loading a sentence-transformer."""

    def __init__(self, hits_by_query: dict[str, list[str]]) -> None:
        self.hits_by_query = hits_by_query
        self.filters_seen: list[dict] = []

    def search(self, query: str, k: int = 8, **kwargs):
        self.filters_seen.append(kwargs)
        ids = self.hits_by_query.get(query, [])[:k]
        hits = [type("E", (), {"review_id": rid})() for rid in ids]
        return type("R", (), {"hits": hits})()


def test_retrieval_is_scored_unfiltered(taxonomy, area_ids) -> None:
    """Passing the area as a filter returns precision 1.0 by construction and
    measures the filter rather than the retriever."""
    labels = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(60)],
            "product_area": [area_ids[0]] * 60,
        }
    )
    retriever = FakeRetriever({})
    retrieval_eval.evaluate_retrieval(
        retriever, labels, taxonomy, k=4, min_support=1,
        queries={area_ids[0]: "a query"},
    )
    assert all("product_area" not in seen for seen in retriever.filters_seen)


def test_precision_is_reported_against_the_base_rate_not_against_zero(taxonomy, area_ids) -> None:
    """19% precision looks respectable in a table and is worthless if 19% of the
    corpus carries the label."""
    labels = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(100)],
            "product_area": [area_ids[0]] * 50 + [area_ids[1]] * 50,
        }
    )
    retriever = FakeRetriever({"q": [f"r{i}" for i in range(4)]})
    frame = retrieval_eval.evaluate_retrieval(
        retriever, labels, taxonomy, k=4, min_support=1, queries={area_ids[0]: "q"}
    )

    row = frame.iloc[0]
    assert row["precision_at_k"] == 1.0
    assert row["base_rate"] == pytest.approx(0.5)
    assert row["lift"] == pytest.approx(2.0)


def test_a_retriever_no_better_than_chance_does_not_clear_the_base_rate(
    taxonomy, area_ids
) -> None:
    labels = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(100)],
            "product_area": [area_ids[0]] * 50 + [area_ids[1]] * 50,
        }
    )
    # Half the hits carry the label -- exactly the base rate.
    retriever = FakeRetriever({"q": ["r0", "r1", "r50", "r51"]})
    frame = retrieval_eval.evaluate_retrieval(
        retriever, labels, taxonomy, k=4, min_support=1, queries={area_ids[0]: "q"}
    )
    assert not frame.iloc[0]["beats_base_rate"]


def test_low_support_areas_are_skipped_rather_than_scored_on_noise(taxonomy, area_ids) -> None:
    labels = pd.DataFrame({"review_id": ["r1"], "product_area": [area_ids[0]]})
    frame = retrieval_eval.evaluate_retrieval(
        FakeRetriever({}), labels, taxonomy, k=4, min_support=50,
        queries={area_ids[0]: "q"},
    )
    assert frame.empty


def test_queries_are_generated_from_the_taxonomy_not_hand_tuned(taxonomy) -> None:
    """A tuned query set measures the person who tuned it."""
    queries = retrieval_eval.build_area_queries(taxonomy)
    assert set(queries) == set(taxonomy.area_ids)
    for area in taxonomy.product_areas:
        assert area.name in queries[area.id]
