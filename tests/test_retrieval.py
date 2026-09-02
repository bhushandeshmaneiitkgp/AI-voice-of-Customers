"""Tests for Phase 6: retrieval, root-cause hypotheses, and orchestration.

No API calls and no model download. The provider and encoder are both injected,
so these run offline in milliseconds.

The failure this layer has to prevent is a fluent, well-cited, wrong answer.
Most of what follows checks the citation boundary: a hypothesis may only cite
reviews it was actually shown, an index and a frame built from different
corpora must refuse to pair, and a retry must change the evidence rather than
reroll the same request.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from voc.graph.rootcause_graph import (
    MAX_ATTEMPTS,
    RETRY_EVIDENCE_STEP,
    hypothesise_node,
    retrieve_node,
    run_pain_point,
    validate_node,
)
from voc.providers.base import CompletionResult
from voc.retrieval import Evidence, Retriever, format_evidence_block
from voc.rootcause import (
    RootCauseHypothesis,
    build_system_prompt,
    build_user_message,
    generate_hypotheses,
    parse_response,
    validate_citations,
)


class FakeIndex:
    """Stands in for FAISS: returns positions by descending dot product."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors
        self.ntotal = len(vectors)

    def search(self, queries, k):
        scores = queries @ self.vectors.T
        order = np.argsort(-scores, axis=1)[:, :k]
        picked = np.take_along_axis(scores, order, axis=1)
        return picked.astype("float32"), order.astype("int64")


class FakeEncoder:
    def __init__(self, mapping: dict[str, list[float]], dims: int = 3) -> None:
        self.mapping = mapping
        self.dims = dims

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        return np.array(
            [self.mapping.get(t, [1.0] + [0.0] * (self.dims - 1)) for t in texts],
            dtype="float32",
        )


def _corpus() -> tuple[pd.DataFrame, np.ndarray]:
    reviews = pd.DataFrame(
        {
            "review_id": ["a1", "b2", "c3", "d4"],
            "review_text": ["late delivery again", "wallet balance stuck",
                            "rude agent on chat", "fast and fresh"],
            "platform": ["zepto", "zepto", "blinkit", "blinkit"],
            "rating": [1, 2, 1, 5],
            "year_month": ["2024-12"] * 4,
            "sentiment": ["negative", "negative", "negative", "positive"],
            "severity": ["high", "critical", "high", "low"],
            "in_comparable_window": [True, True, True, False],
        }
    )
    vectors = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]], dtype="float32"
    )
    return reviews, vectors


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        [("a1", "delivery_reliability"), ("b2", "wallet_and_credits"),
         ("c3", "customer_support"), ("d4", "delivery_speed")],
        columns=["review_id", "product_area"],
    )


def _retriever(**kwargs) -> Retriever:
    reviews, vectors = _corpus()
    encoder = FakeEncoder({}, dims=3)
    return Retriever(FakeIndex(vectors), reviews, "fake", _labels(), encoder=encoder, **kwargs)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_index_and_frame_must_come_from_the_same_corpus() -> None:
    """Otherwise every hit points at the wrong review, with full confidence."""
    reviews, vectors = _corpus()
    with pytest.raises(ValueError, match="not built from the same corpus"):
        Retriever(FakeIndex(vectors[:2]), reviews, "fake")


def test_search_returns_the_nearest_review() -> None:
    reviews, vectors = _corpus()
    encoder = FakeEncoder({"late parcel": [1.0, 0.0, 0.0]})
    retriever = Retriever(FakeIndex(vectors), reviews, "fake", _labels(), encoder=encoder)

    result = retriever.search("late parcel", k=1)
    assert [hit.review_id for hit in result.hits] == ["a1"]


def test_an_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty query"):
        _retriever().search("   ")


def test_platform_filter_excludes_other_platforms() -> None:
    result = _retriever().search("anything", k=4, platform="blinkit")
    assert {hit.platform for hit in result.hits} == {"blinkit"}


def test_area_filter_uses_the_label_table() -> None:
    result = _retriever().search("anything", k=4, product_area="customer_support")
    assert [hit.review_id for hit in result.hits] == ["c3"]


def test_severity_filter_narrows_to_the_requested_levels() -> None:
    result = _retriever().search("anything", k=4, severity=["critical"])
    assert [hit.review_id for hit in result.hits] == ["b2"]


def test_comparable_window_filter_drops_pre_window_reviews() -> None:
    result = _retriever().search("anything", k=4, comparable_window_only=True)
    assert "d4" not in [hit.review_id for hit in result.hits]


def test_excluded_ids_are_not_returned() -> None:
    result = _retriever().search("anything", k=4, exclude_ids=["a1", "b2"])
    assert not {"a1", "b2"} & {hit.review_id for hit in result.hits}


def test_a_selective_filter_reports_truncation_rather_than_padding() -> None:
    """Returning irrelevant matches to fill k would corrupt the evidence set."""
    result = _retriever().search("anything", k=4, product_area="customer_support")
    assert len(result.hits) == 1
    assert result.truncated_by_filters is True


def test_faiss_padding_positions_are_ignored() -> None:
    """FAISS pads with -1 when it holds fewer vectors than requested."""
    reviews, vectors = _corpus()

    class PaddingIndex(FakeIndex):
        def search(self, queries, k):
            return (np.array([[0.9, 0.0]], dtype="float32"),
                    np.array([[0, -1]], dtype="int64"))

    retriever = Retriever(PaddingIndex(vectors), reviews, "fake",
                          encoder=FakeEncoder({}, dims=3))
    assert len(retriever.search("q", k=2).hits) == 1


def test_pain_point_query_is_built_from_taxonomy_ids() -> None:
    """Same construction for every pain point, so no phrasing favours one."""
    result = _retriever().evidence_for_pain_point("customer_support", "unhelpful_agent")
    assert result.query == "customer support unhelpful agent"


def test_citation_carries_the_id_first() -> None:
    hit = Evidence("abc123", "the app crashed", 0.9, "zepto", 1, "2024-12")
    assert hit.citation().startswith("[abc123]")


def test_long_evidence_is_truncated_not_dropped() -> None:
    hit = Evidence("abc123", "x" * 500, 0.9, "zepto", 1, "2024-12")
    rendered = hit.citation(max_chars=50)
    assert rendered.endswith("...")
    assert len(rendered) < 200


# ---------------------------------------------------------------------------
# Citation grounding
# ---------------------------------------------------------------------------


def _hypothesis(**overrides) -> RootCauseHypothesis:
    payload = {
        "hypothesis": "Agents close tickets on first reply.",
        "mechanism": "The queue is measured on response time, not resolution.",
        "supporting_review_ids": ["a1"],
        "disconfirming_evidence": "Some reviews praise a second agent.",
        "proposed_check": "Compare first-reply time against reopen rate.",
        "confidence": 0.6,
    }
    payload.update(overrides)
    return RootCauseHypothesis(**payload)


def test_an_invented_citation_rejects_the_hypothesis() -> None:
    """A model citing a review it was never shown is inventing corroboration."""
    kept, issues = validate_citations(
        [_hypothesis(supporting_review_ids=["a1", "not_supplied"])], ["a1", "b2"], "x/y"
    )
    assert kept == []
    assert issues[0].kind == "invented_citation"


def test_an_invented_citation_is_rejected_not_repaired() -> None:
    """Stripping the bad id would leave a claim whose support does not exist.

    That reads as evidence-backed and is not, which is worse than no claim.
    """
    kept, _ = validate_citations(
        [_hypothesis(supporting_review_ids=["good", "invented"])], ["good"], "x/y"
    )
    assert kept == [], "the surviving good citation must not rescue the hypothesis"


def test_an_uncited_hypothesis_is_rejected() -> None:
    """Uncited means reasoning from general knowledge, not from this corpus."""
    kept, issues = validate_citations(
        [_hypothesis(supporting_review_ids=[])], ["a1"], "x/y"
    )
    assert kept == []
    assert issues[0].kind == "uncited_hypothesis"


def test_a_well_cited_hypothesis_survives() -> None:
    kept, issues = validate_citations([_hypothesis()], ["a1", "b2"], "x/y")
    assert len(kept) == 1
    assert not issues


def test_missing_counter_evidence_is_flagged_but_kept() -> None:
    """An under-examined hypothesis is still a lead worth recording."""
    kept, issues = validate_citations(
        [_hypothesis(disconfirming_evidence="  ")], ["a1"], "x/y"
    )
    assert len(kept) == 1
    assert issues[0].kind == "no_disconfirming_evidence"


def test_duplicate_citations_are_collapsed() -> None:
    item = _hypothesis(supporting_review_ids=["a1", "a1", "b2"])
    assert item.supporting_review_ids == ["a1", "b2"]


def test_a_hypothesis_without_a_check_is_a_schema_violation() -> None:
    """No check means it is too vague to act on, so the schema refuses it."""
    result = parse_response(
        json.dumps({"hypotheses": [{
            "hypothesis": "Support is bad and everyone hates it",
            "mechanism": "Because the agents are not good at their jobs",
            "supporting_review_ids": ["a1"],
            "disconfirming_evidence": "none",
            "proposed_check": "",
            "confidence": 0.5,
        }]}),
        ["a1"], "x/y",
    )
    assert result.hypotheses == []
    assert result.issues[0].kind == "schema_violation"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _valid_payload(review_id: str = "a1") -> str:
    return json.dumps({"hypotheses": [{
        "hypothesis": "Agents close tickets on first reply.",
        "mechanism": "The queue is measured on response time, not resolution.",
        "supporting_review_ids": [review_id],
        "disconfirming_evidence": "Two reviews mention a helpful follow-up.",
        "proposed_check": "Compare first-reply time against ticket reopen rate.",
        "confidence": 0.6,
    }]})


def test_a_fenced_response_parses() -> None:
    result = parse_response(f"```json\n{_valid_payload()}\n```", ["a1"], "x/y")
    assert len(result.hypotheses) == 1


def test_prose_around_the_json_is_tolerated() -> None:
    result = parse_response(f"Here you go:\n{_valid_payload()}\nHope that helps.",
                            ["a1"], "x/y")
    assert len(result.hypotheses) == 1


def test_a_bare_array_is_accepted() -> None:
    """Weaker models routinely drop the wrapping key."""
    inner = json.loads(_valid_payload())["hypotheses"]
    result = parse_response(json.dumps(inner), ["a1"], "x/y")
    assert len(result.hypotheses) == 1


def test_unparseable_output_is_recorded_not_raised() -> None:
    result = parse_response("I cannot help with that.", ["a1"], "x/y")
    assert result.hypotheses == []
    assert result.issues[0].kind == "unparseable_response"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _provider(*texts: str) -> MagicMock:
    provider = MagicMock()
    provider.name = "mock"
    provider.complete.side_effect = [
        CompletionResult(text=t, usage={"input_tokens": 100, "output_tokens": 50})
        for t in texts
    ]
    return provider


def test_generation_returns_grounded_hypotheses() -> None:
    hits = _retriever().search("anything", k=2).hits
    result = generate_hypotheses(
        "customer_support", "unhelpful_agent", {"volume": 954},
        hits, MagicMock(), _provider(_valid_payload(hits[0].review_id)),
    )
    assert len(result.hypotheses) == 1
    assert result.requests_made == 1


def test_generation_with_no_evidence_does_not_call_the_model() -> None:
    """Asking for causes with nothing to reason over invites invention."""
    provider = _provider()
    result = generate_hypotheses("a", "b", {}, [], MagicMock(), provider)

    assert result.issues[0].kind == "no_evidence"
    provider.complete.assert_not_called()


def test_an_api_failure_is_recorded_not_raised() -> None:
    provider = MagicMock()
    provider.complete.side_effect = RuntimeError("502 upstream")
    hits = _retriever().search("anything", k=1).hits

    result = generate_hypotheses("a", "b", {}, hits, MagicMock(), provider)
    assert result.issues[0].kind == "api_error"


def test_the_prompt_carries_aggregate_signal_alongside_quotes() -> None:
    """A mechanism affecting a handful of orders does not explain 954 reviews."""
    hits = _retriever().search("anything", k=2).hits
    message = build_user_message("customer_support", "unhelpful_agent",
                                 {"volume": 954, "escalation_rate": 0.92}, hits)
    assert "954" in message
    assert "92%" in message
    assert hits[0].review_id in message


def test_the_system_prompt_forbids_uncited_ids() -> None:
    prompt = build_system_prompt(3)
    assert "Never cite an id that is not in the supplied set" in prompt


def test_evidence_block_lists_one_citation_per_line() -> None:
    hits = _retriever().search("anything", k=3).hits
    assert len(format_evidence_block(hits).splitlines()) == len(hits)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_retrieve_node_populates_evidence() -> None:
    state = retrieve_node(
        {"product_area": "customer_support", "issue_type": "unhelpful_agent", "k": 4},
        _retriever(),
    )
    assert state["evidence"]


def test_validate_accepts_a_grounded_attempt() -> None:
    retriever = _retriever()
    state = retrieve_node(
        {"product_area": "customer_support", "issue_type": "unhelpful_agent", "k": 4},
        retriever,
    )
    state = hypothesise_node(state, MagicMock(), _provider(_valid_payload("c3")))
    state = validate_node(state)

    assert state["done"] is True
    assert state["result"].hypotheses


def test_an_invented_citation_triggers_a_retry_with_more_evidence() -> None:
    """The retry must change the input, not reroll the same request.

    A model that invented a citation from one evidence set will often invent
    another from the same set.
    """
    retriever = _retriever()
    state = retrieve_node(
        {"product_area": "customer_support", "issue_type": "unhelpful_agent", "k": 8},
        retriever,
    )
    state = hypothesise_node(state, MagicMock(), _provider(_valid_payload("never_supplied")))
    state = validate_node(state)

    assert state["done"] is False
    assert state["k"] == 8 + RETRY_EVIDENCE_STEP
    assert any("invented_citation" in note for note in state["rejected"])


def test_retries_are_bounded() -> None:
    """Beyond a few attempts the problem is the evidence, not the draw."""
    retriever = _retriever()
    state = {"product_area": "customer_support", "issue_type": "unhelpful_agent",
             "k": 8, "attempt": 0, "rejected": []}

    for _ in range(MAX_ATTEMPTS):
        state = retrieve_node(state, retriever)
        state = hypothesise_node(state, MagicMock(), _provider(_valid_payload("bogus")))
        state = validate_node(state)

    assert state["done"] is True, "must stop rather than retry forever"
    assert state["attempt"] == MAX_ATTEMPTS


def test_earlier_rejections_survive_a_later_success() -> None:
    """A run that eventually succeeds still records what went wrong first."""
    retriever = _retriever()
    state = retrieve_node(
        {"product_area": "customer_support", "issue_type": "unhelpful_agent",
         "k": 8, "attempt": 0, "rejected": []},
        retriever,
    )
    state = hypothesise_node(state, MagicMock(), _provider(_valid_payload("bogus")))
    state = validate_node(state)
    state = retrieve_node(state, retriever)
    state = hypothesise_node(state, MagicMock(), _provider(_valid_payload("c3")))
    state = validate_node(state)

    assert state["done"] is True
    assert state["result"].hypotheses
    assert any("invented_citation" in note for note in state["rejected"])


def test_run_pain_point_accepts_an_injected_runner() -> None:
    """The retry decision is the testable part, not the framework."""
    seen = {}

    def runner(state):
        seen.update(state)
        return {**state, "done": True}

    run_pain_point("customer_support", "unhelpful_agent", {"volume": 10},
                   _retriever(), MagicMock(), MagicMock(), runner=runner)

    assert seen["product_area"] == "customer_support"
    assert seen["attempt"] == 0


def test_the_encoder_is_loaded_once_not_per_query(monkeypatch) -> None:
    """Loading a transformer per query is seconds of setup for microseconds of work.

    Tolerable across ten pain points, ruinous behind a UI where every keystroke
    would rebuild the model.
    """
    reviews, vectors = _corpus()
    retriever = Retriever(FakeIndex(vectors), reviews, "fake")

    loads = {"n": 0}

    def counting_loader(model_name):
        loads["n"] += 1
        return FakeEncoder({}, dims=3)

    monkeypatch.setattr("voc.embed.load_encoder", counting_loader)

    for _ in range(5):
        retriever.search("anything", k=1)

    assert loads["n"] == 1, f"encoder was loaded {loads['n']} times across 5 queries"


def test_an_injected_encoder_is_never_replaced() -> None:
    """Tests must stay offline; a lazy loader must not override injection."""
    reviews, vectors = _corpus()
    injected = FakeEncoder({}, dims=3)
    retriever = Retriever(FakeIndex(vectors), reviews, "fake", encoder=injected)

    assert retriever.encoder is injected
