"""Tests for the AI enrichment layer.

No API calls. The client is mocked, so every check runs offline and free:
schema shape, taxonomy validation, grounding verification, id reconciliation,
retry behaviour, caching, and model-specific request construction.

The point of this suite is that the defences work when the model misbehaves.
Anyone can test the happy path; the failures that matter are a model inventing
a category, quoting text that is not there, or silently skipping a review in a
group — each of which would produce a plausible, wrong dataset.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from config.settings import Paths, load_model_registry
from voc.enrich import (
    EnrichmentCache,
    build_run_report,
    cache_key,
    chunk_reviews,
    enrich_sync,
    parse_and_validate,
    stratified_sample,
    to_dataframes,
)
from voc.providers.base import CompletionResult
from voc.enrichment_schemas import (
    AreaLabel,
    ReviewEnrichment,
    build_response_schema,
    validate_against_taxonomy,
    verify_grounding,
)
from voc.llm import estimate_cost
from voc.providers import AnthropicProvider, ProviderError, get_provider, resolve_effort
from voc.prompts import build_system_prompt, build_user_message
from voc.taxonomy import get_taxonomy


@pytest.fixture(scope="module")
def taxonomy():
    return get_taxonomy()


@pytest.fixture()
def reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["aaaa111122223333", "bbbb444455556666"],
            "platform": ["zepto", "blinkit"],
            "review_text": [
                "I have 100 in my wallet and I just wanted to use it, but the option "
                "not available. I emailed zepto but they were not supportive.",
                "Great app, fast delivery and the fruit was fresh every time.",
            ],
            "rating": [1, 5],
            "rating_bucket": ["negative", "positive"],
            "review_date": pd.to_datetime(["2024-12-10", "2024-11-02"]),
            "year_month": ["2024-12", "2024-11"],
            "is_truncated": [False, False],
            "near_dup_group_id": [-1, -1],
            "is_near_dup_representative": [True, True],
            "in_comparable_window": [True, True],
        }
    )


def _enrichment(review_id: str, **overrides) -> dict:
    payload = {
        "review_id": review_id,
        "areas": [
            {
                "product_area": "wallet_and_credits",
                "issue_type": "balance_unusable",
                "strength_type": None,
                "evidence_span": "the option not available",
                "confidence": 0.9,
            }
        ],
        "pain_point": "Wallet balance cannot be spent at checkout.",
        "sentiment": "negative",
        "severity": "high",
        "customer_intent": "complaint",
        "support_escalation": True,
        "overall_confidence": 0.88,
    }
    payload.update(overrides)
    return payload


def _mock_provider(payloads: list[dict]) -> MagicMock:
    """A provider returning one canned completion per call."""
    provider = MagicMock()
    provider.name = "mock"
    provider.supports_batch = False
    provider.complete.side_effect = [
        CompletionResult(
            text=json.dumps({"results": payload}),
            usage={"input_tokens": 100, "output_tokens": 200, "cache_read_input_tokens": 50},
        )
        for payload in payloads
    ]
    return provider


def _failing_provider(exc: Exception) -> MagicMock:
    provider = MagicMock()
    provider.name = "mock"
    provider.supports_batch = False
    provider.complete.side_effect = exc
    return provider


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_system_prompt_contains_every_area(taxonomy) -> None:
    prompt = build_system_prompt(taxonomy)
    for area in taxonomy.product_areas:
        assert area.id in prompt, f"{area.id} missing from prompt"


def test_system_prompt_contains_inclusion_and_exclusion_rules(taxonomy) -> None:
    """The boundary rules are what make multi-label output consistent."""
    prompt = build_system_prompt(taxonomy)
    area = taxonomy.product_areas[0]
    assert area.inclusion[0] in prompt
    assert area.exclusion[0] in prompt


def test_system_prompt_contains_attribute_definitions(taxonomy) -> None:
    prompt = build_system_prompt(taxonomy)
    for value in taxonomy.attribute_values("severity"):
        assert value in prompt


def test_system_prompt_carries_borderline_rules(taxonomy) -> None:
    """The hard distinctions come from the YAML, so prompt and docs cannot drift."""
    prompt = build_system_prompt(taxonomy)
    assert taxonomy.borderline_rules
    for rule in taxonomy.borderline_rules:
        assert rule.strip()[:40] in prompt


def test_system_prompt_is_byte_stable(taxonomy) -> None:
    """Prompt caching requires an identical prefix every request."""
    assert build_system_prompt(taxonomy) == build_system_prompt(taxonomy)


def test_user_message_excludes_the_rating(reviews) -> None:
    """Sentiment must come from text; leaking the rating would contaminate it."""
    message = build_user_message(reviews)
    assert "rating" not in message.lower()
    for review_id in reviews["review_id"]:
        assert review_id in message
    for text in reviews["review_text"]:
        assert text in message


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


def test_response_schema_is_strict(taxonomy) -> None:
    schema = build_response_schema(taxonomy)["schema"]
    enrichment = schema["properties"]["results"]["items"]
    assert enrichment["additionalProperties"] is False
    assert set(enrichment["required"]) == set(enrichment["properties"])


def test_response_schema_enumerates_areas_from_taxonomy(taxonomy) -> None:
    schema = build_response_schema(taxonomy)["schema"]
    area_enum = schema["properties"]["results"]["items"]["properties"]["areas"]["items"][
        "properties"
    ]["product_area"]["enum"]
    assert set(taxonomy.area_ids) <= set(area_enum)
    assert taxonomy.fallback_area.id in area_enum


def test_response_schema_enumerates_attributes(taxonomy) -> None:
    schema = build_response_schema(taxonomy)["schema"]
    properties = schema["properties"]["results"]["items"]["properties"]
    assert properties["sentiment"]["enum"] == taxonomy.attribute_values("sentiment")
    assert properties["customer_intent"]["enum"] == taxonomy.attribute_values("customer_intent")


# ---------------------------------------------------------------------------
# Taxonomy validation -- catching invented labels
# ---------------------------------------------------------------------------


def test_valid_enrichment_produces_no_issues(taxonomy) -> None:
    assert validate_against_taxonomy(ReviewEnrichment(**_enrichment("r1")), taxonomy) == []


def test_invented_area_is_caught(taxonomy) -> None:
    payload = _enrichment("r1")
    payload["areas"][0]["product_area"] = "delivery_speed"  # plausible, not real
    issues = validate_against_taxonomy(ReviewEnrichment(**payload), taxonomy)
    assert any(issue.kind == "unknown_area" for issue in issues)


def test_issue_type_under_wrong_area_is_caught(taxonomy) -> None:
    """A real issue type filed under the wrong parent is the subtle failure."""
    payload = _enrichment("r1")
    payload["areas"][0]["product_area"] = "refunds"
    payload["areas"][0]["issue_type"] = "balance_unusable"  # belongs to wallet
    issues = validate_against_taxonomy(ReviewEnrichment(**payload), taxonomy)
    assert any(issue.kind == "unknown_issue_type" for issue in issues)


def test_conflicting_polarity_is_caught(taxonomy) -> None:
    payload = _enrichment("r1")
    payload["areas"][0]["strength_type"] = "wallet_convenient"
    issues = validate_against_taxonomy(ReviewEnrichment(**payload), taxonomy)
    assert any(issue.kind == "conflicting_polarity" for issue in issues)


def test_missing_polarity_is_caught(taxonomy) -> None:
    payload = _enrichment("r1")
    payload["areas"][0]["issue_type"] = None
    issues = validate_against_taxonomy(ReviewEnrichment(**payload), taxonomy)
    assert any(issue.kind == "missing_polarity" for issue in issues)


def test_invalid_attribute_value_is_caught(taxonomy) -> None:
    payload = _enrichment("r1", sentiment="furious")
    issues = validate_against_taxonomy(ReviewEnrichment(**payload), taxonomy)
    assert any(issue.kind == "invalid_attribute" for issue in issues)


def test_same_area_with_opposite_polarity_is_allowed(taxonomy) -> None:
    """A review may praise and criticise the same surface -- that is not an error."""
    payload = _enrichment("r1")
    payload["areas"].append(
        {
            "product_area": "wallet_and_credits",
            "issue_type": None,
            "strength_type": "wallet_convenient",
            "evidence_span": "cashback is great",
            "confidence": 0.7,
        }
    )
    issues = validate_against_taxonomy(ReviewEnrichment(**payload), taxonomy)
    assert not any(issue.kind == "duplicate_label" for issue in issues)


# ---------------------------------------------------------------------------
# Grounding -- the hallucination detector
# ---------------------------------------------------------------------------


def test_grounded_evidence_verifies(reviews) -> None:
    enrichment = ReviewEnrichment(**_enrichment(reviews.loc[0, "review_id"]))
    issues, rate = verify_grounding(enrichment, reviews.loc[0, "review_text"])
    assert issues == []
    assert rate == 1.0


def test_fabricated_evidence_is_caught(reviews) -> None:
    payload = _enrichment(reviews.loc[0, "review_id"])
    payload["areas"][0]["evidence_span"] = "they refused to refund my money"
    enrichment = ReviewEnrichment(**payload)

    issues, rate = verify_grounding(enrichment, reviews.loc[0, "review_text"])

    assert rate == 0.0
    assert any(issue.kind == "ungrounded_evidence" for issue in issues)


@pytest.mark.parametrize(
    "quoted",
    [
        "The Option Not Available",          # case changed
        "the  option   not  available",      # whitespace collapsed
        "the option not available",          # exact
    ],
)
def test_grounding_tolerates_harmless_requoting(reviews, quoted: str) -> None:
    """Case and whitespace changes do not make a quote invented."""
    payload = _enrichment(reviews.loc[0, "review_id"])
    payload["areas"][0]["evidence_span"] = quoted
    _, rate = verify_grounding(ReviewEnrichment(**payload), reviews.loc[0, "review_text"])
    assert rate == 1.0


def test_elided_evidence_span_is_rejected_at_schema_level() -> None:
    """An ellipsis hides how much context was skipped between fragments."""
    with pytest.raises(Exception):
        AreaLabel(
            product_area="refunds",
            issue_type="refund_not_received",
            evidence_span="I paid online ... never got my money",
            confidence=0.9,
        )


def test_grounding_rate_is_partial_when_some_spans_fail(reviews) -> None:
    payload = _enrichment(reviews.loc[0, "review_id"])
    payload["areas"].append(
        {
            "product_area": "customer_support",
            "issue_type": "no_response",
            "strength_type": None,
            "evidence_span": "nobody ever called me back",  # not in the review
            "confidence": 0.6,
        }
    )
    _, rate = verify_grounding(ReviewEnrichment(**payload), reviews.loc[0, "review_text"])
    assert rate == 0.5


# ---------------------------------------------------------------------------
# Response reconciliation
# ---------------------------------------------------------------------------


def test_parse_and_validate_happy_path(reviews, taxonomy) -> None:
    payload = json.dumps({"results": [_enrichment(rid) for rid in reviews["review_id"]]})
    result = parse_and_validate(payload, reviews, taxonomy)

    assert len(result.enrichments) == 2
    assert result.failed_review_ids == []


def test_omitted_review_is_queued_for_retry(reviews, taxonomy) -> None:
    """A skipped review must be detected, not silently lost."""
    payload = json.dumps({"results": [_enrichment(reviews.loc[0, "review_id"])]})
    result = parse_and_validate(payload, reviews, taxonomy)

    assert len(result.enrichments) == 1
    assert result.failed_review_ids == [reviews.loc[1, "review_id"]]


def test_unrequested_review_id_is_discarded(reviews, taxonomy) -> None:
    """Guards against labels landing on the wrong rows."""
    payload = json.dumps({"results": [_enrichment("ffffffffffffffff")]})
    result = parse_and_validate(payload, reviews, taxonomy)

    assert result.enrichments == []
    assert any(issue.kind == "unexpected_review_id" for issue in result.issues)
    assert set(result.failed_review_ids) == set(reviews["review_id"])


def test_duplicate_review_id_keeps_the_first(reviews, taxonomy) -> None:
    review_id = reviews.loc[0, "review_id"]
    payload = json.dumps({"results": [_enrichment(review_id), _enrichment(review_id)]})
    result = parse_and_validate(payload, reviews, taxonomy)

    assert len(result.enrichments) == 1
    assert any(issue.kind == "duplicate_review_id" for issue in result.issues)


def test_unparseable_response_fails_the_whole_group(reviews, taxonomy) -> None:
    result = parse_and_validate("not json at all", reviews, taxonomy)

    assert result.enrichments == []
    assert set(result.failed_review_ids) == set(reviews["review_id"])
    assert any(issue.kind == "unparseable_response" for issue in result.issues)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_chunking_covers_every_review_exactly_once(reviews) -> None:
    combined = pd.concat(list(chunk_reviews(reviews, 1)))
    assert combined["review_id"].tolist() == reviews["review_id"].tolist()


def test_chunk_size_must_be_positive(reviews) -> None:
    with pytest.raises(ValueError):
        list(chunk_reviews(reviews, 0))


def test_enrich_sync_end_to_end(reviews, taxonomy) -> None:
    profile = load_model_registry()["opus"]
    provider = _mock_provider([[_enrichment(rid) for rid in reviews["review_id"]]])

    result = enrich_sync(reviews, taxonomy, profile, provider, reviews_per_request=5)

    assert len(result.enrichments) == 2
    assert result.requests_made == 1
    assert result.usage["cache_read_input_tokens"] == 50


def test_enrich_sync_retries_omitted_reviews_individually(reviews, taxonomy) -> None:
    """The retry path is what keeps grouping from costing coverage."""
    profile = load_model_registry()["opus"]
    provider = _mock_provider(
        [
            [_enrichment(reviews.loc[0, "review_id"])],   # group call omits one
            [_enrichment(reviews.loc[1, "review_id"])],   # individual retry
        ]
    )

    result = enrich_sync(reviews, taxonomy, profile, provider, reviews_per_request=2)

    assert len(result.enrichments) == 2
    assert result.failed_review_ids == []
    assert provider.complete.call_count == 2


def test_enrich_sync_records_api_errors_without_crashing(reviews, taxonomy) -> None:
    profile = load_model_registry()["opus"]
    provider = _failing_provider(ProviderError("503 overloaded"))

    result = enrich_sync(reviews, taxonomy, profile, provider, reviews_per_request=2, retry_missing=False)

    assert result.enrichments == []
    assert set(result.failed_review_ids) == set(reviews["review_id"])
    assert any(issue.kind == "api_error" for issue in result.issues)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_cache_key_separates_models() -> None:
    """A shared key would serve one model's answers during another's benchmark."""
    registry = load_model_registry()
    assert cache_key("r1", registry["opus"]) != cache_key("r1", registry["haiku"])


def test_cache_round_trips(tmp_path) -> None:
    cache = EnrichmentCache(tmp_path / "cache.json")
    enrichment = ReviewEnrichment(**_enrichment("r1"))
    cache.put("k1", enrichment)
    cache.save()

    reloaded = EnrichmentCache(tmp_path / "cache.json")
    assert reloaded.get("k1") == enrichment
    assert reloaded.get("missing") is None


def test_cache_prevents_repeat_api_calls(reviews, taxonomy, tmp_path) -> None:
    profile = load_model_registry()["opus"]
    cache = EnrichmentCache(tmp_path / "cache.json")
    for review_id in reviews["review_id"]:
        cache.put(cache_key(review_id, profile), ReviewEnrichment(**_enrichment(review_id)))

    provider = MagicMock()
    provider.supports_batch = False
    result = enrich_sync(reviews, taxonomy, profile, provider, cache=cache)

    assert result.cache_hits == 2
    provider.complete.assert_not_called()


def test_corrupt_cache_does_not_crash(tmp_path) -> None:
    path = tmp_path / "cache.json"
    path.write_text("{ broken", encoding="utf-8")
    assert EnrichmentCache(path).get("anything") is None


# ---------------------------------------------------------------------------
# Model-specific request construction
# ---------------------------------------------------------------------------


def _anthropic_provider() -> AnthropicProvider:
    """Provider with an injected mock client, so no key or network is needed."""
    return AnthropicProvider(settings=MagicMock(), client=MagicMock())


def test_adaptive_model_gets_thinking_and_effort(taxonomy) -> None:
    profile = load_model_registry()["opus"]
    params = _anthropic_provider().build_params(
        profile, "sys", "user", build_response_schema(taxonomy)
    )

    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"]["effort"] == profile.default_effort
    assert "budget_tokens" not in json.dumps(params)


def test_budget_style_model_gets_no_effort_parameter(taxonomy) -> None:
    """Haiku 4.5 returns a 400 if given `effort`; the registry prevents that."""
    profile = load_model_registry()["haiku"]
    params = _anthropic_provider().build_params(
        profile, "sys", "user", build_response_schema(taxonomy)
    )

    assert "thinking" not in params
    assert "effort" not in params["output_config"]


def test_system_prompt_is_marked_cacheable(taxonomy) -> None:
    profile = load_model_registry()["opus"]
    params = _anthropic_provider().build_params(
        profile, "sys", "user", build_response_schema(taxonomy)
    )

    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_request_uses_the_model_id_from_the_registry(taxonomy) -> None:
    """Decision 1, checked at the point where the wire request is built."""
    schema = build_response_schema(taxonomy)
    for key, profile in load_model_registry().items():
        provider = get_provider(profile, MagicMock(), client=MagicMock())
        params = provider.build_params(profile, "sys", "user", schema)
        assert params["model"] == profile.model_id, key


def test_extract_text_skips_thinking_blocks() -> None:
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="reasoning here"),
            SimpleNamespace(type="text", text='{"results": []}'),
        ]
    )
    assert AnthropicProvider.extract_text(message) == '{"results": []}'


def test_extract_text_raises_when_no_text_block() -> None:
    message = SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="...")])
    with pytest.raises(ProviderError, match="No text block"):
        AnthropicProvider.extract_text(message)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def test_batch_estimate_is_half_of_standard(taxonomy) -> None:
    profile = load_model_registry()["opus"]
    estimate = estimate_cost(profile, 4620, build_system_prompt(taxonomy))
    assert estimate.usd_batch == pytest.approx(estimate.usd_standard * 0.5)


def test_grouping_reduces_estimated_cost(taxonomy) -> None:
    """The whole reason reviews are grouped per request."""
    profile = load_model_registry()["opus"]
    prompt = build_system_prompt(taxonomy)
    one_at_a_time = estimate_cost(profile, 4620, prompt, reviews_per_request=1)
    grouped = estimate_cost(profile, 4620, prompt, reviews_per_request=5)

    assert grouped.usd_batch < one_at_a_time.usd_batch
    assert grouped.requests < one_at_a_time.requests


def test_cheaper_model_estimates_cheaper(taxonomy) -> None:
    prompt = build_system_prompt(taxonomy)
    registry = load_model_registry()
    assert (
        estimate_cost(registry["haiku"], 1000, prompt).usd_batch
        < estimate_cost(registry["opus"], 1000, prompt).usd_batch
    )


# ---------------------------------------------------------------------------
# Output shaping
# ---------------------------------------------------------------------------


def test_to_dataframes_produces_long_format_labels(reviews, taxonomy) -> None:
    profile = load_model_registry()["opus"]
    provider = _mock_provider([[_enrichment(rid) for rid in reviews["review_id"]]])
    result = enrich_sync(reviews, taxonomy, profile, provider)

    review_frame, label_frame = to_dataframes(result, reviews)

    assert len(review_frame) == 2
    assert len(label_frame) == 2  # one area each
    assert "platform" in label_frame.columns  # deterministic columns carried through
    assert set(label_frame["polarity"]) == {"issue"}


def test_to_dataframes_carries_deterministic_columns(reviews, taxonomy) -> None:
    """Facts we already know must never be re-derived from the model."""
    profile = load_model_registry()["opus"]
    provider = _mock_provider([[_enrichment(rid) for rid in reviews["review_id"]]])
    result = enrich_sync(reviews, taxonomy, profile, provider)

    review_frame, _ = to_dataframes(result, reviews)
    for column in ("platform", "rating", "review_date", "is_truncated", "in_comparable_window"):
        assert column in review_frame.columns


def test_run_report_captures_coverage_and_grounding(reviews, taxonomy) -> None:
    profile = load_model_registry()["opus"]
    provider = _mock_provider([[_enrichment(rid) for rid in reviews["review_id"]]])
    result = enrich_sync(reviews, taxonomy, profile, provider)

    report = build_run_report(result, reviews, profile, use_batch=False)

    assert report["reviews_enriched"] == 2
    assert report["coverage_pct"] == 100.0
    assert report["model"] == profile.model_id
    assert report["grounding"]["mean_rate"] is not None


def test_run_report_records_failures(reviews, taxonomy) -> None:
    profile = load_model_registry()["opus"]
    provider = _failing_provider(ProviderError("boom"))
    result = enrich_sync(reviews, taxonomy, profile, provider, retry_missing=False)

    report = build_run_report(result, reviews, profile, use_batch=False)

    assert report["coverage_pct"] == 0.0
    assert len(report["failed_review_ids"]) == 2
    assert "api_error" in report["issue_counts"]


# ---------------------------------------------------------------------------
# No hardcoding
# ---------------------------------------------------------------------------


def test_no_model_ids_hardcoded_in_enrichment_modules() -> None:
    """Decision 1 must hold in the phase that actually calls the API."""
    offenders = []
    for name in ("llm.py", "enrich.py", "prompts.py", "enrichment_schemas.py"):
        text = (Paths.root / "src" / "voc" / name).read_text(encoding="utf-8")
        if "claude-" in text:
            offenders.append(name)
    assert not offenders, f"hardcoded model ids in {offenders}"


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def _corpus(n: int = 300) -> pd.DataFrame:
    """Skewed like the real corpus: mostly negative, three platforms."""
    platforms = ["blinkit", "zepto", "jiomart"]
    buckets = ["negative"] * 8 + ["positive", "neutral"]
    return pd.DataFrame(
        {
            "review_id": [f"id{i:05d}" for i in range(n)],
            "platform": [platforms[i % 3] for i in range(n)],
            "rating_bucket": [buckets[i % 10] for i in range(n)],
            "review_text": [f"review body number {i}" for i in range(n)],
            "rating": [1 if buckets[i % 10] == "negative" else 5 for i in range(n)],
        }
    )


def test_stratified_sample_preserves_every_column() -> None:
    """Regression guard for a real bug.

    In pandas 3.0 ``groupby().apply()`` operates on each group *excluding* the
    grouping columns, so the obvious implementation silently returned a frame
    with no ``platform``. It surfaced much later while building a request,
    looking like an API failure rather than a sampling one.
    """
    frame = _corpus()
    sample = stratified_sample(frame, 50)

    assert set(sample.columns) == set(frame.columns)
    for column in ("platform", "rating_bucket", "review_text", "review_id"):
        assert column in sample.columns


def test_stratified_sample_returns_exactly_n() -> None:
    assert len(stratified_sample(_corpus(), 50)) == 50


def test_stratified_sample_covers_every_platform() -> None:
    """The point of stratifying: a small sample still exercises each platform."""
    sample = stratified_sample(_corpus(), 30)
    assert sample["platform"].nunique() == 3


def test_stratified_sample_includes_positives_from_a_negative_corpus() -> None:
    """A uniform sample of an 80%-negative corpus barely tests strengths."""
    sample = stratified_sample(_corpus(), 60)
    assert (sample["rating_bucket"] == "positive").sum() > 0


def test_stratified_sample_is_deterministic() -> None:
    frame = _corpus()
    first = stratified_sample(frame, 40, seed=7)
    second = stratified_sample(frame, 40, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_stratified_sample_returns_all_rows_when_n_exceeds_corpus() -> None:
    frame = _corpus(20)
    assert len(stratified_sample(frame, 100)) == 20


def test_stratified_sample_rows_come_from_the_source() -> None:
    frame = _corpus()
    sample = stratified_sample(frame, 25)
    assert set(sample["review_id"]) <= set(frame["review_id"])
    assert sample["review_id"].is_unique
