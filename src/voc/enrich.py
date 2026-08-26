"""
Layer 4 -- AI enrichment orchestration.

Turns cleaned reviews into validated, evidence-backed structured labels.

The pipeline is defensive at every step, because the failure mode that matters
is not a crash — it is a plausible-looking dataset that is quietly wrong:

    group reviews -> call model -> parse JSON -> validate schema
      -> validate against taxonomy -> verify evidence spans
      -> reconcile returned ids against requested ids
      -> retry anything missing, individually

Reviews are grouped to amortise the taxonomy prompt, but every returned
``review_id`` is reconciled against what was requested. A group where the model
skips a review, or invents one, is detected rather than silently shifting labels
onto the wrong rows.

Results are cached to disk keyed by review + model + prompt version, so an
interrupted run resumes instead of re-billing work already done.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd
from pydantic import ValidationError

from config.settings import ModelProfile
from voc.enrichment_schemas import (
    EnrichmentBatchResponse,
    ReviewEnrichment,
    ValidationIssue,
    build_response_schema,
    validate_against_taxonomy,
    verify_grounding,
)
from voc.llm import DEFAULT_REVIEWS_PER_REQUEST, max_tokens_for
from voc.providers import LLMProvider, ProviderError, normalise_usage
from voc.prompts import build_system_prompt, build_user_message
from voc.taxonomy import Taxonomy

logger = logging.getLogger(__name__)

#: Bump when the prompt or schema changes in a way that invalidates cached
#: responses. Part of the cache key, so old entries are simply not reused.
PROMPT_VERSION = "v1"

#: Abort a run after this many consecutive non-retryable failures. A systemic
#: problem -- exhausted credit, revoked key -- fails every request identically,
#: and grinding through the remaining thousands produces nothing but noise and
#: a long wait. Small enough to stop fast, large enough to ride out a blip.
CONSECUTIVE_FAILURE_LIMIT = 5


class RunAborted(RuntimeError):
    """Raised when a run is stopped early because every request is failing."""


@dataclass
class EnrichmentResult:
    """Everything one run produced, including what went wrong."""

    enrichments: list[ReviewEnrichment] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    failed_review_ids: list[str] = field(default_factory=list)
    grounding_rates: dict[str, float] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    requests_made: int = 0
    cache_hits: int = 0
    #: True when a failure is systemic (bad key, no credit) rather than transient.
    fatal: bool = False

    @property
    def enriched_ids(self) -> set[str]:
        return {item.review_id for item in self.enrichments}

    def merge(self, other: "EnrichmentResult") -> None:
        self.enrichments.extend(other.enrichments)
        self.issues.extend(other.issues)
        self.failed_review_ids.extend(other.failed_review_ids)
        self.grounding_rates.update(other.grounding_rates)
        self.requests_made += other.requests_made
        self.cache_hits += other.cache_hits
        self.fatal = self.fatal or other.fatal
        for key, value in other.usage.items():
            self.usage[key] = self.usage.get(key, 0) + value


# ---------------------------------------------------------------------------
# Grouping and caching
# ---------------------------------------------------------------------------


def chunk_reviews(
    frame: pd.DataFrame, size: int = DEFAULT_REVIEWS_PER_REQUEST
) -> Iterator[pd.DataFrame]:
    """Yield consecutive groups of reviews to send per request."""
    if size < 1:
        raise ValueError(f"reviews_per_request must be >= 1, got {size}")
    for start in range(0, len(frame), size):
        yield frame.iloc[start : start + size]


def stratified_sample(
    frame: pd.DataFrame,
    n: int,
    by: Sequence[str] = ("platform", "rating_bucket"),
    seed: int = 42,
) -> pd.DataFrame:
    """Proportional stratified sample that preserves every column.

    Stratifying matters because the corpus is 77.6% negative: a uniform sample
    of 100 would contain barely 20 positive reviews and would hardly exercise
    the strength vocabulary at all.

    Implemented with ``groupby().sample()`` rather than ``groupby().apply()``.
    In pandas 3.0 ``apply`` operates on each group *excluding* the grouping
    columns, so the obvious version silently returns a frame with no
    ``platform`` column -- which then fails much later, while building a
    request, looking like an API problem rather than a sampling one.

    **The result can be slightly smaller than ``n``.** Each stratum is sampled
    at ``frac`` and rounds down independently, so the pieces need not add back
    up: ``n=100`` over this corpus yields 99. That is a property of proportional
    stratification, not a bug -- but it means the requested size is not the
    sample size, and anything reporting on a run must use ``len(frame)``.
    """
    if n >= len(frame):
        return frame.reset_index(drop=True)

    fraction = n / len(frame)
    sampled = frame.groupby(list(by), observed=True, group_keys=False).sample(
        frac=fraction, random_state=seed
    )
    # Shuffle before truncating so trimming to exactly n does not favour
    # whichever stratum happens to sort first.
    return sampled.sample(frac=1.0, random_state=seed).head(n).reset_index(drop=True)


def cache_key(review_id: str, profile: ModelProfile) -> str:
    """Cache identity: the review, the model, and the prompt version.

    Changing any of the three should produce a different label, so all three
    are in the key. A cache that ignores the model would serve Haiku's answers
    for an Opus run and quietly invalidate the whole benchmark.
    """
    payload = f"{review_id}|{profile.model_id}|{PROMPT_VERSION}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


#: Persist the cache after this many new entries. A full-corpus run takes tens
#: of minutes; saving only at the end means a crash at minute 25 loses
#: everything and re-bills it. Small enough that little is ever at risk, large
#: enough that writing is not the bottleneck.
CACHE_AUTOSAVE_EVERY = 25


class EnrichmentCache:
    """Disk cache of validated enrichments, one JSON file per run configuration.

    Saves incrementally rather than only at the end, and writes atomically. A
    long run that dies partway should cost the work in flight, not the whole
    session -- and a process killed mid-write must not leave a half-written
    file that reads as corrupt on the next attempt.

    ``read_through=False`` (the ``--no-cache`` flag) makes every lookup miss, so
    a benchmark measures the model rather than replaying a previous run's
    answers. It is deliberately a *read* bypass only: existing entries are still
    loaded, still written back, and never cleared. A fresh run therefore merges
    into the file instead of replacing it, which keeps the reviews it did not
    touch intact and keeps crash-resume working. Discarding the cache to get a
    clean measurement would throw away paid-for work to buy a number.
    """

    def __init__(
        self,
        path: Path,
        autosave_every: int = CACHE_AUTOSAVE_EVERY,
        read_through: bool = True,
    ) -> None:
        self.path = path
        self.autosave_every = autosave_every
        self.read_through = read_through
        self._entries: dict[str, dict[str, Any]] = {}
        self._unsaved = 0
        if path.exists():
            try:
                self._entries = json.loads(path.read_text(encoding="utf-8"))
                logger.info("Loaded %d cached enrichments from %s", len(self._entries), path.name)
            except json.JSONDecodeError:
                logger.warning("Cache at %s is corrupt; starting empty.", path)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def unsaved(self) -> int:
        return self._unsaved

    def get(self, key: str) -> ReviewEnrichment | None:
        if not self.read_through:
            # Every review must cost a live request, otherwise coverage,
            # token counts and latency describe whichever run filled the cache.
            return None
        raw = self._entries.get(key)
        if raw is None:
            return None
        try:
            return ReviewEnrichment(**raw)
        except ValidationError:
            # A cache entry written by an older schema is not an error; it just
            # cannot be reused.
            return None

    def put(self, key: str, enrichment: ReviewEnrichment) -> None:
        self._entries[key] = enrichment.model_dump()
        self._unsaved += 1
        if self.autosave_every and self._unsaved >= self.autosave_every:
            self.save()

    def save(self) -> None:
        """Write the cache atomically.

        Serialise to a sibling temp file, then ``os.replace`` it over the
        target -- an atomic rename on every platform we support. Writing in
        place would leave a truncated, unparseable file if the process died
        mid-write, turning a recoverable interruption into total data loss.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(self._entries, indent=2), encoding="utf-8")
        os.replace(temp, self.path)
        self._unsaved = 0
        logger.info("Wrote %d cached enrichments to %s", len(self._entries), self.path.name)


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------


def normalise_response_shape(parsed: Any) -> dict[str, Any]:
    """Accept the shape variations models produce around the requested schema.

    The contract asks for ``{"results": [...]}``. Weaker models routinely return
    the array on its own, or wrap it under a differently-named key. That is a
    presentation difference, not a content one -- the same reasoning as
    stripping markdown fences -- so it is normalised rather than rejected.

    Anything genuinely unrecognisable is returned untouched, so the Pydantic
    validation immediately after produces the error message.
    """
    if isinstance(parsed, list):
        return {"results": parsed}

    if isinstance(parsed, dict):
        if "results" in parsed:
            return parsed
        # A single-key object wrapping the array under another name, e.g.
        # {"reviews": [...]} or {"enrichments": [...]}.
        if len(parsed) == 1:
            (only_value,) = parsed.values()
            if isinstance(only_value, list):
                return {"results": only_value}
        # A single enrichment returned unwrapped, without the list.
        if "review_id" in parsed:
            return {"results": [parsed]}

    return parsed


def parse_and_validate(
    payload: str,
    requested: pd.DataFrame,
    taxonomy: Taxonomy,
) -> EnrichmentResult:
    """Parse a model response and run every check against it.

    Reconciles returned ids against requested ids in both directions: a missing
    review is queued for retry, and an id that was never requested is discarded
    with an issue recorded rather than written into the dataset.
    """
    result = EnrichmentResult()

    try:
        parsed = EnrichmentBatchResponse(**normalise_response_shape(json.loads(payload)))
    except Exception as exc:  # noqa: BLE001 - any malformed shape is data, not a crash
        # Deliberately broad. Everything reaching here is untrusted model
        # output, and the only correct response to an unexpected shape is to
        # record it and move on. A narrower clause let a TypeError escape a
        # worker thread and kill a whole run when one model returned a bare
        # JSON array instead of an object.
        logger.warning("Unparseable response for %d review(s): %s", len(requested), exc)
        result.failed_review_ids.extend(requested["review_id"].tolist())
        result.issues.append(
            ValidationIssue(
                review_id="<group>",
                kind="unparseable_response",
                detail=f"{type(exc).__name__}: {exc}"[:300],
            )
        )
        return result

    requested_ids = set(requested["review_id"])
    text_by_id = dict(zip(requested["review_id"], requested["review_text"]))
    seen: set[str] = set()

    for enrichment in parsed.results:
        if enrichment.review_id not in requested_ids:
            result.issues.append(
                ValidationIssue(
                    review_id=enrichment.review_id,
                    kind="unexpected_review_id",
                    detail="Model returned an id that was not requested; discarded.",
                )
            )
            continue
        if enrichment.review_id in seen:
            result.issues.append(
                ValidationIssue(
                    review_id=enrichment.review_id,
                    kind="duplicate_review_id",
                    detail="Model returned the same review twice; kept the first.",
                )
            )
            continue
        seen.add(enrichment.review_id)

        result.issues.extend(validate_against_taxonomy(enrichment, taxonomy))
        grounding_issues, rate = verify_grounding(
            enrichment, text_by_id[enrichment.review_id]
        )
        result.issues.extend(grounding_issues)
        result.grounding_rates[enrichment.review_id] = rate
        result.enrichments.append(enrichment)

    missing = requested_ids - seen
    if missing:
        logger.warning("Model omitted %d of %d requested review(s)", len(missing), len(requested))
        result.failed_review_ids.extend(sorted(missing))

    return result


# ---------------------------------------------------------------------------
# Synchronous path
# ---------------------------------------------------------------------------


def enrich_sync(
    frame: pd.DataFrame,
    taxonomy: Taxonomy,
    profile: ModelProfile,
    provider: LLMProvider,
    reviews_per_request: int = DEFAULT_REVIEWS_PER_REQUEST,
    cache: EnrichmentCache | None = None,
    progress: Callable[[int, int], None] | None = None,
    retry_missing: bool = True,
    effort: str | None = None,
    max_concurrency: int = 1,
) -> EnrichmentResult:
    """Enrich reviews with live API calls, optionally concurrent.

    This is the only path for providers without a batch endpoint, which is
    every open-source option. ``max_concurrency`` above 1 runs groups in a
    thread pool; the work is I/O-bound waiting, so threads are sufficient and
    avoid making the whole pipeline async for one call site.
    """
    system_prompt = build_system_prompt(taxonomy)
    schema = build_response_schema(taxonomy)
    result = EnrichmentResult()

    pending = frame
    if cache is not None:
        cached_rows, uncached = [], []
        for _, row in frame.iterrows():
            hit = cache.get(cache_key(row["review_id"], profile))
            if hit is not None:
                cached_rows.append(hit)
            else:
                uncached.append(row["review_id"])
        if cached_rows:
            result.enrichments.extend(cached_rows)
            result.cache_hits = len(cached_rows)
            logger.info("Reused %d cached enrichment(s)", len(cached_rows))
        pending = frame[frame["review_id"].isin(uncached)]

    groups = list(chunk_reviews(pending, reviews_per_request))

    def run_group(group: pd.DataFrame) -> tuple[pd.DataFrame, EnrichmentResult, dict[str, int]]:
        """Call the provider for one group. Errors become issues, never crashes.

        Returns the group alongside its result so the caller can reconcile even
        when responses arrive out of order under concurrency.

        Nothing may escape this function. It runs inside a thread pool, and an
        uncaught exception surfaces at ``future.result()`` and tears down the
        entire run -- losing every group still in flight over one bad response.
        """
        try:
            completion = provider.complete(
                profile, system_prompt, build_user_message(group), schema,
                effort=effort, max_tokens=max_tokens_for(len(group)),
            )
        except Exception as exc:  # noqa: BLE001 - recorded, run continues
            retryable = getattr(exc, "retryable", True)
            logger.error(
                "Request failed for %d review(s)%s: %s",
                len(group), "" if retryable else " [non-retryable]", exc,
            )
            failure = EnrichmentResult(
                failed_review_ids=group["review_id"].tolist(),
                issues=[
                    ValidationIssue(
                        review_id="<group>",
                        kind="api_error" if retryable else "api_error_fatal",
                        detail=str(exc)[:300],
                    )
                ],
            )
            failure.fatal = not retryable
            return group, failure, {}

        try:
            group_result = parse_and_validate(completion.text, group, taxonomy)
        except Exception as exc:  # noqa: BLE001 - last line of defence
            # parse_and_validate already contains its own broad handler; this
            # exists so that a defect anywhere in validation degrades one group
            # instead of terminating the run from inside a worker thread.
            logger.exception("Validation crashed for %d review(s)", len(group))
            failure = EnrichmentResult(
                failed_review_ids=group["review_id"].tolist(),
                issues=[
                    ValidationIssue(
                        review_id="<group>",
                        kind="validation_crash",
                        detail=f"{type(exc).__name__}: {exc}"[:300],
                    )
                ],
            )
            return group, failure, completion.usage

        group_result.requests_made = 1
        return group, group_result, completion.usage

    consecutive_fatal = 0

    def absorb(group_result: EnrichmentResult, usage: dict[str, int]) -> None:
        """Fold one group's outcome in, tripping the breaker on repeated fatals."""
        nonlocal consecutive_fatal
        consecutive_fatal = consecutive_fatal + 1 if group_result.fatal else 0
        if consecutive_fatal >= CONSECUTIVE_FAILURE_LIMIT:
            raise RunAborted(
                f"{consecutive_fatal} consecutive non-retryable failures -- stopping. "
                "This is an account-level problem (credit, key, or model access), not a "
                "per-request one, so continuing would fail every remaining review. "
                f"Last error: {group_result.issues[-1].detail if group_result.issues else 'unknown'}"
            )
        result.merge(group_result)
        for key, value in usage.items():
            result.usage[key] = result.usage.get(key, 0) + value
        if cache is not None:
            for enrichment in group_result.enrichments:
                cache.put(cache_key(enrichment.review_id, profile), enrichment)

    if max_concurrency > 1 and len(groups) > 1:
        # Providers without a batch endpoint need concurrency to finish in
        # reasonable time: 924 serial requests is over an hour, eight at a time
        # is minutes. Requests are independent, so a thread pool is enough --
        # this is I/O-bound waiting, not computation.
        #
        # Groups are fed through a sliding window rather than submitted all at
        # once. Submitting everything upfront made the breaker advisory: the
        # abort propagates out of the ``with`` block, whose ``__exit__`` calls
        # ``shutdown(wait=True)`` and drains every queued future -- so a run
        # that had already decided to stop still sent the rest. A full-corpus
        # run hit exactly that: the breaker tripped after 5 failures and all
        # 868 groups were dispatched anyway. Only what is already in flight can
        # still be sent, which is bounded by ``max_concurrency``.
        completed = 0
        queued = iter(groups)
        in_flight: set[Future] = set()
        aborted: RunAborted | None = None

        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:

            def fill_window() -> None:
                """Refill the window, unless the breaker has already tripped."""
                while aborted is None and len(in_flight) < max_concurrency:
                    group = next(queued, None)
                    if group is None:
                        return
                    in_flight.add(pool.submit(run_group, group))

            fill_window()
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                in_flight.difference_update(done)
                for future in done:
                    _, group_result, usage = future.result()
                    completed += 1
                    try:
                        absorb(group_result, usage)
                    except RunAborted as exc:
                        # Stop feeding the pool, but keep draining what is
                        # already running: those requests are billed either
                        # way, so a success among them still belongs in the
                        # cache. The first abort is the one that explains the
                        # run, so later ones do not overwrite it.
                        if aborted is None:
                            aborted = exc
                    if progress:
                        progress(completed, len(groups))
                fill_window()

        if aborted is not None:
            raise aborted
    else:
        for index, group in enumerate(groups, start=1):
            _, group_result, usage = run_group(group)
            absorb(group_result, usage)
            if progress:
                progress(index, len(groups))

    # A review the model skipped in a group usually succeeds on its own, so
    # retry individually rather than losing it. Skipped entirely when the run
    # hit a systemic failure: retrying a credit or auth error once per review
    # turns one problem into thousands of identical doomed requests.
    if result.fatal:
        logger.error(
            "Skipping individual retries: the failure is account-level, not per-request."
        )
    elif retry_missing and result.failed_review_ids:
        retry_ids = list(dict.fromkeys(result.failed_review_ids))
        retry_frame = frame[frame["review_id"].isin(retry_ids)]
        if not retry_frame.empty:
            logger.info("Retrying %d review(s) individually", len(retry_frame))
            result.failed_review_ids = []
            retry_result = enrich_sync(
                retry_frame,
                taxonomy,
                profile,
                provider,
                reviews_per_request=1,
                cache=cache,
                retry_missing=False,
                effort=effort,
                max_concurrency=max_concurrency,
            )
            result.merge(retry_result)

    return result


# ---------------------------------------------------------------------------
# Batch path
# ---------------------------------------------------------------------------


def submit_batch(
    frame: pd.DataFrame,
    taxonomy: Taxonomy,
    profile: ModelProfile,
    provider: LLMProvider,
    reviews_per_request: int = DEFAULT_REVIEWS_PER_REQUEST,
    effort: str | None = None,
) -> tuple[str, dict[str, list[str]]]:
    """Submit the corpus as one Batch API job.

    Returns the batch id and a mapping of ``custom_id`` to the review ids in
    that group, which is what lets results be reconciled on retrieval.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    system_prompt = build_system_prompt(taxonomy)
    schema = build_response_schema(taxonomy)

    requests: list[Request] = []
    group_map: dict[str, list[str]] = {}

    for index, group in enumerate(chunk_reviews(frame, reviews_per_request)):
        custom_id = f"group-{index:05d}"
        group_map[custom_id] = group["review_id"].tolist()
        params = provider.build_params(
            profile, system_prompt, build_user_message(group), schema, effort=effort
        )
        requests.append(
            Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(**params))
        )

    batch = provider.client.messages.batches.create(requests=requests)
    logger.info(
        "Submitted batch %s: %d requests covering %d reviews",
        batch.id, len(requests), len(frame),
    )
    return batch.id, group_map


def poll_batch(
    provider: LLMProvider,
    batch_id: str,
    interval_seconds: int = 60,
    timeout_seconds: int = 24 * 3600,
    on_poll: Callable[[Any], None] | None = None,
) -> Any:
    """Block until a batch finishes. Most complete well inside an hour."""
    started = time.monotonic()
    while True:
        batch = provider.client.messages.batches.retrieve(batch_id)
        if on_poll:
            on_poll(batch)
        if batch.processing_status == "ended":
            return batch
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(
                f"Batch {batch_id} still {batch.processing_status} after "
                f"{timeout_seconds}s. It keeps running; retrieve it later with its id."
            )
        time.sleep(interval_seconds)


def collect_batch_results(
    provider: LLMProvider,
    batch_id: str,
    frame: pd.DataFrame,
    group_map: dict[str, list[str]],
    taxonomy: Taxonomy,
    cache: EnrichmentCache | None = None,
    profile: ModelProfile | None = None,
) -> EnrichmentResult:
    """Retrieve, parse, and validate every result in a finished batch.

    Results arrive in arbitrary order, so everything is keyed by ``custom_id``
    and never by position.
    """
    result = EnrichmentResult()
    by_id = frame.set_index("review_id")

    for entry in provider.client.messages.batches.results(batch_id):
        review_ids = group_map.get(entry.custom_id, [])
        if not review_ids:
            result.issues.append(
                ValidationIssue(
                    review_id="<group>",
                    kind="unknown_custom_id",
                    detail=f"Batch returned unrecognised custom_id {entry.custom_id!r}",
                )
            )
            continue

        requested = by_id.loc[by_id.index.intersection(review_ids)].reset_index()

        outcome = entry.result.type
        if outcome != "succeeded":
            detail = outcome
            if outcome == "errored":
                detail = f"errored: {getattr(entry.result.error, 'type', 'unknown')}"
            logger.warning("Batch entry %s %s", entry.custom_id, detail)
            result.failed_review_ids.extend(review_ids)
            result.issues.append(
                ValidationIssue(review_id="<group>", kind="batch_failure", detail=detail)
            )
            continue

        result.requests_made += 1
        for key, value in normalise_usage(getattr(entry.result.message, "usage", None)).items():
            result.usage[key] = result.usage.get(key, 0) + value

        group_result = parse_and_validate(
            provider.extract_text(entry.result.message), requested, taxonomy
        )
        result.merge(group_result)

        if cache is not None and profile is not None:
            for enrichment in group_result.enrichments:
                cache.put(cache_key(enrichment.review_id, profile), enrichment)

    return result


# ---------------------------------------------------------------------------
# Output shaping
# ---------------------------------------------------------------------------


def to_dataframes(
    result: EnrichmentResult, frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten enrichments into two tidy tables.

    Long-format labels (one row per review-area) rather than a wide table with
    list columns, because every downstream question — frequency by area,
    platform comparison, trend by month — is a ``groupby`` on this shape.
    """
    review_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    for enrichment in result.enrichments:
        review_rows.append(
            {
                "review_id": enrichment.review_id,
                "sentiment": enrichment.sentiment,
                "severity": enrichment.severity,
                "customer_intent": enrichment.customer_intent,
                "support_escalation": enrichment.support_escalation,
                "pain_point": enrichment.pain_point,
                "n_areas": len(enrichment.areas),
                "overall_confidence": enrichment.overall_confidence,
                "grounding_rate": result.grounding_rates.get(enrichment.review_id, 1.0),
            }
        )
        for label in enrichment.areas:
            label_rows.append(
                {
                    "review_id": enrichment.review_id,
                    "product_area": label.product_area,
                    "issue_type": label.issue_type,
                    "strength_type": label.strength_type,
                    "polarity": "issue" if label.issue_type else "strength",
                    "evidence_span": label.evidence_span,
                    "confidence": label.confidence,
                }
            )

    reviews = pd.DataFrame(review_rows)
    labels = pd.DataFrame(label_rows)

    # Carry the deterministic columns through so downstream analysis never has
    # to re-join, and never has to ask the model for a fact we already know.
    carry = frame[
        [
            "review_id", "platform", "rating", "rating_bucket", "review_date",
            "year_month", "review_text", "is_truncated", "near_dup_group_id",
            "is_near_dup_representative", "in_comparable_window",
        ]
    ]
    if not reviews.empty:
        reviews = carry.merge(reviews, on="review_id", how="inner")
    if not labels.empty:
        labels = labels.merge(
            carry[["review_id", "platform", "rating", "year_month", "in_comparable_window"]],
            on="review_id",
            how="left",
        )

    return reviews, labels


def build_run_report(
    result: EnrichmentResult,
    frame: pd.DataFrame,
    profile: ModelProfile,
    use_batch: bool,
    cache_bypassed: bool = False,
) -> dict[str, Any]:
    """Summarise a run: coverage, quality, and cost. Written to JSON.

    ``cache_bypassed`` is recorded because it decides how the numbers may be
    read. A run that served most reviews from cache has honest coverage but
    meaningless token, cost and latency figures, and the report is the only
    place a later reader can tell the two apart.
    """
    issue_counts: dict[str, int] = {}
    for issue in result.issues:
        issue_counts[issue.kind] = issue_counts.get(issue.kind, 0) + 1

    rates = list(result.grounding_rates.values())
    fully_grounded = sum(1 for rate in rates if rate >= 1.0)

    return {
        "model": profile.model_id,
        "model_key": profile.key,
        "prompt_version": PROMPT_VERSION,
        "transport": "batch" if use_batch else "sync",
        "cache_bypassed": cache_bypassed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reviews_requested": len(frame),
        "reviews_enriched": len(result.enrichments),
        "coverage_pct": round(len(result.enrichments) / max(1, len(frame)) * 100, 2),
        "failed_review_ids": result.failed_review_ids,
        "requests_made": result.requests_made,
        "cache_hits": result.cache_hits,
        "usage": result.usage,
        "grounding": {
            "mean_rate": round(sum(rates) / len(rates), 4) if rates else None,
            "fully_grounded_reviews": fully_grounded,
            "fully_grounded_pct": round(fully_grounded / len(rates) * 100, 2) if rates else None,
        },
        "issue_counts": issue_counts,
        "issues_sample": [issue.model_dump() for issue in result.issues[:40]],
    }
