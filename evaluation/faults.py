"""
Fault injection: measuring what the validators actually catch.

The pipeline reports 98.4% grounding, zero unknown areas, and a handful of
schema failures. Those are statements about what the *detectors found*, and on
their own they are equally consistent with two very different worlds: a model
that rarely errs, or detectors that rarely fire. Nothing in the run report can
tell the two apart, because a fault nobody detects is indistinguishable from a
fault that never happened.

So this module manufactures the errors. It takes real enrichments the pipeline
accepted, injects a known fault of a known kind, and asks the pipeline's own
validators what they make of it. The result is a **capture rate** per fault
kind, which is the number that licenses reading 98.4% as a measurement.

Two properties make the study honest:

**It uses the real validator, not a copy.** Detection runs through
``voc.enrich.parse_and_validate`` -- the same function the live run calls,
including id reconciliation. A reimplementation here would drift from the
pipeline and would end up measuring this file.

**It measures false positives too.** A detector that flags everything has a
100% capture rate and is worthless, so the study runs the unmutated
enrichments through the identical path first and records what fires on
known-good input. Capture rate without that figure is half a result.

The honest limit: injected faults are synthetic. A capture rate is an upper
bound on real-world detection, because a real model's mistakes can be subtler
than a mutation designed to represent them -- most obviously for paraphrase,
where a model that stays close to the wording is harder to catch than any
deterministic edit. What the study proves is that each defence fires on the
class of error it was built for, and does not fire otherwise.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

import pandas as pd

from voc.enrich import parse_and_validate
from voc.taxonomy import Taxonomy

from evaluation.metrics import Rate

logger = logging.getLogger(__name__)

#: Recorded when the model omits a review entirely -- not a ``ValidationIssue``
#: kind, but a detection, and the fault study must not treat it as a miss.
OMITTED = "omitted_review"

Payload = dict[str, Any]
Mutator = Callable[[Payload, str, Taxonomy, random.Random], Payload | None]


@dataclass(frozen=True)
class Fault:
    """One class of error, the mutation that creates it, and who should catch it."""

    kind: str
    description: str
    #: The ``ValidationIssue`` kind this fault ought to produce.
    expected: str
    #: Which of the three defences owns it, for reporting by layer.
    defence: str
    mutate: Mutator


# ---------------------------------------------------------------------------
# Mutations
#
# Each returns None when the fault cannot be applied to this enrichment -- a
# review with no areas cannot have its evidence fabricated. Inapplicable is not
# the same as undetected, and the study counts them separately.
# ---------------------------------------------------------------------------


def _first_area(payload: Payload) -> dict[str, Any] | None:
    areas = payload.get("areas") or []
    return areas[0] if areas else None


def _fabricate_span(payload: Payload, text: str, taxonomy: Taxonomy, rng: random.Random) -> Payload | None:
    """Replace a quote with words that are not in the review at all.

    The blunt case: outright invention. If grounding misses this it misses
    everything.
    """
    mutated = copy.deepcopy(payload)
    area = _first_area(mutated)
    if area is None:
        return None
    area["evidence_span"] = f"the courier {rng.randrange(10**6)} apologised profusely"
    return mutated


def _truncate_span(payload: Payload, text: str, taxonomy: Taxonomy, rng: random.Random) -> Payload | None:
    """Drop an interior word from a real quote.

    The subtle case, and the one that matters: a span that is *almost* verbatim,
    of the kind a model produces when it tidies a quote rather than copying it.
    It carries no ellipsis, so the schema's elision check cannot see it, and
    substring matching is the only thing standing between it and the dataset.
    """
    mutated = copy.deepcopy(payload)
    area = _first_area(mutated)
    if area is None:
        return None
    words = str(area.get("evidence_span", "")).split()
    if len(words) < 4:
        return None
    cut = rng.randrange(1, len(words) - 1)
    area["evidence_span"] = " ".join(words[:cut] + words[cut + 1:])
    return mutated


def _stitch_span(payload: Payload, text: str, taxonomy: Taxonomy, rng: random.Random) -> Payload | None:
    """Join two distant fragments with an ellipsis."""
    mutated = copy.deepcopy(payload)
    area = _first_area(mutated)
    if area is None:
        return None
    words = str(area.get("evidence_span", "")).split()
    if len(words) < 4:
        return None
    area["evidence_span"] = f"{' '.join(words[:2])}...{' '.join(words[-2:])}"
    return mutated


def _invent_area(payload: Payload, text: str, taxonomy: Taxonomy, rng: random.Random) -> Payload | None:
    """File the label under a plausible category that is not in the taxonomy.

    Derived from a real id at runtime rather than written out here, so the
    module names no taxonomy term and cannot drift from the YAML.
    """
    mutated = copy.deepcopy(payload)
    area = _first_area(mutated)
    if area is None:
        return None
    known = set(taxonomy.area_ids)
    candidate = f"{rng.choice(sorted(known))}_experience"
    while candidate in known:
        candidate += "_x"
    area["product_area"] = candidate
    return mutated


def _misfile_issue_type(payload: Payload, text: str, taxonomy: Taxonomy, rng: random.Random) -> Payload | None:
    """Keep a real issue type but move it under an area that does not own it.

    The failure schema validation cannot see: both values are in the enum, and
    only the parent-child relationship is wrong.
    """
    mutated = copy.deepcopy(payload)
    area = _first_area(mutated)
    if area is None or not area.get("issue_type"):
        return None

    issue = area["issue_type"]
    elsewhere = [
        spec.id
        for spec in taxonomy.product_areas
        if issue not in {item.id for item in spec.issue_types}
    ]
    if not elsewhere:
        return None
    area["product_area"] = rng.choice(sorted(elsewhere))
    return mutated


def _conflict_polarity(payload: Payload, text: str, taxonomy: Taxonomy, rng: random.Random) -> Payload | None:
    """Claim the review both criticises and praises the same area in one label."""
    mutated = copy.deepcopy(payload)
    area = _first_area(mutated)
    if area is None:
        return None
    spec = taxonomy.area(area["product_area"]) if area.get("product_area") in set(taxonomy.area_ids) else None
    if spec is None or not spec.issue_types or not spec.strength_types:
        return None
    area["issue_type"] = spec.issue_types[0].id
    area["strength_type"] = spec.strength_types[0].id
    return mutated


def _drop_polarity(payload: Payload, text: str, taxonomy: Taxonomy, rng: random.Random) -> Payload | None:
    """Name an area with neither an issue nor a strength -- a label with no content."""
    mutated = copy.deepcopy(payload)
    area = _first_area(mutated)
    if area is None:
        return None
    area["issue_type"] = None
    area["strength_type"] = None
    return mutated


def _invalid_sentiment(payload: Payload, text: str, taxonomy: Taxonomy, rng: random.Random) -> Payload | None:
    """Return a sentiment outside the enum."""
    mutated = copy.deepcopy(payload)
    mutated["sentiment"] = "ambivalent_leaning_positive"
    return mutated


def _duplicate_label(payload: Payload, text: str, taxonomy: Taxonomy, rng: random.Random) -> Payload | None:
    """Repeat a label verbatim, inflating that area's volume by one."""
    mutated = copy.deepcopy(payload)
    area = _first_area(mutated)
    if area is None:
        return None
    mutated["areas"].append(copy.deepcopy(area))
    return mutated


def _confidence_out_of_range(payload: Payload, text: str, taxonomy: Taxonomy, rng: random.Random) -> Payload | None:
    """Report a confidence above 1.0."""
    mutated = copy.deepcopy(payload)
    area = _first_area(mutated)
    if area is None:
        return None
    area["confidence"] = 1.5
    return mutated


def _wrong_review_id(payload: Payload, text: str, taxonomy: Taxonomy, rng: random.Random) -> Payload | None:
    """Answer about a review nobody asked about.

    The highest-consequence fault in the set: undetected, every label lands on
    the wrong review and the whole dataset is quietly misaligned rather than
    merely wrong. It is why ids are echoed back at all.
    """
    mutated = copy.deepcopy(payload)
    mutated["review_id"] = f"{rng.randrange(16**16):016x}"
    return mutated


FAULTS: tuple[Fault, ...] = (
    Fault("fabricated_span", "Quote invented outright", "ungrounded_evidence",
          "grounding", _fabricate_span),
    Fault("truncated_span", "Quote missing an interior word", "ungrounded_evidence",
          "grounding", _truncate_span),
    Fault("stitched_span", "Two fragments joined by an ellipsis", "unparseable_response",
          "schema", _stitch_span),
    Fault("invented_area", "Category not in the taxonomy", "unknown_area",
          "taxonomy", _invent_area),
    Fault("misfiled_issue_type", "Real issue type under the wrong area", "unknown_issue_type",
          "taxonomy", _misfile_issue_type),
    Fault("conflicting_polarity", "Issue and strength on one label", "conflicting_polarity",
          "taxonomy", _conflict_polarity),
    Fault("missing_polarity", "Area named with neither issue nor strength", "missing_polarity",
          "taxonomy", _drop_polarity),
    Fault("invalid_sentiment", "Attribute outside the enum", "invalid_attribute",
          "taxonomy", _invalid_sentiment),
    Fault("duplicate_label", "Identical label repeated", "duplicate_label",
          "taxonomy", _duplicate_label),
    Fault("confidence_out_of_range", "Confidence above 1.0", "unparseable_response",
          "schema", _confidence_out_of_range),
    Fault("misaligned_review_id", "Answer echoes an id nobody requested", "unexpected_review_id",
          "reconciliation", _wrong_review_id),
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@contextmanager
def _quiet_validator() -> Iterator[None]:
    """Silence the enrichment logger for the duration of the study.

    Every injection is *meant* to fail validation, so the pipeline's warnings
    are the study working rather than something going wrong. Left on, one run
    prints several thousand stack-trace-like messages and buries the actual
    result.
    """
    pipeline_logger = logging.getLogger("voc.enrich")
    previous = pipeline_logger.level
    pipeline_logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        pipeline_logger.setLevel(previous)


def detect(payload: Payload, review_id: str, review_text: str, taxonomy: Taxonomy) -> set[str]:
    """Run one enrichment through the pipeline's real validation path.

    Wrapped as a single-review group response because that is the shape the
    pipeline actually parses; going through ``parse_and_validate`` rather than
    calling the validators directly means id reconciliation is exercised too,
    and means this study cannot pass while the live path fails.
    """
    requested = pd.DataFrame([{"review_id": review_id, "review_text": review_text}])
    result = parse_and_validate(json.dumps({"results": [payload]}), requested, taxonomy)

    kinds = {issue.kind for issue in result.issues}
    if result.failed_review_ids:
        kinds.add(OMITTED)
    return kinds


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Baseline:
    """What the validators say about enrichments nobody tampered with.

    This is the control, and a capture rate is only half a result without it: a
    validator that raised on every input would score 100% on the study below and
    be worthless. The only way to tell a detector from an alarm is to feed it
    known-good data and count what fires.

    ``clean_ids`` falls out of the same pass and is what the study injects into,
    so the two halves are computed together rather than validating the corpus
    twice.
    """

    flagged: int
    examined: int
    kinds: dict[str, int]
    clean_ids: list[str]

    @property
    def rate(self) -> Rate:
        return Rate(self.flagged, self.examined)


def clean_baseline(
    enrichments: Mapping[str, Payload],
    texts: Mapping[str, str],
    taxonomy: Taxonomy,
    sample: Sequence[str] | None = None,
) -> Baseline:
    """Validate untouched enrichments: how many raise anything, and which ids do not."""
    ids = list(sample) if sample is not None else sorted(enrichments)
    flagged = 0
    kinds: dict[str, int] = {}
    clean: list[str] = []
    with _quiet_validator():
        for review_id in ids:
            found = detect(
                enrichments[review_id], review_id, texts.get(review_id, ""), taxonomy
            )
            if found:
                flagged += 1
                for kind in found:
                    kinds[kind] = kinds.get(kind, 0) + 1
            else:
                clean.append(review_id)
    return Baseline(flagged=flagged, examined=len(ids), kinds=kinds, clean_ids=clean)


def clean_pool(
    enrichments: Mapping[str, Payload],
    texts: Mapping[str, str],
    taxonomy: Taxonomy,
) -> list[str]:
    """Reviews whose stored enrichment raises nothing at all.

    The fault study runs only on these, and the reason is worth stating because
    the obvious alternative is wrong. Injecting into an enrichment that already
    carries an ungrounded span means the validator was already flagging that
    label, and no amount of further damage changes the verdict -- the fault
    would score as undetected purely because the detector had beaten the study
    to it. Filtering the pool removes the confound instead of correcting for it,
    and it sharpens the claim: *of the enrichments this pipeline currently
    accepts, how many faults would it catch?*

    Thin wrapper over ``clean_baseline``, which computes this in the same pass
    as the false-positive control. Call that directly when both are wanted.
    """
    return clean_baseline(enrichments, texts, taxonomy).clean_ids


def run_fault_study(
    enrichments: Mapping[str, Payload],
    texts: Mapping[str, str],
    taxonomy: Taxonomy,
    per_kind: int = 200,
    seed: int = 42,
    faults: Sequence[Fault] = FAULTS,
    pool: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Inject each fault into accepted enrichments and record what the validators catch.

    Every subject is drawn from ``clean_pool``, so the unmutated enrichment
    raises nothing and any issue after mutation is unambiguously the study's
    doing. ``detected`` is therefore "the validators objected", with no
    before-and-after bookkeeping to get wrong.

    ``attributed`` is the stricter reading -- the detector that fired was the one
    responsible for this fault class. A fault caught by the wrong validator is
    still caught, but it means the defence that owns it has a hole.

    A mutation that leaves the payload unchanged is counted ``inapplicable``
    rather than injected. Clearing a polarity that was already absent is not a
    fault the study created, and scoring it as a miss would penalise the
    validators for the mutator's no-op.
    """
    if not enrichments:
        return pd.DataFrame()

    # Accepting a pre-computed pool because deciding it costs one validation
    # pass over the whole cache, and the caller usually wants its size for the
    # report anyway.
    pool = list(pool) if pool is not None else clean_pool(enrichments, texts, taxonomy)
    if not pool:
        logger.warning(
            "No enrichment in the cache passes validation cleanly; the fault study "
            "has no unambiguous subjects and is skipped."
        )
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for fault in faults:
        rng = random.Random(f"{seed}:{fault.kind}")
        attempted = inapplicable = detected = attributed = 0

        # Sampled without replacement where the pool allows, so a small corpus
        # does not have one review standing in for two hundred.
        order = rng.sample(pool, k=min(per_kind, len(pool)))
        while len(order) < per_kind and pool:
            order.append(rng.choice(pool))

        with _quiet_validator():
            for review_id in order:
                payload = enrichments[review_id]
                text = texts.get(review_id, "")
                mutated = fault.mutate(payload, text, taxonomy, rng)
                if mutated is None or mutated == payload:
                    inapplicable += 1
                    continue

                attempted += 1
                found = detect(mutated, review_id, text, taxonomy)
                if found:
                    detected += 1
                    if fault.expected in found:
                        attributed += 1

        capture = Rate(detected, attempted)
        attribution = Rate(attributed, attempted)
        rows.append(
            {
                "fault": fault.kind,
                "description": fault.description,
                "defence": fault.defence,
                "expected_detector": fault.expected,
                "injected": attempted,
                "inapplicable": inapplicable,
                "detected": detected,
                "capture_rate": capture.value,
                "capture_ci_low": capture.interval[0] if capture.total else None,
                "capture_ci_high": capture.interval[1] if capture.total else None,
                "correctly_attributed": attributed,
                "attribution_rate": attribution.value,
            }
        )

    return pd.DataFrame(rows)


def summarise(study: pd.DataFrame) -> dict[str, Any]:
    """Aggregate the study, weighted by injections rather than by fault kind.

    Weighted because an unweighted mean over eleven fault kinds would let a
    class that could only be injected twice move the headline as much as one
    injected two hundred times.
    """
    if study.empty:
        return {}
    injected = int(study["injected"].sum())
    detected = int(study["detected"].sum())
    attributed = int(study["correctly_attributed"].sum())
    overall = Rate(detected, injected)
    return {
        "injected": injected,
        "detected": detected,
        "capture_rate": overall.as_dict(),
        "attribution_rate": Rate(attributed, injected).as_dict(),
        "fault_kinds": int(len(study)),
        "kinds_fully_captured": int((study["capture_rate"] == 1.0).sum()),
        "weakest_fault": (
            study.sort_values("capture_rate").iloc[0]["fault"]
            if study["capture_rate"].notna().any() else None
        ),
    }
