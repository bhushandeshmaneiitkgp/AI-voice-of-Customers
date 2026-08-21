"""
Layer 4 contracts -- what the LLM must return for each review.

Three defences sit between the model and the dataset, because a structured
response is not the same thing as a *correct* one:

1. **Schema validation** (Pydantic) -- the response has the right shape.
2. **Taxonomy validation** -- every label exists in ``config/taxonomy.yaml`` and
   every issue type belongs to the area it was filed under. Models invent
   plausible-sounding categories; this catches it deterministically.
3. **Grounding verification** -- every ``evidence_span`` is a verbatim substring
   of the review it came from. This is the cheapest hallucination detector
   available and it needs no second model to run.

The taxonomy is never hardcoded here. Enums are injected at runtime from the
YAML, so the contract follows the taxonomy rather than duplicating it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from voc.taxonomy import Taxonomy


# ---------------------------------------------------------------------------
# Model output contracts
# ---------------------------------------------------------------------------


class AreaLabel(BaseModel):
    """One product area the review touches, with the evidence for it.

    Either ``issue_type`` or ``strength_type`` is set, never both: an area is
    polarity-neutral, and this is where the polarity for *this mention* lands.
    A review can criticise delivery speed and praise product range in one
    breath, which is two AreaLabels with opposite polarity.
    """

    model_config = ConfigDict(frozen=True)

    product_area: str
    issue_type: str | None = None
    strength_type: str | None = None
    evidence_span: str = Field(
        ...,
        min_length=3,
        description="Verbatim substring of the review justifying this label.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("evidence_span")
    @classmethod
    def _no_ellipsis(cls, value: str) -> str:
        """Reject stitched-together quotes.

        A span containing an ellipsis is usually the model joining two distant
        fragments, which defeats substring verification and hides how much
        context was skipped.
        """
        if "..." in value or "…" in value:
            raise ValueError("evidence_span must be a contiguous quote, not an elision")
        return value


class ReviewEnrichment(BaseModel):
    """The complete enrichment for one review."""

    model_config = ConfigDict(frozen=True)

    review_id: str = Field(..., description="Echoed back so responses cannot be misaligned.")
    areas: list[AreaLabel] = Field(default_factory=list)
    pain_point: str | None = Field(
        default=None,
        description="One sentence naming the specific problem in the customer's terms.",
    )
    sentiment: str
    severity: str | None = None
    customer_intent: str
    support_escalation: bool = False
    overall_confidence: float = Field(..., ge=0.0, le=1.0)

    @property
    def product_areas(self) -> list[str]:
        return [area.product_area for area in self.areas]


class EnrichmentBatchResponse(BaseModel):
    """Wrapper for a multi-review request.

    Several reviews are sent per API call to amortise the taxonomy prompt.
    Each result carries its own ``review_id`` so a mismatch is detectable rather
    than silently shifting every label by one row.
    """

    results: list[ReviewEnrichment]


# ---------------------------------------------------------------------------
# Validation against the taxonomy
# ---------------------------------------------------------------------------


class ValidationIssue(BaseModel):
    """One problem found in a model response. Reported, never silently dropped."""

    review_id: str
    kind: str
    detail: str


def validate_against_taxonomy(
    enrichment: ReviewEnrichment, taxonomy: Taxonomy
) -> list[ValidationIssue]:
    """Check every label exists and sits under the right parent.

    Catches the failure mode where a model returns a confident, well-formed
    label that simply is not in our taxonomy -- a plausible synonym of a real
    area, or a real issue type filed under the wrong parent area.
    """
    issues: list[ValidationIssue] = []
    known_areas = set(taxonomy.area_ids)

    def add(kind: str, detail: str) -> None:
        issues.append(ValidationIssue(review_id=enrichment.review_id, kind=kind, detail=detail))

    for label in enrichment.areas:
        if label.product_area not in known_areas:
            add("unknown_area", f"{label.product_area!r} is not a product area")
            continue

        area = taxonomy.area(label.product_area)

        if label.issue_type and label.strength_type:
            add(
                "conflicting_polarity",
                f"{label.product_area}: both issue_type and strength_type set",
            )
        if not label.issue_type and not label.strength_type:
            add("missing_polarity", f"{label.product_area}: neither issue nor strength set")

        if label.issue_type:
            valid = {item.id for item in area.issue_types}
            if label.issue_type not in valid:
                add(
                    "unknown_issue_type",
                    f"{label.issue_type!r} is not an issue type of {label.product_area}",
                )
        if label.strength_type:
            valid = {item.id for item in area.strength_types}
            if label.strength_type not in valid:
                add(
                    "unknown_strength_type",
                    f"{label.strength_type!r} is not a strength type of {label.product_area}",
                )

    duplicates = {
        area for area in enrichment.product_areas
        if enrichment.product_areas.count(area) > 1
    }
    # Duplicates are allowed only when polarity differs -- a review can praise
    # and criticise the same surface.
    for area_id in duplicates:
        labels = [label for label in enrichment.areas if label.product_area == area_id]
        polarities = {(label.issue_type, label.strength_type) for label in labels}
        if len(polarities) != len(labels):
            add("duplicate_label", f"{area_id} labelled identically more than once")

    for field, attribute in (
        ("sentiment", enrichment.sentiment),
        ("customer_intent", enrichment.customer_intent),
    ):
        allowed = taxonomy.attribute_values(field)
        if attribute not in allowed:
            add("invalid_attribute", f"{field}={attribute!r} not in {allowed}")

    if enrichment.severity is not None:
        allowed = taxonomy.attribute_values("severity")
        if enrichment.severity not in allowed:
            add("invalid_attribute", f"severity={enrichment.severity!r} not in {allowed}")

    return issues


# ---------------------------------------------------------------------------
# Grounding verification
# ---------------------------------------------------------------------------


def _normalise_for_matching(text: str) -> str:
    """Fold away differences that do not change what was quoted.

    Models routinely straighten curly quotes, collapse whitespace, or change
    case when quoting. None of those make a quote invented, so matching on the
    raw string would produce false hallucination reports.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_grounding(
    enrichment: ReviewEnrichment, review_text: str
) -> tuple[list[ValidationIssue], float]:
    """Confirm every evidence span really appears in the review.

    Returns the issues found and the share of spans that verified. This is the
    project's primary hallucination metric: it is deterministic, needs no judge
    model, and directly answers "is this insight actually supported by the
    review it cites".
    """
    haystack = _normalise_for_matching(review_text)
    issues: list[ValidationIssue] = []
    verified = 0

    for label in enrichment.areas:
        needle = _normalise_for_matching(label.evidence_span)
        if needle and needle in haystack:
            verified += 1
        else:
            issues.append(
                ValidationIssue(
                    review_id=enrichment.review_id,
                    kind="ungrounded_evidence",
                    detail=(
                        f"{label.product_area}: quoted span not found in review "
                        f"-- {label.evidence_span[:80]!r}"
                    ),
                )
            )

    rate = verified / len(enrichment.areas) if enrichment.areas else 1.0
    return issues, rate


# ---------------------------------------------------------------------------
# JSON schema for the API request
# ---------------------------------------------------------------------------


def build_response_schema(taxonomy: Taxonomy) -> dict[str, Any]:
    """Build the strict JSON schema sent as ``output_config.format``.

    Enum values are injected from the taxonomy at runtime rather than written
    out here, so the schema cannot drift from ``config/taxonomy.yaml``.

    Constraining areas and attributes to enums at the API level removes a whole
    class of error before it reaches our validators. Issue types stay free
    strings because a per-area conditional schema would be unreadable; they are
    checked by ``validate_against_taxonomy`` instead.
    """
    area_ids = taxonomy.area_ids + [taxonomy.fallback_area.id]
    issue_ids = sorted(
        {item.id for area in taxonomy.product_areas for item in area.issue_types}
    )
    strength_ids = sorted(
        {item.id for area in taxonomy.product_areas for item in area.strength_types}
    )

    area_label = {
        "type": "object",
        "properties": {
            "product_area": {"type": "string", "enum": area_ids},
            "issue_type": {"type": ["string", "null"], "enum": [*issue_ids, None]},
            "strength_type": {"type": ["string", "null"], "enum": [*strength_ids, None]},
            "evidence_span": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": [
            "product_area",
            "issue_type",
            "strength_type",
            "evidence_span",
            "confidence",
        ],
        "additionalProperties": False,
    }

    enrichment = {
        "type": "object",
        "properties": {
            "review_id": {"type": "string"},
            "areas": {"type": "array", "items": area_label},
            "pain_point": {"type": ["string", "null"]},
            "sentiment": {"type": "string", "enum": taxonomy.attribute_values("sentiment")},
            "severity": {
                "type": ["string", "null"],
                "enum": [*taxonomy.attribute_values("severity"), None],
            },
            "customer_intent": {
                "type": "string",
                "enum": taxonomy.attribute_values("customer_intent"),
            },
            "support_escalation": {"type": "boolean"},
            "overall_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": [
            "review_id",
            "areas",
            "pain_point",
            "sentiment",
            "severity",
            "customer_intent",
            "support_escalation",
            "overall_confidence",
        ],
        "additionalProperties": False,
    }

    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {"results": {"type": "array", "items": enrichment}},
            "required": ["results"],
            "additionalProperties": False,
        },
    }
