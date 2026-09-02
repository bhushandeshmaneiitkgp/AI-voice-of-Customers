"""
Inter-model agreement, computed rather than narrated.

``docs/MODEL_BENCHMARK.md`` reports that two models agree on sentiment 96.8% of
the time and on the exact product-area set 61% of the time. Those figures were
originally worked out by hand and written into prose, which makes them
unreproducible and unfalsifiable -- exactly the property this project spends its
effort avoiding everywhere else. This module derives them from the per-model
caches instead, so the benchmark document quotes a computation rather than a
memory.

**Agreement is not accuracy, and the vocabulary here refuses to blur it.**
Nothing in this module returns a field called accuracy, the serialised output
carries an explicit ``is_accuracy: false``, and the two models are named
``left`` and ``right`` rather than reference and prediction. Two models can be
wrong together; on the areas where they most often differ, at least one of them
is wrong and neither can say which.

What agreement *is* good for is triage. The reviews where two independently
prompted models produce different area sets are the reviews most likely to be
genuinely ambiguous, and a limited annotation budget should spend itself there
first. That is what ``goldset.disagreement_ids`` consumes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from evaluation.goldset import format_label_tokens
from evaluation.metrics import (
    CategoricalScore,
    MultiLabelScore,
    score_categorical,
    score_multilabel,
)

logger = logging.getLogger(__name__)

#: Single-valued fields worth comparing across models. Confidence is excluded
#: deliberately: models are not calibrated against each other, so a difference
#: in stated confidence says nothing about whether either is right.
COMPARED_FIELDS: tuple[str, ...] = (
    "sentiment",
    "severity",
    "customer_intent",
    "support_escalation",
)


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Read a per-model enrichment cache, keyed by review id.

    The cache is keyed on a hash of review, model and prompt version, which is
    right for lookup and useless for comparison. Every stored payload carries
    its own ``review_id``, so re-keying on that is what makes two caches
    joinable without reconstructing anybody's hash.
    """
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Cache at %s is not readable JSON", path)
        return {}

    by_review: dict[str, dict[str, Any]] = {}
    for payload in raw.values():
        review_id = payload.get("review_id")
        if review_id:
            by_review[str(review_id)] = payload
    return by_review


def area_sets(payloads: Mapping[str, dict[str, Any]]) -> dict[str, set[str]]:
    """Product areas per review, de-duplicated."""
    return {
        review_id: {
            str(area["product_area"])
            for area in (payload.get("areas") or [])
            if area.get("product_area")
        }
        for review_id, payload in payloads.items()
    }


def area_label_sets(payloads: Mapping[str, dict[str, Any]]) -> dict[str, set[str]]:
    """Area plus its issue or strength, in the gold set's token syntax.

    Shares the renderer with the gold path so the fine-grained comparison means
    the same thing whichever reference it is run against.
    """
    sets: dict[str, set[str]] = {}
    for review_id, payload in payloads.items():
        tokens = set()
        for area in payload.get("areas") or []:
            if not area.get("product_area"):
                continue
            tokens.add(
                format_label_tokens(
                    [(str(area["product_area"]), area.get("issue_type"), area.get("strength_type"))]
                )
            )
        sets[review_id] = tokens
    return sets


def field_values(payloads: Mapping[str, dict[str, Any]], name: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for review_id, payload in payloads.items():
        value = payload.get(name)
        if value is None:
            continue
        values[review_id] = value if isinstance(value, bool) else str(value).lower()
    return values


@dataclass
class ModelAgreement:
    """How two models compare on the reviews they both labelled."""

    left: str
    right: str
    overlap: int
    left_only: int
    right_only: int
    areas: MultiLabelScore | None = None
    area_labels: MultiLabelScore | None = None
    fields: dict[str, CategoricalScore] = field(default_factory=dict)
    mean_areas_left: float | None = None
    mean_areas_right: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            # Stated in the artefact, not just the prose, so a downstream reader
            # cannot pick these numbers up and label them accuracy.
            "is_accuracy": False,
            "note": (
                "Agreement between two models. Neither side is ground truth; "
                "where they differ at least one is wrong and this cannot say which."
            ),
            "overlap": self.overlap,
            "left_only": self.left_only,
            "right_only": self.right_only,
            "product_area": self.areas.as_dict() if self.areas else None,
            "area_and_type": self.area_labels.as_dict() if self.area_labels else None,
            "fields": {k: v.as_dict() for k, v in self.fields.items()},
            "mean_areas_left": self.mean_areas_left,
            "mean_areas_right": self.mean_areas_right,
        }


def compare_models(
    left_payloads: Mapping[str, dict[str, Any]],
    right_payloads: Mapping[str, dict[str, Any]],
    left_name: str,
    right_name: str,
) -> ModelAgreement:
    """Score two models against each other over the reviews both covered.

    ``left_only`` and ``right_only`` are reported rather than folded in. One
    model covering 4,568 reviews and another covering 95 is a fact about the
    runs, not a disagreement, and averaging it into the score would make the
    smaller run look like a failure to agree.
    """
    left_areas, right_areas = area_sets(left_payloads), area_sets(right_payloads)
    shared = set(left_areas) & set(right_areas)

    agreement = ModelAgreement(
        left=left_name,
        right=right_name,
        overlap=len(shared),
        left_only=len(set(left_areas) - shared),
        right_only=len(set(right_areas) - shared),
    )
    if not shared:
        return agreement

    agreement.areas = score_multilabel(
        {k: v for k, v in left_areas.items() if k in shared},
        {k: v for k, v in right_areas.items() if k in shared},
    )
    left_pairs, right_pairs = area_label_sets(left_payloads), area_label_sets(right_payloads)
    agreement.area_labels = score_multilabel(
        {k: v for k, v in left_pairs.items() if k in shared},
        {k: v for k, v in right_pairs.items() if k in shared},
    )

    for name in COMPARED_FIELDS:
        left_values = field_values(left_payloads, name)
        right_values = field_values(right_payloads, name)
        paired = shared & set(left_values) & set(right_values)
        if not paired:
            continue
        agreement.fields[name] = score_categorical(
            {k: left_values[k] for k in paired},
            {k: right_values[k] for k in paired},
        )

    agreement.mean_areas_left = sum(len(left_areas[k]) for k in shared) / len(shared)
    agreement.mean_areas_right = sum(len(right_areas[k]) for k in shared) / len(shared)
    return agreement
