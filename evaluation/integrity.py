"""
Dataset integrity: auditing the shipped artefacts rather than the run that made them.

The fault study asks whether the validators *can* catch an error. This asks a
different and blunter question: given that they caught things, what happened to
those labels? And the answer, which nothing else in the project states plainly,
is that **the validators report, they do not gate.**

``parse_and_validate`` records an issue and then appends the enrichment anyway.
That is a defensible design -- dropping a review because one of its three labels
is misfiled loses two good labels and a whole row of coverage -- but it means
"110 unknown issue types were found" and "110 unknown issue types are in the
dataset" are the same sentence, and every reader so far has been free to assume
the first meant they had been removed.

So this module counts what actually landed. It reads the shipped parquet files,
not the run report, because the run report describes requests and the question
here is about rows.

It also runs the one check the taxonomy asks for and nobody was performing: the
fallback area carries a ``monitoring_rule`` saying that if the model files more
than a stated share of reviews there, the taxonomy has a real gap. That rule was
written in Phase 2 and has never been evaluated against a full run.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from voc.taxonomy import Taxonomy

from evaluation.metrics import Rate

logger = logging.getLogger(__name__)

#: Fallback for parsing the threshold out of the taxonomy's monitoring rule,
#: used only if the rule states no percentage of its own.
DEFAULT_FALLBACK_THRESHOLD = 0.10


@dataclass(frozen=True)
class IntegrityFinding:
    """One class of invalid label that reached the shipped dataset."""

    key: str
    description: str
    affected_labels: int
    total_labels: int
    consequence: str

    @property
    def share(self) -> Rate:
        return Rate(self.affected_labels, self.total_labels)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "affected_labels": self.affected_labels,
            "total_labels": self.total_labels,
            "share": self.share.as_dict(),
            "consequence": self.consequence,
        }


def fallback_threshold(taxonomy: Taxonomy) -> float:
    """The share at which the taxonomy declares itself to have a gap.

    Read out of the taxonomy's own ``monitoring_rule`` rather than restated
    here, so the check enforces the rule as written and cannot drift from it.
    """
    rule = getattr(taxonomy.fallback_area, "monitoring_rule", "") or ""
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", rule)
    return float(match.group(1)) / 100 if match else DEFAULT_FALLBACK_THRESHOLD


def audit(
    reviews: pd.DataFrame,
    labels: pd.DataFrame,
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    """Count invalid labels that survived validation and reached the dataset."""
    if labels.empty:
        return {}

    total = len(labels)
    known_areas = set(taxonomy.area_ids)
    fallback = taxonomy.fallback_area.id
    valid_issues = {(a.id, i.id) for a in taxonomy.product_areas for i in a.issue_types}
    valid_strengths = {(a.id, s.id) for a in taxonomy.product_areas for s in a.strength_types}

    def pair_invalid(column: str, allowed: set[tuple[str, str]]) -> pd.Series:
        """Rows whose (area, label) pairing is not in the taxonomy.

        The fallback area is excluded: it has no sub-label vocabulary to check
        against, so anything filed there is unvalidatable rather than invalid.
        """
        return pd.Series(
            [
                bool(label) and not (isinstance(label, float) and pd.isna(label))
                and area != fallback and (area, label) not in allowed
                for area, label in zip(labels["product_area"], labels[column])
            ],
            index=labels.index,
        )

    # Masks rather than counts, because one label can trip two findings at once
    # -- an invented area with no polarity is both -- and summing the counts to
    # get "how many are clean" then subtracts it twice.
    masks = {
        "unknown_area": ~labels["product_area"].isin(known_areas | {fallback}),
        "invalid_issue_type": pair_invalid("issue_type", valid_issues),
        "invalid_strength_type": pair_invalid("strength_type", valid_strengths),
        "no_polarity": labels["issue_type"].isna() & labels["strength_type"].isna(),
    }
    descriptions = {
        "unknown_area": (
            "Product area not in the taxonomy",
            "the area would appear as its own row in every downstream table",
        ),
        "invalid_issue_type": (
            "Issue type absent from the taxonomy, or filed under the wrong area",
            "counts toward a pain point that the taxonomy does not define",
        ),
        "invalid_strength_type": (
            "Strength type absent from the taxonomy, or filed under the wrong area",
            "inflates the praise side of an area it does not belong to",
        ),
        "no_polarity": (
            "Area named with neither an issue nor a strength",
            "carries no product meaning; excluded from every issue-side analysis",
        ),
    }

    findings = [
        IntegrityFinding(key, descriptions[key][0], int(mask.sum()), total, descriptions[key][1])
        for key, mask in masks.items()
    ]

    any_invalid = pd.Series(False, index=labels.index)
    for mask in masks.values():
        any_invalid |= mask.fillna(False)

    fallback_labels = int((labels["product_area"] == fallback).sum())
    fallback_reviews = (
        labels.loc[labels["product_area"] == fallback, "review_id"].nunique()
        if fallback_labels else 0
    )
    threshold = fallback_threshold(taxonomy)
    review_total = len(reviews) if not reviews.empty else labels["review_id"].nunique()
    fallback_share = fallback_reviews / review_total if review_total else 0.0

    grounding = {}
    if "grounding_rate" in reviews.columns and not reviews.empty:
        imperfect = int((reviews["grounding_rate"] < 1.0).sum())
        grounding = {
            "reviews_below_full": imperfect,
            "reviews": len(reviews),
            "share": Rate(imperfect, len(reviews)).as_dict(),
            "mean_rate": float(reviews["grounding_rate"].mean()),
        }

    return {
        "total_labels": total,
        "findings": [f.as_dict() for f in findings if f.affected_labels],
        "clean_labels": int((~any_invalid).sum()),
        "fallback_area": {
            "id": fallback,
            "labels": fallback_labels,
            "reviews": fallback_reviews,
            "share_of_reviews": fallback_share,
            "threshold": threshold,
            "breached": fallback_share > threshold,
            "rule": (getattr(taxonomy.fallback_area, "monitoring_rule", "") or "").strip(),
        },
        "grounding": grounding,
    }
