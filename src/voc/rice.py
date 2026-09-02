"""
Layer 11 -- RICE prioritisation.

RICE = Reach x Impact x Confidence / Effort.

Three of those four this corpus can speak to. **Effort it cannot**, and that is
the load-bearing fact in this module.

Reach, Impact and Confidence are all properties of the problem, and the problem
is what customer reviews describe. Effort is a property of *the solution and the
codebase that would carry it* -- how many services it touches, what the
migration looks like, who is free next sprint. No amount of review text contains
that, and a model asked to guess it will produce a confident number anyway.

So effort is a required human input. Without it this module reports **RIC**, an
explicitly partial score, and refuses to present a final ranking. That is
deliberately inconvenient: a RICE table whose denominator was invented ranks
work by fiction while looking quantitative, and it is *more* dangerous than no
table because the arithmetic lends it authority.

Confidence is the interesting term here, because for once it can be grounded in
something. Standard practice is to pick 100%/80%/50% by feel. This derives it
from measurable evidence quality -- how well the labels were grounded, how large
the sample was, whether a mechanism was proposed, whether a competitive
difference survived correction -- so the number answers "how much do we know"
rather than "how sure does someone feel today".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: Standard RICE impact scale. Keeping the canonical values means a score here
#: is comparable with one from any other RICE table, rather than being a
#: private scale that happens to share a name.
IMPACT_SCALE: dict[str, float] = {
    "minimal": 0.25,
    "low": 0.5,
    "medium": 1.0,
    "high": 2.0,
    "massive": 3.0,
}

#: Severity is ordinal; these are the cut points from mean severity (1-4) to
#: the impact scale. A pain point averaging "high" maps to high impact.
SEVERITY_TO_IMPACT: tuple[tuple[float, str], ...] = (
    (3.5, "massive"),
    (3.0, "high"),
    (2.5, "medium"),
    (2.0, "low"),
    (0.0, "minimal"),
)

#: A stated intent to leave is worth more than its frequency suggests: churn is
#: the one outcome in this dataset with direct revenue meaning. This lifts
#: impact by one scale step when churn intent clears the bar.
CHURN_ESCALATION_THRESHOLD = 0.02

#: Confidence components and their weights. They sum to 1.0, and each is a
#: measurement rather than a judgement -- which is the whole point of deriving
#: confidence instead of picking it.
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "grounding": 0.30,      # were the labels' quotes actually in the reviews
    "sample": 0.25,         # is the volume enough to be a pattern
    "label_confidence": 0.20,  # what the enrichment model reported
    "mechanism": 0.15,      # did Phase 6 find a grounded root cause
    "competitive": 0.10,    # did a platform difference survive correction
}

#: Volume at which the sample term saturates. Beyond this, more reviews stop
#: buying meaningful certainty about whether the pattern is real.
SAMPLE_SATURATION = 300


@dataclass
class RiceInputs:
    """Everything that feeds one score, kept separable so each is auditable."""

    product_area: str
    issue_type: str
    reach_per_month: float
    impact: float
    impact_label: str
    confidence: float
    confidence_parts: dict[str, float]
    #: None until a human supplies it. The module will not invent it.
    effort_person_weeks: float | None = None

    @property
    def has_effort(self) -> bool:
        return self.effort_person_weeks is not None and self.effort_person_weeks > 0

    @property
    def ric(self) -> float:
        """Reach x Impact x Confidence -- the part the data supports."""
        return self.reach_per_month * self.impact * self.confidence

    @property
    def rice(self) -> float | None:
        """The full score, or None when effort is unknown."""
        if not self.has_effort:
            return None
        return self.ric / float(self.effort_person_weeks)


def reach_per_month(volume: int, months_observed: int) -> float:
    """Reviews per month raising this issue.

    **This is reviews, not customers.** People who write app-store reviews are
    a small and self-selecting slice of users, skewed toward the annoyed. The
    figure is a consistent relative signal across pain points measured the same
    way, which is what RICE actually needs -- but multiplying it by a user base
    to get "customers affected" would be inventing a number.
    """
    if months_observed <= 0:
        raise ValueError("months_observed must be positive to express a rate")
    return volume / months_observed


def impact_from_signals(
    mean_severity: float, churn_rate: float, escalation_rate: float
) -> tuple[float, str]:
    """Map severity, churn and escalation onto the RICE impact scale.

    Severity sets the base. Churn intent promotes one step, because a customer
    saying they will leave is a different class of outcome from one who is
    annoyed. Escalation promotes only from the bottom of the scale: a support
    contact is a real cost, but a routine one.
    """
    label = "minimal"
    for threshold, candidate in SEVERITY_TO_IMPACT:
        if mean_severity >= threshold:
            label = candidate
            break

    order = list(IMPACT_SCALE)
    position = order.index(label)

    if churn_rate >= CHURN_ESCALATION_THRESHOLD:
        position = min(position + 1, len(order) - 1)
    elif escalation_rate >= 0.5 and position < order.index("medium"):
        position = min(position + 1, len(order) - 1)

    label = order[position]
    return IMPACT_SCALE[label], label


def confidence_from_evidence(
    grounding_rate: float,
    volume: int,
    mean_label_confidence: float,
    has_mechanism: bool,
    competitive_significant: bool,
) -> tuple[float, dict[str, float]]:
    """Derive RICE confidence from measurable evidence quality.

    Returns the score and its components, because a confidence number nobody
    can decompose is a feeling with a decimal point.
    """
    parts = {
        "grounding": float(max(0.0, min(1.0, grounding_rate))),
        "sample": float(min(1.0, volume / SAMPLE_SATURATION)),
        "label_confidence": float(max(0.0, min(1.0, mean_label_confidence))),
        "mechanism": 1.0 if has_mechanism else 0.0,
        "competitive": 1.0 if competitive_significant else 0.0,
    }
    score = sum(CONFIDENCE_WEIGHTS[key] * value for key, value in parts.items())
    return score, parts


def build_rice_inputs(
    pain_points: pd.DataFrame,
    reviews: pd.DataFrame,
    months_observed: int,
    root_causes: pd.DataFrame | None = None,
    area_rates: pd.DataFrame | None = None,
    effort: dict[tuple[str, str], float] | None = None,
) -> list[RiceInputs]:
    """Assemble RICE inputs for each pain point.

    ``effort`` maps ``(product_area, issue_type)`` to person-weeks. Anything
    absent stays None and scores as RIC only.
    """
    effort = effort or {}

    with_mechanism: set[tuple[str, str]] = set()
    if root_causes is not None and not root_causes.empty:
        with_mechanism = set(zip(root_causes["product_area"], root_causes["issue_type"]))

    significant_areas: set[str] = set()
    if area_rates is not None and not area_rates.empty and "significant" in area_rates:
        significant_areas = set(
            area_rates[area_rates["significant"]]["product_area"].unique()
        )

    grounding_by_review = (
        reviews.set_index("review_id")["grounding_rate"].to_dict()
        if "grounding_rate" in reviews.columns
        else {}
    )

    rows: list[RiceInputs] = []
    for row in pain_points.itertuples():
        key = (row.product_area, row.issue_type)

        cited = str(getattr(row, "evidence_review_ids", "") or "").split()
        grounding = [grounding_by_review.get(r) for r in cited]
        grounding = [g for g in grounding if g is not None]
        # Fall back to the corpus mean rather than assuming perfection: an
        # optimistic default here would inflate confidence for exactly the
        # pain points whose evidence could not be checked.
        mean_grounding = (
            sum(grounding) / len(grounding) if grounding
            else float(reviews["grounding_rate"].mean()) if "grounding_rate" in reviews
            else 0.0
        )

        impact, impact_label = impact_from_signals(
            float(row.mean_severity), float(row.churn_rate), float(row.escalation_rate)
        )
        confidence, parts = confidence_from_evidence(
            grounding_rate=mean_grounding,
            volume=int(row.volume),
            mean_label_confidence=float(row.mean_confidence),
            has_mechanism=key in with_mechanism,
            competitive_significant=row.product_area in significant_areas,
        )

        rows.append(
            RiceInputs(
                product_area=row.product_area,
                issue_type=row.issue_type,
                reach_per_month=reach_per_month(int(row.volume), months_observed),
                impact=impact,
                impact_label=impact_label,
                confidence=confidence,
                confidence_parts=parts,
                effort_person_weeks=effort.get(key),
            )
        )
    return rows


def to_frame(inputs: list[RiceInputs]) -> pd.DataFrame:
    """Rank the inputs. Sorts by RICE where effort is known, RIC where not.

    The two groups are never interleaved: a scored item and an unscored one are
    not comparable, and sorting them into one list would imply they are.
    """
    if not inputs:
        return pd.DataFrame()

    rows = [
        {
            "product_area": item.product_area,
            "issue_type": item.issue_type,
            "reach_per_month": round(item.reach_per_month, 2),
            "impact": item.impact,
            "impact_label": item.impact_label,
            "confidence": round(item.confidence, 4),
            "effort_person_weeks": item.effort_person_weeks,
            "ric": round(item.ric, 2),
            "rice": round(item.rice, 2) if item.rice is not None else None,
            "scored": item.has_effort,
            **{f"confidence_{k}": round(v, 3) for k, v in item.confidence_parts.items()},
        }
        for item in inputs
    ]
    frame = pd.DataFrame(rows)

    scored = frame[frame["scored"]].sort_values("rice", ascending=False)
    unscored = frame[~frame["scored"]].sort_values("ric", ascending=False)
    ordered = pd.concat([scored, unscored], ignore_index=True)
    ordered.insert(0, "rank", range(1, len(ordered) + 1))
    return ordered


def load_effort(path) -> dict[tuple[str, str], float]:
    """Read a CSV of person-week estimates.

    Columns: product_area, issue_type, effort_person_weeks. Rows with a blank
    or non-positive estimate are skipped rather than coerced to zero -- a zero
    denominator would send an item to the top of the ranking.
    """
    frame = pd.read_csv(path)
    required = {"product_area", "issue_type", "effort_person_weeks"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"effort file is missing column(s): {sorted(missing)}")

    estimates: dict[tuple[str, str], float] = {}
    for row in frame.itertuples():
        value = pd.to_numeric(row.effort_person_weeks, errors="coerce")
        if pd.isna(value) or value <= 0:
            continue
        estimates[(row.product_area, row.issue_type)] = float(value)

    logger.info("Loaded %d effort estimate(s) from %s", len(estimates), path)
    return estimates


def write_effort_template(inputs: list[RiceInputs], path) -> None:
    """Write a CSV for a human to fill in.

    The template carries reach, impact and confidence alongside each row, so
    whoever estimates effort can see what they are trading against rather than
    working from a bare list of identifiers.
    """
    frame = pd.DataFrame(
        [
            {
                "product_area": item.product_area,
                "issue_type": item.issue_type,
                "reach_per_month": round(item.reach_per_month, 2),
                "impact_label": item.impact_label,
                "confidence": round(item.confidence, 3),
                "ric": round(item.ric, 2),
                "effort_person_weeks": "",
            }
            for item in inputs
        ]
    )
    frame.to_csv(path, index=False)
