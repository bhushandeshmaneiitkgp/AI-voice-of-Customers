"""
The hand-labelled reference set: how it is sampled, written, validated and scored.

Every caveat in every report this project produces ends at the same sentence --
*the labels are model output and no ground truth exists*. This module is the
apparatus for removing that sentence. It cannot remove it on its own, because
the one input it needs is a human reading reviews, and nothing here can
manufacture that.

What it can do is make sure that when the labels arrive they are worth having:

**The sample is drawn before anyone looks at it.** Proportional stratification
by platform and rating bucket, under a fixed seed, so the sample is
reproducible and a 78%-negative corpus does not leave the positive vocabulary
unmeasured.

**The template never shows the model's answer.** Anchoring is not a hypothetical
risk here -- shown a plausible label, an annotator agrees with it far more often
than they would have chosen it, and the resulting gold set drifts toward the
system it is supposed to audit. The measured accuracy then rises for a reason
that has nothing to do with the model being right. A test enforces this.

**The two strata are never pooled.** The random stratum is the only one that can
support a corpus-level claim. The disagreement stratum is enriched for hard
cases on purpose, so it finds failure modes faster and its accuracy is biased
downward by construction. Averaging them produces a number that describes no
population at all.

**Bad gold is rejected, not absorbed.** An annotator typo that invents an area
would otherwise be scored as a model error, and the model would be blamed for
the reference's mistake.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from voc.enrich import stratified_sample
from voc.taxonomy import Taxonomy

from evaluation.metrics import (
    CategoricalScore,
    MultiLabelScore,
    score_categorical,
    score_multilabel,
)

logger = logging.getLogger(__name__)

GOLD_SCHEMA_VERSION = "v1"

#: Separates one label from the next in the annotator's single label column.
LABEL_SEPARATOR = ";"

#: Separates a product area from the issue or strength within it.
AREA_LABEL_SEPARATOR = "/"

#: Marks the label after it as a strength rather than an issue. A single column
#: with a polarity marker beats two parallel columns the annotator has to keep
#: aligned by hand, which is the format that produces silent off-by-one gold.
STRENGTH_MARKER = "+"

#: Below this many labelled reviews a stratum is reported but not treated as a
#: result. At n=20 the Wilson interval around 90% spans roughly 70-97%, which
#: is compatible with almost any claim anyone would want to make.
MIN_GOLD_ITEMS = 30

#: Columns the annotator fills in. Everything else in the template is context.
ANNOTATION_COLUMNS: tuple[str, ...] = (
    "gold_labels",
    "gold_sentiment",
    "gold_severity",
    "gold_customer_intent",
    "gold_support_escalation",
    "annotator",
    "notes",
)

#: Columns the template shows for context. Conspicuously absent: anything the
#: model produced for this review.
CONTEXT_COLUMNS: tuple[str, ...] = (
    "review_id",
    "platform",
    "rating",
    "review_date",
    "review_text",
)

#: Single-valued fields scored against gold, paired with the enriched column
#: they are compared to.
CATEGORICAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("gold_sentiment", "sentiment"),
    ("gold_severity", "severity"),
    ("gold_customer_intent", "customer_intent"),
    ("gold_support_escalation", "support_escalation"),
)


# ---------------------------------------------------------------------------
# The label mini-language
# ---------------------------------------------------------------------------


@dataclass
class ParsedLabels:
    """One review's labels, in both the coarse and fine views, plus complaints."""

    areas: set[str] = field(default_factory=set)
    #: ``area/label`` pairs -- the same judgement scored at full resolution.
    area_labels: set[str] = field(default_factory=set)
    problems: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.problems


def parse_label_tokens(raw: Any, taxonomy: Taxonomy) -> ParsedLabels:
    """Turn ``area/issue; area/+strength`` into label sets, checking every part.

    Validated against the taxonomy rather than merely split, because a typo in
    the reference is indistinguishable from a model error once the two are in
    the same scoring function -- and the model is the one that gets blamed.
    """
    parsed = ParsedLabels()
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return parsed
    text = str(raw).strip()
    if not text:
        return parsed

    known_areas = set(taxonomy.area_ids) | {taxonomy.fallback_area.id}

    for token in text.split(LABEL_SEPARATOR):
        token = token.strip()
        if not token:
            continue
        area, separator, label = token.partition(AREA_LABEL_SEPARATOR)
        area = area.strip()
        label = label.strip()

        if area not in known_areas:
            parsed.problems.append(f"{area!r} is not a product area")
            continue
        parsed.areas.add(area)

        if not separator or not label:
            parsed.problems.append(
                f"{token!r} names an area but no issue or strength within it"
            )
            continue

        if area == taxonomy.fallback_area.id:
            # The fallback area exists precisely for reviews that fit nothing;
            # requiring a sub-label there would force an invented one.
            parsed.area_labels.add(f"{area}{AREA_LABEL_SEPARATOR}{label}")
            continue

        spec = taxonomy.area(area)
        if label.startswith(STRENGTH_MARKER):
            valid = {item.id for item in spec.strength_types}
            bare = label[len(STRENGTH_MARKER):]
            if bare not in valid:
                parsed.problems.append(f"{bare!r} is not a strength of {area}")
                continue
        else:
            valid = {item.id for item in spec.issue_types}
            if label not in valid:
                parsed.problems.append(f"{label!r} is not an issue type of {area}")
                continue

        parsed.area_labels.add(f"{area}{AREA_LABEL_SEPARATOR}{label}")

    return parsed


def format_label_tokens(pairs: Iterable[tuple[str, str | None, str | None]]) -> str:
    """Inverse of the parser: render (area, issue, strength) triples as tokens.

    Used to show worked examples in the annotation guide, generated from the
    live taxonomy so the guide cannot describe a syntax the parser rejects.
    """
    tokens = []
    for area, issue, strength in pairs:
        if strength:
            tokens.append(f"{area}{AREA_LABEL_SEPARATOR}{STRENGTH_MARKER}{strength}")
        elif issue:
            tokens.append(f"{area}{AREA_LABEL_SEPARATOR}{issue}")
        else:
            tokens.append(area)
    return f"{LABEL_SEPARATOR} ".join(tokens)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def disagreement_ids(
    left: Mapping[str, set[str]],
    right: Mapping[str, set[str]],
) -> list[str]:
    """Reviews where two models assigned different area sets.

    The benchmark found exact-set agreement at 61%, which makes the remaining
    39% the reviews most likely to be genuinely ambiguous -- and therefore the
    ones a limited annotation budget should reach first. Sorted so the selection
    is reproducible.
    """
    return sorted(key for key in set(left) & set(right) if left[key] != right[key])


def build_gold_sample(
    reviews: pd.DataFrame,
    n_random: int,
    hard_ids: Sequence[str] = (),
    n_hard: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Draw the reviews a human should label, tagged with which stratum they came from.

    The random stratum is stratified by platform and rating bucket, for the same
    reason the model benchmark was: a uniform draw from a 78%-negative corpus
    would contain barely twenty positive reviews and would leave the strength
    vocabulary essentially unmeasured.

    The hard stratum is drawn from ``hard_ids`` *after* removing anything the
    random stratum already took, so a review cannot appear twice and the two
    strata stay independently scoreable.
    """
    if reviews.empty:
        return pd.DataFrame()

    random_part = stratified_sample(reviews, n_random, seed=seed)
    random_part = random_part.assign(stratum="random")

    taken = set(random_part["review_id"])
    remaining = [rid for rid in hard_ids if rid not in taken]

    hard_part = pd.DataFrame()
    if n_hard and remaining:
        pool = reviews[reviews["review_id"].isin(remaining)]
        if not pool.empty:
            hard_part = pool.sample(
                n=min(n_hard, len(pool)), random_state=seed
            ).assign(stratum="disagreement")
            if len(hard_part) < n_hard:
                logger.warning(
                    "Disagreement stratum wanted %d reviews but only %d were available. "
                    "It is capped by the overlap between models, not by the corpus.",
                    n_hard,
                    len(hard_part),
                )

    combined = pd.concat([random_part, hard_part], ignore_index=True)
    # Shuffled so the annotator cannot infer the stratum from position and
    # start treating the second half as "the hard ones".
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


def build_template(sample: pd.DataFrame) -> pd.DataFrame:
    """The annotator-facing sheet: context, empty answer columns, nothing else.

    The stratum is stripped here and kept in the provenance file. Knowing that a
    review was selected because two models disagreed about it changes how it is
    read, and the whole value of the reference is that it was not read that way.
    """
    columns = [c for c in CONTEXT_COLUMNS if c in sample.columns]
    template = sample[columns].copy()
    for column in ANNOTATION_COLUMNS:
        template[column] = ""
    return template


def build_provenance(sample: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Which stratum each review came from, and under what seed."""
    columns = [c for c in ("review_id", "platform", "rating", "stratum") if c in sample.columns]
    provenance = sample[columns].copy()
    provenance["seed"] = seed
    provenance["schema_version"] = GOLD_SCHEMA_VERSION
    return provenance


def build_annotation_guide(taxonomy: Taxonomy, sample_size: int, strata: Mapping[str, int]) -> str:
    """Generate the labelling instructions from the live taxonomy.

    Generated rather than written by hand so the vocabulary in the guide cannot
    drift from the vocabulary the validator enforces -- a guide that documents a
    label the parser rejects wastes the most expensive resource in this phase,
    which is a human's afternoon.
    """
    lines: list[str] = [
        "# Annotation guide — gold reference set",
        "",
        f"Schema `{GOLD_SCHEMA_VERSION}` · {sample_size} reviews · "
        + " · ".join(f"{count} {name}" for name, count in sorted(strata.items())),
        "",
        "Fill in `gold_template.csv` and save it as `gold_labels.csv` in the same",
        "folder. Do not reorder or delete rows; scoring joins on `review_id`.",
        "",
        "## Why this exists",
        "",
        "Every number this pipeline reports about label quality is currently a",
        "measure of agreement, not correctness. Two models agreeing proves only that",
        "they share a bias. These labels are the reference that turns agreement into",
        "accuracy, so they need to be *your* reading of the review, not a check of",
        "somebody else's — which is why the model's answer is deliberately not shown.",
        "",
        "## Filling in `gold_labels`",
        "",
        f"One entry per thing the review is about, separated by `{LABEL_SEPARATOR}`.",
        f"Each entry is `area{AREA_LABEL_SEPARATOR}issue_type`, or",
        f"`area{AREA_LABEL_SEPARATOR}{STRENGTH_MARKER}strength_type` when the review",
        "is *praising* that area.",
        "",
        "A review can criticise one area and praise another; record both. A review",
        "that says nothing about any area gets an empty cell — that is a real answer,",
        "not a skipped row.",
        "",
    ]

    example_pairs: list[tuple[str, str | None, str | None]] = []
    for area in taxonomy.product_areas[:2]:
        if area.issue_types:
            example_pairs.append((area.id, area.issue_types[0].id, None))
    for area in taxonomy.product_areas:
        if area.strength_types:
            example_pairs.append((area.id, None, area.strength_types[0].id))
            break
    if example_pairs:
        lines += ["Example:", "", "```", format_label_tokens(example_pairs), "```", ""]

    lines += ["## Vocabulary", "", "| Area | Issue types | Strength types |", "|---|---|---|"]
    for area in taxonomy.product_areas:
        issues = ", ".join(f"`{item.id}`" for item in area.issue_types) or "—"
        strengths = ", ".join(f"`{item.id}`" for item in area.strength_types) or "—"
        lines.append(f"| `{area.id}` | {issues} | {strengths} |")
    lines += [
        f"| `{taxonomy.fallback_area.id}` | *(free text; for reviews that fit nothing above)* | — |",
        "",
        "## The other columns",
        "",
        "| Column | Allowed values |",
        "|---|---|",
    ]
    for column, _ in CATEGORICAL_FIELDS:
        attribute = column.removeprefix("gold_")
        if attribute == "support_escalation":
            allowed = "`true` / `false`"
        else:
            allowed = ", ".join(f"`{v}`" for v in taxonomy.attribute_values(attribute))
        lines.append(f"| `{column}` | {allowed} |")

    lines += [
        "| `annotator` | your initials — needed to measure inter-annotator agreement |",
        "| `notes` | anything the vocabulary could not express |",
        "",
        "## When you are unsure",
        "",
        "Write the note. A review you found genuinely ambiguous is a finding about",
        "the taxonomy, and a label guessed to fill the cell is worse than a blank one:",
        "it enters the reference silently and every future accuracy figure inherits it.",
        "",
        "Leave a row blank rather than guessing. Blank rows are counted and reported,",
        "not silently dropped.",
        "",
        "## Second annotator",
        "",
        "If a second person labels the same reviews, save their file as",
        "`gold_labels_<initials>.csv`. Two independent passes give a Cohen's kappa",
        "between annotators, which is the only thing that says whether the task is",
        "well-defined enough for the model's score to mean anything. Without it, an",
        "accuracy of 85% cannot be told apart from a task where humans agree 85% of",
        "the time.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading and validating
# ---------------------------------------------------------------------------


@dataclass
class GoldIssue:
    review_id: str
    kind: str
    detail: str


@dataclass
class GoldSet:
    """Loaded human labels, with everything wrong with them made explicit."""

    frame: pd.DataFrame
    issues: list[GoldIssue] = field(default_factory=list)
    unlabelled: list[str] = field(default_factory=list)

    @property
    def labelled(self) -> int:
        return len(self.frame)

    @property
    def usable(self) -> bool:
        return self.labelled > 0

    def by_stratum(self, stratum: str) -> pd.DataFrame:
        if "stratum" not in self.frame.columns:
            return self.frame
        return self.frame[self.frame["stratum"] == stratum]


def load_gold(
    path: Path,
    taxonomy: Taxonomy,
    provenance: pd.DataFrame | None = None,
) -> GoldSet:
    """Read the annotator's file, validate every label, and report the rest.

    Rows the annotator left blank are removed and listed, never treated as "no
    areas". A blank row means nobody has looked at it; an empty ``gold_labels``
    cell with the categorical fields filled in means somebody looked and found
    nothing. Conflating them would score the model against reviews that were
    never labelled.
    """
    if not path.exists():
        return GoldSet(pd.DataFrame())

    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    issues: list[GoldIssue] = []
    unlabelled: list[str] = []
    rows: list[dict[str, Any]] = []

    allowed = {
        column: set(taxonomy.attribute_values(column.removeprefix("gold_")))
        for column, _ in CATEGORICAL_FIELDS
        if column != "gold_support_escalation"
    }

    for record in raw.to_dict("records"):
        review_id = str(record.get("review_id", "")).strip()
        if not review_id:
            issues.append(GoldIssue("", "missing_review_id", "row has no review_id"))
            continue

        touched = any(
            str(record.get(column, "")).strip() for column in ANNOTATION_COLUMNS
        )
        if not touched:
            unlabelled.append(review_id)
            continue

        parsed = parse_label_tokens(record.get("gold_labels"), taxonomy)
        for problem in parsed.problems:
            issues.append(GoldIssue(review_id, "invalid_label", problem))

        row: dict[str, Any] = {
            "review_id": review_id,
            "gold_areas": parsed.areas,
            "gold_area_labels": parsed.area_labels,
            "annotator": str(record.get("annotator", "")).strip(),
            "notes": str(record.get("notes", "")).strip(),
        }

        for column, _ in CATEGORICAL_FIELDS:
            value = str(record.get(column, "")).strip().lower()
            if not value:
                row[column] = None
                continue
            if column == "gold_support_escalation":
                if value in ("true", "yes", "1"):
                    row[column] = True
                elif value in ("false", "no", "0"):
                    row[column] = False
                else:
                    issues.append(
                        GoldIssue(review_id, "invalid_attribute", f"{column}={value!r}")
                    )
                    row[column] = None
                continue
            if value not in allowed[column]:
                issues.append(
                    GoldIssue(review_id, "invalid_attribute", f"{column}={value!r}")
                )
                row[column] = None
            else:
                row[column] = value

        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty and provenance is not None and not provenance.empty:
        frame = frame.merge(
            provenance[["review_id", "stratum"]], on="review_id", how="left"
        )
        frame["stratum"] = frame["stratum"].fillna("unknown")

    duplicates = frame["review_id"].duplicated() if not frame.empty else pd.Series(dtype=bool)
    if bool(duplicates.any()):
        for review_id in frame.loc[duplicates, "review_id"]:
            issues.append(GoldIssue(review_id, "duplicate_row", "labelled more than once"))
        frame = frame[~duplicates].reset_index(drop=True)

    return GoldSet(frame=frame, issues=issues, unlabelled=unlabelled)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class StratumScore:
    """One stratum's results. Never merged with another stratum's."""

    stratum: str
    n: int
    sufficient: bool
    areas: MultiLabelScore | None = None
    area_labels: MultiLabelScore | None = None
    categorical: dict[str, CategoricalScore] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stratum": self.stratum,
            "n": self.n,
            "sufficient": self.sufficient,
            "minimum_for_reporting": MIN_GOLD_ITEMS,
            "product_area": self.areas.as_dict() if self.areas else None,
            "area_and_type": self.area_labels.as_dict() if self.area_labels else None,
            "categorical": {k: v.as_dict() for k, v in self.categorical.items()},
        }


def score_gold(
    gold: GoldSet,
    reviews: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, StratumScore]:
    """Score the model against the human labels, one stratum at a time.

    Returns an empty mapping when nothing has been labelled. That is the honest
    output for an unlabelled gold set, and it is what keeps the report from
    printing an accuracy section that describes zero reviews.
    """
    if not gold.usable or reviews.empty:
        return {}

    predicted_areas = _predicted_area_sets(labels)
    predicted_pairs = _predicted_area_label_sets(labels)
    enriched = reviews.set_index("review_id")

    strata = (
        sorted(gold.frame["stratum"].dropna().unique())
        if "stratum" in gold.frame.columns
        else ["all"]
    )

    results: dict[str, StratumScore] = {}
    for stratum in strata:
        subset = (
            gold.frame[gold.frame["stratum"] == stratum]
            if "stratum" in gold.frame.columns
            else gold.frame
        )
        ids = [rid for rid in subset["review_id"] if rid in enriched.index]
        if not ids:
            continue

        reference_areas = dict(zip(subset["review_id"], subset["gold_areas"]))
        reference_pairs = dict(zip(subset["review_id"], subset["gold_area_labels"]))
        # Every gold review the model did enrich must appear, even with no
        # labels, so "the model found nothing here" is scored rather than skipped.
        candidate_areas = {rid: predicted_areas.get(rid, set()) for rid in ids}
        candidate_pairs = {rid: predicted_pairs.get(rid, set()) for rid in ids}

        score = StratumScore(
            stratum=stratum,
            n=len(ids),
            sufficient=len(ids) >= MIN_GOLD_ITEMS,
            areas=score_multilabel(
                {k: v for k, v in reference_areas.items() if k in candidate_areas},
                candidate_areas,
            ),
            area_labels=score_multilabel(
                {k: v for k, v in reference_pairs.items() if k in candidate_pairs},
                candidate_pairs,
            ),
        )

        for column, enriched_column in CATEGORICAL_FIELDS:
            if column not in subset.columns or enriched_column not in reviews.columns:
                continue
            reference = {
                rid: value
                for rid, value in zip(subset["review_id"], subset[column])
                if value is not None and rid in enriched.index
            }
            if not reference:
                continue
            candidate = {
                rid: _normalise_value(enriched.loc[rid, enriched_column])
                for rid in reference
            }
            reference = {k: _normalise_value(v) for k, v in reference.items()}
            score.categorical[enriched_column] = score_categorical(reference, candidate)

        results[stratum] = score

    return results


def _normalise_value(value: Any) -> Any:
    """Fold booleans and NaN into forms that compare cleanly across sources."""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value).strip().lower()


def _predicted_area_sets(labels: pd.DataFrame) -> dict[str, set[str]]:
    if labels.empty:
        return {}
    return (
        labels.groupby("review_id")["product_area"].apply(lambda s: set(s)).to_dict()
    )


def _predicted_area_label_sets(labels: pd.DataFrame) -> dict[str, set[str]]:
    """Area plus its issue or strength, in the annotator's own token syntax.

    Rendered through the same format the parser accepts so that the fine-grained
    comparison is genuinely like-for-like rather than two spellings of the same
    judgement being scored as a disagreement.
    """
    if labels.empty:
        return {}

    sets: dict[str, set[str]] = {}
    for record in labels.to_dict("records"):
        review_id = str(record["review_id"])
        area = record.get("product_area")
        if not area or (isinstance(area, float) and pd.isna(area)):
            continue
        issue = record.get("issue_type")
        strength = record.get("strength_type")
        issue = None if issue is None or (isinstance(issue, float) and pd.isna(issue)) else issue
        strength = (
            None if strength is None or (isinstance(strength, float) and pd.isna(strength))
            else strength
        )
        token = format_label_tokens([(str(area), issue, strength)])
        sets.setdefault(review_id, set()).add(token)
    return sets
