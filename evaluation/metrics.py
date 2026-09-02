"""
Scoring primitives, deliberately blind to where the reference came from.

Nothing in this module knows whether it is being handed human gold labels or a
second model's output. That is the point: accuracy and inter-model agreement
are the *same arithmetic*, and the only thing that makes one an accuracy claim
is the provenance of the reference. Writing two near-identical scorers would
invite the two to drift, and would hide the fact that the difference is
epistemic rather than mathematical.

So the functions here take ``reference`` and ``candidate``, and the caller is
responsible for saying which is which in the report.

Three rules the metrics follow throughout:

1. **Every rate carries an interval.** ``n`` is small in this phase -- 100
   hand-labelled reviews at best -- and a bare "91%" invites a precision the
   sample cannot support.
2. **Only items present in both sides are scored.** A review the candidate
   failed to label is a coverage failure, not an agreement; counting it either
   way corrupts the number. It is reported separately.
3. **A metric that is undefined returns ``None``, never a default.** Cohen's
   kappa on a single-valued reference and precision with no predictions are
   both undefined, and 0.0 would read as "measured, and bad".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from voc.trends import wilson_interval


@dataclass(frozen=True)
class Rate:
    """A proportion, its sample size, and the interval that qualifies it.

    Bundled rather than returned as three loose numbers because the interval is
    the part that gets dropped in transit, and a rate that arrives without one
    is exactly how a 100-review sample turns into a confident claim.
    """

    hits: int
    total: int

    @property
    def value(self) -> float | None:
        return self.hits / self.total if self.total else None

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson_interval(self.hits, self.total) if self.total else None

    def as_dict(self) -> dict[str, Any]:
        interval = self.interval
        return {
            "value": self.value,
            "hits": self.hits,
            "total": self.total,
            "ci_low": interval[0] if interval else None,
            "ci_high": interval[1] if interval else None,
        }

    def __str__(self) -> str:
        if self.total == 0:
            return "n/a (no comparable items)"
        low, high = self.interval  # type: ignore[misc]
        return f"{self.value * 100:.1f}% ({low * 100:.1f}-{high * 100:.1f}), n={self.total}"


# ---------------------------------------------------------------------------
# Multi-label scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiLabelScore:
    """How well a candidate's label *sets* match a reference's.

    Product areas are multi-label -- a review can be about delivery and support
    at once -- so a single accuracy number does not exist. Three views, because
    each hides something the others show:

    * micro P/R/F1 counts every (review, label) pair, so a review with five
      labels weighs five times one with a single label. This is the figure that
      answers "of the labels it applied, how many belong".
    * exact set match is the strictest reading and the one a reader intuitively
      assumes a table is showing. It is always the lowest of the three.
    * mean Jaccard sits between them and is the least misreadable: partial
      credit for partial overlap, per review rather than per label.
    """

    true_positives: int
    false_positives: int
    false_negatives: int
    exact_match: Rate
    mean_jaccard: float | None
    compared: int
    reference_only: int
    candidate_only: int

    @property
    def precision(self) -> float | None:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else None

    @property
    def recall(self) -> float | None:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else None

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    def as_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "exact_match": self.exact_match.as_dict(),
            "mean_jaccard": self.mean_jaccard,
            "compared": self.compared,
            "reference_only": self.reference_only,
            "candidate_only": self.candidate_only,
        }


def jaccard(a: set[str], b: set[str]) -> float:
    """Overlap over union. Two empty sets agree completely, and score 1.0.

    The empty-empty case is a real one -- a review with no product area at all
    -- and calling it 0.0 would penalise the candidate for correctly finding
    nothing.
    """
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def score_multilabel(
    reference: Mapping[str, set[str]],
    candidate: Mapping[str, set[str]],
) -> MultiLabelScore:
    """Compare two mappings of item id -> label set.

    Scored strictly over the intersection of keys. An item the candidate never
    labelled is a coverage failure and is counted in ``reference_only``, not
    folded in as a miss: pooling the two would let a model improve its recall by
    crashing on the reviews it would have got wrong.
    """
    shared = sorted(set(reference) & set(candidate))

    tp = fp = fn = exact = 0
    jaccards: list[float] = []

    for key in shared:
        gold, pred = set(reference[key]), set(candidate[key])
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        exact += int(gold == pred)
        jaccards.append(jaccard(gold, pred))

    return MultiLabelScore(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        exact_match=Rate(exact, len(shared)),
        mean_jaccard=sum(jaccards) / len(jaccards) if jaccards else None,
        compared=len(shared),
        reference_only=len(set(reference) - set(candidate)),
        candidate_only=len(set(candidate) - set(reference)),
    )


def per_label_scores(
    reference: Mapping[str, set[str]],
    candidate: Mapping[str, set[str]],
) -> pd.DataFrame:
    """Break the micro score out by label, sorted by reference support.

    The aggregate hides the shape of the errors. A model can post a respectable
    micro-F1 while being unusable on the third-largest area, and that is the
    failure a product team needs to know about, because that area is somebody's
    roadmap item.
    """
    shared = sorted(set(reference) & set(candidate))
    labels = sorted({label for key in shared for label in reference[key] | candidate[key]})

    rows = []
    for label in labels:
        tp = sum(1 for k in shared if label in reference[k] and label in candidate[k])
        fp = sum(1 for k in shared if label not in reference[k] and label in candidate[k])
        fn = sum(1 for k in shared if label in reference[k] and label not in candidate[k])
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        rows.append(
            {
                "label": label,
                "reference_support": tp + fn,
                "candidate_support": tp + fp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("reference_support", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Single-label scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoricalScore:
    """Accuracy for a one-value-per-item field, with the two things that qualify it.

    Raw accuracy on this corpus is close to meaningless on its own: 78% of
    reviews are negative, so a classifier that answers "negative" every time
    scores 78%. Both companions are therefore mandatory rather than optional.

    * ``majority_baseline`` is what guessing the commonest reference value
      would score. Accuracy below it is worse than a constant.
    * ``kappa`` is agreement corrected for that chance agreement. It is the
      number to quote when the class balance is this skewed.
    """

    accuracy: Rate
    kappa: float | None
    majority_baseline: float | None
    majority_value: str | None
    compared: int

    @property
    def lift_over_baseline(self) -> float | None:
        accuracy, baseline = self.accuracy.value, self.majority_baseline
        if accuracy is None or baseline is None:
            return None
        return accuracy - baseline

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy.as_dict(),
            "kappa": self.kappa,
            "majority_baseline": self.majority_baseline,
            "majority_value": self.majority_value,
            "lift_over_baseline": self.lift_over_baseline,
            "compared": self.compared,
        }


def cohen_kappa(reference: Sequence[Any], candidate: Sequence[Any]) -> float | None:
    """Agreement above what chance would produce, given each side's own biases.

    Returns ``None`` when chance agreement is total -- a reference and candidate
    that both use exactly one value agree perfectly for no informative reason,
    and kappa's denominator is zero there. Reporting 0.0 or 1.0 in that case
    would both be assertions the data does not support.
    """
    if len(reference) != len(candidate):
        raise ValueError(
            f"reference has {len(reference)} items, candidate has {len(candidate)} -- "
            "kappa is only defined over paired judgements"
        )
    n = len(reference)
    if n == 0:
        return None

    observed = sum(1 for a, b in zip(reference, candidate) if a == b) / n

    reference_counts = Counter(reference)
    candidate_counts = Counter(candidate)
    expected = sum(
        (reference_counts[value] / n) * (candidate_counts.get(value, 0) / n)
        for value in reference_counts
    )
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def score_categorical(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> CategoricalScore:
    """Accuracy, kappa and the majority baseline for one single-valued field."""
    shared = sorted(set(reference) & set(candidate))
    gold = [reference[key] for key in shared]
    pred = [candidate[key] for key in shared]

    hits = sum(1 for a, b in zip(gold, pred) if a == b)
    counts = Counter(gold)
    majority_value, majority_count = counts.most_common(1)[0] if counts else (None, 0)

    return CategoricalScore(
        accuracy=Rate(hits, len(shared)),
        kappa=cohen_kappa(gold, pred),
        majority_baseline=majority_count / len(shared) if shared else None,
        majority_value=str(majority_value) if majority_value is not None else None,
        compared=len(shared),
    )


def confusion_matrix(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> pd.DataFrame:
    """Reference values down the rows, candidate values across the columns.

    Which way the errors go matters more than how many there are. A sentiment
    classifier that calls neutral reviews negative and one that calls negative
    reviews neutral post the same accuracy and have opposite consequences for
    every downstream rate in the product.
    """
    shared = sorted(set(reference) & set(candidate))
    if not shared:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "reference": [reference[key] for key in shared],
            "candidate": [candidate[key] for key in shared],
        }
    )
    return pd.crosstab(frame["reference"], frame["candidate"])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def format_optional(value: float | None, digits: int = 3) -> str:
    """Render a metric that may be undefined, without inventing a zero."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_percent(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def label_sets(
    frame: pd.DataFrame,
    key_column: str,
    label_column: str,
    ids: Iterable[str] | None = None,
) -> dict[str, set[str]]:
    """Collapse a long label table into item id -> label set.

    ``ids`` forces items with no labels at all to appear as empty sets rather
    than vanishing. A review the model examined and assigned nothing is a
    prediction of "no areas", which is answerable; a review that is simply
    absent is not, and the two must not look alike.
    """
    sets: dict[str, set[str]] = {str(i): set() for i in ids} if ids is not None else {}
    for key, label in zip(frame[key_column], frame[label_column]):
        if label is None or (isinstance(label, float) and pd.isna(label)):
            continue
        sets.setdefault(str(key), set()).add(str(label))
    return sets
