"""
Layer 5 -- pain-point discovery and scoring.

Enrichment produces 10,790 labels across 4,568 reviews. A PM cannot act on
that, and a raw frequency count answers the wrong question: the most *common*
complaint is not automatically the most *costly* one. A high-volume annoyance
and a low-volume reason people delete the app deserve different attention.

A pain point here is a ``(product_area, issue_type)`` pair -- the unit the
taxonomy already defines -- scored on five signals that are cheap to explain
and derived only from fields the pipeline already validated:

    volume       how many distinct reviews raise it
    severity     how bad the reviewer said it was
    escalation   how often it drove a support contact
    churn        how often it came with a stated intent to leave
    negativity   how consistently it appears in negative reviews

The weights below are a product judgement, not a discovered constant, so they
live in one place and are printed in the report. Anyone who disagrees can
change one number and re-run rather than argue with a black box.

Trend is deliberately *never scored*, and on this corpus it is not even
reported: outside the comparable window review volume tracks when the data was
scraped rather than when problems happened, and the window itself is only three
months long. ``add_trend`` refuses rather than publishing a ratio that would be
a collection artefact wearing a trend's clothes.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: Severity is ordinal, not numeric, so the mapping is explicit. The gaps are
#: even because nothing in the data justifies claiming that critical is, say,
#: 2.5x worse than high rather than 1.33x.
SEVERITY_RANK: dict[str, int] = {"low": 1, "medium": 2, "high": 3, "critical": 4}

#: Score weights. They sum to 1.0 so the composite stays in [0, 1] and a
#: reader can see at a glance what the score mostly is: volume and severity,
#: with escalation and churn as multiplying evidence of real cost.
WEIGHTS: dict[str, float] = {
    "volume": 0.35,
    "severity": 0.25,
    "escalation": 0.15,
    "churn": 0.15,
    "negativity": 0.10,
}

#: Intent value that signals a customer stating they will stop using the app.
CHURN_INTENT = "churn_warning"


def _normalise(series: pd.Series) -> pd.Series:
    """Min-max to [0, 1]. A flat column becomes 0, not NaN.

    Without the guard, a corpus where every pain point has identical severity
    divides by zero and takes the whole composite to NaN -- turning a boring
    edge case into a blank report.
    """
    span = series.max() - series.min()
    if span <= 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / span


def build_pain_points(
    labels: pd.DataFrame,
    reviews: pd.DataFrame,
    min_volume: int = 15,
) -> pd.DataFrame:
    """Aggregate labels into scored, ranked pain points.

    ``labels`` is the per-area table and ``reviews`` the per-review table, both
    from Phase 3. They are joined rather than merged upstream because one
    review contributes to several pain points and its severity must count once
    per pain point, not once overall.
    """
    if labels.empty:
        return pd.DataFrame()

    # Only issues are pain points. A `strength` label is a compliment, and
    # scoring it as a problem would put "fast delivery" on the fix-it list.
    issues = labels[labels["polarity"] == "issue"].copy()
    if issues.empty:
        logger.warning("No issue-polarity labels; nothing to score.")
        return pd.DataFrame()

    review_facts = reviews.set_index("review_id")
    issues["severity_rank"] = (
        issues["review_id"].map(review_facts["severity"]).map(SEVERITY_RANK)
    )
    issues["escalated"] = issues["review_id"].map(review_facts["support_escalation"])
    issues["is_churn"] = (
        issues["review_id"].map(review_facts["customer_intent"]) == CHURN_INTENT
    )
    issues["is_negative"] = (
        issues["review_id"].map(review_facts["sentiment"]) == "negative"
    )

    grouped = issues.groupby(["product_area", "issue_type"], observed=True)
    frame = pd.DataFrame(
        {
            # nunique, not size: a model that returned the same area twice for
            # one review must not double its apparent volume.
            "volume": grouped["review_id"].nunique(),
            "mean_severity": grouped["severity_rank"].mean(),
            "escalation_rate": grouped["escalated"].mean(),
            "churn_rate": grouped["is_churn"].mean(),
            "negative_share": grouped["is_negative"].mean(),
            "mean_confidence": grouped["confidence"].mean(),
            "platforms": grouped["platform"].nunique(),
            "top_platform": grouped["platform"].agg(lambda s: s.mode().iat[0]),
        }
    ).reset_index()

    below = frame[frame["volume"] < min_volume]
    if not below.empty:
        logger.info(
            "Dropping %d pain point(s) below the volume floor of %d",
            len(below), min_volume,
        )
    frame = frame[frame["volume"] >= min_volume].copy()
    if frame.empty:
        return frame

    # Written out term by term so each contribution is readable and reviewable.
    # Rates are already in [0, 1] and are used directly; counts and means are
    # min-maxed, because "how big is 1,922 mentions" only means something
    # relative to the other pain points in the same corpus.
    frame["score"] = (
        WEIGHTS["volume"] * _normalise(frame["volume"])
        + WEIGHTS["severity"] * _normalise(frame["mean_severity"])
        + WEIGHTS["escalation"] * frame["escalation_rate"]
        + WEIGHTS["churn"] * _normalise(frame["churn_rate"])
        + WEIGHTS["negativity"] * frame["negative_share"]
    )

    for column in ("mean_severity", "escalation_rate", "churn_rate",
                   "negative_share", "mean_confidence", "score"):
        frame[column] = frame[column].round(4)

    frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def add_trend(
    pain_points: pd.DataFrame,
    labels: pd.DataFrame,
    recent_months: int = 3,
    min_prior_months: int = 3,
) -> pd.DataFrame:
    """Attach a recent-vs-prior volume ratio, **or refuse to**.

    Refusing is the normal outcome on this corpus, and that is the finding.

    Review volume here is dominated by when the data was collected, not by when
    problems happened: 50 months are present, but 42 of the earliest 47 hold
    fewer than ten reviews each, while the final three hold 3,475. Dividing a
    recent per-month rate by that prior rate produced ratios up to 197x on the
    first attempt -- pure collection artefact, published as if it were customer
    behaviour.

    Phase 1 already drew this line: ``in_comparable_window`` marks the period
    from 2024-10 in which every platform is meaningfully present. Restricted to
    that window the corpus is three months long, which is shorter than any
    honest trend needs, so this returns no trend column at all.

    It is left in place, rather than deleted, because the guard is the point:
    when a longer window exists the ratio becomes computable, and until then
    the absence of the column is more informative than a number would be.
    """
    if pain_points.empty or labels.empty or "year_month" not in labels.columns:
        return pain_points

    issues = labels[labels["polarity"] == "issue"]
    if "in_comparable_window" in issues.columns:
        # Anything before the window compares a platform against noise.
        issues = issues[issues["in_comparable_window"]]

    months = sorted(issues["year_month"].dropna().unique())
    prior_months = len(months) - recent_months
    if prior_months < min_prior_months:
        logger.warning(
            "Not computing trend: %d comparable month(s) available, need %d "
            "(%d recent + %d prior). Review volume outside the comparable "
            "window reflects collection, not customer behaviour.",
            len(months), recent_months + min_prior_months, recent_months, min_prior_months,
        )
        return pain_points

    recent_set = set(months[-recent_months:])
    recent = issues[issues["year_month"].isin(recent_set)]
    prior = issues[~issues["year_month"].isin(recent_set)]

    def per_month(subset: pd.DataFrame, n_months: int) -> pd.Series:
        counts = subset.groupby(["product_area", "issue_type"], observed=True)[
            "review_id"
        ].nunique()
        return counts / max(1, n_months)

    recent_rate = per_month(recent, len(recent_set))
    prior_rate = per_month(prior, prior_months)

    ratios = []
    for key in zip(pain_points["product_area"], pain_points["issue_type"]):
        before = prior_rate.get(key, 0.0)
        after = recent_rate.get(key, 0.0)
        # No prior mentions means new, not infinitely worse.
        ratios.append(round(float(after / before), 2) if before > 0 else None)

    pain_points = pain_points.copy()
    pain_points["trend_ratio"] = ratios
    return pain_points


def attach_evidence(
    pain_points: pd.DataFrame,
    labels: pd.DataFrame,
    n: int = 3,
) -> pd.DataFrame:
    """Attach the highest-confidence verbatim spans for each pain point.

    Every claim in the report should be one click from the customer's own
    words. Evidence spans were verified verbatim against the source text at
    enrichment time, so quoting them here cannot invent a quote.
    """
    if pain_points.empty or labels.empty:
        return pain_points

    issues = labels[labels["polarity"] == "issue"]
    quotes, quote_ids = [], []
    for area, issue_type in zip(pain_points["product_area"], pain_points["issue_type"]):
        subset = issues[
            (issues["product_area"] == area) & (issues["issue_type"] == issue_type)
        ].nlargest(n, "confidence")
        quotes.append([str(s)[:200] for s in subset["evidence_span"].tolist()])
        quote_ids.append(subset["review_id"].tolist())

    pain_points = pain_points.copy()
    pain_points["evidence"] = quotes
    pain_points["evidence_review_ids"] = quote_ids
    return pain_points
