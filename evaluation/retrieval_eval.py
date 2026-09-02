"""
Retrieval evaluation: does the RAG layer return reviews that are actually about the query?

Every root-cause hypothesis in Phase 6 is built from retrieved evidence, so a
retriever that returns plausible-looking but off-topic reviews would produce
hypotheses that cite real ids, pass the citation check, and still be about
nothing. Citation validation cannot catch that; only relevance can.

There is no human relevance judgement to score against, so this uses the corpus
labels as the reference. That is weaker than gold, and it is *not* circular in
the way it first looks: the retriever ranks by MiniLM sentence embeddings, the
labels come from a 70B instruction model, and neither ever sees the other's
output. Agreement between two independent systems about which reviews concern
delivery is real evidence about both -- it is just evidence about consistency
rather than about truth.

The comparison that matters is against the **base rate**, not against zero. If
19% of the corpus mentions an area, a retriever returning 19% relevant hits has
done nothing at all, and precision@8 of 0.19 would still look respectable in a
table. Every row here therefore carries its own base rate and the lift over it,
and the base rate is the analytic share rather than a sampled control, because
the expectation is exactly computable and sampling would only add noise to the
thing being compared against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from voc.taxonomy import Taxonomy

from evaluation.metrics import Rate

logger = logging.getLogger(__name__)

#: Areas below this many labelled reviews are skipped. Precision@8 against a
#: base rate of 0.2% is dominated by whether the index happens to hold nine
#: matching reviews rather than eight.
MIN_AREA_SUPPORT = 50


def build_area_queries(taxonomy: Taxonomy) -> dict[str, str]:
    """A natural-language query per product area, taken from the taxonomy itself.

    Built from the area's own name and definition rather than hand-written
    probes, for two reasons: the queries cannot drift from the vocabulary being
    measured, and nobody gets to tune a query until the retriever looks good on
    it. A hand-tuned query set measures the tuner.
    """
    queries: dict[str, str] = {}
    for area in taxonomy.product_areas:
        parts = [area.name]
        definition = (area.definition or "").strip()
        if definition:
            parts.append(definition)
        queries[area.id] = " ".join(parts).strip()
    return queries


@dataclass
class RetrievalRow:
    """One query's result: what came back, and what chance alone would have given."""

    area: str
    query: str
    retrieved: int
    relevant: int
    base_rate: float

    @property
    def precision(self) -> Rate:
        return Rate(self.relevant, self.retrieved)

    @property
    def lift(self) -> float | None:
        value = self.precision.value
        if value is None or self.base_rate <= 0:
            return None
        return value / self.base_rate

    @property
    def beats_base_rate(self) -> bool:
        """True only when the interval clears the base rate.

        A point estimate above the base rate at k=8 is worth very little -- one
        extra hit moves precision by 12.5 points. Requiring the lower bound to
        clear it is the difference between a result and a coin.
        """
        interval = self.precision.interval
        return bool(interval and interval[0] > self.base_rate)


def evaluate_retrieval(
    retriever: Any,
    labels: pd.DataFrame,
    taxonomy: Taxonomy,
    k: int = 8,
    min_support: int = MIN_AREA_SUPPORT,
    queries: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Score unfiltered retrieval against the corpus labels, one area at a time.

    Deliberately does **not** pass ``product_area`` to the search. Filtering by
    the very label being scored would return precision 1.0 by construction and
    would measure the filter, not the retriever.
    """
    if labels.empty:
        return pd.DataFrame()

    area_members = labels.groupby("product_area")["review_id"].apply(set).to_dict()
    corpus_size = labels["review_id"].nunique()
    query_map = dict(queries) if queries is not None else build_area_queries(taxonomy)

    rows: list[dict[str, Any]] = []
    for area, query in sorted(query_map.items()):
        members = area_members.get(area, set())
        if len(members) < min_support:
            continue
        if not query.strip():
            logger.warning("Area %s has no text to build a query from; skipped.", area)
            continue

        result = retriever.search(query, k=k)
        hits = [evidence.review_id for evidence in result.hits]
        row = RetrievalRow(
            area=area,
            query=query,
            retrieved=len(hits),
            relevant=sum(1 for review_id in hits if review_id in members),
            base_rate=len(members) / corpus_size if corpus_size else 0.0,
        )
        interval = row.precision.interval
        rows.append(
            {
                "area": row.area,
                "support": len(members),
                "retrieved": row.retrieved,
                "relevant": row.relevant,
                "precision_at_k": row.precision.value,
                "ci_low": interval[0] if interval else None,
                "ci_high": interval[1] if interval else None,
                "base_rate": row.base_rate,
                "lift": row.lift,
                "beats_base_rate": row.beats_base_rate,
            }
        )

    return pd.DataFrame(rows)


def summarise(evaluation: pd.DataFrame) -> dict[str, Any]:
    """Pool the per-area rows into one precision figure, plus how many cleared chance.

    Pooled over hits rather than averaged over areas: a mean of per-area
    precisions weights an area with three retrieved reviews the same as one with
    eight, and the pooled figure is the one with a defensible interval.
    """
    if evaluation.empty:
        return {}
    retrieved = int(evaluation["retrieved"].sum())
    relevant = int(evaluation["relevant"].sum())
    pooled = Rate(relevant, retrieved)
    return {
        "areas_evaluated": int(len(evaluation)),
        "pooled_precision": pooled.as_dict(),
        "mean_base_rate": float(evaluation["base_rate"].mean()),
        "mean_lift": (
            float(evaluation["lift"].mean()) if evaluation["lift"].notna().any() else None
        ),
        "areas_beating_base_rate": int(evaluation["beats_base_rate"].sum()),
        "weakest_area": (
            evaluation.sort_values("precision_at_k").iloc[0]["area"]
            if evaluation["precision_at_k"].notna().any() else None
        ),
    }
