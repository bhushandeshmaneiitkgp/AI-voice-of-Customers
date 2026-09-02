"""
Layer 8 -- RAG evidence retrieval.

Phase 4 built the vector index; this is what reads from it. Given a question or
a pain point, return the reviews that actually bear on it, each carrying the
identifier needed to check the quote against the source.

The retrieval contract is deliberately narrow: **it returns evidence, never
answers**. Layer 9 forms hypotheses from what comes back, and every hypothesis
is later checked against these ``review_id`` values. A retriever that
summarised, reranked by relevance-to-a-conclusion, or silently dropped
inconvenient hits would break that check while appearing to work.

Filtering happens *after* the vector search, over an over-fetched candidate
set, rather than by rebuilding a filtered index per query. Pre-filtering would
be exact but costs an index build per filter combination; over-fetching costs
one flat scan of 4,568 vectors, which is microseconds. The tradeoff is that a
very selective filter can return fewer than ``k`` hits -- which is reported
honestly rather than padded with irrelevant matches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from voc.embed import encode_texts, load_index, normalise

logger = logging.getLogger(__name__)

#: Multiplier applied to ``k`` before filtering. A filter that keeps one review
#: in ten still needs ten times the candidates to fill the result set. Higher
#: costs nothing measurable on a flat index; lower silently returns short.
OVERFETCH = 12

#: Hard ceiling on candidates pulled from the index, so a pathological filter
#: cannot turn one query into a full-corpus scan and sort.
MAX_CANDIDATES = 2000


@dataclass
class Evidence:
    """One retrieved review, with everything needed to verify it."""

    review_id: str
    text: str
    score: float
    platform: str
    rating: int
    year_month: str
    sentiment: str | None = None
    severity: str | None = None
    product_areas: list[str] = field(default_factory=list)

    def citation(self, max_chars: int = 220) -> str:
        """Render for a prompt: id first, so a citation is copyable."""
        body = self.text if len(self.text) <= max_chars else self.text[:max_chars] + "..."
        return f"[{self.review_id}] ({self.platform}, {self.rating}*, {self.year_month}) {body}"


@dataclass
class RetrievalResult:
    """What a query returned, and what it could not."""

    query: str
    hits: list[Evidence]
    #: Candidates examined before filtering, for diagnosing an empty result.
    candidates_examined: int = 0
    #: True when filters left fewer than k hits -- the caller should say so
    #: rather than treat a short list as "all the evidence there is".
    truncated_by_filters: bool = False


class Retriever:
    """Semantic search over the enriched corpus.

    Holds the FAISS index alongside the review frame, because a hit is a
    position in the index and is meaningless without the row it points at.
    Constructing the two separately and pairing them later is how retrieval
    systems end up quoting the wrong review with full confidence.
    """

    def __init__(
        self,
        index: Any,
        reviews: pd.DataFrame,
        model_name: str,
        labels: pd.DataFrame | None = None,
        encoder: Any | None = None,
    ) -> None:
        if index.ntotal != len(reviews):
            raise ValueError(
                f"index holds {index.ntotal} vectors but the frame has {len(reviews)} "
                "rows -- they were not built from the same corpus, and every hit "
                "would point at the wrong review"
            )
        self.index = index
        self.reviews = reviews.reset_index(drop=True)
        self.model_name = model_name
        self._encoder = encoder
        self._areas: dict[str, list[str]] = {}
        if labels is not None and not labels.empty:
            self._areas = (
                labels.groupby("review_id")["product_area"].apply(list).to_dict()
            )

    @property
    def encoder(self) -> Any:
        """Load the sentence-transformer once, on first query.

        Loading it per call costs seconds of model construction for a
        millisecond of encoding. That is tolerable across ten pain points and
        ruinous behind a UI, where every keystroke would rebuild a transformer.
        """
        if self._encoder is None:
            from voc.embed import load_encoder

            self._encoder = load_encoder(self.model_name)
        return self._encoder

    @classmethod
    def from_paths(
        cls,
        index_path,
        reviews: pd.DataFrame,
        model_name: str,
        labels: pd.DataFrame | None = None,
    ) -> "Retriever":
        return cls(load_index(index_path), reviews, model_name, labels)

    def _row_to_evidence(self, position: int, score: float) -> Evidence:
        row = self.reviews.iloc[position]
        review_id = str(row["review_id"])
        return Evidence(
            review_id=review_id,
            text=str(row["review_text"]),
            score=float(score),
            platform=str(row["platform"]),
            rating=int(row["rating"]),
            year_month=str(row["year_month"]),
            sentiment=row.get("sentiment"),
            severity=row.get("severity"),
            product_areas=self._areas.get(review_id, []),
        )

    def search(
        self,
        query: str,
        k: int = 8,
        platform: str | None = None,
        product_area: str | None = None,
        severity: Sequence[str] | None = None,
        comparable_window_only: bool = False,
        exclude_ids: Sequence[str] = (),
    ) -> RetrievalResult:
        """Return up to ``k`` reviews most similar to ``query``.

        Scores are cosine similarities, because the index is inner-product over
        normalised vectors.
        """
        if not query.strip():
            raise ValueError("empty query -- retrieval needs something to match on")

        vector = encode_texts([query], self.model_name, encoder=self.encoder)
        candidates = min(MAX_CANDIDATES, max(k * OVERFETCH, k), self.index.ntotal)
        scores, positions = self.index.search(
            np.ascontiguousarray(normalise(vector), dtype="float32"), candidates
        )

        excluded = set(exclude_ids)
        hits: list[Evidence] = []
        for score, position in zip(scores[0], positions[0]):
            # FAISS pads with -1 when it has fewer vectors than requested.
            if position < 0:
                continue
            evidence = self._row_to_evidence(int(position), float(score))
            if evidence.review_id in excluded:
                continue
            if platform and evidence.platform != platform:
                continue
            if product_area and product_area not in evidence.product_areas:
                continue
            if severity and evidence.severity not in severity:
                continue
            if comparable_window_only and not bool(
                self.reviews.iloc[int(position)].get("in_comparable_window", True)
            ):
                continue
            hits.append(evidence)
            if len(hits) >= k:
                break

        if len(hits) < k:
            logger.info(
                "Query %r returned %d of %d requested after filtering %d candidates",
                query[:60], len(hits), k, candidates,
            )

        return RetrievalResult(
            query=query,
            hits=hits,
            candidates_examined=candidates,
            truncated_by_filters=len(hits) < k,
        )

    def evidence_for_pain_point(
        self,
        product_area: str,
        issue_type: str,
        k: int = 8,
        platform: str | None = None,
    ) -> RetrievalResult:
        """Evidence for a Phase 4 pain point.

        The query is built from the taxonomy identifiers rather than a
        hand-written question, so every pain point is retrieved the same way and
        no phrasing choice quietly favours one over another. The area filter
        does the precision work; the embedding supplies the ordering within it.
        """
        query = f"{product_area.replace('_', ' ')} {issue_type.replace('_', ' ')}"
        return self.search(query, k=k, product_area=product_area, platform=platform)


def format_evidence_block(hits: Sequence[Evidence], max_chars: int = 220) -> str:
    """Render hits for a prompt, one citation per line."""
    return "\n".join(hit.citation(max_chars) for hit in hits)
