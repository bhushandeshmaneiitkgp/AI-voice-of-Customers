"""
Layer 6b -- clustering reviews into themes.

The taxonomy (Phase 2) says what categories exist; enrichment (Phase 3) says
which apply to each review. Neither says *what the recurring complaints
actually are* inside a category. The largest product area in this corpus carries
close to two thousand labels -- useful for counting, useless as a brief.
Clustering the embeddings splits that mass into themes a PM can act on,
discovered from the text rather than declared in advance.

Category names are deliberately absent from this module. They belong to
``config/taxonomy.yaml``, and a test enforces that they never leak into Python
-- including into a docstring, which is how this paragraph originally failed.

k is chosen by silhouette score over a candidate range, not picked by eye. A
hand-chosen k is an unfalsifiable claim about the data; a scored one can be
argued with, and the score is written into the report so a reader can.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

#: Silhouette on every row is O(n^2) in memory. Above this many reviews it is
#: computed on a random subsample -- the score is a model-selection heuristic,
#: not a reported metric, and a stable estimate is worth more than an exact one
#: that cannot be computed at all.
SILHOUETTE_SAMPLE_CAP = 2000


@dataclass
class ClusterModel:
    """A fitted clustering plus the evidence for its k."""

    k: int
    labels: np.ndarray
    centroids: np.ndarray
    silhouette: float
    #: (k, score) for every candidate tried, so the choice is auditable.
    scores: list[tuple[int, float]] = field(default_factory=list)


def choose_k(
    vectors: np.ndarray,
    k_min: int,
    k_max: int,
    seed: int = 42,
) -> list[tuple[int, float]]:
    """Score each candidate k by silhouette. Returns (k, score), best first.

    Silhouette rather than inertia: inertia falls monotonically with k, so the
    "elbow" is a judgement call dressed up as a measurement. Silhouette has an
    interior optimum and can therefore actually choose.
    """
    n = len(vectors)
    if n < 3:
        raise ValueError(f"need at least 3 rows to cluster, got {n}")

    # k must stay below n, and silhouette is undefined at k=1.
    upper = min(k_max, n - 1)
    candidates = [k for k in range(max(2, k_min), upper + 1)]
    if not candidates:
        raise ValueError(f"no valid k in [{k_min}, {k_max}] for {n} rows")

    rng = np.random.default_rng(seed)
    if n > SILHOUETTE_SAMPLE_CAP:
        sample_idx = rng.choice(n, SILHOUETTE_SAMPLE_CAP, replace=False)
    else:
        sample_idx = np.arange(n)

    scored: list[tuple[int, float]] = []
    for k in candidates:
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = model.fit_predict(vectors)
        # A degenerate fit that collapses to one populated cluster cannot be
        # scored; treat it as the worst possible rather than crashing.
        if len(np.unique(labels[sample_idx])) < 2:
            scored.append((k, -1.0))
            continue
        score = silhouette_score(vectors[sample_idx], labels[sample_idx], metric="cosine")
        scored.append((k, float(score)))
        logger.info("k=%d silhouette=%.4f", k, score)

    return sorted(scored, key=lambda pair: pair[1], reverse=True)


def fit_clusters(
    vectors: np.ndarray,
    k_min: int,
    k_max: int,
    seed: int = 42,
) -> ClusterModel:
    """Choose k, then fit at that k.

    Warns when the winner sits on a boundary of the search range. A k at the
    edge is not a discovered optimum -- it is the best of what was offered, and
    the real peak may lie outside. Silently reporting it as "chosen by
    silhouette" would dress up the range default as a finding.
    """
    scores = choose_k(vectors, k_min, k_max, seed)
    best_k, best_score = scores[0]

    tried = sorted(k for k, _ in scores)
    if len(tried) > 1 and best_k in (tried[0], tried[-1]):
        edge = "lower" if best_k == tried[0] else "upper"
        logger.warning(
            "k=%d is the %s bound of the search range [%d, %d]. The optimum may "
            "lie outside it -- widen the range before treating this k as chosen "
            "by the data.",
            best_k, edge, tried[0], tried[-1],
        )

    logger.info("Chose k=%d (silhouette=%.4f)", best_k, best_score)

    model = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    labels = model.fit_predict(vectors)
    return ClusterModel(
        k=best_k,
        labels=labels,
        centroids=model.cluster_centers_,
        silhouette=best_score,
        scores=scores,
    )


def exemplars(
    vectors: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    cluster_id: int,
    n: int = 3,
) -> list[int]:
    """Row positions of the reviews closest to a cluster's centroid.

    These are what a human reads to decide whether a cluster is real. Picking
    the nearest to the centroid rather than a random sample means the examples
    shown are the ones the cluster is actually *about*.
    """
    members = np.flatnonzero(labels == cluster_id)
    if not len(members):
        return []
    # Vectors are normalised, so inner product with the centroid ranks by
    # cosine similarity without another normalisation pass.
    similarity = vectors[members] @ centroids[cluster_id]
    order = np.argsort(similarity)[::-1][:n]
    return members[order].tolist()


def summarise_clusters(
    frame: pd.DataFrame,
    model: ClusterModel,
    vectors: np.ndarray,
    labels_frame: pd.DataFrame | None = None,
    n_exemplars: int = 3,
) -> pd.DataFrame:
    """One row per cluster: size, composition, and what it is about.

    ``labels_frame`` is the per-area label table from enrichment. When present,
    each cluster gets its dominant product_area and issue_type, which is what
    turns an opaque "cluster 7" into a named theme in the report.
    """
    working = frame.copy()
    working["cluster_id"] = model.labels

    severity_rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}

    rows = []
    for cluster_id in range(model.k):
        members = working[working["cluster_id"] == cluster_id]
        if members.empty:
            continue

        picks = exemplars(vectors, model.labels, model.centroids, cluster_id, n_exemplars)

        dominant_area = dominant_issue = None
        if labels_frame is not None and not labels_frame.empty:
            member_labels = labels_frame[labels_frame["review_id"].isin(members["review_id"])]
            if not member_labels.empty:
                dominant_area = member_labels["product_area"].mode().iat[0]
                issues = member_labels["issue_type"].dropna()
                if not issues.empty:
                    dominant_issue = issues.mode().iat[0]

        severities = members["severity"].map(severity_rank).dropna()

        rows.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "share_pct": round(len(members) / len(working) * 100, 2),
                "dominant_area": dominant_area,
                "dominant_issue": dominant_issue,
                "mean_severity": round(float(severities.mean()), 2) if len(severities) else None,
                "escalation_rate": round(float(members["support_escalation"].mean()), 3),
                "negative_share": round(
                    float((members["sentiment"] == "negative").mean()), 3
                ),
                "mean_rating": round(float(members["rating"].mean()), 2),
                "top_platform": members["platform"].mode().iat[0],
                "exemplar_review_ids": [working.iloc[i]["review_id"] for i in picks],
                "exemplar_texts": [working.iloc[i]["review_text"][:200] for i in picks],
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values("size", ascending=False).reset_index(drop=True)
