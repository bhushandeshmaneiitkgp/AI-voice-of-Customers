"""Tests for Phase 4: embeddings, clustering, and pain-point scoring.

No model download and no torch import. The encoder is injected, so every check
here runs offline in milliseconds -- the same discipline the enrichment tests
apply to the LLM.

What matters in this layer is not that a score is produced but that it is the
*right* score: that a review contributing to three pain points counts once in
each, that compliments never enter the fix-it list, and that vectors stay
attached to the review they were computed from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voc.cluster import choose_k, exemplars, fit_clusters, summarise_clusters
from voc.embed import EmbeddingStore, embed_reviews, encode_texts, normalise
from voc.painpoints import (
    SEVERITY_RANK,
    WEIGHTS,
    add_trend,
    attach_evidence,
    build_pain_points,
)


class FakeEncoder:
    """Deterministic stand-in: hashes text to a fixed-width vector."""

    def __init__(self, dims: int = 8) -> None:
        self.dims = dims
        self.calls: list[list[str]] = []

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self.dims), dtype="float32")
        for i, text in enumerate(texts):
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            out[i] = rng.normal(size=self.dims)
        return out


def _reviews(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(n)],
            "review_text": [f"review text number {i}" for i in range(n)],
            "platform": ["zepto", "blinkit"] * (n // 2),
            "rating": [1, 5] * (n // 2),
            "sentiment": ["negative", "positive"] * (n // 2),
            "severity": ["high", "low"] * (n // 2),
            "customer_intent": ["complaint", "praise"] * (n // 2),
            "support_escalation": [True, False] * (n // 2),
        }
    )


def _labels(rows: list[tuple]) -> pd.DataFrame:
    """rows: (review_id, product_area, issue_type, polarity, confidence, month)."""
    return pd.DataFrame(
        rows,
        columns=["review_id", "product_area", "issue_type", "polarity",
                 "confidence", "year_month"],
    ).assign(platform="zepto", evidence_span="quoted span", strength_type=None)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_vectors_are_normalised_so_inner_product_is_cosine() -> None:
    """FAISS offers inner product, not cosine; normalising is what bridges them."""
    vectors = encode_texts(["a", "b", "c"], "fake", encoder=FakeEncoder())
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_zero_vector_does_not_become_nan() -> None:
    """A NaN in the index poisons every later search with no error message."""
    result = normalise(np.zeros((2, 4), dtype="float32"))
    assert not np.isnan(result).any()


def test_store_returns_vectors_in_the_callers_order() -> None:
    """Storage order is insertion order; callers align against a DataFrame.

    Returning storage order would attach each review to a neighbour's vector --
    a silent corruption that still produces clusters.
    """
    store = EmbeddingStore(pytest.importorskip("pathlib").Path("nonexistent.npz"), "m")
    store.add(["a", "b", "c"], np.array([[1.0], [2.0], [3.0]], dtype="float32"))

    assert store.vectors_for(["c", "a"]).ravel().tolist() == [3.0, 1.0]


def test_store_refuses_to_misalign_ids_and_vectors(tmp_path) -> None:
    store = EmbeddingStore(tmp_path / "e.npz", "m")
    with pytest.raises(ValueError, match="refusing to misalign"):
        store.add(["a", "b"], np.array([[1.0]], dtype="float32"))


def test_store_raises_on_an_unknown_review(tmp_path) -> None:
    store = EmbeddingStore(tmp_path / "e.npz", "m")
    store.add(["a"], np.array([[1.0]], dtype="float32"))
    with pytest.raises(KeyError):
        store.vectors_for(["a", "missing"])


def test_store_round_trips_through_disk(tmp_path) -> None:
    path = tmp_path / "e.npz"
    store = EmbeddingStore(path, "model-x")
    store.add(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    store.save()

    reloaded = EmbeddingStore(path, "model-x")
    assert len(reloaded) == 2
    assert reloaded.vectors_for(["b"]).ravel().tolist() == [0.0, 1.0]


def test_cache_from_another_model_is_ignored(tmp_path) -> None:
    """Two embedding models produce incomparable spaces.

    Serving one model's vectors to another yields clusters that look plausible
    and mean nothing -- the embedding equivalent of the enrichment cache key
    including the model id.
    """
    path = tmp_path / "e.npz"
    built = EmbeddingStore(path, "model-a")
    built.add(["a"], np.array([[1.0, 0.0]], dtype="float32"))
    built.save()

    assert len(EmbeddingStore(path, "model-b")) == 0


def test_only_uncached_reviews_are_encoded(tmp_path) -> None:
    """Re-embedding what is already on disk is pure waste."""
    frame = _reviews(4)
    encoder = FakeEncoder()
    store = EmbeddingStore(tmp_path / "e.npz", "m")

    embed_reviews(frame, "m", store=store, encoder=encoder)
    assert len(encoder.calls[0]) == 4

    embed_reviews(frame, "m", store=store, encoder=encoder)
    assert len(encoder.calls) == 1, "second pass should have encoded nothing"


def test_embeddings_stay_aligned_to_frame_order(tmp_path) -> None:
    """The reason the store reorders: rows must keep their own vector."""
    frame = _reviews(6)
    store = EmbeddingStore(tmp_path / "e.npz", "m")
    encoder = FakeEncoder()

    first = embed_reviews(frame, "m", store=store, encoder=encoder)
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    second = embed_reviews(shuffled, "m", store=store, encoder=encoder)

    assert np.allclose(first[::-1], second)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _separable(n_per: int = 20, dims: int = 6, groups: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    centres = np.eye(groups, dims) * 10
    return normalise(
        np.vstack([centres[g] + rng.normal(scale=0.1, size=(n_per, dims))
                   for g in range(groups)])
    )


def test_silhouette_prefers_the_true_group_count() -> None:
    """k is scored, not chosen by eye -- so it must actually find the answer."""
    scored = choose_k(_separable(), k_min=2, k_max=6)
    assert scored[0][0] == 3


def test_every_candidate_k_is_scored_and_recorded() -> None:
    """The report prints the losing scores, so the choice can be argued with."""
    model = fit_clusters(_separable(), k_min=2, k_max=5)
    # Ranked best-first, so compare the set of candidates, not the order.
    assert sorted(k for k, _ in model.scores) == [2, 3, 4, 5]
    assert model.scores[0][0] == model.k == 3


def test_k_cannot_exceed_the_row_count() -> None:
    """KMeans with k >= n raises; clamping keeps a small corpus workable."""
    tiny = normalise(np.random.default_rng(0).normal(size=(5, 4)))
    model = fit_clusters(tiny, k_min=2, k_max=50)
    assert model.k < len(tiny)


def test_clustering_rejects_a_corpus_too_small_to_cluster() -> None:
    with pytest.raises(ValueError, match="at least 3 rows"):
        choose_k(np.zeros((2, 4), dtype="float32"), k_min=2, k_max=4)


def test_exemplars_are_nearest_the_centroid() -> None:
    """Exemplars are what a human reads to judge whether a cluster is real."""
    vectors = normalise(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype="float32"))
    labels = np.array([0, 0, 1])
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")

    assert exemplars(vectors, labels, centroids, 0, n=1) == [0]


def test_cluster_summary_reports_composition() -> None:
    frame = _reviews(6)
    vectors = _separable(n_per=2, dims=4, groups=3)
    model = fit_clusters(vectors, k_min=2, k_max=3)
    labels = _labels(
        [(f"r{i}", "delivery_reliability", "late_delivery", "issue", 0.9, "2024-12")
         for i in range(6)]
    )

    summary = summarise_clusters(frame, model, vectors, labels)

    assert summary["size"].sum() == 6
    assert set(summary.columns) >= {"cluster_id", "size", "share_pct",
                                    "dominant_area", "escalation_rate"}
    assert (summary["dominant_area"] == "delivery_reliability").all()


# ---------------------------------------------------------------------------
# Pain-point scoring
# ---------------------------------------------------------------------------


def test_compliments_never_become_pain_points() -> None:
    """Scoring a strength as a problem puts 'fast delivery' on the fix-it list."""
    reviews = _reviews(4)
    labels = _labels(
        [("r0", "delivery_speed", None, "strength", 0.9, "2024-12"),
         ("r1", "delivery_speed", None, "strength", 0.9, "2024-12")]
    )

    assert build_pain_points(labels, reviews, min_volume=1).empty


def test_volume_counts_reviews_not_labels() -> None:
    """A model repeating an area for one review must not double its volume."""
    reviews = _reviews(2)
    labels = _labels(
        [("r0", "customer_support", "unhelpful_agent", "issue", 0.9, "2024-12"),
         ("r0", "customer_support", "unhelpful_agent", "issue", 0.8, "2024-12"),
         ("r1", "customer_support", "unhelpful_agent", "issue", 0.9, "2024-12")]
    )

    result = build_pain_points(labels, reviews, min_volume=1)
    assert result.iloc[0]["volume"] == 2


def test_low_volume_pain_points_are_dropped() -> None:
    """Below the floor a pattern is an anecdote and buries the real signal."""
    reviews = _reviews(6)
    labels = _labels(
        [("r0", "app_experience", "crash", "issue", 0.9, "2024-12")]
        + [(f"r{i}", "customer_support", "unhelpful_agent", "issue", 0.9, "2024-12")
           for i in range(1, 6)]
    )

    result = build_pain_points(labels, reviews, min_volume=3)
    assert list(result["product_area"]) == ["customer_support"]


def test_severity_outranks_volume_when_volume_ties() -> None:
    """The point of the score: the most common issue is not the most costly."""
    reviews = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(6)],
            "review_text": ["t"] * 6,
            "platform": ["zepto"] * 6,
            "rating": [1] * 6,
            "sentiment": ["negative"] * 6,
            # First three critical, last three low.
            "severity": ["critical"] * 3 + ["low"] * 3,
            "customer_intent": ["complaint"] * 6,
            "support_escalation": [False] * 6,
        }
    )
    labels = _labels(
        [(f"r{i}", "payments", "charged_twice", "issue", 0.9, "2024-12") for i in range(3)]
        + [(f"r{i}", "app_experience", "slow_ui", "issue", 0.9, "2024-12") for i in range(3, 6)]
    )

    result = build_pain_points(labels, reviews, min_volume=1)
    assert result.iloc[0]["issue_type"] == "charged_twice"
    assert result.iloc[0]["rank"] == 1


def test_score_stays_within_zero_and_one() -> None:
    """Weights sum to 1.0 and every term is bounded, so the composite is too."""
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    reviews = _reviews(6)
    labels = _labels(
        [(f"r{i}", "customer_support", "unhelpful_agent", "issue", 0.9, "2024-12")
         for i in range(6)]
    )
    result = build_pain_points(labels, reviews, min_volume=1)
    assert result["score"].between(0.0, 1.0).all()


def test_severity_ranking_is_ordinal_and_complete() -> None:
    assert list(SEVERITY_RANK) == ["low", "medium", "high", "critical"]
    assert sorted(SEVERITY_RANK.values()) == [1, 2, 3, 4]


def test_identical_pain_points_do_not_produce_nan_scores() -> None:
    """A flat column divides by zero and blanks the whole report."""
    reviews = _reviews(4)
    labels = _labels(
        [("r0", "a_area", "x", "issue", 0.9, "2024-12"),
         ("r1", "a_area", "x", "issue", 0.9, "2024-12"),
         ("r2", "b_area", "y", "issue", 0.9, "2024-12"),
         ("r3", "b_area", "y", "issue", 0.9, "2024-12")]
    )

    result = build_pain_points(labels, reviews, min_volume=1)
    assert not result["score"].isna().any()


def test_trend_is_reported_but_does_not_move_the_ranking() -> None:
    """Trend is a Phase 5 question; folding it in here would weaken the score."""
    reviews = _reviews(6)
    labels = _labels(
        [(f"r{i}", "customer_support", "unhelpful_agent", "issue", 0.9, "2024-0" + str(i + 1))
         for i in range(6)]
    )

    base = build_pain_points(labels, reviews, min_volume=1)
    with_trend = add_trend(base, labels, recent_months=2)

    assert "trend_ratio" in with_trend.columns
    assert list(base["score"]) == list(with_trend["score"])


def test_a_brand_new_pain_point_has_no_trend_ratio() -> None:
    """No prior mentions means new, not infinitely worse."""
    reviews = _reviews(6)
    labels = _labels(
        [("r0", "old", "x", "issue", 0.9, "2024-01"),
         ("r1", "old", "x", "issue", 0.9, "2024-02"),
         ("r2", "old", "x", "issue", 0.9, "2024-03"),
         ("r4", "old", "x", "issue", 0.9, "2024-04"),
         ("r3", "brand_new", "y", "issue", 0.9, "2024-04")]
    )

    result = add_trend(build_pain_points(labels, reviews, min_volume=1), labels,
                       recent_months=1, min_prior_months=3)
    new_row = result[result["product_area"] == "brand_new"].iloc[0]
    # pandas stores the None as NaN in a float column; either way it is absent.
    assert pd.isna(new_row["trend_ratio"])


def test_trend_refuses_when_there_is_too_little_history() -> None:
    """A ratio over three months of a ramping scrape measures the scrape.

    The real corpus produced ratios up to 197x this way. No column is more
    informative than a number that looks like a finding and is not one.
    """
    reviews = _reviews(4)
    labels = _labels(
        [("r0", "support", "x", "issue", 0.9, "2024-10"),
         ("r1", "support", "x", "issue", 0.9, "2024-11"),
         ("r2", "support", "x", "issue", 0.9, "2024-12")]
    )

    result = add_trend(build_pain_points(labels, reviews, min_volume=1), labels,
                       recent_months=3, min_prior_months=3)
    assert "trend_ratio" not in result.columns


def test_trend_ignores_months_outside_the_comparable_window() -> None:
    """Phase 1 drew this line; the trend must not step over it.

    Pre-window months hold a handful of reviews each. Counting them as the
    baseline is what manufactured the original blow-up.
    """
    reviews = _reviews(6)
    rows = [("r0", "support", "x", "issue", 0.9, "2020-07"),
            ("r1", "support", "x", "issue", 0.9, "2020-08"),
            ("r2", "support", "x", "issue", 0.9, "2020-09"),
            ("r3", "support", "x", "issue", 0.9, "2024-10"),
            ("r4", "support", "x", "issue", 0.9, "2024-11"),
            ("r5", "support", "x", "issue", 0.9, "2024-12")]
    labels = _labels(rows)
    labels["in_comparable_window"] = [False, False, False, True, True, True]

    # Only 3 in-window months survive, so it must refuse -- if the pre-window
    # months leaked in there would be 6 and it would happily compute.
    result = add_trend(build_pain_points(labels, reviews, min_volume=1), labels,
                       recent_months=1, min_prior_months=3)
    assert "trend_ratio" not in result.columns


def test_evidence_comes_from_the_matching_pain_point() -> None:
    """Every claim in the report must be one click from the customer's words."""
    reviews = _reviews(4)
    labels = pd.DataFrame(
        [
            ("r0", "payments", "charged_twice", "issue", 0.95, "2024-12", "billed me twice"),
            ("r1", "payments", "charged_twice", "issue", 0.90, "2024-12", "double charge"),
            ("r2", "app_experience", "slow_ui", "issue", 0.99, "2024-12", "app is slow"),
        ],
        columns=["review_id", "product_area", "issue_type", "polarity",
                 "confidence", "year_month", "evidence_span"],
    ).assign(platform="zepto", strength_type=None)

    result = attach_evidence(
        build_pain_points(labels, reviews, min_volume=1), labels, n=2
    )
    payments = result[result["issue_type"] == "charged_twice"].iloc[0]

    assert "billed me twice" in payments["evidence"]
    assert "app is slow" not in payments["evidence"]


def test_empty_input_returns_empty_rather_than_raising() -> None:
    assert build_pain_points(pd.DataFrame(), _reviews(2)).empty


def test_a_k_on_the_range_boundary_is_flagged(caplog) -> None:
    """An edge-of-range k is the best of what was offered, not an optimum.

    Reporting it as "chosen by silhouette" would dress up the range default as
    a finding. Three well-separated groups scored over k=3..5 must win at the
    lower bound and say so.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="voc.cluster"):
        model = fit_clusters(_separable(), k_min=3, k_max=5)

    assert model.k == 3
    assert any("lower bound" in record.getMessage() for record in caplog.records)


def test_an_interior_k_is_not_flagged(caplog) -> None:
    """The warning must stay quiet when the peak is genuinely bracketed."""
    import logging

    with caplog.at_level(logging.WARNING, logger="voc.cluster"):
        model = fit_clusters(_separable(), k_min=2, k_max=6)

    assert model.k == 3
    assert not [r for r in caplog.records
                if "bound of the search range" in r.getMessage()]
