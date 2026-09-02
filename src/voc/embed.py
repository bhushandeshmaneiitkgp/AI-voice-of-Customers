"""
Layer 6a -- sentence embeddings and the vector index.

Turns review text into vectors so that Phase 4 can group reviews by what they
actually say, and Phase 6 can retrieve evidence for a claim.

Embeddings are computed **locally** with sentence-transformers rather than
through a hosted endpoint. The enrichment layer pays for an API because only a
large model can do that job; embedding short English text does not need one,
and routing it through a provider would add a per-run cost, a rate limit and a
network failure mode to a step that is deterministic and finishes in under a
minute on CPU.

Vectors are cached to disk keyed by review_id and model, for the same reason
enrichment responses are: recomputing 4,600 embeddings on every run is waste,
and a cache keyed without the model name would silently mix two vector spaces.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Cosine similarity is the right metric for sentence embeddings, but FAISS
#: offers inner product, not cosine. On L2-normalised vectors the two are
#: identical, so every vector is normalised once at creation and the index uses
#: plain inner product. Normalising at query time instead would be a silent
#: correctness bug the first time someone forgot.
_EPS = 1e-12


def normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise row-wise so inner product equals cosine similarity.

    Zero vectors would divide by zero; they are left untouched instead, which
    makes them orthogonal to everything rather than NaN. A NaN in the index
    poisons every subsequent search with no error message.
    """
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, _EPS)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def load_encoder(model_name: str) -> Any:
    """Load the sentence-transformer, importing it lazily.

    The import costs seconds and pulls in torch, so it happens here rather than
    at module import. Layers 1-4 must keep working -- and their tests must keep
    running -- on a machine that has never installed torch.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "sentence-transformers is not installed. Phase 4 needs it:\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "    pip install sentence-transformers faiss-cpu"
        ) from exc

    logger.info("Loading embedding model %s", model_name)
    return SentenceTransformer(model_name)


def encode_texts(
    texts: Sequence[str],
    model_name: str,
    batch_size: int = 64,
    encoder: Any | None = None,
) -> np.ndarray:
    """Embed texts and return normalised float32 vectors.

    ``encoder`` is injectable so tests can exercise the surrounding logic --
    caching, ordering, normalisation -- without downloading a model.
    """
    if not len(texts):
        return np.zeros((0, 0), dtype="float32")

    encoder = encoder or load_encoder(model_name)
    vectors = encoder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return normalise(vectors)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class EmbeddingStore:
    """Disk cache of review vectors, keyed by review_id.

    Mirrors ``EnrichmentCache``: load what exists, compute only what is
    missing, write atomically. The model name is stored alongside the vectors
    so a cache built by one model is never served to another -- two embedding
    models produce incomparable spaces, and mixing them yields clusters that
    look plausible and mean nothing.
    """

    def __init__(self, path: Path, model_name: str) -> None:
        self.path = path
        self.model_name = model_name
        self._ids: list[str] = []
        self._vectors: np.ndarray = np.zeros((0, 0), dtype="float32")

        if path.exists():
            payload = np.load(path, allow_pickle=False)
            stored_model = str(payload["model"].item())
            if stored_model != model_name:
                logger.warning(
                    "Embedding cache at %s was built by %r, not %r; ignoring it.",
                    path.name, stored_model, model_name,
                )
            else:
                self._ids = [str(x) for x in payload["ids"]]
                self._vectors = payload["vectors"].astype("float32")
                logger.info("Loaded %d cached embeddings from %s", len(self._ids), path.name)

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def index_of(self) -> dict[str, int]:
        return {review_id: i for i, review_id in enumerate(self._ids)}

    def missing(self, review_ids: Iterable[str]) -> list[str]:
        known = set(self._ids)
        # dict.fromkeys preserves order while de-duplicating, so the encode
        # batch is deterministic run to run.
        return list(dict.fromkeys(r for r in review_ids if r not in known))

    def add(self, review_ids: Sequence[str], vectors: np.ndarray) -> None:
        if not len(review_ids):
            return
        vectors = np.asarray(vectors, dtype="float32")
        if len(review_ids) != len(vectors):
            raise ValueError(
                f"{len(review_ids)} ids but {len(vectors)} vectors -- refusing to "
                "misalign the cache"
            )
        if len(self._vectors):
            self._vectors = np.vstack([self._vectors, vectors])
        else:
            self._vectors = vectors
        self._ids.extend(review_ids)

    def vectors_for(self, review_ids: Sequence[str]) -> np.ndarray:
        """Return vectors in the caller's order, not storage order.

        Callers align vectors against a DataFrame. Returning them in insertion
        order would attach each review to a neighbour's embedding -- a silent
        corruption that produces clusters rather than an exception.
        """
        position = self.index_of
        missing = [r for r in review_ids if r not in position]
        if missing:
            raise KeyError(f"{len(missing)} review(s) not in the store, e.g. {missing[:3]}")
        return self._vectors[[position[r] for r in review_ids]]

    def save(self) -> None:
        """Write atomically -- a truncated .npz is unreadable, not partial."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        # Write through an open handle, not a path. Given a path, np.savez
        # appends ".npz" if it is not already the extension -- so the temp file
        # lands at "<name>.npz.tmp.npz" and the rename below targets something
        # that was never written. A file object suppresses that rewriting.
        with open(temp, "wb") as handle:
            np.savez(
                handle,
                ids=np.array(self._ids, dtype=object).astype(str),
                vectors=self._vectors,
                model=np.array(self.model_name),
            )
        temp.replace(self.path)
        logger.info("Wrote %d embeddings to %s", len(self._ids), self.path.name)


def embed_reviews(
    frame: pd.DataFrame,
    model_name: str,
    store: EmbeddingStore | None = None,
    batch_size: int = 64,
    encoder: Any | None = None,
) -> np.ndarray:
    """Embed every review in ``frame``, reusing whatever the store already has.

    Returns vectors aligned to ``frame`` row order.
    """
    review_ids = frame["review_id"].tolist()

    if store is None:
        return encode_texts(frame["review_text"].tolist(), model_name, batch_size, encoder)

    outstanding = store.missing(review_ids)
    if outstanding:
        logger.info("Embedding %d new review(s); %d reused", len(outstanding), len(store))
        pending = frame[frame["review_id"].isin(outstanding)]
        # Reindex to `outstanding` order so ids and vectors cannot drift apart.
        pending = pending.set_index("review_id").loc[outstanding].reset_index()
        vectors = encode_texts(
            pending["review_text"].tolist(), model_name, batch_size, encoder
        )
        store.add(outstanding, vectors)
    else:
        logger.info("All %d embeddings served from cache", len(review_ids))

    return store.vectors_for(review_ids)


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------


def build_index(vectors: np.ndarray) -> Any:
    """Build a flat inner-product index over normalised vectors.

    Flat means exhaustive search: exact results, no training step, no recall
    cliff. At 4,620 vectors an approximate index would save microseconds and
    cost correctness -- IVF/HNSW start paying off around a million rows.
    """
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("faiss-cpu is not installed. Run: pip install faiss-cpu") from exc

    vectors = np.ascontiguousarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index: Any, path: Path) -> None:
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info("Wrote FAISS index (%d vectors) to %s", index.ntotal, path.name)


def load_index(path: Path) -> Any:
    import faiss

    return faiss.read_index(str(path))


def search(index: Any, queries: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, positions) for each query row.

    Scores are cosine similarities because the vectors are normalised.
    """
    queries = np.ascontiguousarray(normalise(np.atleast_2d(queries)), dtype="float32")
    return index.search(queries, k)
