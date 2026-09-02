"""
Build sentence embeddings and the FAISS index over the enriched corpus.

Runs locally on CPU and costs nothing -- no API key required. The first run
downloads the model (~90MB) from HuggingFace; later runs are offline.

    # embed every enriched review and build the index
    python scripts/05_build_embeddings.py

    # a different embedding model, without touching code
    VOC_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2 \
        python scripts/05_build_embeddings.py

    # recompute from scratch instead of reusing the vector cache
    python scripts/05_build_embeddings.py --no-cache

Reads  : data/processed/reviews_enriched.parquet
Writes : artifacts/embeddings.npz      (vector cache, keyed by review_id)
         artifacts/reviews.faiss       (index for Phase 6 retrieval)
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- must precede project imports
import argparse
import logging
import sys
import time

import pandas as pd

from config.settings import Paths, get_settings
from voc.embed import EmbeddingStore, build_index, embed_reviews, save_index


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore cached vectors and re-encode every review. The cache file "
             "is still rewritten; nothing is deleted.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=settings.embedding_batch_size,
        help="Encoder batch size. Lower it if memory is tight.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format="%(levelname)-8s %(name)s | %(message)s")
    log = logging.getLogger("embed")
    Paths.ensure_output_dirs()

    if not Paths.enriched_reviews.exists():
        log.error("Enriched corpus not found. Run: python scripts/04_run_enrichment.py --all")
        return 1

    frame = pd.read_parquet(Paths.enriched_reviews)
    model_name = settings.embedding_model

    print()
    print("=" * 78)
    print("  EMBEDDINGS")
    print("=" * 78)
    print(f"  Model      : {model_name}")
    print(f"  Reviews    : {len(frame):,}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Cache      : " + ("BYPASSED (--no-cache)" if args.no_cache
                                else f"read-through -> {Paths.embeddings.name}"))
    print("-" * 78)

    # A bypass is a read bypass, matching the enrichment cache: existing
    # vectors are still rewritten, never discarded, so an aborted run costs
    # only the work in flight.
    store = None if args.no_cache else EmbeddingStore(Paths.embeddings, model_name)
    if store is not None:
        print(f"  Cached     : {len(store):,} vector(s) already on disk")

    started = time.perf_counter()
    try:
        vectors = embed_reviews(
            frame, model_name, store=store, batch_size=args.batch_size
        )
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    elapsed = time.perf_counter() - started

    if store is None:
        # --no-cache still persists, so the next run is fast again.
        store = EmbeddingStore(Paths.embeddings, model_name)
        store.add(frame["review_id"].tolist(), vectors)
    store.save()

    index = build_index(vectors)
    save_index(index, Paths.faiss_index)

    print(f"  Encoded    : {vectors.shape[0]:,} x {vectors.shape[1]} dims in {elapsed:.1f}s")
    print("-" * 78)
    print(f"  Vectors : {Paths.embeddings}")
    print(f"  Index   : {Paths.faiss_index}  ({index.ntotal:,} vectors)")
    print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
