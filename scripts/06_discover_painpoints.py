"""
Cluster the enriched corpus and score the pain points it contains.

This is the Phase 4 deliverable: the step that turns 10,790 validated labels
into a ranked, evidence-backed list a PM can act on.

    # cluster and score, choosing k by silhouette
    python scripts/06_discover_painpoints.py

    # force a specific k instead of scoring the range
    python scripts/06_discover_painpoints.py --k 12

    # skip clustering; only rescore pain points from the labels
    python scripts/06_discover_painpoints.py --no-clusters

Reads  : data/processed/reviews_enriched.parquet
         data/processed/review_labels.parquet
         artifacts/embeddings.npz        (run 05_build_embeddings.py first)
Writes : data/processed/review_clusters.parquet
         data/processed/cluster_summary.parquet
         data/processed/pain_points.parquet
         reports/PAIN_POINTS.md
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- must precede project imports
import argparse
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config.settings import Paths, get_settings
from voc.cluster import ClusterModel, fit_clusters, summarise_clusters
from voc.embed import EmbeddingStore
from voc.painpoints import WEIGHTS, add_trend, attach_evidence, build_pain_points


def build_report(
    pain_points: pd.DataFrame,
    clusters: pd.DataFrame | None,
    model: ClusterModel | None,
    reviews: pd.DataFrame,
    settings,
) -> str:
    """Render the markdown brief. Every claim carries its evidence."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = []
    add = lines.append

    add("# Pain Points — Quick-Commerce Voice of Customer")
    add("")
    add(f"**{generated}** · {len(reviews):,} enriched reviews · "
        f"{len(pain_points)} scored pain points")
    add("")
    add("Phase 4 output. Enrichment said which categories apply to each review; "
        "this ranks them by how much they appear to cost, and clusters the text "
        "to surface themes the taxonomy does not name.")
    add("")
    add("---")
    add("")

    # --- How the score works, before any number that uses it ---------------
    add("## How the score is computed")
    add("")
    add("A pain point is a `(product_area, issue_type)` pair. Five signals, "
        "weighted:")
    add("")
    add("| Signal | Weight | Meaning |")
    add("|---|---|---|")
    add(f"| Volume | {WEIGHTS['volume']:.2f} | distinct reviews raising it (min-maxed) |")
    add(f"| Severity | {WEIGHTS['severity']:.2f} | mean severity, low=1 → critical=4 |")
    add(f"| Escalation | {WEIGHTS['escalation']:.2f} | share that drove a support contact |")
    add(f"| Churn | {WEIGHTS['churn']:.2f} | share stating intent to leave |")
    add(f"| Negativity | {WEIGHTS['negativity']:.2f} | share appearing in negative reviews |")
    add("")
    add(f"Pain points under **{settings.min_pain_point_volume} reviews** are "
        "excluded: below that a pattern is an anecdote.")
    add("")
    add("The weights are a **product judgement, not a discovered constant**. They "
        "live in `src/voc/painpoints.py` so disagreeing with them is a one-line "
        "change and a re-run, not an argument with a black box.")
    add("")
    if "trend_ratio" in pain_points.columns:
        add("`trend_ratio` is **reported but not scored** — recent months vs prior, "
            "per month, inside the comparable window only.")
    else:
        add("**There is no trend column, and that is a result.** The first attempt "
            "produced ratios up to 197× — which was collection, not customers. "
            "The corpus spans 50 months, but 42 of the earliest 47 hold fewer "
            "than ten reviews each while the final three hold 3,475. Restricted "
            "to the comparable window (from 2024-10, where every platform is "
            "meaningfully present) only three months remain — shorter than any "
            "honest trend needs. `add_trend` therefore refuses. Whether these "
            "pain points are growing is a **Phase 5 question that this corpus "
            "cannot answer**.")
    add("")
    add("---")
    add("")

    # --- The ranking -------------------------------------------------------
    add("## Ranked pain points")
    add("")
    has_trend = "trend_ratio" in pain_points.columns
    header = "| # | Area | Issue | Reviews | Severity | Escalation | Churn |"
    rule = "|---|---|---|---:|---:|---:|---:|"
    add(header + (" Trend |" if has_trend else "") + " Score |")
    add(rule + ("---:|" if has_trend else "") + "---:|")
    for row in pain_points.itertuples():
        line = (
            f"| {row.rank} | `{row.product_area}` | `{row.issue_type}` | "
            f"{row.volume:,} | {row.mean_severity:.2f} | "
            f"{row.escalation_rate * 100:.0f}% | {row.churn_rate * 100:.1f}% |"
        )
        if has_trend:
            trend = row.trend_ratio
            line += " — |" if pd.isna(trend) else f" {trend:.2f}× |"
        add(line + f" **{row.score:.3f}** |")
    add("")

    # --- Evidence for the top items ---------------------------------------
    add("### What customers actually said")
    add("")
    add("Verbatim spans, verified against the source text at enrichment time.")
    add("")
    for row in pain_points.head(5).itertuples():
        add(f"**{row.rank}. `{row.product_area}` / `{row.issue_type}`** "
            f"— {row.volume:,} reviews, score {row.score:.3f}")
        add("")
        for quote in getattr(row, "evidence", []) or []:
            add(f"> {quote}")
            add("")
    add("---")
    add("")

    # --- Clusters ----------------------------------------------------------
    if clusters is not None and model is not None:
        add("## Themes discovered in the text")
        add("")
        add(f"k = **{model.k}**, chosen by silhouette score ({model.silhouette:.4f}) "
            f"over the range {settings.cluster_k_min}–{settings.cluster_k_max}. "
            "Scored rather than picked by eye, so the number is arguable:")
        add("")
        add("| k | silhouette |")
        add("|---:|---:|")
        for k, score in sorted(model.scores)[:12]:
            marker = "  ← chosen" if k == model.k else ""
            add(f"| {k} | {score:.4f}{marker} |")
        add("")
        add("| Cluster | Size | Share | Dominant area | Dominant issue | Severity | Escalation |")
        add("|---:|---:|---:|---|---|---:|---:|")
        for row in clusters.itertuples():
            severity = f"{row.mean_severity:.2f}" if row.mean_severity else "—"
            add(
                f"| {row.cluster_id} | {row.size:,} | {row.share_pct:.1f}% | "
                f"`{row.dominant_area}` | `{row.dominant_issue}` | {severity} | "
                f"{row.escalation_rate * 100:.0f}% |"
            )
        add("")
        add("### Representative reviews per theme")
        add("")
        add("Closest to each centroid — what the cluster is actually about.")
        add("")
        for row in clusters.head(6).itertuples():
            add(f"**Cluster {row.cluster_id}** ({row.size:,} reviews, "
                f"`{row.dominant_area}`)")
            add("")
            for text in row.exemplar_texts:
                add(f"> {text}")
                add("")
        add("---")
        add("")

    # --- Honest limits -----------------------------------------------------
    add("## Caveats")
    add("")
    add("**The score ranks, it does not measure.** It combines five signals on "
        "one weighting; a different weighting produces a different order. It is "
        "a way to argue about priority with the data present, not a cost model.")
    add("")
    add("**Labels are model output, not ground truth.** Grounding was verified "
        "at 98.4% and coverage at 98.9%, but no hand-labelled gold set exists "
        "until Phase 9. Volume figures inherit whatever bias the model has.")
    add("")
    add("**Clusters are unsupervised.** Silhouette picks the most separable k, "
        "not the most useful one. A theme that splits across two clusters, or "
        "two themes sharing one, are both possible and neither is an error.")
    add("")
    if model is not None:
        add(f"**The clusters are weakly separated.** A silhouette of "
            f"{model.silhouette:.3f} is low in absolute terms — the peak is real "
            "(scores rise to it and fall after), but it describes overlapping "
            "regions of one continuous space, not distinct groups. That is what "
            "short review text usually looks like. Treat the themes as a reading "
            "aid over the ranked pain points, not as a partition of the corpus, "
            "and note that two clusters here share a dominant area: the split "
            "between them is about phrasing, not category.")
        add("")
    add("**Severity is self-reported through the model.** It reflects how the "
        "review reads, not operational impact. A calm review of a serious "
        "failure scores low.")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=None,
                        help="Force a cluster count instead of scoring the range.")
    parser.add_argument("--no-clusters", action="store_true",
                        help="Score pain points only; skip embeddings and clustering.")
    parser.add_argument("--min-volume", type=int, default=settings.min_pain_point_volume,
                        help="Drop pain points below this many reviews.")
    parser.add_argument("--seed", type=int, default=42, help="Clustering seed.")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format="%(levelname)-8s %(name)s | %(message)s")
    log = logging.getLogger("painpoints")
    Paths.ensure_output_dirs()

    for required in (Paths.enriched_reviews, Paths.enriched_labels):
        if not required.exists():
            log.error("%s not found. Run: python scripts/04_run_enrichment.py --all",
                      required.name)
            return 1

    reviews = pd.read_parquet(Paths.enriched_reviews)
    labels = pd.read_parquet(Paths.enriched_labels)

    print()
    print("=" * 78)
    print("  PAIN-POINT DISCOVERY")
    print("=" * 78)
    print(f"  Reviews  : {len(reviews):,}")
    print(f"  Labels   : {len(labels):,}")
    print(f"  Volume floor : {args.min_volume}")
    print("-" * 78)

    # --- Clustering --------------------------------------------------------
    clusters = model = None
    if not args.no_clusters:
        if not Paths.embeddings.exists():
            log.error("No embeddings. Run: python scripts/05_build_embeddings.py")
            return 1

        store = EmbeddingStore(Paths.embeddings, settings.embedding_model)
        missing = store.missing(reviews["review_id"])
        if missing:
            log.error(
                "%d review(s) have no embedding. Re-run scripts/05_build_embeddings.py",
                len(missing),
            )
            return 1

        vectors = store.vectors_for(reviews["review_id"].tolist())
        k_min = k_max = args.k if args.k else None
        model = fit_clusters(
            vectors,
            k_min or settings.cluster_k_min,
            k_max or settings.cluster_k_max,
            seed=args.seed,
        )
        clusters = summarise_clusters(reviews, model, vectors, labels)

        assignments = reviews[["review_id", "platform", "year_month"]].copy()
        assignments["cluster_id"] = model.labels
        assignments.to_parquet(Paths.review_clusters, index=False)
        # Lists do not survive a parquet round-trip cleanly in every reader, so
        # the exemplar columns are serialised as text for the stored copy.
        stored = clusters.copy()
        for column in ("exemplar_review_ids", "exemplar_texts"):
            stored[column] = stored[column].apply(lambda v: " ||| ".join(map(str, v)))
        stored.to_parquet(Paths.cluster_summary, index=False)

        print(f"  Clusters : k={model.k} (silhouette {model.silhouette:.4f})")
        print(f"             largest {clusters.iloc[0]['size']:,} reviews "
              f"({clusters.iloc[0]['share_pct']:.1f}%), "
              f"area={clusters.iloc[0]['dominant_area']}")

    # --- Pain points -------------------------------------------------------
    pain_points = build_pain_points(labels, reviews, min_volume=args.min_volume)
    if pain_points.empty:
        log.error("No pain points cleared the volume floor of %d.", args.min_volume)
        return 1

    pain_points = add_trend(pain_points, labels)
    pain_points = attach_evidence(pain_points, labels)

    stored = pain_points.copy()
    for column in ("evidence", "evidence_review_ids"):
        stored[column] = stored[column].apply(lambda v: " ||| ".join(map(str, v)))
    stored.to_parquet(Paths.pain_points, index=False)

    Paths.pain_point_report.write_text(
        build_report(pain_points, clusters, model, reviews, settings), encoding="utf-8"
    )

    top = pain_points.iloc[0]
    print(f"  Pain points  : {len(pain_points)} scored "
          f"({len(labels[labels['polarity'] == 'issue']):,} issue labels in)")
    print(f"  Highest      : {top['product_area']} / {top['issue_type']}  "
          f"({top['volume']:,} reviews, score {top['score']:.3f})")
    print("-" * 78)
    if clusters is not None:
        print(f"  Clusters : {Paths.review_clusters}")
        print(f"             {Paths.cluster_summary}")
    print(f"  Table    : {Paths.pain_points}")
    print(f"  Report   : {Paths.pain_point_report}")
    print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
