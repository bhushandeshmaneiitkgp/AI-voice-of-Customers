"""
Build the cleaned dataset from the immutable raw CSV.

Run:
    python scripts/01_build_clean.py

Reads  : data/raw/reviews.csv            (read-only)
Writes : data/interim/reviews_clean.parquet
         data/interim/cleaning_report.json

Overrides (useful when tuning, no code edit required):
    python scripts/01_build_clean.py --near-dup-threshold 0.85
    python scripts/01_build_clean.py --limit 500        # fast dev loop
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- must precede project imports
import argparse
import json
import logging
import sys

from config.settings import Paths, get_settings
from voc.clean import build_clean_dataset
from voc.ingest import ingest


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--near-dup-threshold",
        type=float,
        default=settings.near_dup_threshold,
        help="Cosine similarity at/above which reviews are near-duplicates.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=settings.sample_limit,
        help="Process only the first N rows (0 = all). For fast iteration.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(levelname)-8s %(name)s | %(message)s",
    )
    log = logging.getLogger("clean")

    Paths.ensure_output_dirs()

    # ---- Layer 1: ingest --------------------------------------------------
    raw, ingest_report = ingest(Paths.raw_reviews)
    if args.limit and args.limit > 0:
        log.warning("SAMPLE MODE: limiting to first %d rows", args.limit)
        raw = raw.head(args.limit)

    # ---- Layer 2: clean ---------------------------------------------------
    clean, clean_report = build_clean_dataset(
        raw,
        truncation_cap_chars=settings.truncation_cap_chars,
        truncation_tolerance=settings.truncation_tolerance,
        near_dup_threshold=args.near_dup_threshold,
        min_review_chars=settings.min_review_chars,
    )

    # ---- Persist ----------------------------------------------------------
    clean.to_parquet(Paths.clean_reviews, index=False, compression="snappy")

    combined_report = {
        "ingest": ingest_report.model_dump(),
        "cleaning": clean_report.model_dump(),
    }
    Paths.clean_report.write_text(
        json.dumps(combined_report, indent=2, default=str), encoding="utf-8"
    )

    # ---- Summary ----------------------------------------------------------
    size_kb = Paths.clean_reviews.stat().st_size / 1024
    print()
    print("=" * 66)
    print("  CLEANING COMPLETE")
    print("=" * 66)
    print(f"  Rows in / out         : {clean_report.rows_in:,} -> {clean_report.rows_out:,}")
    print(f"  Rows dropped          : {clean_report.rows_dropped:,} {clean_report.drop_reasons or ''}")
    print(f"  Whitespace normalised : {clean_report.whitespace_normalised:,}")
    print(f"  Truncated (flagged)   : {clean_report.truncated_reviews:,}")
    print(f"  Near-dup groups       : {clean_report.near_dup_groups:,} "
          f"covering {clean_report.near_dup_members:,} reviews "
          f"(largest group: {clean_report.largest_near_dup_group})")
    print(f"  Date range            : {clean_report.date_min} -> {clean_report.date_max}")
    print(f"  In comparable window  : {clean_report.rows_in_comparable_window:,}")
    print(f"  Platforms             : {clean_report.platform_counts}")
    print("-" * 66)
    print(f"  Parquet : {Paths.clean_reviews}  ({size_kb:,.0f} KB)")
    print(f"  Report  : {Paths.clean_report}")
    print("=" * 66)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
