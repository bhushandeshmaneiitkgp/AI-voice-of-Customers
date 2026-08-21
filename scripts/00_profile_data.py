"""
Profile the immutable raw dataset and write reports/data_profile.md.

Run:
    python scripts/00_profile_data.py

Reads  : data/raw/reviews.csv   (read-only)
Writes : reports/data_profile.md

This script is the reproducible replacement for exploratory notebook cells.
Any figure quoted in the README should come from a run of this script.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  -- must precede project imports
import argparse
import logging
import sys

from config.settings import Paths, get_settings
from voc.ingest import ingest
from voc.profiling import build_profile_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Paths.data_profile),
        help="Where to write the Markdown report.",
    )
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(levelname)-8s %(name)s | %(message)s",
    )
    log = logging.getLogger("profile")

    Paths.ensure_output_dirs()

    log.info("Reading immutable source: %s", Paths.raw_reviews)
    raw, ingest_report = ingest(Paths.raw_reviews)

    if ingest_report.rows_rejected:
        log.warning(
            "%d row(s) rejected during ingestion; profiling the %d valid rows.",
            ingest_report.rows_rejected,
            ingest_report.rows_valid,
        )

    report_markdown = build_profile_report(
        raw,
        source_path=str(Paths.raw_reviews.relative_to(Paths.root)),
        checksum=ingest_report.source_sha256,
        source_bytes=ingest_report.source_bytes,
        truncation_cap_chars=settings.truncation_cap_chars,
        truncation_tolerance=settings.truncation_tolerance,
    )

    output_path = Paths.reports_dir / "data_profile.md"
    output_path.write_text(report_markdown, encoding="utf-8")

    log.info("Wrote profile: %s (%d lines)", output_path, report_markdown.count("\n"))
    print()
    print(f"  Records profiled : {ingest_report.rows_valid:,}")
    print(f"  Source SHA-256   : {ingest_report.source_sha256}")
    print(f"  Report           : {output_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
