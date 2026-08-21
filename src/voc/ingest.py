"""
Layer 1 -- Data ingestion.

Single job: read the immutable raw CSV, prove which bytes were read, validate
that every row matches the contract, and hand a DataFrame to the cleaning layer.

The immutability guarantee is enforced structurally, not by convention:
this module opens the raw file in binary/read mode only, and nothing else in
the codebase references ``Paths.raw_reviews`` for writing. ``verify_raw_integrity``
lets any later run assert the source has not drifted since it was profiled.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from voc.schemas import RawReviewRow, IngestReport

logger = logging.getLogger(__name__)

#: Columns the raw file must contain. Order does not matter; presence does.
EXPECTED_COLUMNS: set[str] = {"rating", "date", "review", "platform"}

#: Checksum of the dataset this pipeline was built and profiled against.
#: If it changes, the profiling report and any gold labels may no longer apply.
KNOWN_SOURCE_SHA256: str = (
    "5d3bb24ee7185612e66f57cf2e21d99e9d628a15d3d68989b8cf560996e9f138"
)


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256. Read-only by construction."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_integrity(path: Path, expected_sha256: str | None = None) -> str:
    """Confirm the raw file is present and unchanged.

    Returns the checksum. Warns rather than raises on mismatch: a deliberately
    updated dataset is a legitimate event, but it must never pass unnoticed.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Phase 1 expects the immutable source at data/raw/reviews.csv."
        )

    actual = compute_sha256(path)
    expected = expected_sha256 or KNOWN_SOURCE_SHA256
    if expected and actual != expected:
        logger.warning(
            "Raw dataset checksum changed.\n"
            "  expected: %s\n"
            "  actual  : %s\n"
            "The profiling report, gold labels, and any cached enrichment were "
            "produced against the previous file. Re-run profiling before trusting "
            "downstream outputs.",
            expected,
            actual,
        )
    return actual


def load_raw_reviews(path: Path) -> tuple[pd.DataFrame, str]:
    """Read the raw CSV into a DataFrame of plain strings.

    Everything is read as ``str`` with NA conversion disabled, so pandas cannot
    coerce an empty review into ``NaN`` or a rating into a float behind our
    back. Type conversion is the cleaning layer's job, done explicitly.
    """
    checksum = verify_raw_integrity(path)

    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,  # do not invent NaN from empty strings
        na_filter=False,
        encoding="utf-8",
    )

    missing = EXPECTED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Raw file is missing required column(s): {sorted(missing)}. "
            f"Found: {list(frame.columns)}"
        )

    logger.info("Read %d rows x %d columns from %s", len(frame), frame.shape[1], path.name)
    return frame, checksum


def validate_raw_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate every row against ``RawReviewRow``.

    Returns the valid subset plus human-readable messages for anything rejected.
    Rejected rows are reported, never silently dropped -- a pipeline that hides
    its losses is a pipeline whose numbers cannot be defended.
    """
    valid_indices: list[int] = []
    problems: list[str] = []

    for index, row in frame.iterrows():
        try:
            RawReviewRow(
                rating=row["rating"],
                date=row["date"],
                review=row["review"],
                platform=row["platform"],
            )
            valid_indices.append(index)
        except ValidationError as exc:
            first_error = exc.errors()[0]
            field = ".".join(str(part) for part in first_error["loc"])
            problems.append(f"row {index}: {field}: {first_error['msg']}")

    if problems:
        logger.warning("%d row(s) failed raw validation", len(problems))
        for message in problems[:10]:
            logger.warning("  %s", message)

    return frame.loc[valid_indices].copy(), problems


def ingest(path: Path) -> tuple[pd.DataFrame, IngestReport]:
    """Full ingestion: read, checksum, validate, report."""
    frame, checksum = load_raw_reviews(path)
    rows_read = len(frame)

    valid, problems = validate_raw_rows(frame)

    report = IngestReport(
        source_path=str(path),
        source_sha256=checksum,
        source_bytes=path.stat().st_size,
        rows_read=rows_read,
        rows_valid=len(valid),
        rows_rejected=rows_read - len(valid),
        rejection_examples=problems[:20],
        loaded_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    logger.info(
        "Ingested %d/%d rows (%d rejected) | sha256=%s",
        report.rows_valid,
        report.rows_read,
        report.rows_rejected,
        checksum[:12],
    )
    return valid, report
