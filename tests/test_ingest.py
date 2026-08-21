"""Tests for the ingestion layer, including the raw-data immutability guarantee."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from config.settings import Paths
from voc.ingest import (
    EXPECTED_COLUMNS,
    KNOWN_SOURCE_SHA256,
    compute_sha256,
    ingest,
    load_raw_reviews,
    validate_raw_rows,
    verify_raw_integrity,
)

CSV_HEADER = "rating,date,review,platform\n"


@pytest.fixture()
def tiny_csv(tmp_path: Path) -> Path:
    """A small well-formed CSV, including a multi-line quoted review field."""
    path = tmp_path / "reviews.csv"
    path.write_text(
        CSV_HEADER
        + '1,30 December 2024,"delivery late\n    and no support",zepto\n'
        + "5,15 November 2024,fast and fresh,blinkit\n"
        + "3,21 July 2020,average experience overall,jiomart\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# checksums
# ---------------------------------------------------------------------------


def test_compute_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    payload = b"voice of customer" * 1000
    path.write_bytes(payload)

    assert compute_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_verify_raw_integrity_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_raw_integrity(tmp_path / "nope.csv")


def test_verify_raw_integrity_warns_on_drift(tiny_csv: Path, caplog) -> None:
    """A changed source file must never pass unnoticed."""
    with caplog.at_level("WARNING"):
        checksum = verify_raw_integrity(tiny_csv, expected_sha256="0" * 64)

    assert checksum == compute_sha256(tiny_csv)
    assert "checksum changed" in caplog.text.lower()


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_load_raw_reviews_reads_all_columns_as_strings(tiny_csv: Path) -> None:
    frame, checksum = load_raw_reviews(tiny_csv)

    assert set(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 3
    assert len(checksum) == 64
    # Types must stay str so pandas cannot coerce anything behind our back.
    assert all(frame[column].map(type).eq(str).all() for column in frame.columns)


def test_load_raw_reviews_preserves_multiline_fields(tiny_csv: Path) -> None:
    """Quoted newlines are part of a field, not a record separator."""
    frame, _ = load_raw_reviews(tiny_csv)
    assert "\n" in frame.loc[0, "review"]


def test_load_raw_reviews_rejects_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("rating,date,review\n1,30 December 2024,text\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required column"):
        load_raw_reviews(path)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_validate_raw_rows_separates_good_from_bad(tmp_path: Path) -> None:
    path = tmp_path / "mixed.csv"
    path.write_text(
        CSV_HEADER
        + "1,30 December 2024,valid review text,zepto\n"
        + "9,30 December 2024,rating out of range,zepto\n"
        + "2,30 December 2024,unknown platform,instamart\n",
        encoding="utf-8",
    )
    frame, _ = load_raw_reviews(path)

    valid, problems = validate_raw_rows(frame)

    assert len(valid) == 1
    assert len(problems) == 2
    # Problems must name the row and the field, so they are actionable.
    assert all("row " in message for message in problems)


def test_ingest_reports_provenance(tiny_csv: Path) -> None:
    frame, report = ingest(tiny_csv)

    assert len(frame) == 3
    assert report.rows_read == 3
    assert report.rows_valid == 3
    assert report.rows_rejected == 0
    assert report.source_sha256 == compute_sha256(tiny_csv)
    assert report.source_bytes == tiny_csv.stat().st_size
    assert report.loaded_at_utc.endswith("+00:00")


# ---------------------------------------------------------------------------
# Immutability guarantee (runs against the real dataset)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Paths.raw_reviews.exists(), reason="raw dataset not present in this checkout"
)
def test_raw_dataset_is_unmodified() -> None:
    """The raw file must still be byte-identical to what the project was built on.

    This is the executable form of the immutability rule: if any pipeline run
    ever writes to data/raw/, this test fails on the next run.
    """
    assert compute_sha256(Paths.raw_reviews) == KNOWN_SOURCE_SHA256


@pytest.mark.skipif(
    not Paths.raw_reviews.exists(), reason="raw dataset not present in this checkout"
)
def test_real_dataset_passes_validation() -> None:
    """Every row of the actual source file satisfies the raw contract."""
    frame, report = ingest(Paths.raw_reviews)

    assert report.rows_read == 4620
    assert report.rows_rejected == 0
    assert len(frame) == 4620
