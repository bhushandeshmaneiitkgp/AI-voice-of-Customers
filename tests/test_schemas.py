"""Tests for the data contracts.

These are the boundaries that make bad assumptions fail loudly instead of
propagating into a chart a PM would then act on.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from voc.schemas import (
    COMPARABLE_WINDOW_START,
    PLATFORMS,
    CleanReview,
    RatingBucket,
    RawReviewRow,
)


# ---------------------------------------------------------------------------
# RatingBucket
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        (1, RatingBucket.NEGATIVE),
        (2, RatingBucket.NEGATIVE),
        (3, RatingBucket.NEUTRAL),
        (4, RatingBucket.POSITIVE),
        (5, RatingBucket.POSITIVE),
    ],
)
def test_rating_bucket_boundaries(rating: int, expected: RatingBucket) -> None:
    assert RatingBucket.from_rating(rating) is expected


# ---------------------------------------------------------------------------
# RawReviewRow
# ---------------------------------------------------------------------------


def test_raw_row_accepts_valid_input() -> None:
    row = RawReviewRow(
        rating=1, date="30 December 2024", review="wallet money vanished", platform="zepto"
    )
    assert row.rating == 1
    assert row.platform == "zepto"


def test_raw_row_normalises_platform_case() -> None:
    row = RawReviewRow(
        rating=5, date="1 July 2024", review="good app", platform="  BlinkIt  "
    )
    assert row.platform == "blinkit"


@pytest.mark.parametrize("rating", [0, 6, -1, 100])
def test_raw_row_rejects_out_of_range_rating(rating: int) -> None:
    with pytest.raises(ValidationError):
        RawReviewRow(
            rating=rating, date="30 December 2024", review="text", platform="zepto"
        )


def test_raw_row_rejects_unknown_platform() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RawReviewRow(
            rating=3, date="30 December 2024", review="text", platform="swiggy_instamart"
        )
    # The message must tell a future maintainer what to do, not just that it failed.
    assert "PLATFORMS" in str(excinfo.value)


@pytest.mark.parametrize("field", ["date", "review", "platform"])
def test_raw_row_rejects_empty_strings(field: str) -> None:
    payload = {
        "rating": 1,
        "date": "30 December 2024",
        "review": "text",
        "platform": "zepto",
    }
    payload[field] = ""
    with pytest.raises(ValidationError):
        RawReviewRow(**payload)


def test_raw_row_is_immutable() -> None:
    row = RawReviewRow(rating=1, date="1 July 2024", review="t", platform="zepto")
    with pytest.raises(ValidationError):
        row.rating = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CleanReview
# ---------------------------------------------------------------------------


def _clean_payload(**overrides):
    payload = {
        "review_id": "a1b2c3d4e5f60718",
        "source_row_index": 0,
        "platform": "zepto",
        "rating": 1,
        "rating_bucket": RatingBucket.NEGATIVE,
        "review_date": date(2024, 12, 30),
        "year": 2024,
        "month": 12,
        "year_month": "2024-12",
        "review_raw": "raw\n text",
        "review_text": "raw text",
        "char_len": 8,
        "word_count": 2,
        "is_truncated": False,
        "ends_without_terminal_punct": True,
        "has_non_latin": False,
        "near_dup_group_id": -1,
        "near_dup_group_size": 1,
        "is_near_dup_representative": True,
        "in_comparable_window": True,
    }
    payload.update(overrides)
    return payload


def test_clean_review_accepts_valid_payload() -> None:
    review = CleanReview(**_clean_payload())
    assert review.review_id == "a1b2c3d4e5f60718"
    assert review.rating_bucket is RatingBucket.NEGATIVE


@pytest.mark.parametrize("bad_year_month", ["2024-1", "202412", "2024/12", "Dec-2024"])
def test_clean_review_enforces_year_month_pattern(bad_year_month: str) -> None:
    with pytest.raises(ValidationError):
        CleanReview(**_clean_payload(year_month=bad_year_month))


def test_clean_review_rejects_unknown_platform() -> None:
    with pytest.raises(ValidationError):
        CleanReview(**_clean_payload(platform="instamart"))


def test_clean_review_rejects_short_review_id() -> None:
    with pytest.raises(ValidationError):
        CleanReview(**_clean_payload(review_id="tooshort"))


def test_clean_review_rejects_group_size_below_one() -> None:
    with pytest.raises(ValidationError):
        CleanReview(**_clean_payload(near_dup_group_size=0))


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_platform_constant_matches_dataset() -> None:
    assert set(PLATFORMS) == {"blinkit", "jiomart", "zepto"}


def test_comparable_window_start_is_documented_value() -> None:
    """Derived from profiling: before this date Blinkit has 12 rows, Zepto 1."""
    assert COMPARABLE_WINDOW_START == date(2024, 10, 1)
