"""Tests for the cleaning layer.

Each transform is a pure function, which is why it can be tested directly.
The invariant these tests protect is the one the whole project rests on:
cleaning **flags** problems, it does not silently delete rows.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from voc.clean import (
    build_clean_dataset,
    detect_truncation,
    ends_without_terminal_punct,
    find_near_duplicates,
    has_non_latin,
    make_review_id,
    normalize_whitespace,
    parse_review_date,
)

CLEAN_KWARGS = dict(
    truncation_cap_chars=500,
    truncation_tolerance=5,
    near_dup_threshold=0.80,
    min_review_chars=10,
)


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hello world", "hello world"),
        ("hello    world", "hello world"),
        ("hello\n    world", "hello world"),
        ("  padded  ", "padded"),
        ("line\n\n\nbreak", "line break"),
        ("tab\tsep", "tab sep"),
        # The exact shape seen in the source file: wrapped, indented lines.
        ("I was using it\n            for long time", "I was using it for long time"),
    ],
)
def test_normalize_whitespace(raw: str, expected: str) -> None:
    assert normalize_whitespace(raw) == expected


def test_normalize_whitespace_is_idempotent() -> None:
    once = normalize_whitespace("a\n  b   c ")
    assert normalize_whitespace(once) == once


# ---------------------------------------------------------------------------
# make_review_id
# ---------------------------------------------------------------------------


def test_review_id_is_deterministic() -> None:
    """Gold labels are keyed on this ID; instability would void the eval set."""
    first = make_review_id("zepto", "2024-12-30", 1, "the wallet balance vanished")
    second = make_review_id("zepto", "2024-12-30", 1, "the wallet balance vanished")
    assert first == second


def test_review_id_shape() -> None:
    review_id = make_review_id("blinkit", "2024-10-01", 5, "great app")
    assert len(review_id) == 16
    assert all(char in "0123456789abcdef" for char in review_id)


@pytest.mark.parametrize(
    "changed",
    [
        ("jiomart", "2024-12-30", 1, "same text"),
        ("zepto", "2024-12-31", 1, "same text"),
        ("zepto", "2024-12-30", 2, "same text"),
        ("zepto", "2024-12-30", 1, "different text"),
    ],
)
def test_review_id_changes_with_every_field(changed: tuple) -> None:
    baseline = make_review_id("zepto", "2024-12-30", 1, "same text")
    assert make_review_id(*changed) != baseline


# ---------------------------------------------------------------------------
# parse_review_date
# ---------------------------------------------------------------------------


def test_parse_review_date_source_format() -> None:
    assert parse_review_date("30 December 2024") == date(2024, 12, 30)
    assert parse_review_date("  1 July 2020  ") == date(2020, 7, 1)


@pytest.mark.parametrize("bad", ["2024-12-30", "30/12/2024", "Dec 30 2024", "", "soon"])
def test_parse_review_date_rejects_other_formats(bad: str) -> None:
    """Strict on purpose: a format change means the data changed."""
    with pytest.raises(ValueError):
        parse_review_date(bad)


# ---------------------------------------------------------------------------
# quality flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("length", "expected"),
    [(500, True), (496, True), (495, True), (494, False), (300, False), (0, False)],
)
def test_detect_truncation(length: int, expected: bool) -> None:
    assert detect_truncation("x" * length, 500, 5) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("finished.", False),
        ("really?", False),
        ("wow!", False),
        ('he said "no"', False),
        ("cut off mid sen", True),
        ("trailing comma,", True),
    ],
)
def test_ends_without_terminal_punct(text: str, expected: bool) -> None:
    assert ends_without_terminal_punct(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain english review", False),
        ("numbers 123 and symbols !@#", False),
        ("mixed कुछ text", True),  # Devanagari
        ("", False),
    ],
)
def test_has_non_latin(text: str, expected: bool) -> None:
    assert has_non_latin(text) is expected


# ---------------------------------------------------------------------------
# near-duplicate detection
# ---------------------------------------------------------------------------


def test_near_duplicates_groups_identical_texts() -> None:
    texts = [
        "the delivery was late and support never replied to my complaint",
        "the delivery was late and support never replied to my complaint",
        "fresh vegetables arrived quickly and the packaging was excellent",
    ]
    group_id, group_size, is_rep = find_near_duplicates(texts, threshold=0.80)

    assert group_id[0] == group_id[1] != -1
    assert group_id[2] == -1
    assert list(group_size) == [2, 2, 1]
    # Exactly one representative inside the group.
    assert sum(is_rep[:2]) == 1
    # A unique review is always its own representative.
    assert is_rep[2]


def test_near_duplicates_leaves_distinct_texts_alone() -> None:
    texts = [
        "my refund has not arrived after fourteen days of waiting",
        "the mobile application crashes every time I open the cart",
        "prices went up and the handling fee is unreasonable now",
    ]
    group_id, group_size, is_rep = find_near_duplicates(texts, threshold=0.80)

    assert list(group_id) == [-1, -1, -1]
    assert list(group_size) == [1, 1, 1]
    assert all(is_rep)


def test_near_duplicate_representative_is_longest() -> None:
    """The longest member retains the most information after truncation."""
    short = "delivery was very late and the order never arrived at all"
    long = "delivery was very late and the order never arrived at all unfortunately"
    group_id, _, is_rep = find_near_duplicates([short, long], threshold=0.80)

    assert group_id[0] == group_id[1] != -1
    assert is_rep[1] and not is_rep[0]


def test_near_duplicates_handles_tiny_inputs() -> None:
    for texts in ([], ["only one review here"]):
        group_id, group_size, is_rep = find_near_duplicates(texts, threshold=0.80)
        assert len(group_id) == len(texts)
        assert all(size == 1 for size in group_size)
        assert all(is_rep)


# ---------------------------------------------------------------------------
# build_clean_dataset (integration over the layer)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rating": ["1", "5", "1", "3"],
            "date": [
                "30 December 2024",
                "15 November 2024",
                "30 December 2024",
                "21 July 2020",
            ],
            "review": [
                "The delivery was\n    late again and no one from support replied.",
                "Great app, fast delivery and the fruit was fresh.",
                "The delivery was late again and no one from support replied.",
                "x" * 500,
            ],
            "platform": ["zepto", "blinkit", "zepto", "jiomart"],
        }
    )


def test_build_clean_dataset_shape_and_columns(sample_raw: pd.DataFrame) -> None:
    clean, report = build_clean_dataset(sample_raw, **CLEAN_KWARGS)

    assert len(clean) == 4
    assert report.rows_in == 4
    assert report.rows_out == 4
    assert report.rows_dropped == 0
    assert clean["review_id"].is_unique


def test_build_clean_dataset_normalises_and_preserves_raw(sample_raw: pd.DataFrame) -> None:
    """Normalised text for analysis; original text kept for verifiable evidence."""
    clean, report = build_clean_dataset(sample_raw, **CLEAN_KWARGS)
    first = clean.iloc[0]

    assert "\n" not in first["review_text"]
    assert "\n" in first["review_raw"]
    assert report.whitespace_normalised == 1


def test_build_clean_dataset_flags_truncation(sample_raw: pd.DataFrame) -> None:
    clean, report = build_clean_dataset(sample_raw, **CLEAN_KWARGS)

    assert report.truncated_reviews == 1
    assert clean.loc[clean["char_len"] == 500, "is_truncated"].all()


def test_build_clean_dataset_flags_near_duplicates(sample_raw: pd.DataFrame) -> None:
    """Rows 0 and 2 are the same complaint with different whitespace."""
    clean, report = build_clean_dataset(sample_raw, **CLEAN_KWARGS)

    assert report.near_dup_groups == 1
    assert report.near_dup_members == 2
    duplicated = clean[clean["near_dup_group_id"] >= 0]
    assert len(duplicated) == 2
    assert duplicated["is_near_dup_representative"].sum() == 1


def test_build_clean_dataset_derives_scoping_fields(sample_raw: pd.DataFrame) -> None:
    clean, _ = build_clean_dataset(sample_raw, **CLEAN_KWARGS)

    assert set(clean["rating_bucket"]) == {"negative", "positive", "neutral"}
    # The 2020 JioMart row predates the all-platform comparable window.
    assert clean["in_comparable_window"].sum() == 3
    assert clean.loc[clean["year"] == 2020, "in_comparable_window"].eq(False).all()


def test_build_clean_dataset_drops_only_unusable_rows(sample_raw: pd.DataFrame) -> None:
    """Dropping is reserved for structurally unusable rows, and is always counted."""
    with_junk = pd.concat(
        [
            sample_raw,
            pd.DataFrame(
                {
                    "rating": ["1"],
                    "date": ["30 December 2024"],
                    "review": ["bad"],  # below min_review_chars
                    "platform": ["zepto"],
                }
            ),
        ],
        ignore_index=True,
    )

    clean, report = build_clean_dataset(with_junk, **CLEAN_KWARGS)

    assert report.rows_in == 5
    assert report.rows_out == 4
    assert report.drop_reasons == {"review_too_short": 1}
    assert len(clean) == 4


def test_build_clean_dataset_is_reproducible(sample_raw: pd.DataFrame) -> None:
    """Two runs over the same input must produce byte-identical IDs and flags."""
    first, _ = build_clean_dataset(sample_raw, **CLEAN_KWARGS)
    second, _ = build_clean_dataset(sample_raw, **CLEAN_KWARGS)

    pd.testing.assert_frame_equal(first, second)
