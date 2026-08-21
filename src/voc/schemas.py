"""
Data contracts for the pipeline.

These Pydantic models are the spine of the project. Every layer either produces
or consumes one of them, which means a bad assumption fails loudly at a schema
boundary instead of silently propagating into a chart a PM would then trust.

Phase 1 defines the two contracts that matter now:
  * ``RawReviewRow``  - what we accept from the immutable source CSV.
  * ``CleanReview``   - what the cleaning layer guarantees to every later phase.

The AI enrichment contracts (sentiment, product_area, severity, ...) arrive in
Phase 3 and will extend ``CleanReview`` rather than replace it.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config.settings import get_dataset_config

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

def allowed_platforms() -> tuple[str, ...]:
    """Platform values this dataset accepts, from ``config/dataset.yaml``.

    Read at call time rather than frozen at import, so processing a different
    corpus is a config change. Validation stays strict either way: anything
    outside this set is rejected at ingestion.
    """
    return get_dataset_config().platform_ids


def accepted_date_formats() -> tuple[str, ...]:
    """strptime patterns to try, in order, from ``config/dataset.yaml``.

    A list rather than a single format so a second corpus can declare its own,
    but still a closed set -- a date matching none of them is an error, not a
    silently dropped row.
    """
    return tuple(get_dataset_config().date_formats)

#: Start of the window in which all three platforms are meaningfully present.
#: Before 2024-10, Blinkit has 12 reviews and Zepto has 1 -- any cross-platform
#: comparison earlier than this is comparing a platform against noise. Rows
#: outside this window are kept (never deleted) but flagged so the UI can scope
#: competitive claims honestly.
COMPARABLE_WINDOW_START: date = date(2024, 10, 1)


class RatingBucket(str, Enum):
    """Coarse polarity derived from the star rating.

    Derived from the rating only -- NOT from model output. Keeping this separate
    from AI-predicted sentiment is what lets Phase 9 measure the model against
    an independent signal instead of against itself.
    """

    NEGATIVE = "negative"  # 1-2 stars
    NEUTRAL = "neutral"  # 3 stars
    POSITIVE = "positive"  # 4-5 stars

    @classmethod
    def from_rating(cls, rating: int) -> "RatingBucket":
        if rating <= 2:
            return cls.NEGATIVE
        if rating == 3:
            return cls.NEUTRAL
        return cls.POSITIVE


# ---------------------------------------------------------------------------
# Layer 1 contract: the raw source row
# ---------------------------------------------------------------------------


class RawReviewRow(BaseModel):
    """One row exactly as it appears in ``data/raw/reviews.csv``.

    Validation here is intentionally strict. If the source file ever changes
    shape -- a new platform, a different date format, a rating of 0 -- we want
    ingestion to fail with a precise message rather than quietly produce a
    dataset with a hole in it.
    """

    model_config = ConfigDict(str_strip_whitespace=False, frozen=True)

    rating: int = Field(..., ge=1, le=5, description="Star rating, 1-5 inclusive.")
    date: str = Field(..., min_length=1, description="Raw date string, unparsed.")
    review: str = Field(..., min_length=1, description="Raw review text, unmodified.")
    platform: str = Field(..., min_length=1)

    @field_validator("platform")
    @classmethod
    def _known_platform(cls, value: str) -> str:
        normalised = value.strip().lower()
        allowed = allowed_platforms()
        if normalised not in allowed:
            raise ValueError(
                f"Unknown platform {value!r}. Expected one of {allowed}. "
                "If this corpus legitimately has another platform, declare it "
                "under `platforms:` in config/dataset.yaml -- do not loosen "
                "this check."
            )
        return normalised


# ---------------------------------------------------------------------------
# Layer 2 contract: the cleaned row
# ---------------------------------------------------------------------------


class CleanReview(BaseModel):
    """One cleaned review. The guaranteed input to every phase after this one.

    Note that ``review_raw`` is carried through untouched alongside the
    normalised ``review_text``. Evidence shown to a PM in the UI should quote
    text that provably came from the source, and keeping both means we can
    always prove that.
    """

    model_config = ConfigDict(frozen=True)

    # --- Identity ----------------------------------------------------------
    review_id: str = Field(
        ...,
        min_length=16,
        description=(
            "Deterministic content hash. Stable across runs and machines, so "
            "evidence links and gold labels survive a pipeline re-run."
        ),
    )
    source_row_index: int = Field(
        ..., ge=0, description="0-based position in the raw CSV, for traceability."
    )

    # --- Core fields -------------------------------------------------------
    platform: str
    rating: int = Field(..., ge=1, le=5)
    rating_bucket: RatingBucket
    review_date: date
    year: int
    month: int
    year_month: str = Field(..., pattern=r"^\d{4}-\d{2}$")

    # --- Text --------------------------------------------------------------
    review_raw: str = Field(..., min_length=1)
    review_text: str = Field(..., min_length=1)
    char_len: int = Field(..., ge=0)
    word_count: int = Field(..., ge=0)

    # --- Quality flags -----------------------------------------------------
    is_truncated: bool = Field(
        ...,
        description=(
            "Text hit the source scraper's character cap. Truncated reviews may "
            "be missing their resolution clause, which biases severity and "
            "sentiment negative -- exclude them from outcome-dependent labels."
        ),
    )
    ends_without_terminal_punct: bool
    has_non_latin: bool

    # --- Near-duplicate flags ---------------------------------------------
    near_dup_group_id: int = Field(
        ...,
        description="Shared id for near-identical reviews; -1 when the review is unique.",
    )
    near_dup_group_size: int = Field(..., ge=1)
    is_near_dup_representative: bool = Field(
        ...,
        description=(
            "True for exactly one review per near-duplicate group. Pain-point "
            "frequency is reported both raw and representative-only, so a "
            "templated complaint cannot silently manufacture a trend."
        ),
    )

    # --- Analysis scoping --------------------------------------------------
    in_comparable_window: bool = Field(
        ...,
        description=(
            "Review falls on/after COMPARABLE_WINDOW_START, the period in which "
            "all three platforms are meaningfully present."
        ),
    )

    @field_validator("platform")
    @classmethod
    def _known_platform(cls, value: str) -> str:
        allowed = allowed_platforms()
        if value not in allowed:
            raise ValueError(f"Unknown platform {value!r}; expected one of {allowed}")
        return value


# ---------------------------------------------------------------------------
# Run reports (serialised to JSON so every run is auditable)
# ---------------------------------------------------------------------------


class IngestReport(BaseModel):
    """Provenance record for one ingestion run."""

    source_path: str
    source_sha256: str = Field(
        ..., description="Checksum of the raw file, proving which bytes were read."
    )
    source_bytes: int
    rows_read: int
    rows_valid: int
    rows_rejected: int
    rejection_examples: list[str] = Field(default_factory=list)
    loaded_at_utc: str


class CleaningReport(BaseModel):
    """What the cleaning layer actually did, in numbers.

    Written to ``data/interim/cleaning_report.json`` on every run. This is the
    audit trail that makes the pipeline defensible: anyone can check that no
    rows were silently dropped.
    """

    rows_in: int
    rows_out: int
    rows_dropped: int
    drop_reasons: dict[str, int] = Field(default_factory=dict)

    whitespace_normalised: int
    exact_duplicate_texts: int
    truncated_reviews: int
    ends_without_terminal_punct: int
    non_latin_reviews: int

    near_dup_threshold: float
    near_dup_groups: int
    near_dup_members: int
    largest_near_dup_group: int

    date_min: str
    date_max: str
    rows_in_comparable_window: int

    platform_counts: dict[str, int] = Field(default_factory=dict)
    rating_counts: dict[str, int] = Field(default_factory=dict)

    settings_snapshot: dict[str, float | int | str] = Field(default_factory=dict)
    generated_at_utc: str
