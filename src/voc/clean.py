"""
Layer 2 -- Data cleaning.

Turns validated raw rows into ``data/interim/reviews_clean.parquet``, the
dataset every later phase reads. Five things happen here, each of which exists
because profiling found a specific problem in the source data:

1. **Whitespace normalisation** -- 99.5% of reviews carry newlines and runs of
   spaces from the source page's indentation. Cosmetic, but it corrupts token
   counts and embedding quality if left in.

2. **Deterministic review IDs** -- the raw file has no identifier column. Without
   a stable ID, evidence links and hand-labelled gold data break on every
   re-run. The ID is a content hash, so it is stable across runs and machines.

3. **Truncation detection** -- the scraper capped text at 500 characters. A
   truncated review can lose its resolution clause ("...but support fixed it"),
   which biases sentiment and severity negative. Flagged, never dropped.

4. **Near-duplicate detection** -- exact duplicates are almost absent (1 pair),
   but ~21 Zepto wallet complaints share a near-identical template. Those would
   inflate a cluster and manufacture a pain-point signal that is really one
   complaint copied many times. Flagged and grouped, never dropped.

5. **Comparable-window flagging** -- before 2024-10, Blinkit has 12 reviews and
   Zepto has 1. Cross-platform comparison outside that window compares a
   platform against noise.

The consistent principle: **flag, do not delete.** Every row that enters
leaves, annotated. Dropping is reserved for rows that are structurally unusable,
and every drop is counted in ``CleaningReport.drop_reasons``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections.abc import Sequence
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from voc.schemas import (
    COMPARABLE_WINDOW_START,
    accepted_date_formats,
    CleaningReport,
    CleanReview,
    RatingBucket,
)

logger = logging.getLogger(__name__)

#: Collapses any run of whitespace (spaces, tabs, newlines) into one space.
_WHITESPACE_RUN = re.compile(r"\s+")

#: Characters that plausibly end a finished sentence.
_TERMINAL_PUNCT = frozenset('.!?"\')')


# ---------------------------------------------------------------------------
# Field-level transforms (pure functions -- each is unit-tested)
# ---------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends.

    >>> normalize_whitespace("hello\\n    world  ")
    'hello world'
    """
    return _WHITESPACE_RUN.sub(" ", text).strip()


def make_review_id(platform: str, date_iso: str, rating: int, text: str) -> str:
    """Deterministic 16-hex-character content hash.

    All four source fields participate, so two genuinely different reviews that
    happen to share text (posted on different days) still get distinct IDs.

    Determinism matters more than it looks: gold labels created in Phase 9 are
    keyed on this ID, so a non-reproducible ID would silently invalidate the
    evaluation set on the next pipeline run.
    """
    payload = f"{platform}|{date_iso}|{rating}|{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def parse_review_date(value: str, formats: Sequence[str] | None = None) -> date:
    """Parse a source date against the dataset's declared formats, in order.

    Strict by design, and still strict now that the format list is configurable:
    a value matching none of the declared patterns raises rather than being
    coerced or skipped. The error names every pattern tried, because the usual
    cause is a corpus whose ``date_formats`` need declaring in
    ``config/dataset.yaml`` -- not a corrupt row.
    """
    patterns = tuple(formats) if formats is not None else accepted_date_formats()
    text = value.strip()
    for pattern in patterns:
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    raise ValueError(
        f"Date {value!r} matches none of the configured formats {patterns}. "
        "Declare the corpus's format under `date_formats:` in config/dataset.yaml."
    )


def detect_truncation(text: str, cap_chars: int, tolerance: int) -> bool:
    """True when the text sits at the source scraper's character ceiling.

    Evidence for ``cap_chars=500``: the raw distribution shows 84 reviews at
    exactly 500 characters and 622 in the 480-500 band, against only 159 in
    460-480 and 124 in 440-460. Natural writing length does not pile up like
    that; a hard cap does.
    """
    return len(text) >= (cap_chars - tolerance)


def ends_without_terminal_punct(text: str) -> bool:
    """True when the text does not end on sentence-ending punctuation."""
    return bool(text) and text[-1] not in _TERMINAL_PUNCT


def has_non_latin(text: str) -> bool:
    """True when any alphabetic character falls outside the Latin script.

    Only 2 reviews in the source contain Devanagari, so multi-language handling
    is explicitly out of scope -- but the flag makes that a measured decision
    rather than an unnoticed gap.
    """
    for char in text:
        if char.isalpha():
            try:
                if not unicodedata.name(char).startswith("LATIN"):
                    return True
            except ValueError:  # unnamed codepoint
                continue
    return False


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------


def find_near_duplicates(
    texts: list[str],
    threshold: float,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group near-identical reviews via TF-IDF cosine similarity.

    Method, and why this one:
      * TF-IDF over word 1- and 2-grams gives each review a sparse vector.
        Vectors are L2-normalised by scikit-learn, so the dot product of two
        rows *is* their cosine similarity -- no separate distance step needed.
      * Pairs scoring at or above ``threshold`` become edges in a graph, and
        connected components become duplicate groups. Using components rather
        than raw pairs means A~B and B~C put all three in one group, which is
        what "these are the same complaint" actually means.
      * The similarity matrix is computed in row chunks so memory stays flat
        regardless of corpus size.

    Chosen over MinHash/LSH because at ~4.6k documents the exact computation
    takes under a second, and exactness removes a tuning parameter we would
    otherwise have to defend.

    Returns three arrays aligned to ``texts``:
      ``group_id``      -- shared id per group, ``-1`` for unique reviews
      ``group_size``    -- size of that review's group, ``1`` when unique
      ``is_representative`` -- ``True`` for exactly one member of each group
    """
    count = len(texts)
    group_id = np.full(count, -1, dtype=np.int32)
    group_size = np.ones(count, dtype=np.int32)
    is_representative = np.ones(count, dtype=bool)

    if count < 2:
        return group_id, group_size, is_representative

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(texts)

    edge_rows: list[int] = []
    edge_cols: list[int] = []

    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        # Rows are L2-normalised, so this block IS cosine similarity.
        similarity = (matrix[start:end] @ matrix.T).toarray()
        for local_index in range(similarity.shape[0]):
            global_index = start + local_index
            # Upper triangle only: avoids self-matches and double-counting.
            candidates = np.nonzero(similarity[local_index] >= threshold)[0]
            for other in candidates[candidates > global_index]:
                edge_rows.append(global_index)
                edge_cols.append(int(other))

    if not edge_rows:
        logger.info("No near-duplicate pairs found at threshold %.2f", threshold)
        return group_id, group_size, is_representative

    adjacency = coo_matrix(
        (np.ones(len(edge_rows), dtype=np.int8), (edge_rows, edge_cols)),
        shape=(count, count),
    )
    _, labels = connected_components(adjacency, directed=False)

    # Keep only components with 2+ members; singletons stay flagged as unique.
    label_counts = np.bincount(labels)
    multi_member_labels = {
        int(label) for label in np.nonzero(label_counts >= 2)[0]
    }

    next_group_id = 0
    for label in sorted(multi_member_labels):
        members = np.nonzero(labels == label)[0]
        group_id[members] = next_group_id
        group_size[members] = len(members)

        # Representative = longest text (most information survived truncation),
        # tie-broken by lowest index so the choice is reproducible.
        best = max(members, key=lambda idx: (len(texts[idx]), -idx))
        is_representative[members] = False
        is_representative[best] = True

        next_group_id += 1

    logger.info(
        "Near-duplicate detection: %d group(s) covering %d review(s) at threshold %.2f",
        next_group_id,
        int((group_id >= 0).sum()),
        threshold,
    )
    return group_id, group_size, is_representative


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_clean_dataset(
    raw: pd.DataFrame,
    *,
    truncation_cap_chars: int,
    truncation_tolerance: int,
    near_dup_threshold: float,
    min_review_chars: int,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning pass and produce the validated clean dataset."""
    rows_in = len(raw)
    drop_reasons: dict[str, int] = {}
    records: list[dict] = []
    whitespace_changed = 0

    for source_index, row in raw.iterrows():
        raw_text = row["review"]
        text = normalize_whitespace(raw_text)
        if text != raw_text:
            whitespace_changed += 1

        # --- structural drops (the only rows that do not survive) ---
        if len(text) < min_review_chars:
            drop_reasons["review_too_short"] = drop_reasons.get("review_too_short", 0) + 1
            continue

        try:
            review_date = parse_review_date(row["date"])
        except ValueError:
            drop_reasons["unparseable_date"] = drop_reasons.get("unparseable_date", 0) + 1
            continue

        rating = int(row["rating"])
        platform = row["platform"].strip().lower()
        date_iso = review_date.isoformat()

        records.append(
            {
                "review_id": make_review_id(platform, date_iso, rating, text),
                "source_row_index": int(source_index),
                "platform": platform,
                "rating": rating,
                "rating_bucket": RatingBucket.from_rating(rating).value,
                "review_date": review_date,
                "year": review_date.year,
                "month": review_date.month,
                "year_month": f"{review_date.year}-{review_date.month:02d}",
                "review_raw": raw_text,
                "review_text": text,
                "char_len": len(text),
                "word_count": len(text.split()),
                "is_truncated": detect_truncation(
                    text, truncation_cap_chars, truncation_tolerance
                ),
                "ends_without_terminal_punct": ends_without_terminal_punct(text),
                "has_non_latin": has_non_latin(text),
                "in_comparable_window": review_date >= COMPARABLE_WINDOW_START,
            }
        )

    if not records:
        raise ValueError("Cleaning produced zero rows -- check the raw input.")

    frame = pd.DataFrame.from_records(records)

    # --- ID uniqueness -----------------------------------------------------
    # Expected to be a no-op: profiling found zero rows identical across all
    # four source columns. Handled anyway, loudly, because a silent ID clash
    # would corrupt evidence links in a way that is very hard to notice later.
    duplicated_ids = frame["review_id"].duplicated(keep=False)
    if duplicated_ids.any():
        clash_count = int(duplicated_ids.sum())
        logger.warning(
            "%d review_id collision(s) detected; disambiguating with an occurrence suffix.",
            clash_count,
        )
        occurrence = frame.groupby("review_id").cumcount()
        frame.loc[occurrence > 0, "review_id"] = (
            frame.loc[occurrence > 0, "review_id"]
            + "-"
            + occurrence[occurrence > 0].astype(str)
        )

    # --- near-duplicate grouping ------------------------------------------
    group_id, group_size, is_representative = find_near_duplicates(
        frame["review_text"].tolist(), threshold=near_dup_threshold
    )
    frame["near_dup_group_id"] = group_id
    frame["near_dup_group_size"] = group_size
    frame["is_near_dup_representative"] = is_representative

    # --- contract validation ----------------------------------------------
    # Every row is checked against CleanReview. This is the boundary that keeps
    # bad assumptions from reaching the analysis layers.
    validation_errors: list[str] = []
    for record in frame.to_dict(orient="records"):
        try:
            CleanReview(**record)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            validation_errors.append(f"{record.get('review_id')}: {exc}")
            if len(validation_errors) >= 10:
                break
    if validation_errors:
        raise ValueError(
            "Cleaned rows failed CleanReview validation:\n  "
            + "\n  ".join(validation_errors)
        )

    frame = _apply_dtypes(frame)

    report = CleaningReport(
        rows_in=rows_in,
        rows_out=len(frame),
        rows_dropped=rows_in - len(frame),
        drop_reasons=drop_reasons,
        whitespace_normalised=whitespace_changed,
        exact_duplicate_texts=int(frame["review_text"].duplicated().sum()),
        truncated_reviews=int(frame["is_truncated"].sum()),
        ends_without_terminal_punct=int(frame["ends_without_terminal_punct"].sum()),
        non_latin_reviews=int(frame["has_non_latin"].sum()),
        near_dup_threshold=near_dup_threshold,
        near_dup_groups=int(frame.loc[frame["near_dup_group_id"] >= 0, "near_dup_group_id"].nunique()),
        near_dup_members=int((frame["near_dup_group_id"] >= 0).sum()),
        largest_near_dup_group=int(frame["near_dup_group_size"].max()),
        date_min=str(frame["review_date"].min().date()),
        date_max=str(frame["review_date"].max().date()),
        rows_in_comparable_window=int(frame["in_comparable_window"].sum()),
        platform_counts={
            str(k): int(v) for k, v in frame["platform"].value_counts().items()
        },
        rating_counts={
            str(k): int(v) for k, v in frame["rating"].value_counts().sort_index().items()
        },
        settings_snapshot={
            "truncation_cap_chars": truncation_cap_chars,
            "truncation_tolerance": truncation_tolerance,
            "near_dup_threshold": near_dup_threshold,
            "min_review_chars": min_review_chars,
            "comparable_window_start": COMPARABLE_WINDOW_START.isoformat(),
        },
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    return frame, report


def _apply_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Set compact, explicit dtypes so the parquet round-trips predictably."""
    frame = frame.copy()
    frame["review_date"] = pd.to_datetime(frame["review_date"])
    conversions = {
        "review_id": "string",
        "platform": "category",
        "rating": "int8",
        "rating_bucket": "category",
        "year": "int16",
        "month": "int8",
        "year_month": "string",
        "review_raw": "string",
        "review_text": "string",
        "char_len": "int16",
        "word_count": "int16",
        "source_row_index": "int32",
        "near_dup_group_id": "int32",
        "near_dup_group_size": "int32",
    }
    for column, dtype in conversions.items():
        frame[column] = frame[column].astype(dtype)

    column_order = [
        "review_id",
        "source_row_index",
        "platform",
        "rating",
        "rating_bucket",
        "review_date",
        "year",
        "month",
        "year_month",
        "review_text",
        "review_raw",
        "char_len",
        "word_count",
        "is_truncated",
        "ends_without_terminal_punct",
        "has_non_latin",
        "near_dup_group_id",
        "near_dup_group_size",
        "is_near_dup_representative",
        "in_comparable_window",
    ]
    return frame[column_order]
