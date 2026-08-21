"""
Phase 2 -- Taxonomy discovery analysis.

Measures how the corpus distributes across the proposed product areas, using
the keyword probes stored in ``config/taxonomy.yaml``.

**What these numbers are.** Lexical probe matches: high precision, mediocre
recall. They exist to size areas, expose platform differences, and let a human
audit why the taxonomy looks the way it does. They are explicitly NOT the
classification result -- Phase 3 replaces them with LLM labels validated
against a gold set.

**Why keep them at all after Phase 3.** They become a cheap, deterministic
baseline. If LLM classification and probe matching disagree wildly on an
area's size, one of them is wrong and it is worth finding out which.

The three measurements that decided the taxonomy:

* **Platform spread** (max/min area share across platforms) -- an area with a
  large spread is carrying competitive signal that a merged parent would hide.
* **Positive:negative ratio** -- separates surfaces customers praise from
  surfaces that only ever fail, which is why strengths and issues are modelled
  separately rather than as one signed axis.
* **Support lift** -- P(support | area) against the base rate, which showed
  support to be a downstream amplifier rather than an independent driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from voc.taxonomy import Taxonomy


@dataclass
class AreaStats:
    """Discovery measurements for one product area."""

    area_id: str
    name: str
    domain: str
    matched: int
    share_pct: float
    share_negative_pct: float
    share_positive_pct: float
    positive_negative_ratio: float
    platform_share_pct: dict[str, float] = field(default_factory=dict)
    platform_spread: float = 0.0
    support_lift: float = 0.0
    representative_share_pct: float = 0.0
    near_dup_inflation_pp: float = 0.0


@dataclass
class DiscoveryResult:
    n_reviews: int
    areas: list[AreaStats]
    unmatched: int
    unmatched_pct: float
    unmatched_mean_rating: float
    mean_areas_per_review: float
    areas_per_review_dist: dict[int, int]
    cooccurrence: pd.DataFrame
    support_base_rate: float
    support_area_id: str
    corpus_mean_rating: float


def apply_probes(frame: pd.DataFrame, taxonomy: Taxonomy) -> pd.DataFrame:
    """Return a boolean matrix of shape (n_reviews, n_areas).

    One column per product area, aligned to ``frame``'s index. This is the
    substrate every other measurement in this module is computed from.
    """
    text = frame["review_text"].astype(str)
    return pd.DataFrame(
        {
            area_id: text.str.contains(pattern, regex=True)
            for area_id, pattern in taxonomy.compiled_probes().items()
        },
        index=frame.index,
    )


def analyse(frame: pd.DataFrame, taxonomy: Taxonomy) -> DiscoveryResult:
    """Run the full discovery analysis over the cleaned corpus."""
    hits = apply_probes(frame, taxonomy)

    total = len(frame)
    is_negative = frame["rating"] <= 2
    is_positive = frame["rating"] >= 4
    n_negative = int(is_negative.sum())
    n_positive = int(is_positive.sum())

    platforms = sorted(frame["platform"].astype(str).unique())
    platform_totals = frame["platform"].astype(str).value_counts()

    # Which area represents support is declared in the taxonomy, not here, so
    # renaming the area does not require a code change.
    support_id = taxonomy.special_area("support_area")
    support_hits = hits[support_id]
    support_base = float(support_hits.mean())

    representatives = frame["is_near_dup_representative"]

    stats: list[AreaStats] = []
    for area in taxonomy.product_areas:
        column = hits[area.id]
        matched = int(column.sum())

        share_neg = float(column[is_negative].mean() * 100) if n_negative else 0.0
        share_pos = float(column[is_positive].mean() * 100) if n_positive else 0.0
        ratio = (share_pos / share_neg) if share_neg > 0 else float("inf")

        per_platform = {
            platform: float(
                column[frame["platform"].astype(str) == platform].sum()
                / platform_totals[platform]
                * 100
            )
            for platform in platforms
        }
        # Floor the denominator so a near-zero area cannot report an
        # astronomically large, meaningless spread.
        spread = max(per_platform.values()) / max(min(per_platform.values()), 0.1)

        # Does this area pull support contact above the corpus base rate?
        lift = (
            float(support_hits[column].mean()) / support_base
            if matched and support_base
            else 0.0
        )

        rep_share = float(column[representatives].mean() * 100)

        stats.append(
            AreaStats(
                area_id=area.id,
                name=area.name,
                domain=area.domain,
                matched=matched,
                share_pct=matched / total * 100,
                share_negative_pct=share_neg,
                share_positive_pct=share_pos,
                positive_negative_ratio=ratio,
                platform_share_pct=per_platform,
                platform_spread=spread,
                support_lift=lift,
                representative_share_pct=rep_share,
                near_dup_inflation_pp=(matched / total * 100) - rep_share,
            )
        )

    per_review = hits.sum(axis=1)
    unmatched_mask = per_review == 0

    return DiscoveryResult(
        n_reviews=total,
        areas=stats,
        unmatched=int(unmatched_mask.sum()),
        unmatched_pct=float(unmatched_mask.mean() * 100),
        unmatched_mean_rating=float(frame.loc[unmatched_mask, "rating"].mean()),
        mean_areas_per_review=float(per_review.mean()),
        areas_per_review_dist={int(k): int(v) for k, v in per_review.value_counts().sort_index().items()},
        cooccurrence=_jaccard_matrix(hits),
        support_base_rate=support_base,
        support_area_id=support_id,
        corpus_mean_rating=float(frame["rating"].mean()),
    )


def _jaccard_matrix(hits: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Jaccard overlap between areas.

    High overlap flags a merge candidate; low overlap across the board is
    evidence the areas are genuinely distinct surfaces rather than synonyms.
    """
    matrix = hits.to_numpy(dtype=bool)
    intersection = matrix.T.astype(int) @ matrix.astype(int)
    counts = matrix.sum(axis=0)
    union = counts[:, None] + counts[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, intersection / union, 0.0)
    return pd.DataFrame(jaccard, index=hits.columns, columns=hits.columns)


def top_cooccurring_pairs(
    result: DiscoveryResult, limit: int = 12
) -> list[tuple[str, str, float]]:
    """Highest-overlap area pairs, as (area_a, area_b, jaccard)."""
    matrix = result.cooccurrence
    pairs: list[tuple[str, str, float]] = []
    columns = list(matrix.columns)
    for i, area_a in enumerate(columns):
        for area_b in columns[i + 1 :]:
            pairs.append((area_a, area_b, float(matrix.loc[area_a, area_b])))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:limit]


def to_dataframe(result: DiscoveryResult) -> pd.DataFrame:
    """Flatten area stats into a tidy frame for CSV export."""
    rows = []
    for area in result.areas:
        row = {
            "area_id": area.area_id,
            "name": area.name,
            "domain": area.domain,
            "matched": area.matched,
            "share_pct": round(area.share_pct, 2),
            "share_of_negative_pct": round(area.share_negative_pct, 2),
            "share_of_positive_pct": round(area.share_positive_pct, 2),
            "positive_negative_ratio": round(area.positive_negative_ratio, 3),
            "platform_spread": round(area.platform_spread, 2),
            "support_lift": round(area.support_lift, 3),
            "near_dup_inflation_pp": round(area.near_dup_inflation_pp, 3),
        }
        for platform, share in area.platform_share_pct.items():
            row[f"share_{platform}_pct"] = round(share, 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("share_pct", ascending=False)
