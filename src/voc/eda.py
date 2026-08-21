"""
Layer 3 -- Exploratory data analysis and product intelligence profiling.

Answers what the cleaned corpus actually contains, so the AI pipeline is aimed
at real patterns rather than assumed ones. No LLM is involved at this stage by
design: everything here is deterministic and reproducible from the parquet.

The analytical spine of this module is one distinction, applied everywhere:

    DATASET SAMPLING PATTERN   how these reviews came to be collected
    CUSTOMER BEHAVIOUR         what customers actually did

Review volume rising from 83 in July to 2,212 in December is a *scraping*
pattern. It says nothing about demand. Conflating the two is the single
easiest way to produce a confident, wrong product decision from this data,
so every temporal and comparative function here reports normalised shares and
carries its own coverage caveat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from voc.discovery import apply_probes
from voc.taxonomy import Taxonomy

# A platform-month needs at least this many reviews to be worth comparing.
# Below it, a single templated complaint can move the share by several points.
MIN_REVIEWS_FOR_COMPARISON = 50


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class DatasetProfile:
    total_reviews: int
    by_platform: dict[str, int]
    by_rating: dict[int, int]
    mean_rating: float
    mean_rating_by_platform: dict[str, float]
    rating_bucket_share: dict[str, float]
    date_min: date
    date_max: date
    span_days: int
    truncated: int
    truncated_pct: float
    near_dup_groups: int
    near_dup_members: int
    non_latin: int
    char_len_stats: dict[str, float]
    word_count_stats: dict[str, float]


@dataclass
class TemporalProfile:
    monthly_volume: pd.DataFrame          # month x platform counts
    platform_date_ranges: dict[str, tuple[date, date]]
    december_share_pct: float
    months_meeting_threshold: list[str]
    comparison_window: tuple[date, date] | None
    comparison_window_reviews: int
    comparison_window_pct: float
    window_rationale: str
    monthly_negative_share: pd.DataFrame  # month x %neg, the confound evidence


@dataclass
class RatingProfile:
    distribution: pd.DataFrame            # rating x platform, counts and shares
    by_platform: pd.DataFrame
    over_time: pd.DataFrame               # month x platform mean rating
    full_vs_window: pd.DataFrame          # sampling-confound check
    ranking_changes_in_window: bool       # True = full-corpus comparison is unsafe
    reordered_platforms: list[str]
    confound_note: str


@dataclass
class TextProfile:
    length_by_platform: pd.DataFrame
    very_short: pd.DataFrame
    very_long: pd.DataFrame
    truncated_comparison: dict[str, float]
    top_terms_negative: list[tuple[str, int]]
    top_terms_positive: list[tuple[str, int]]
    template_groups: pd.DataFrame


@dataclass
class ThemeProfile:
    """Keyword-probe theme shares. EXPLORATORY ESTIMATES, never final labels."""

    share_full: pd.DataFrame              # area x platform, whole corpus
    share_window: pd.DataFrame            # area x platform, comparison window only
    window_vs_full_shift: pd.DataFrame    # does time coverage explain the gaps?
    shared_themes: list[str]
    platform_specific: dict[str, list[str]]


@dataclass
class EDAResult:
    profile: DatasetProfile
    temporal: TemporalProfile
    ratings: RatingProfile
    text: TextProfile
    themes: ThemeProfile
    reconciliation: dict[str, object]


# ---------------------------------------------------------------------------
# 1. Dataset profile
# ---------------------------------------------------------------------------


def _describe(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "p10": float(series.quantile(0.10)),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "p90": float(series.quantile(0.90)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }


def profile_dataset(frame: pd.DataFrame) -> DatasetProfile:
    total = len(frame)
    dup = frame[frame["near_dup_group_id"] >= 0]

    return DatasetProfile(
        total_reviews=total,
        by_platform={str(k): int(v) for k, v in frame["platform"].value_counts().items()},
        by_rating={int(k): int(v) for k, v in frame["rating"].value_counts().sort_index().items()},
        mean_rating=float(frame["rating"].mean()),
        mean_rating_by_platform={
            str(k): float(v) for k, v in frame.groupby("platform", observed=True)["rating"].mean().items()
        },
        rating_bucket_share={
            str(k): float(v * 100)
            for k, v in frame["rating_bucket"].value_counts(normalize=True).items()
        },
        date_min=frame["review_date"].min().date(),
        date_max=frame["review_date"].max().date(),
        span_days=int((frame["review_date"].max() - frame["review_date"].min()).days),
        truncated=int(frame["is_truncated"].sum()),
        truncated_pct=float(frame["is_truncated"].mean() * 100),
        near_dup_groups=int(dup["near_dup_group_id"].nunique()),
        near_dup_members=len(dup),
        non_latin=int(frame["has_non_latin"].sum()),
        char_len_stats=_describe(frame["char_len"]),
        word_count_stats=_describe(frame["word_count"]),
    )


# ---------------------------------------------------------------------------
# 2. Temporal analysis
# ---------------------------------------------------------------------------


def find_comparison_window(frame: pd.DataFrame) -> tuple[tuple[date, date] | None, list[str], str]:
    """Find the months where every platform is present in comparable volume.

    A cross-platform claim is only defensible where all platforms actually have
    data. Blinkit contributes 12 reviews before October and Zepto 1 -- comparing
    across those months would be comparing a platform against noise, and any
    difference found would be an artefact of when the scrape ran.
    """
    counts = (
        frame.groupby(["year_month", "platform"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    qualifying = counts[(counts >= MIN_REVIEWS_FOR_COMPARISON).all(axis=1)]

    if qualifying.empty:
        return None, [], (
            "No month has at least "
            f"{MIN_REVIEWS_FOR_COMPARISON} reviews for every platform. "
            "Cross-platform comparison is not supportable on this dataset."
        )

    months = list(qualifying.index)
    subset = frame[frame["year_month"].isin(months)]
    start, end = subset["review_date"].min().date(), subset["review_date"].max().date()

    rationale = (
        f"Months where all {counts.shape[1]} platforms each have at least "
        f"{MIN_REVIEWS_FOR_COMPARISON} reviews: {', '.join(months)}. "
        f"This yields {len(subset):,} reviews ({len(subset) / len(frame) * 100:.1f}% of the corpus). "
        "Outside this window the corpus is effectively single-platform, so any "
        "cross-platform difference would measure scrape coverage rather than "
        "customer experience."
    )
    return (start, end), months, rationale


def analyse_temporal(frame: pd.DataFrame) -> TemporalProfile:
    monthly = (
        frame.groupby(["year_month", "platform"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    monthly["total"] = monthly.sum(axis=1)

    ranges = {
        str(platform): (group["review_date"].min().date(), group["review_date"].max().date())
        for platform, group in frame.groupby("platform", observed=True)
    }

    december = frame[frame["year_month"] == "2024-12"]

    # Negative share by month. This is the confound that makes raw complaint
    # COUNTS meaningless: the mix of ratings being scraped changes over time,
    # so a rising count can reflect scrape composition, not worsening service.
    neg_share = (
        frame.assign(is_neg=frame["rating"] <= 2)
        .groupby("year_month", observed=True)
        .agg(reviews=("review_id", "size"), negative_share_pct=("is_neg", lambda s: s.mean() * 100))
        .sort_index()
    )

    window, months, rationale = find_comparison_window(frame)
    in_window = frame[frame["year_month"].isin(months)] if months else frame.iloc[0:0]

    return TemporalProfile(
        monthly_volume=monthly,
        platform_date_ranges=ranges,
        december_share_pct=float(len(december) / len(frame) * 100),
        months_meeting_threshold=months,
        comparison_window=window,
        comparison_window_reviews=len(in_window),
        comparison_window_pct=float(len(in_window) / len(frame) * 100),
        window_rationale=rationale,
        monthly_negative_share=neg_share,
    )


# ---------------------------------------------------------------------------
# 3. Rating analysis
# ---------------------------------------------------------------------------


def analyse_ratings(frame: pd.DataFrame, window_months: list[str]) -> RatingProfile:
    distribution = (
        frame.groupby(["rating", "platform"], observed=True).size().unstack(fill_value=0)
    )

    by_platform = frame.groupby("platform", observed=True).agg(
        reviews=("review_id", "size"),
        mean_rating=("rating", "mean"),
        pct_negative=("rating", lambda s: (s <= 2).mean() * 100),
        pct_neutral=("rating", lambda s: (s == 3).mean() * 100),
        pct_positive=("rating", lambda s: (s >= 4).mean() * 100),
    )

    over_time = (
        frame.groupby(["year_month", "platform"], observed=True)["rating"]
        .mean()
        .unstack()
        .sort_index()
    )

    # The confound test: if a platform's rating advantage survives restricting
    # to the shared window, it is not purely an artefact of when it was scraped.
    windowed = frame[frame["year_month"].isin(window_months)] if window_months else frame
    full_vs_window = pd.DataFrame(
        {
            "mean_rating_full": frame.groupby("platform", observed=True)["rating"].mean(),
            "mean_rating_window": windowed.groupby("platform", observed=True)["rating"].mean(),
            "n_full": frame.groupby("platform", observed=True).size(),
            "n_window": windowed.groupby("platform", observed=True).size(),
        }
    )
    full_vs_window["shift"] = (
        full_vs_window["mean_rating_window"] - full_vs_window["mean_rating_full"]
    )

    # Does the platform ORDER change once date coverage is equalised? This is
    # computed rather than asserted, because it is the sharpest available test
    # of whether a cross-platform rating claim is safe to make.
    rank_full = full_vs_window["mean_rating_full"].rank(ascending=False)
    rank_window = full_vs_window["mean_rating_window"].rank(ascending=False)
    reordered = [
        str(platform)
        for platform in full_vs_window.index
        if rank_full[platform] != rank_window[platform]
    ]
    largest_shift = full_vs_window["shift"].abs().max()

    if reordered:
        swapped = ", ".join(sorted(reordered))
        confound_note = (
            f"**The platform ranking CHANGES when date coverage is equalised.** Restricting "
            f"to the comparison window moves {swapped} relative to each other, with the "
            f"largest single shift being {largest_shift:+.2f} stars. This is direct evidence "
            "that full-corpus rating comparisons on this dataset are contaminated by *when* "
            "each platform was scraped, not only by how customers rated it. Any cross-platform "
            "rating statement must therefore be restricted to the comparison window. Even "
            "then it describes THIS SAMPLE of reviews, not customer satisfaction: the corpus "
            "is complaint-biased and self-selected, and platforms may have been scraped under "
            "different sort orders or filters."
        )
    else:
        confound_note = (
            f"The platform ranking is unchanged when date coverage is equalised (largest "
            f"shift {largest_shift:+.2f} stars), so the ordering is not purely an artefact of "
            "differing coverage. It still describes THIS SAMPLE of reviews rather than "
            "customer satisfaction."
        )

    return RatingProfile(
        distribution=distribution,
        by_platform=by_platform,
        over_time=over_time,
        full_vs_window=full_vs_window,
        ranking_changes_in_window=bool(reordered),
        reordered_platforms=sorted(reordered),
        confound_note=confound_note,
    )


# ---------------------------------------------------------------------------
# 4. Review text analysis
# ---------------------------------------------------------------------------

_STOPWORDS = set(
    """a an the and or but if of to in on at for with is are was were be been being it its
    i my me we our you your they them their he she this that these those have has had do does
    did not no so very just too also can could would should will shall may might must am as by
    from about into over under again then than there here when where what which who whom how
    why all any both each more most other some such only own same now app get got getting use
    using used one two because after before while during out up down off ive dont didnt cant
    im theyre its dont didnt ive cant wont thats""".split()
)


def _top_terms(series: pd.Series, limit: int = 25) -> list[tuple[str, int]]:
    """Document frequency of terms, not raw term frequency.

    Document frequency answers "how many customers mentioned this", which is the
    product question. Raw frequency lets one long ranting review dominate.
    """
    from collections import Counter
    import re

    counter: Counter[str] = Counter()
    for text in series:
        tokens = {
            token
            for token in re.findall(r"[a-z']+", str(text).lower())
            if len(token) > 2 and token not in _STOPWORDS
        }
        counter.update(tokens)
    return counter.most_common(limit)


def analyse_text(frame: pd.DataFrame) -> TextProfile:
    length_by_platform = frame.groupby("platform", observed=True).agg(
        median_chars=("char_len", "median"),
        mean_chars=("char_len", "mean"),
        median_words=("word_count", "median"),
        pct_truncated=("is_truncated", lambda s: s.mean() * 100),
    )

    short_cutoff = frame["char_len"].quantile(0.05)
    long_cutoff = frame["char_len"].quantile(0.95)
    very_short = frame[frame["char_len"] <= short_cutoff]
    very_long = frame[frame["char_len"] >= long_cutoff]

    truncated = frame[frame["is_truncated"]]
    intact = frame[~frame["is_truncated"]]
    truncated_comparison = {
        "truncated_n": float(len(truncated)),
        "truncated_mean_rating": float(truncated["rating"].mean()),
        "intact_mean_rating": float(intact["rating"].mean()),
        "truncated_pct_negative": float((truncated["rating"] <= 2).mean() * 100),
        "intact_pct_negative": float((intact["rating"] <= 2).mean() * 100),
    }

    templates = (
        frame[frame["near_dup_group_id"] >= 0]
        .groupby("near_dup_group_id")
        .agg(
            members=("review_id", "size"),
            platforms=("platform", lambda s: ", ".join(sorted(set(s.astype(str))))),
            mean_rating=("rating", "mean"),
            sample=("review_text", lambda s: str(s.iloc[0])[:110]),
        )
        .sort_values("members", ascending=False)
    )

    return TextProfile(
        length_by_platform=length_by_platform,
        very_short=very_short,
        very_long=very_long,
        truncated_comparison=truncated_comparison,
        top_terms_negative=_top_terms(frame.loc[frame["rating"] <= 2, "review_text"]),
        top_terms_positive=_top_terms(frame.loc[frame["rating"] >= 4, "review_text"]),
        template_groups=templates,
    )


# ---------------------------------------------------------------------------
# 5. Theme exploration (probe-based estimates)
# ---------------------------------------------------------------------------


def analyse_themes(
    frame: pd.DataFrame, taxonomy: Taxonomy, window_months: list[str]
) -> ThemeProfile:
    """Theme shares by platform, on the full corpus and the comparison window.

    These are EXPLORATORY ESTIMATES from keyword probes. They size areas and
    surface candidate differences; they are not the classification, and nothing
    downstream treats them as labels.

    Computing both views matters: if a platform gap disappears once date
    coverage is equalised, the gap was a sampling artefact.
    """
    def shares(subset: pd.DataFrame) -> pd.DataFrame:
        hits = apply_probes(subset, taxonomy)
        rows = {}
        for platform, index in subset.groupby("platform", observed=True).groups.items():
            rows[str(platform)] = hits.loc[index].mean() * 100
        result = pd.DataFrame(rows)
        result["all"] = hits.mean() * 100
        return result

    full = shares(frame)
    windowed = shares(frame[frame["year_month"].isin(window_months)]) if window_months else full

    platforms = [column for column in full.columns if column != "all"]
    shift = pd.DataFrame(
        {platform: windowed[platform] - full[platform] for platform in platforms}
    )

    # A theme is "shared" when every platform sits within a narrow band of the
    # others; "platform-specific" when one platform stands well clear.
    shared: list[str] = []
    specific: dict[str, list[str]] = {platform: [] for platform in platforms}
    for area in windowed.index:
        values = windowed.loc[area, platforms]
        spread = values.max() / max(values.min(), 0.1)
        if spread < 1.5:
            shared.append(str(area))
        elif spread >= 2.0:
            specific[str(values.idxmax())].append(str(area))

    return ThemeProfile(
        share_full=full,
        share_window=windowed,
        window_vs_full_shift=shift,
        shared_themes=shared,
        platform_specific=specific,
    )


# ---------------------------------------------------------------------------
# 6. Validation / reconciliation
# ---------------------------------------------------------------------------


def reconcile(frame: pd.DataFrame, cleaning_report: dict) -> dict[str, object]:
    """Check the EDA input against what Phase 1 said it produced.

    Guards against the quiet failure mode where an analysis silently runs on a
    stale, partial, or re-derived dataset and every number after it is wrong.
    """
    cleaning = cleaning_report["cleaning"]
    checks: dict[str, object] = {}

    checks["row_count_matches"] = len(frame) == cleaning["rows_out"]
    checks["rows_expected"] = cleaning["rows_out"]
    checks["rows_found"] = len(frame)

    observed_platforms = {str(k): int(v) for k, v in frame["platform"].value_counts().items()}
    checks["platform_counts_match"] = observed_platforms == cleaning["platform_counts"]
    checks["platform_counts"] = observed_platforms

    observed_ratings = {
        str(k): int(v) for k, v in frame["rating"].value_counts().sort_index().items()
    }
    checks["rating_counts_match"] = observed_ratings == cleaning["rating_counts"]
    checks["rating_counts"] = observed_ratings

    checks["null_cells"] = int(frame.isna().sum().sum())
    checks["no_nulls"] = checks["null_cells"] == 0

    checks["review_ids_unique"] = bool(frame["review_id"].is_unique)
    checks["dates_valid"] = bool(
        frame["review_date"].notna().all()
        and str(frame["review_date"].min().date()) == cleaning["date_min"]
        and str(frame["review_date"].max().date()) == cleaning["date_max"]
    )
    checks["ratings_in_range"] = bool(frame["rating"].between(1, 5).all())
    checks["truncated_matches"] = int(frame["is_truncated"].sum()) == cleaning["truncated_reviews"]

    checks["all_passed"] = all(
        bool(checks[key])
        for key in (
            "row_count_matches",
            "platform_counts_match",
            "rating_counts_match",
            "no_nulls",
            "review_ids_unique",
            "dates_valid",
            "ratings_in_range",
            "truncated_matches",
        )
    )
    return checks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_eda(frame: pd.DataFrame, taxonomy: Taxonomy, cleaning_report: dict) -> EDAResult:
    temporal = analyse_temporal(frame)
    return EDAResult(
        profile=profile_dataset(frame),
        temporal=temporal,
        ratings=analyse_ratings(frame, temporal.months_meeting_threshold),
        text=analyse_text(frame),
        themes=analyse_themes(frame, taxonomy, temporal.months_meeting_threshold),
        reconciliation=reconcile(frame, cleaning_report),
    )
