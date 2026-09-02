"""
Artefact loading for the dashboard.

Kept out of the Streamlit file on purpose. Everything here is a plain function
over paths and DataFrames, so the part of the UI that can actually be wrong --
which phase has run, what to show when it has not, how a rate becomes a label --
is testable without launching a browser.

The dashboard is the first place a reader meets these numbers without having
read the report that qualifies them, so the loading layer carries the
availability logic too: a page whose inputs are missing says which script would
produce them, rather than rendering an empty chart that looks like a finding of
zero.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Artefact:
    """One pipeline output and the command that produces it."""

    key: str
    path: Path
    produced_by: str
    phase: str

    @property
    def available(self) -> bool:
        return self.path.exists()


@dataclass
class PipelineState:
    """Which phases have produced output, and what to run for the rest."""

    artefacts: dict[str, Artefact] = field(default_factory=dict)

    def available(self, *keys: str) -> bool:
        return all(self.artefacts[k].available for k in keys if k in self.artefacts)

    def missing(self, *keys: str) -> list[Artefact]:
        return [
            self.artefacts[k] for k in keys
            if k in self.artefacts and not self.artefacts[k].available
        ]

    def instructions_for(self, *keys: str) -> list[str]:
        """De-duplicated commands that would produce the missing inputs.

        De-duplicated because one script often produces several artefacts, and
        telling someone to run it three times is how a clear message becomes
        noise.
        """
        return list(dict.fromkeys(a.produced_by for a in self.missing(*keys)))


def build_state(paths) -> PipelineState:
    """Describe every artefact the dashboard can read.

    ``paths`` is ``config.settings.Paths``; passed in rather than imported so a
    test can point the whole dashboard at a temporary directory.
    """
    definitions = [
        ("clean", paths.clean_reviews, "python scripts/01_build_clean.py", "1"),
        ("reviews", paths.enriched_reviews, "python scripts/04_run_enrichment.py --all", "3"),
        ("labels", paths.enriched_labels, "python scripts/04_run_enrichment.py --all", "3"),
        ("enrichment_report", paths.enrichment_report, "python scripts/04_run_enrichment.py --all", "3"),
        ("embeddings", paths.embeddings, "python scripts/05_build_embeddings.py", "4"),
        ("faiss", paths.faiss_index, "python scripts/05_build_embeddings.py", "4"),
        ("pain_points", paths.pain_points, "python scripts/06_discover_painpoints.py", "4"),
        ("clusters", paths.cluster_summary, "python scripts/06_discover_painpoints.py", "4"),
        ("platform_metrics", paths.platform_metrics, "python scripts/07_analyse_trends.py", "5"),
        ("comparisons", paths.platform_comparisons, "python scripts/07_analyse_trends.py", "5"),
        ("area_rates", paths.area_rates, "python scripts/07_analyse_trends.py", "5"),
        ("root_causes", paths.root_causes, "python scripts/08_root_cause.py", "6"),
        ("opportunities", paths.opportunities, "python scripts/09_build_roadmap.py", "7"),
        ("rice", paths.rice_scores, "python scripts/09_build_roadmap.py", "7"),
        ("experiments", paths.experiment_plans, "python scripts/09_build_roadmap.py", "7"),
    ]
    return PipelineState(
        {key: Artefact(key, path, command, phase) for key, path, command, phase in definitions}
    )


def load_table(artefact: Artefact) -> pd.DataFrame:
    """Read one parquet artefact, or an empty frame if it is not there.

    Empty rather than raising: a dashboard that dies because Phase 7 has not run
    is less useful than one that shows Phases 1-6 and says so.
    """
    if not artefact.available:
        return pd.DataFrame()
    try:
        return pd.read_parquet(artefact.path)
    except Exception as exc:  # noqa: BLE001 - a corrupt artefact must not kill the app
        logger.error("Could not read %s: %s", artefact.path.name, exc)
        return pd.DataFrame()


def load_report(artefact: Artefact) -> dict:
    if not artefact.available:
        return {}
    try:
        return json.loads(artefact.path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Could not read %s: %s", artefact.path.name, exc)
        return {}


def split_packed(value: object, separator: str = " ||| ") -> list[str]:
    """Unpack a list column that was flattened for parquet storage.

    Phases 4 and 6 join lists into one string because list columns do not round
    trip cleanly through every parquet reader. This is the inverse.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value)
    if not text.strip():
        return []
    parts = text.split(separator) if separator in text else text.split()
    return [p for p in (part.strip() for part in parts) if p]


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------


def corpus_summary(reviews: pd.DataFrame, labels: pd.DataFrame, report: dict) -> dict:
    """Headline numbers for the overview page."""
    if reviews.empty:
        return {}

    window = (
        reviews[reviews["in_comparable_window"]]
        if "in_comparable_window" in reviews.columns else reviews
    )
    return {
        "reviews": len(reviews),
        "labels": len(labels),
        "areas_per_review": round(len(labels) / len(reviews), 2) if len(reviews) else 0.0,
        "platforms": reviews["platform"].nunique() if "platform" in reviews else 0,
        "months": reviews["year_month"].nunique() if "year_month" in reviews else 0,
        "comparable_reviews": len(window),
        "comparable_months": window["year_month"].nunique() if "year_month" in window else 0,
        "coverage_pct": report.get("coverage_pct"),
        "grounding_pct": (
            round(report["grounding"]["mean_rate"] * 100, 1)
            if report.get("grounding", {}).get("mean_rate") is not None else None
        ),
        "negative_share": (
            round((reviews["sentiment"] == "negative").mean() * 100, 1)
            if "sentiment" in reviews else None
        ),
    }


def significance_label(significant: object) -> str:
    """Turn the corrected verdict into words a reader cannot skim past.

    The dashboard is where an insignificant difference is most likely to be
    read as a finding: it sits in a table next to real ones, formatted
    identically. Saying "not established" rather than showing a bare p-value
    costs a column and prevents the most likely misreading in the product.
    """
    return "established" if bool(significant) else "not established"


def format_rate(rate: object, low: object = None, high: object = None) -> str:
    """Render a rate with its interval, so precision is visible at a glance."""
    if rate is None or (isinstance(rate, float) and pd.isna(rate)):
        return "—"
    text = f"{float(rate) * 100:.1f}%"
    if low is not None and high is not None and not pd.isna(low) and not pd.isna(high):
        text += f" ({float(low) * 100:.1f}–{float(high) * 100:.1f})"
    return text


def scoring_mode(rice: pd.DataFrame) -> str:
    """Whether the roadmap is a real RICE ranking or the partial one.

    Surfaced prominently in the UI: a RIC table and a RICE table look alike,
    and the difference is whether anyone has estimated effort.
    """
    if rice.empty or "scored" not in rice.columns:
        return "none"
    if bool(rice["scored"].any()):
        return "rice"
    return "ric"
