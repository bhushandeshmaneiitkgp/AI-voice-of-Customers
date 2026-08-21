"""
Central configuration for the VoC Copilot.

Two responsibilities, deliberately kept separate from anything that touches data:

1. ``Paths``    - every filesystem location the pipeline uses, derived from the
                  repo root, so scripts behave identically no matter which
                  directory you run them from.
2. ``Settings`` - tunables and secrets, loaded from the environment / ``.env``.

Design rules enforced here:
  * No secret is ever a literal in source. The API key is read from the
    environment only, and is optional until Phase 3 needs it.
  * No model ID is hardcoded in pipeline code. Models come from
    ``config/models.yaml`` and are selected by environment variable, so the
    same code can be benchmarked across models (Decision 1).
  * Nothing in this module reads or writes data files.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# config/settings.py -> config/ -> <repo root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


class Paths:
    """Filesystem layout. All paths are absolute and derived from PROJECT_ROOT."""

    root = PROJECT_ROOT

    config_dir = PROJECT_ROOT / "config"
    model_registry = PROJECT_ROOT / "config" / "models.yaml"
    taxonomy = PROJECT_ROOT / "config" / "taxonomy.yaml"

    docs_dir = PROJECT_ROOT / "docs"

    data = PROJECT_ROOT / "data"

    # ---- IMMUTABLE. Nothing in this codebase may open these for writing. ----
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_reviews = PROJECT_ROOT / "data" / "raw" / "reviews.csv"

    # ---- Derived, regenerable outputs. ----
    interim_dir = PROJECT_ROOT / "data" / "interim"
    clean_reviews = PROJECT_ROOT / "data" / "interim" / "reviews_clean.parquet"
    clean_report = PROJECT_ROOT / "data" / "interim" / "cleaning_report.json"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    eval_dir = PROJECT_ROOT / "data" / "eval"

    artifacts_dir = PROJECT_ROOT / "artifacts"
    reports_dir = PROJECT_ROOT / "reports"
    data_profile = PROJECT_ROOT / "reports" / "data_profile.md"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    eda_report = PROJECT_ROOT / "reports" / "EDA_REPORT.md"
    taxonomy_discovery_report = PROJECT_ROOT / "reports" / "taxonomy_discovery.md"
    taxonomy_discovery_areas = PROJECT_ROOT / "data" / "processed" / "taxonomy_discovery_areas.csv"
    taxonomy_discovery_matrix = PROJECT_ROOT / "data" / "processed" / "taxonomy_discovery_cooccurrence.csv"

    @classmethod
    def ensure_output_dirs(cls) -> None:
        """Create every writable directory. Never touches ``raw_dir``."""
        for directory in (
            cls.interim_dir,
            cls.processed_dir,
            cls.eval_dir,
            cls.artifacts_dir,
            cls.reports_dir,
            cls.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model registry  (Decision 1)
# ---------------------------------------------------------------------------


class ModelProfile(BaseModel):
    """One selectable model configuration, loaded from ``config/models.yaml``."""

    # pydantic reserves the ``model_`` prefix for its own attributes. We keep
    # ``model_id`` because that is the API's name for the field, so we opt out
    # of the protected namespace rather than rename it to something misleading.
    model_config = ConfigDict(protected_namespaces=(), frozen=True)

    key: str
    model_id: str
    display_name: str
    tier: str
    context_window: int
    input_price_per_mtok: float
    output_price_per_mtok: float
    batch_discount: float = 0.5
    thinking_style: Literal["adaptive", "budget"] = "adaptive"
    supports_effort: bool = True
    default_effort: str | None = "high"
    notes: str = ""

    def estimate_cost_usd(
        self,
        input_tokens: int,
        output_tokens: int,
        use_batch: bool = False,
    ) -> float:
        """Estimated USD cost for a given token workload.

        Used to print a cost estimate *before* spending money, so a
        misconfigured run is caught at the estimate rather than at the bill.
        """
        multiplier = self.batch_discount if use_batch else 1.0
        dollars = (
            input_tokens / 1_000_000 * self.input_price_per_mtok
            + output_tokens / 1_000_000 * self.output_price_per_mtok
        )
        return multiplier * dollars


@lru_cache(maxsize=1)
def load_model_registry(path: Path | None = None) -> dict[str, ModelProfile]:
    """Parse ``config/models.yaml`` into validated ``ModelProfile`` objects."""
    registry_path = path or Paths.model_registry
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. "
            "It is required so that model IDs are never hardcoded in pipeline code."
        )
    raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    profiles = (raw or {}).get("profiles") or {}
    if not profiles:
        raise ValueError(f"No 'profiles:' block found in {registry_path}")
    return {key: ModelProfile(key=key, **cfg) for key, cfg in profiles.items()}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Runtime settings, read from environment variables and ``.env``.

    Every field below is overridable without editing code, e.g.::

        VOC_ENRICHMENT_MODEL=haiku python scripts/03_run_enrichment.py
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="VOC_",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Secrets -----------------------------------------------------------
    # No VOC_ prefix: this is the SDK's conventional variable name.
    # Optional here because Phase 1 (ingest/clean) must run without a key.
    anthropic_api_key: str | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )

    # --- Model selection ---------------------------------------------------
    enrichment_model: str = "opus"
    synthesis_model: str = "opus"
    use_batch_api: bool = True

    # --- Cleaning parameters ----------------------------------------------
    # Justification for 500: the raw file shows a hard pile-up at exactly
    # 500 characters (84 reviews at 500, 622 in 480-500, vs 159 in 460-480).
    # That is a scraper cap, not natural writing length.
    truncation_cap_chars: int = 500
    truncation_tolerance: int = 5

    # Cosine similarity at or above which two reviews are near-duplicates.
    near_dup_threshold: float = 0.80

    # Reviews shorter than this after normalisation are unusable.
    min_review_chars: int = 10

    # --- Dev conveniences --------------------------------------------------
    sample_limit: int = 0  # 0 = process everything
    log_level: str = "INFO"

    @field_validator("near_dup_threshold")
    @classmethod
    def _check_threshold(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError(f"near_dup_threshold must be in (0, 1], got {value}")
        return value

    # --- Derived accessors -------------------------------------------------

    @property
    def enrichment_profile(self) -> ModelProfile:
        return self._resolve(self.enrichment_model, "VOC_ENRICHMENT_MODEL")

    @property
    def synthesis_profile(self) -> ModelProfile:
        return self._resolve(self.synthesis_model, "VOC_SYNTHESIS_MODEL")

    @staticmethod
    def _resolve(key: str, var_name: str) -> ModelProfile:
        registry = load_model_registry()
        if key not in registry:
            raise ValueError(
                f"{var_name}={key!r} is not defined in config/models.yaml. "
                f"Available: {sorted(registry)}"
            )
        return registry[key]

    def require_api_key(self) -> str:
        """Return the API key, or fail with an actionable message.

        Called only by phases that actually make API calls, so the Phase 1
        pipeline stays runnable with no credentials configured at all.
        """
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set.\n"
                "  1. Copy .env.example to .env\n"
                "  2. Add your key from https://console.anthropic.com/settings/keys\n"
                "Never paste the key into a tracked source file."
            )
        return self.anthropic_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Prefer this over instantiating ``Settings``."""
    return Settings()
