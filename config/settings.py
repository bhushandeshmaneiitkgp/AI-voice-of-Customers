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
from typing import ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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
    dataset_config = PROJECT_ROOT / "config" / "dataset.yaml"

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
    enriched_reviews = PROJECT_ROOT / "data" / "processed" / "reviews_enriched.parquet"
    enriched_labels = PROJECT_ROOT / "data" / "processed" / "review_labels.parquet"
    enrichment_report = PROJECT_ROOT / "data" / "processed" / "enrichment_report.json"
    eval_dir = PROJECT_ROOT / "data" / "eval"

    @classmethod
    def enrichment_cache(cls, model_key: str) -> Path:
        """Per-model response cache, so a model benchmark cannot cross-contaminate."""
        return cls.artifacts_dir / f"enrichment_cache_{model_key}.json"

    # ---- Phase 4: embeddings, index, clusters, pain points ----
    embeddings = PROJECT_ROOT / "artifacts" / "embeddings.npz"
    faiss_index = PROJECT_ROOT / "artifacts" / "reviews.faiss"
    review_clusters = PROJECT_ROOT / "data" / "processed" / "review_clusters.parquet"
    cluster_summary = PROJECT_ROOT / "data" / "processed" / "cluster_summary.parquet"
    pain_points = PROJECT_ROOT / "data" / "processed" / "pain_points.parquet"
    pain_point_report = PROJECT_ROOT / "reports" / "PAIN_POINTS.md"

    # ---- Phase 5: competitive metrics and trend ----
    platform_metrics = PROJECT_ROOT / "data" / "processed" / "platform_metrics.parquet"
    platform_comparisons = PROJECT_ROOT / "data" / "processed" / "platform_comparisons.parquet"
    area_rates = PROJECT_ROOT / "data" / "processed" / "area_rates_by_platform.parquet"
    monthly_rates = PROJECT_ROOT / "data" / "processed" / "monthly_rates.parquet"
    competitive_report = PROJECT_ROOT / "reports" / "COMPETITIVE.md"

    # ---- Phase 6: retrieval and root-cause hypotheses ----
    root_causes = PROJECT_ROOT / "data" / "processed" / "root_causes.parquet"
    root_cause_report = PROJECT_ROOT / "reports" / "ROOT_CAUSES.md"

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
# Dataset profile
# ---------------------------------------------------------------------------


class PlatformSpec(BaseModel):
    """One competitor/source the dataset covers."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str


class DomainSpec(BaseModel):
    """What this corpus is about, in the words the LLM prompt needs."""

    model_config = ConfigDict(frozen=True)

    description: str = Field(..., min_length=3)
    entity_noun: str = "platform"
    reviewer_context: str = ""


class DatasetConfig(BaseModel):
    """The dataset-specific half of the pipeline, loaded from config/dataset.yaml.

    Three things live here because they are properties of a *corpus*, not of the
    pipeline: which platforms exist, how its dates are written, and what domain
    the reviews come from. Everything else in the codebase is domain-agnostic.

    This makes validation configurable, not optional. An unlisted platform is
    still rejected; an unparseable date is still an error. What changes is that
    the allowed set travels with the dataset instead of being frozen in Python.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str
    description: str = ""
    platforms: list[PlatformSpec] = Field(..., min_length=1)
    date_formats: list[str] = Field(..., min_length=1)
    domain: DomainSpec

    @field_validator("date_formats")
    @classmethod
    def _formats_are_usable(cls, values: list[str]) -> list[str]:
        """Reject a pattern strptime cannot use, at load time rather than row 4,000."""
        from datetime import datetime

        for pattern in values:
            try:
                datetime.strptime(datetime(2024, 12, 30).strftime(pattern), pattern)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"date_format {pattern!r} is not a usable strptime pattern: {exc}")
        return values

    @model_validator(mode="after")
    def _platform_ids_unique(self) -> "DatasetConfig":
        ids = [platform.id for platform in self.platforms]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"Duplicate platform ids: {sorted(duplicates)}")
        return self

    @property
    def platform_ids(self) -> tuple[str, ...]:
        """Canonical ids, sorted so error messages and reports are stable."""
        return tuple(sorted(platform.id for platform in self.platforms))

    @property
    def platform_display_names(self) -> tuple[str, ...]:
        return tuple(platform.display_name for platform in self.platforms)

    def display_name_for(self, platform_id: str) -> str:
        for platform in self.platforms:
            if platform.id == platform_id:
                return platform.display_name
        return platform_id


def load_dataset_config(path: Path | None = None) -> DatasetConfig:
    """Read and validate ``config/dataset.yaml``."""
    config_path = path or Paths.dataset_config
    if not config_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found at {config_path}. It declares the platforms, "
            "date formats and domain for the corpus being processed."
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not raw:
        raise ValueError(f"{config_path} is empty")
    return DatasetConfig(**raw)


@lru_cache(maxsize=1)
def get_dataset_config() -> DatasetConfig:
    """Cached dataset profile.

    Honours ``VOC_DATASET_CONFIG`` so another corpus can be processed by
    pointing at a different YAML file, with no code change::

        VOC_DATASET_CONFIG=config/dataset_othercorp.yaml python scripts/01_build_clean.py
    """
    import os

    override = os.environ.get("VOC_DATASET_CONFIG")
    return load_dataset_config(Path(override) if override else None)


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

    #: Which vendor serves this model. "anthropic" uses the first-party SDK;
    #: anything else is treated as OpenAI-compatible and needs a base_url.
    provider: str = "anthropic"
    base_url: str | None = None

    #: How strongly the provider can constrain output shape. Drives whether the
    #: JSON schema is enforced by the API or merely described in the prompt.
    structured_output: Literal["json_schema", "json_object", "none"] = "json_schema"

    #: Only Anthropic offers a discounted async batch endpoint.
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
        # Fields with a validation_alias can otherwise ONLY be set by that
        # alias. Most alias names happen to case-match their field name, but
        # LLM_API_KEY does not -- so without this, constructing Settings in a
        # test or script silently ignores that argument.
        populate_by_name=True,
    )

    # --- Secrets -----------------------------------------------------------
    # No VOC_ prefix: this is the SDK's conventional variable name.
    # Optional here because Phase 1 (ingest/clean) must run without a key.
    anthropic_api_key: str | None = Field(
        default=None, validation_alias="ANTHROPIC_API_KEY"
    )
    openrouter_api_key: str | None = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )
    # Catch-all for other OpenAI-compatible endpoints (Groq, Together,
    # DeepSeek, Fireworks). Local servers such as Ollama need no key.
    openai_compatible_api_key: str | None = Field(
        default=None, validation_alias="LLM_API_KEY"
    )

    # --- Model selection ---------------------------------------------------
    # Defaults mirror `default_enrichment` / `default_synthesis` in
    # config/models.yaml; a test asserts the two stay in agreement.
    enrichment_model: str = "llama70b"
    synthesis_model: str = "opus"
    use_batch_api: bool = True

    # Overrides the model's default effort for the enrichment pass only.
    # Classification is a narrow task, so lower effort is often the right
    # trade: thinking tokens are billed as output and dominate the bill at
    # high effort. Ignored by models that do not support effort.
    enrichment_effort: str | None = None

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

    # --- Embeddings and clustering (Phase 4) -------------------------------
    # Local sentence-transformers, not an API. Embedding 4,620 reviews through
    # a hosted endpoint would add a per-run cost and a network dependency to
    # something that is deterministic and runs in under a minute on CPU.
    # all-MiniLM-L6-v2: 384 dims, ~90MB, and the standard baseline for short
    # English text. Overridable because the right model is an empirical
    # question, exactly as it is for enrichment.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    # Candidate cluster counts to score. Silhouette picks between them rather
    # than a k chosen by eye, so the number is defensible and reproducible.
    # Lower bound is 3, not the observed optimum of 6: a range that starts at
    # the answer cannot demonstrate it is the answer. Silhouette on this corpus
    # rises 0.102 (k=3) to 0.123 (k=6) and falls after, so the peak is bracketed
    # on both sides and the choice is the data's rather than the default's.
    cluster_k_min: int = 3
    cluster_k_max: int = 24

    # A pain point below this many reviews is not a pattern, it is an anecdote.
    # Reporting singletons would bury the real signal under noise.
    min_pain_point_volume: int = 15

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

    #: Where to get a key for each provider, shown when one is missing.
    _KEY_SOURCES: ClassVar[dict[str, tuple[str, str]]] = {
        "anthropic": ("ANTHROPIC_API_KEY", "https://console.anthropic.com/settings/keys"),
        "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/keys"),
    }

    def require_api_key(self, provider: str = "anthropic") -> str:
        """Return the API key for a provider, or fail with an actionable message.

        Called only by phases that make API calls, so the Phase 1 and 2
        pipelines stay runnable with no credentials configured at all.

        Local providers (Ollama, vLLM) need no real key, so a placeholder is
        returned rather than blocking a run that would have worked.
        """
        candidates = {
            "anthropic": self.anthropic_api_key,
            "openrouter": self.openrouter_api_key,
        }
        key = candidates.get(provider) or self.openai_compatible_api_key
        if key:
            return key

        if provider in ("ollama", "local", "vllm"):
            return "not-needed"

        variable, url = self._KEY_SOURCES.get(
            provider, (f"{provider.upper()}_API_KEY or LLM_API_KEY", "the provider's dashboard")
        )
        raise RuntimeError(
            f"No API key found for provider {provider!r}.\n"
            f"  1. Copy .env.example to .env\n"
            f"  2. Set {variable} using a key from {url}\n"
            "Never paste a key into a tracked source file."
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Prefer this over instantiating ``Settings``."""
    return Settings()
