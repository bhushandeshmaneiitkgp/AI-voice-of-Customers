"""Tests for configuration and the model registry (Decision 1).

The point of the registry is that a model can be swapped by environment
variable, so the same pipeline can be benchmarked across models in Phase 9.
These tests protect that property.
"""

from __future__ import annotations

import pytest

from config.settings import Paths, Settings, load_model_registry


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_loads_expected_profiles() -> None:
    registry = load_model_registry()
    assert {"opus", "sonnet", "haiku"} <= set(registry)


def test_registry_profiles_are_well_formed() -> None:
    for key, profile in load_model_registry().items():
        assert profile.key == key
        assert profile.model_id, f"{key} has no model_id"
        # Locally hosted models are genuinely free, so zero is valid there and
        # only there -- a hosted model priced at zero is a transcription error.
        floor = 0.0 if profile.tier == "local" else 0.0001
        assert profile.input_price_per_mtok >= floor, key
        assert profile.output_price_per_mtok >= floor, key
        assert 0 < profile.batch_discount <= 1
        assert profile.context_window > 0


def test_settings_defaults_match_the_registry_defaults() -> None:
    """The YAML declares defaults and Settings mirrors them; drift is silent.

    Without this, ``default_enrichment`` in models.yaml could say one thing
    while an unconfigured run quietly used another model.
    """
    import yaml

    from config.settings import Settings

    raw = yaml.safe_load(Paths.model_registry.read_text(encoding="utf-8"))
    settings = Settings(_env_file=None)

    assert settings.enrichment_model == raw["default_enrichment"]
    assert settings.synthesis_model == raw["default_synthesis"]


def test_declared_defaults_exist_in_the_registry() -> None:
    import yaml

    raw = yaml.safe_load(Paths.model_registry.read_text(encoding="utf-8"))
    registry = load_model_registry()
    assert raw["default_enrichment"] in registry
    assert raw["default_synthesis"] in registry


def test_no_model_id_is_hardcoded_in_pipeline_code() -> None:
    """Model IDs must live in config/models.yaml, never in a .py file.

    This is the executable form of Decision 1. If someone later pastes a model
    string into pipeline code, the benchmark story quietly breaks -- so the
    test fails instead.
    """
    offenders: list[str] = []
    for path in list((Paths.root / "src").rglob("*.py")) + list(
        (Paths.root / "scripts").rglob("*.py")
    ):
        text = path.read_text(encoding="utf-8")
        if "claude-" in text:
            offenders.append(str(path.relative_to(Paths.root)))
    assert not offenders, f"Hardcoded model IDs found in: {offenders}"


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def test_cost_estimate_scales_with_tokens() -> None:
    opus = load_model_registry()["opus"]
    one_million_in = opus.estimate_cost_usd(1_000_000, 0)
    assert one_million_in == pytest.approx(opus.input_price_per_mtok)


def test_batch_api_halves_the_estimate() -> None:
    haiku = load_model_registry()["haiku"]
    standard = haiku.estimate_cost_usd(500_000, 700_000, use_batch=False)
    batched = haiku.estimate_cost_usd(500_000, 700_000, use_batch=True)
    assert batched == pytest.approx(standard * haiku.batch_discount)


def test_relative_model_costs_are_ordered() -> None:
    """Sanity check that the pricing table was not transcribed wrongly."""
    registry = load_model_registry()
    workload = dict(input_tokens=600_000, output_tokens=700_000, use_batch=True)
    assert (
        registry["haiku"].estimate_cost_usd(**workload)
        < registry["sonnet"].estimate_cost_usd(**workload)
        < registry["opus"].estimate_cost_usd(**workload)
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_model_selection_resolves_through_environment(monkeypatch) -> None:
    monkeypatch.setenv("VOC_ENRICHMENT_MODEL", "haiku")
    monkeypatch.setenv("VOC_SYNTHESIS_MODEL", "opus")

    settings = Settings(_env_file=None)

    assert settings.enrichment_profile.model_id == load_model_registry()["haiku"].model_id
    assert settings.synthesis_profile.model_id == load_model_registry()["opus"].model_id


def test_unknown_model_key_fails_with_available_options(monkeypatch) -> None:
    monkeypatch.setenv("VOC_ENRICHMENT_MODEL", "gpt-9")
    settings = Settings(_env_file=None)

    with pytest.raises(ValueError) as excinfo:
        _ = settings.enrichment_profile
    assert "Available:" in str(excinfo.value)


def test_missing_api_key_message_is_actionable(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(_env_file=None, anthropic_api_key=None)

    with pytest.raises(RuntimeError) as excinfo:
        settings.require_api_key()
    message = str(excinfo.value)
    assert ".env.example" in message
    assert "console.anthropic.com" in message


def test_phase1_runs_without_any_api_key(monkeypatch) -> None:
    """Ingestion and cleaning must work with zero credentials configured."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(_env_file=None, anthropic_api_key=None)
    assert settings.anthropic_api_key is None


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_near_dup_threshold_is_validated(bad: float) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, near_dup_threshold=bad)


def test_paths_never_expose_a_writable_raw_directory() -> None:
    """ensure_output_dirs must not create or target anything under data/raw."""
    writable = {
        Paths.interim_dir,
        Paths.processed_dir,
        Paths.eval_dir,
        Paths.artifacts_dir,
        Paths.reports_dir,
    }
    assert Paths.raw_dir not in writable
    assert all(Paths.raw_dir not in directory.parents for directory in writable)
