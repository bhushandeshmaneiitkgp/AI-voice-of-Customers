"""Tests for the dataset profile (config/dataset.yaml).

Two jobs, and both matter equally:

1. **The current corpus is unchanged.** Making platforms and date formats
   configurable must be a no-op for the 4,620-review dataset. If any of these
   fail, the refactor altered behaviour rather than relocating it.
2. **Another business's corpus can be declared.** Proven by loading a synthetic
   config with different platforms and a different date format and pushing rows
   through the real validators.

Validation stays strict throughout. Several tests below exist specifically to
prove the checks still *reject* things — configurability is not permissiveness.
"""

from __future__ import annotations

from datetime import date

import pytest
import yaml
from pydantic import ValidationError

from config.settings import (
    DatasetConfig,
    Paths,
    get_dataset_config,
    load_dataset_config,
)
from voc.clean import parse_review_date
from voc.prompts import build_system_prompt
from voc.schemas import RawReviewRow, accepted_date_formats, allowed_platforms
from voc.taxonomy import get_taxonomy


@pytest.fixture(scope="module")
def dataset() -> DatasetConfig:
    return get_dataset_config()


def _write_config(tmp_path, **overrides) -> DatasetConfig:
    """Build a synthetic dataset profile for a different business."""
    payload = {
        "name": "food_delivery_demo",
        "display_name": "Food Delivery Demo",
        "description": "Synthetic corpus for reuse testing.",
        "platforms": [
            {"id": "swiggy", "display_name": "Swiggy"},
            {"id": "zomato", "display_name": "Zomato"},
        ],
        "date_formats": ["%Y-%m-%d"],
        "domain": {
            "description": "Indian food delivery apps",
            "entity_noun": "platform",
            "reviewer_context": "",
        },
    }
    payload.update(overrides)
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_dataset_config(path)


# ---------------------------------------------------------------------------
# The shipped config is valid and unchanged
# ---------------------------------------------------------------------------


def test_dataset_config_file_exists() -> None:
    assert Paths.dataset_config.exists()


def test_shipped_config_loads(dataset: DatasetConfig) -> None:
    assert dataset.name == "quickcommerce_in"
    assert dataset.display_name


def test_current_platforms_are_preserved(dataset: DatasetConfig) -> None:
    """Regression guard: the refactor must not change the existing corpus."""
    assert dataset.platform_ids == ("blinkit", "jiomart", "zepto")
    assert allowed_platforms() == ("blinkit", "jiomart", "zepto")


def test_current_date_format_is_preserved(dataset: DatasetConfig) -> None:
    """Exactly one format, the same one as before the refactor.

    Declaring formats the corpus does not contain would widen what ingestion
    accepts, which is a behaviour change dressed up as configurability.
    """
    assert dataset.date_formats == ["%d %B %Y"]
    assert parse_review_date("30 December 2024") == date(2024, 12, 30)


def test_current_domain_description_is_preserved(dataset: DatasetConfig) -> None:
    assert "quick-commerce" in dataset.domain.description


def test_display_names_are_declared(dataset: DatasetConfig) -> None:
    """The prompt needs proper brand casing, not lowercase ids."""
    assert set(dataset.platform_display_names) == {"Blinkit", "JioMart", "Zepto"}
    assert dataset.display_name_for("jiomart") == "JioMart"


def test_display_name_falls_back_to_the_id(dataset: DatasetConfig) -> None:
    assert dataset.display_name_for("unknown_platform") == "unknown_platform"


# ---------------------------------------------------------------------------
# Requirement 1: configurable platforms -- still STRICT
# ---------------------------------------------------------------------------


def test_a_different_corpus_can_declare_its_own_platforms(tmp_path) -> None:
    config = _write_config(tmp_path)
    assert config.platform_ids == ("swiggy", "zomato")


def test_platform_validation_is_configurable_not_removed(tmp_path, monkeypatch) -> None:
    """The whole point: the gate moves, it does not open.

    Under the food-delivery profile, 'swiggy' is accepted and 'blinkit' is
    rejected -- the exact inverse of the shipped config. A permissive
    implementation would accept both.
    """
    config = _write_config(tmp_path)
    monkeypatch.setattr("voc.schemas.get_dataset_config", lambda: config)

    accepted = RawReviewRow(
        rating=1, date="2025-03-14", review="courier never arrived", platform="swiggy"
    )
    assert accepted.platform == "swiggy"

    with pytest.raises(ValidationError):
        RawReviewRow(
            rating=1, date="2025-03-14", review="text", platform="blinkit"
        )


def test_unknown_platform_message_points_at_the_config() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RawReviewRow(rating=1, date="30 December 2024", review="text", platform="deliveroo")

    message = str(excinfo.value)
    assert "config/dataset.yaml" in message
    assert "do not loosen" in message.lower()


def test_duplicate_platform_ids_are_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError, match="Duplicate platform ids"):
        _write_config(
            tmp_path,
            platforms=[
                {"id": "swiggy", "display_name": "Swiggy"},
                {"id": "swiggy", "display_name": "Swiggy Again"},
            ],
        )


def test_empty_platform_list_is_rejected(tmp_path) -> None:
    """A corpus with no declared platforms would accept nothing at all."""
    with pytest.raises(ValidationError):
        _write_config(tmp_path, platforms=[])


def test_platform_ids_must_be_lowercase_slugs(tmp_path) -> None:
    with pytest.raises(ValidationError):
        _write_config(tmp_path, platforms=[{"id": "Swiggy", "display_name": "Swiggy"}])


# ---------------------------------------------------------------------------
# Requirement 2: configurable date formats -- still STRICT
# ---------------------------------------------------------------------------


def test_alternative_date_format_parses(tmp_path) -> None:
    config = _write_config(tmp_path)
    assert parse_review_date("2025-03-14", config.date_formats) == date(2025, 3, 14)


def test_formats_are_tried_in_order(tmp_path) -> None:
    config = _write_config(tmp_path, date_formats=["%d %B %Y", "%Y-%m-%d"])
    assert parse_review_date("30 December 2024", config.date_formats) == date(2024, 12, 30)
    assert parse_review_date("2025-03-14", config.date_formats) == date(2025, 3, 14)


@pytest.mark.parametrize("bad", ["30/12/2024", "Dec 30 2024", "", "next tuesday"])
def test_unparseable_dates_still_raise(bad: str) -> None:
    """Configurable does not mean permissive: an undeclared format is an error."""
    with pytest.raises(ValueError, match="matches none of the configured formats"):
        parse_review_date(bad)


def test_date_error_names_the_patterns_tried() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_review_date("2025-03-14", ["%d %B %Y"])
    assert "%d %B %Y" in str(excinfo.value)
    assert "config/dataset.yaml" in str(excinfo.value)


def test_a_format_the_shipped_config_excludes_is_rejected() -> None:
    """US-style dates are not declared, so they must not silently parse."""
    with pytest.raises(ValueError):
        parse_review_date("12/30/2024")


def test_unusable_strptime_pattern_is_caught_at_load(tmp_path) -> None:
    """Fail on the config file, not on row 4,000 of the corpus."""
    with pytest.raises(ValidationError, match="not a usable strptime pattern"):
        _write_config(tmp_path, date_formats=["%Q not a real directive %"])


def test_empty_date_format_list_is_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError):
        _write_config(tmp_path, date_formats=[])


# ---------------------------------------------------------------------------
# Requirement 3: configurable domain description in the prompt
# ---------------------------------------------------------------------------


def test_prompt_uses_the_configured_domain(dataset: DatasetConfig) -> None:
    prompt = build_system_prompt(get_taxonomy(), dataset)
    assert dataset.domain.description in prompt


def test_prompt_brands_come_from_config_not_python(tmp_path) -> None:
    """The decisive test: swap the config, and the prompt's domain sentence swaps.

    Scoped to the domain sentence on purpose. The taxonomy is *also*
    domain-specific and is *also* swappable, but via its own YAML file -- a
    foreign corpus supplies both. Asserting the whole prompt is brand-free would
    be asserting something this function does not control, and would fail on
    taxonomy text like "Zepto Cash" that is correctly there.
    """
    config = _write_config(tmp_path)
    header = "\n".join(build_system_prompt(get_taxonomy(), config).splitlines()[:4])

    assert "Indian food delivery apps" in header
    assert "Swiggy" in header and "Zomato" in header
    for leaked in ("Blinkit", "Zepto", "JioMart", "quick-commerce"):
        assert leaked not in header, f"{leaked!r} leaked into a foreign-corpus prompt"


def test_shipped_config_produces_the_original_domain_sentence(dataset: DatasetConfig) -> None:
    """Behaviour preserved: the current corpus still gets its original framing."""
    header = "\n".join(build_system_prompt(get_taxonomy(), dataset).splitlines()[:4])

    assert "quick-commerce" in header
    for brand in ("Blinkit", "JioMart", "Zepto"):
        assert brand in header


def test_no_brand_names_are_hardcoded_in_the_prompt_module() -> None:
    """Executable form of the requirement: brands live in YAML, not Python."""
    source = (Paths.root / "src" / "voc" / "prompts.py").read_text(encoding="utf-8")
    for brand in ("Blinkit", "Zepto", "JioMart", "quick-commerce"):
        assert brand not in source, f"{brand!r} is hardcoded in prompts.py"


def test_reviewer_context_is_optional(tmp_path) -> None:
    config = _write_config(tmp_path)
    assert config.domain.reviewer_context == ""
    prompt = build_system_prompt(get_taxonomy(), config)
    assert prompt.startswith("You are a product analyst")


def test_reviewer_context_reaches_the_prompt(tmp_path) -> None:
    config = _write_config(
        tmp_path,
        domain={
            "description": "Indian food delivery apps",
            "entity_noun": "platform",
            "reviewer_context": "SENTINEL CONTEXT LINE",
        },
    )
    assert "SENTINEL CONTEXT LINE" in build_system_prompt(get_taxonomy(), config)


def test_prompt_remains_byte_stable_for_caching(dataset: DatasetConfig) -> None:
    """Config-driven text must not introduce per-call variation."""
    taxonomy = get_taxonomy()
    assert build_system_prompt(taxonomy, dataset) == build_system_prompt(taxonomy, dataset)


# ---------------------------------------------------------------------------
# Loader behaviour
# ---------------------------------------------------------------------------


def test_missing_config_file_is_actionable(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="platforms"):
        load_dataset_config(tmp_path / "absent.yaml")


def test_empty_config_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "dataset.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_dataset_config(path)


def test_dataset_name_must_be_a_slug(tmp_path) -> None:
    """The name is intended for output namespacing, so keep it filesystem-safe."""
    with pytest.raises(ValidationError):
        _write_config(tmp_path, name="Food Delivery Demo")


def test_accepted_date_formats_reads_from_config(dataset: DatasetConfig) -> None:
    assert accepted_date_formats() == tuple(dataset.date_formats)
