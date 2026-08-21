"""
Taxonomy loading and validation.

The taxonomy lives entirely in ``config/taxonomy.yaml``. This module is the
only thing that reads it, and it validates the structure on load so a typo in
YAML fails immediately rather than producing a silently mislabelled corpus in
Phase 3.

Nothing here hardcodes an area name, an issue type, or a keyword pattern --
a test (``test_no_taxonomy_terms_hardcoded_in_python``) enforces that rule
across the whole codebase. The reason is practical: the taxonomy will change
as we learn, and a definition that exists in two places will drift.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from config.settings import Paths


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class Domain(BaseModel):
    """A dashboard grouping. Rolls up from product areas; never classified to."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str


class LabelType(BaseModel):
    """An issue_type or strength_type: the polarity-specific label under an area."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    evidence_phrases: list[str] = Field(default_factory=list)


class ProductArea(BaseModel):
    """One product area (L1) with its issue types and strength types (L2).

    ``inclusion`` and ``exclusion`` are not documentation garnish -- they are
    fed verbatim into the Phase 3 classification prompt. Boundary rules the
    model can read are what keep multi-label output consistent.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    domain: str
    definition: str
    inclusion: list[str]
    exclusion: list[str]
    discovery_estimate_pct: float = Field(..., ge=0.0, le=100.0)
    positive_negative_ratio: float = Field(..., ge=0.0)
    issue_types: list[LabelType]
    strength_types: list[LabelType] = Field(default_factory=list)
    platform_note: str | None = None
    downstream_note: str | None = None

    @model_validator(mode="after")
    def _label_ids_unique_within_area(self) -> "ProductArea":
        for label, items in (("issue_types", self.issue_types), ("strength_types", self.strength_types)):
            ids = [item.id for item in items]
            duplicates = {i for i in ids if ids.count(i) > 1}
            if duplicates:
                raise ValueError(f"{self.id}.{label} has duplicate ids: {sorted(duplicates)}")
        return self


class AttributeValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    definition: str


class Attribute(BaseModel):
    """A review-level attribute, deliberately kept out of the area vocabulary."""

    model_config = ConfigDict(frozen=True)

    id: str
    description: str
    type: str
    values: list[AttributeValue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enums_have_values(self) -> "Attribute":
        if self.type == "single_select" and not self.values:
            raise ValueError(f"attribute {self.id!r} is single_select but defines no values")
        return self


class FallbackArea(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    definition: str
    monitoring_rule: str


class Taxonomy(BaseModel):
    """The validated taxonomy. Loaded once, treated as read-only everywhere."""

    model_config = ConfigDict(frozen=True)

    version: str
    derived_from: dict
    domains: list[Domain]
    product_areas: list[ProductArea]
    attributes: list[Attribute]
    borderline_rules: list[str]
    special_areas: dict[str, str]
    fallback_area: FallbackArea
    dataset_caveats: dict[str, str]
    discovery_probes: dict[str, str]

    # -- integrity checks ---------------------------------------------------

    @model_validator(mode="after")
    def _validate_graph(self) -> "Taxonomy":
        area_ids = [area.id for area in self.product_areas]

        duplicates = {i for i in area_ids if area_ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"Duplicate product_area ids: {sorted(duplicates)}")

        if self.fallback_area.id in area_ids:
            raise ValueError(
                f"fallback_area id {self.fallback_area.id!r} collides with a real product area"
            )

        # Every area must point at a domain that exists.
        domain_ids = {domain.id for domain in self.domains}
        orphans = {area.id: area.domain for area in self.product_areas if area.domain not in domain_ids}
        if orphans:
            raise ValueError(f"Areas reference unknown domains: {orphans}")

        # Issue and strength type ids must be globally unique, so a label can
        # be resolved back to exactly one parent area without ambiguity.
        seen: dict[str, str] = {}
        for area in self.product_areas:
            for item in [*area.issue_types, *area.strength_types]:
                if item.id in seen:
                    raise ValueError(
                        f"Label id {item.id!r} appears in both {seen[item.id]!r} and {area.id!r}; "
                        "ids must be globally unique so a label maps to one parent area."
                    )
                seen[item.id] = area.id

        # Special-area pointers must resolve to real areas.
        for role, target in self.special_areas.items():
            if target not in area_ids:
                raise ValueError(
                    f"special_areas.{role} points at unknown area {target!r}. Known: {area_ids}"
                )

        # Probes must cover exactly the declared areas -- no orphan probe, no
        # area silently missing from the discovery numbers.
        probe_ids = set(self.discovery_probes)
        missing = set(area_ids) - probe_ids
        extra = probe_ids - set(area_ids)
        if missing:
            raise ValueError(f"Areas with no discovery probe: {sorted(missing)}")
        if extra:
            raise ValueError(f"Probes with no matching product area: {sorted(extra)}")

        for area_id, pattern in self.discovery_probes.items():
            # Guard against a subtle YAML trap: inside a DOUBLE-quoted scalar,
            # "\b" is the backspace escape, so a word-boundary anchor silently
            # becomes a control character and that alternative matches nothing.
            # The failure is invisible -- the regex still compiles and still
            # returns plausible counts. Probes with backslashes must therefore
            # use single-quoted YAML, and this check enforces it.
            control_chars = {char for char in pattern if ord(char) < 32}
            if control_chars:
                raise ValueError(
                    f"Probe for {area_id!r} contains control character(s) "
                    f"{[hex(ord(c)) for c in control_chars]}. This almost always means "
                    "a regex escape like \\b was written inside a double-quoted YAML "
                    "scalar and parsed as an escape sequence. Use single quotes."
                )
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Probe for {area_id!r} is not a valid regex: {exc}") from exc

        return self

    # -- accessors ----------------------------------------------------------

    @property
    def area_ids(self) -> list[str]:
        return [area.id for area in self.product_areas]

    def area(self, area_id: str) -> ProductArea:
        for area in self.product_areas:
            if area.id == area_id:
                return area
        raise KeyError(f"Unknown product area {area_id!r}. Known: {self.area_ids}")

    def issue_type_ids(self, area_id: str) -> list[str]:
        return [item.id for item in self.area(area_id).issue_types]

    def parent_area_of(self, label_id: str) -> str:
        """Resolve an issue_type or strength_type id back to its product area."""
        for area in self.product_areas:
            for item in [*area.issue_types, *area.strength_types]:
                if item.id == label_id:
                    return area.id
        raise KeyError(f"Unknown issue/strength type {label_id!r}")

    def attribute(self, attribute_id: str) -> Attribute:
        for attribute in self.attributes:
            if attribute.id == attribute_id:
                return attribute
        raise KeyError(f"Unknown attribute {attribute_id!r}")

    def attribute_values(self, attribute_id: str) -> list[str]:
        return [value.id for value in self.attribute(attribute_id).values]

    def areas_in_domain(self, domain_id: str) -> list[ProductArea]:
        return [area for area in self.product_areas if area.domain == domain_id]

    def special_area(self, role: str) -> str:
        """Resolve a named role (e.g. ``support_area``) to its product area id."""
        try:
            return self.special_areas[role]
        except KeyError as exc:
            raise KeyError(
                f"No special_areas entry for {role!r}. Declared: {sorted(self.special_areas)}"
            ) from exc

    def compiled_probes(self) -> dict[str, re.Pattern[str]]:
        """Case-insensitive compiled probes, for the discovery script only."""
        return {
            area_id: re.compile(pattern, re.IGNORECASE)
            for area_id, pattern in self.discovery_probes.items()
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _normalise(raw: dict) -> dict:
    """Convert the YAML's id-keyed mappings into the id-carrying lists Pydantic wants.

    The YAML uses mappings for ``domains`` and ``attributes`` because that reads
    better by hand; the models use lists because ordering matters downstream.
    """
    data = dict(raw)
    data["domains"] = [{"id": key, **value} for key, value in raw.get("domains", {}).items()]
    data["attributes"] = [{"id": key, **value} for key, value in raw.get("attributes", {}).items()]
    return data


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    """Read and validate ``config/taxonomy.yaml``."""
    taxonomy_path = path or Paths.taxonomy
    if not taxonomy_path.exists():
        raise FileNotFoundError(
            f"Taxonomy not found at {taxonomy_path}. It is the single source of "
            "truth for product areas; pipeline code must never define them inline."
        )
    raw = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    if not raw:
        raise ValueError(f"{taxonomy_path} is empty")
    return Taxonomy(**_normalise(raw))


@lru_cache(maxsize=1)
def get_taxonomy() -> Taxonomy:
    """Cached taxonomy singleton. Prefer this in pipeline code."""
    return load_taxonomy()
