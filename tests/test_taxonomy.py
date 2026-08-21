"""Tests for the taxonomy configuration and discovery analysis.

The taxonomy is data, not code, so these tests are the only thing standing
between a YAML typo and a silently mislabelled corpus in Phase 3.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest
import yaml

from config.settings import Paths
from voc.discovery import analyse, apply_probes, to_dataframe
from voc.taxonomy import Taxonomy, get_taxonomy, load_taxonomy


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return get_taxonomy()


# ---------------------------------------------------------------------------
# Syntactic validity
# ---------------------------------------------------------------------------


def test_taxonomy_file_exists() -> None:
    assert Paths.taxonomy.exists(), f"taxonomy.yaml missing at {Paths.taxonomy}"


def test_taxonomy_is_valid_yaml() -> None:
    raw = yaml.safe_load(Paths.taxonomy.read_text(encoding="utf-8"))
    assert isinstance(raw, dict) and raw, "taxonomy.yaml did not parse to a non-empty mapping"


def test_taxonomy_loads_and_validates(taxonomy: Taxonomy) -> None:
    assert taxonomy.version
    assert taxonomy.product_areas
    assert taxonomy.domains
    assert taxonomy.attributes


def test_version_is_semver(taxonomy: Taxonomy) -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", taxonomy.version), taxonomy.version


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------


def test_area_ids_are_unique(taxonomy: Taxonomy) -> None:
    ids = taxonomy.area_ids
    assert len(ids) == len(set(ids))


def test_domain_ids_are_unique(taxonomy: Taxonomy) -> None:
    ids = [domain.id for domain in taxonomy.domains]
    assert len(ids) == len(set(ids))


def test_issue_and_strength_ids_are_globally_unique(taxonomy: Taxonomy) -> None:
    """A label must resolve to exactly one parent area, with no ambiguity."""
    seen: dict[str, str] = {}
    for area in taxonomy.product_areas:
        for item in [*area.issue_types, *area.strength_types]:
            assert item.id not in seen, (
                f"{item.id!r} appears in both {seen.get(item.id)!r} and {area.id!r}"
            )
            seen[item.id] = area.id


def test_area_ids_are_snake_case(taxonomy: Taxonomy) -> None:
    for area in taxonomy.product_areas:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", area.id), area.id
        for item in [*area.issue_types, *area.strength_types]:
            assert re.fullmatch(r"[a-z][a-z0-9_]*", item.id), item.id


def test_fallback_area_does_not_collide(taxonomy: Taxonomy) -> None:
    assert taxonomy.fallback_area.id not in taxonomy.area_ids


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------


def test_every_area_maps_to_a_declared_domain(taxonomy: Taxonomy) -> None:
    domain_ids = {domain.id for domain in taxonomy.domains}
    for area in taxonomy.product_areas:
        assert area.domain in domain_ids, f"{area.id} -> unknown domain {area.domain!r}"


def test_every_domain_has_at_least_one_area(taxonomy: Taxonomy) -> None:
    for domain in taxonomy.domains:
        assert taxonomy.areas_in_domain(domain.id), f"domain {domain.id!r} has no areas"


def test_parent_area_resolution_round_trips(taxonomy: Taxonomy) -> None:
    for area in taxonomy.product_areas:
        for item in area.issue_types:
            assert taxonomy.parent_area_of(item.id) == area.id


def test_unknown_ids_raise_keyerror(taxonomy: Taxonomy) -> None:
    with pytest.raises(KeyError):
        taxonomy.area("no_such_area")
    with pytest.raises(KeyError):
        taxonomy.parent_area_of("no_such_issue")


# ---------------------------------------------------------------------------
# Required fields and content quality
# ---------------------------------------------------------------------------


def test_every_area_has_required_fields(taxonomy: Taxonomy) -> None:
    for area in taxonomy.product_areas:
        assert area.name.strip(), f"{area.id} has no name"
        assert len(area.definition.strip()) > 40, f"{area.id} definition is too thin"
        assert area.inclusion, f"{area.id} has no inclusion criteria"
        assert area.exclusion, f"{area.id} has no exclusion criteria"
        assert area.issue_types, f"{area.id} has no issue types"


def test_exclusion_criteria_reference_real_areas(taxonomy: Taxonomy) -> None:
    """Exclusions point elsewhere with '-> area_id'; those targets must exist.

    These strings go verbatim into the Phase 3 prompt, so a stale pointer would
    actively teach the model a category that does not exist.
    """
    known = set(taxonomy.area_ids) | {taxonomy.fallback_area.id}
    for area in taxonomy.product_areas:
        for rule in area.exclusion:
            for referenced in re.findall(r"->\s*([a-z][a-z0-9_]*)", rule):
                assert referenced in known, (
                    f"{area.id} exclusion points at unknown area {referenced!r}"
                )


def test_discovery_estimates_are_plausible(taxonomy: Taxonomy) -> None:
    for area in taxonomy.product_areas:
        assert 0.0 <= area.discovery_estimate_pct <= 100.0
        assert area.positive_negative_ratio >= 0.0


# ---------------------------------------------------------------------------
# Attributes stay separate from the area vocabulary
# ---------------------------------------------------------------------------


def test_required_attributes_are_declared(taxonomy: Taxonomy) -> None:
    declared = {attribute.id for attribute in taxonomy.attributes}
    required = {
        "sentiment",
        "severity",
        "customer_intent",
        "support_escalation",
        "evidence_span",
        "confidence",
    }
    assert required <= declared, f"missing attributes: {sorted(required - declared)}"


def test_enum_attributes_have_defined_values(taxonomy: Taxonomy) -> None:
    for attribute_id in ("sentiment", "severity", "customer_intent"):
        values = taxonomy.attribute_values(attribute_id)
        assert values, f"{attribute_id} has no values"
        assert len(values) == len(set(values)), f"{attribute_id} has duplicate values"


def test_sentiment_includes_mixed(taxonomy: Taxonomy) -> None:
    """Mixed is not the same as neutral, and the corpus contains plenty of it."""
    assert "mixed" in taxonomy.attribute_values("sentiment")


def test_attribute_values_do_not_collide_with_area_ids(taxonomy: Taxonomy) -> None:
    """Attributes describe HOW; areas describe WHERE. The vocabularies stay disjoint."""
    area_ids = set(taxonomy.area_ids)
    for attribute in taxonomy.attributes:
        for value in attribute.values:
            assert value.id not in area_ids, (
                f"attribute {attribute.id!r} value {value.id!r} collides with a product area"
            )


def test_support_escalation_is_a_boolean_attribute(taxonomy: Taxonomy) -> None:
    """Support is downstream, so it needs a flag as well as an area."""
    assert taxonomy.attribute("support_escalation").type == "boolean"
    assert taxonomy.special_area("support_area") in taxonomy.area_ids


def test_borderline_rules_are_declared(taxonomy: Taxonomy) -> None:
    """The hard distinctions are taxonomy data, feeding both docs and the prompt."""
    assert len(taxonomy.borderline_rules) >= 5
    for rule in taxonomy.borderline_rules:
        assert len(rule.strip()) > 40


def test_borderline_rules_reference_real_areas(taxonomy: Taxonomy) -> None:
    """A rule naming a renamed area would silently teach the model a dead label."""
    known = set(taxonomy.area_ids) | {taxonomy.fallback_area.id}
    joined = " ".join(taxonomy.borderline_rules)
    referenced = {
        token for token in re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", joined)
        if token.endswith(tuple(area.split("_")[-1] for area in known))
    }
    unknown = {
        token for token in referenced
        if token not in known
        and token not in {"support_escalation", "missing_feature", "return_or_refund"}
    }
    assert not unknown, f"borderline rules reference unknown identifiers: {sorted(unknown)}"


def test_special_areas_resolve_to_real_areas(taxonomy: Taxonomy) -> None:
    for role, target in taxonomy.special_areas.items():
        assert target in taxonomy.area_ids, f"special_areas.{role} -> unknown area {target!r}"


def test_loader_rejects_unknown_special_area(tmp_path) -> None:
    raw = yaml.safe_load(Paths.taxonomy.read_text(encoding="utf-8"))
    raw["special_areas"]["support_area"] = "not_a_real_area"

    broken = tmp_path / "taxonomy.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="special_areas.support_area"):
        load_taxonomy(broken)


# ---------------------------------------------------------------------------
# Discovery probes
# ---------------------------------------------------------------------------


def test_probes_cover_exactly_the_declared_areas(taxonomy: Taxonomy) -> None:
    assert set(taxonomy.discovery_probes) == set(taxonomy.area_ids)


def test_probes_compile(taxonomy: Taxonomy) -> None:
    for area_id, pattern in taxonomy.discovery_probes.items():
        re.compile(pattern)  # raises on failure


def test_probes_contain_no_control_characters(taxonomy: Taxonomy) -> None:
    r"""Regression guard for a real bug found in Phase 2.

    Inside a DOUBLE-quoted YAML scalar, ``\b`` is the backspace escape, so a
    word-boundary anchor silently becomes a control character that matches
    nothing. The regex still compiles and still returns plausible-looking
    counts, so the failure is invisible without this check. Probes containing
    backslashes must use single-quoted YAML.
    """
    for area_id, pattern in taxonomy.discovery_probes.items():
        offenders = [hex(ord(c)) for c in pattern if ord(c) < 32]
        assert not offenders, f"probe {area_id!r} contains control chars {offenders}"


def test_word_boundary_probes_survived_yaml_parsing(taxonomy: Taxonomy) -> None:
    """At least one probe must still carry a literal \\b after parsing."""
    joined = " ".join(taxonomy.discovery_probes.values())
    assert r"\b" in joined, "word-boundary anchors were lost during YAML parsing"


def test_loader_rejects_control_characters_in_probes(tmp_path) -> None:
    """The guard must actually fire, not just exist."""
    raw = yaml.safe_load(Paths.taxonomy.read_text(encoding="utf-8"))
    first_area = raw["product_areas"][0]["id"]
    raw["discovery_probes"][first_area] = "bad\x08probe"

    broken = tmp_path / "taxonomy.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="control character"):
        load_taxonomy(broken)


def test_loader_rejects_probe_for_unknown_area(tmp_path) -> None:
    raw = yaml.safe_load(Paths.taxonomy.read_text(encoding="utf-8"))
    raw["discovery_probes"]["not_a_real_area"] = "foo"

    broken = tmp_path / "taxonomy.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="no matching product area"):
        load_taxonomy(broken)


def test_loader_rejects_duplicate_area_ids(tmp_path) -> None:
    raw = yaml.safe_load(Paths.taxonomy.read_text(encoding="utf-8"))
    raw["product_areas"].append(dict(raw["product_areas"][0]))

    broken = tmp_path / "taxonomy.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate product_area ids"):
        load_taxonomy(broken)


def test_loader_rejects_area_with_unknown_domain(tmp_path) -> None:
    raw = yaml.safe_load(Paths.taxonomy.read_text(encoding="utf-8"))
    raw["product_areas"][0]["domain"] = "nonexistent_domain"

    broken = tmp_path / "taxonomy.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown domains"):
        load_taxonomy(broken)


# ---------------------------------------------------------------------------
# No hardcoding anywhere else
# ---------------------------------------------------------------------------


def _project_python_files() -> list:
    files = []
    for folder in ("src", "scripts"):
        files.extend((Paths.root / folder).rglob("*.py"))
    return files


def test_no_taxonomy_terms_hardcoded_in_python(taxonomy: Taxonomy) -> None:
    """Area and issue ids must exist only in config/taxonomy.yaml.

    Executable form of requirement 10. A definition living in two places will
    drift, and the drift is silent. No module is exempt: where an analysis
    genuinely needs to single out one area (the support-lift baseline), it
    resolves the id through ``special_areas`` in the YAML rather than naming
    it inline.
    """
    tracked = [area.id for area in taxonomy.product_areas if "_" in area.id]

    offenders: list[str] = []
    for path in _project_python_files():
        text = path.read_text(encoding="utf-8")
        for area_id in tracked:
            if area_id in text:
                offenders.append(f"{path.relative_to(Paths.root)} contains {area_id!r}")

    assert not offenders, "Taxonomy terms hardcoded outside config/taxonomy.yaml:\n  " + "\n  ".join(
        offenders
    )


def test_no_probe_patterns_hardcoded_in_python(taxonomy: Taxonomy) -> None:
    """The regexes themselves must not be duplicated into a module."""
    offenders = []
    for path in _project_python_files():
        text = path.read_text(encoding="utf-8")
        for area_id, pattern in taxonomy.discovery_probes.items():
            if pattern in text:
                offenders.append(f"{path.relative_to(Paths.root)} duplicates probe {area_id!r}")
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# Discovery analysis (runs against the real cleaned corpus)
# ---------------------------------------------------------------------------


requires_corpus = pytest.mark.skipif(
    not Paths.clean_reviews.exists(),
    reason="cleaned corpus not built; run scripts/01_build_clean.py",
)


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    frame = pd.read_parquet(Paths.clean_reviews)
    frame["platform"] = frame["platform"].astype(str)
    return frame


@requires_corpus
def test_probe_matrix_shape(corpus: pd.DataFrame, taxonomy: Taxonomy) -> None:
    hits = apply_probes(corpus, taxonomy)
    assert hits.shape == (len(corpus), len(taxonomy.product_areas))
    assert hits.dtypes.eq(bool).all()


@requires_corpus
def test_every_area_matches_something(corpus: pd.DataFrame, taxonomy: Taxonomy) -> None:
    """An area matching nothing would be a taxonomy invented rather than discovered."""
    hits = apply_probes(corpus, taxonomy)
    empty = [area for area in hits.columns if hits[area].sum() == 0]
    assert not empty, f"areas with zero matches: {empty}"


@requires_corpus
def test_every_area_is_populated_on_every_platform(
    corpus: pd.DataFrame, taxonomy: Taxonomy
) -> None:
    """Required for the single-platform mode to have no empty categories."""
    hits = apply_probes(corpus, taxonomy)
    gaps = []
    for platform in corpus["platform"].unique():
        mask = corpus["platform"] == platform
        for area in hits.columns:
            if hits.loc[mask, area].sum() == 0:
                gaps.append(f"{platform}/{area}")
    assert not gaps, f"empty area/platform combinations: {gaps}"


@requires_corpus
def test_corpus_is_multi_label(corpus: pd.DataFrame, taxonomy: Taxonomy) -> None:
    """The design rests on this: single-label classification would be wrong."""
    hits = apply_probes(corpus, taxonomy)
    assert hits.sum(axis=1).mean() > 1.5


@requires_corpus
def test_unmatched_share_is_within_monitoring_threshold(
    corpus: pd.DataFrame, taxonomy: Taxonomy
) -> None:
    """The YAML sets a 10% ceiling for the fallback area; probes should clear it."""
    result = analyse(corpus, taxonomy)
    assert result.unmatched_pct < 10.0, (
        f"{result.unmatched_pct:.1f}% unmatched suggests a genuine taxonomy gap"
    )


@requires_corpus
def test_wallet_area_discriminates_platforms(corpus: pd.DataFrame, taxonomy: Taxonomy) -> None:
    """Guards the single most consequential discovery finding.

    wallet_and_credits was absent from the 8-category hypothesis and is the
    strongest platform discriminator in the corpus. If a future taxonomy edit
    dissolves that signal, this test says so.
    """
    result = analyse(corpus, taxonomy)
    wallet_area = taxonomy.parent_area_of("balance_unusable")
    wallet = next(a for a in result.areas if a.area_id == wallet_area)
    assert wallet.platform_spread > 10.0
    assert max(wallet.platform_share_pct.values()) > 20.0


@requires_corpus
def test_support_is_downstream_of_operational_failures(
    corpus: pd.DataFrame, taxonomy: Taxonomy
) -> None:
    """Justifies support_escalation existing as a separate attribute."""
    result = analyse(corpus, taxonomy)
    by_id = {area.area_id: area for area in result.areas}
    operational = by_id[taxonomy.parent_area_of("missing_items")]
    value_perception = by_id[taxonomy.parent_area_of("handling_or_platform_fee")]
    assert operational.support_lift > 1.3
    assert value_perception.support_lift < 1.0


@requires_corpus
def test_near_duplicates_do_not_distort_area_shares(
    corpus: pd.DataFrame, taxonomy: Taxonomy
) -> None:
    """Templated reviews must not be what the taxonomy is built on."""
    result = analyse(corpus, taxonomy)
    worst = max(abs(area.near_dup_inflation_pp) for area in result.areas)
    assert worst < 1.0, f"near-duplicate inflation reached {worst:.2f} pp"


@requires_corpus
def test_discovery_is_reproducible(corpus: pd.DataFrame, taxonomy: Taxonomy) -> None:
    first = to_dataframe(analyse(corpus, taxonomy))
    second = to_dataframe(analyse(corpus, taxonomy))
    pd.testing.assert_frame_equal(first, second)


@requires_corpus
def test_yaml_estimates_match_measured_values(corpus: pd.DataFrame, taxonomy: Taxonomy) -> None:
    """The documented estimates must reflect what the probes actually produce.

    Catches the case where someone edits a probe but leaves the published
    number, which would quietly make docs/TAXONOMY.md wrong.
    """
    result = analyse(corpus, taxonomy)
    measured = {area.area_id: area.share_pct for area in result.areas}
    drift = {
        area.id: (area.discovery_estimate_pct, round(measured[area.id], 1))
        for area in taxonomy.product_areas
        if abs(area.discovery_estimate_pct - measured[area.id]) > 0.5
    }
    assert not drift, f"declared vs measured share drift (declared, measured): {drift}"
