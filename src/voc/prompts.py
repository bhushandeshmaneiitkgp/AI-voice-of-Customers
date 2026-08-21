"""
Prompt construction for AI enrichment.

The system prompt is **generated from** ``config/taxonomy.yaml`` rather than
written alongside it. Two reasons:

* A prompt that repeats the taxonomy in prose will drift from the YAML the
  first time an area is renamed, and the drift is silent.
* The inclusion/exclusion rules and borderline cases were derived from real
  corpus evidence in Phase 2. Feeding them to the model verbatim is what makes
  multi-label output consistent instead of vibes-based.

The prompt is deliberately stable across requests so it can be cached: reviews
go in the user turn, never in the system block.
"""

from __future__ import annotations

import pandas as pd

from config.settings import DatasetConfig, get_dataset_config
from voc.taxonomy import Taxonomy


def _format_areas(taxonomy: Taxonomy) -> str:
    """Render the taxonomy as the reference section of the prompt."""
    blocks: list[str] = []
    for domain in taxonomy.domains:
        areas = taxonomy.areas_in_domain(domain.id)
        if not areas:
            continue
        blocks.append(f"\n### Domain: {domain.name}")
        for area in areas:
            lines = [
                f"\n**{area.id}** — {area.name}",
                f"  Definition: {area.definition.strip()}",
                "  Use when:",
            ]
            lines += [f"    - {rule}" for rule in area.inclusion]
            lines.append("  Do NOT use when:")
            lines += [f"    - {rule}" for rule in area.exclusion]
            lines.append(
                "  issue_type options: "
                + ", ".join(item.id for item in area.issue_types)
            )
            if area.strength_types:
                lines.append(
                    "  strength_type options: "
                    + ", ".join(item.id for item in area.strength_types)
                )
            blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _format_attributes(taxonomy: Taxonomy) -> str:
    blocks: list[str] = []
    for attribute_id in ("sentiment", "severity", "customer_intent"):
        attribute = taxonomy.attribute(attribute_id)
        lines = [f"\n**{attribute_id}** — {attribute.description.strip()}"]
        lines += [f"    - `{value.id}`: {value.definition.strip()}" for value in attribute.values]
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def build_system_prompt(taxonomy: Taxonomy, dataset: DatasetConfig | None = None) -> str:
    """Assemble the full classification prompt from the taxonomy and dataset.

    The domain sentence comes from ``config/dataset.yaml``, not from a literal
    here. Naming one industry in this module would mean the model is told it is
    reading that industry's reviews no matter which corpus it was actually
    handed -- biasing every label, invisibly.

    Note the taxonomy is separately domain-specific and separately swappable:
    a different business supplies both its own ``dataset.yaml`` and its own
    ``taxonomy.yaml``. This function only guarantees the *dataset* half.

    Kept byte-stable across requests so prompt caching can hit: nothing
    review-specific, no timestamps, no counters.
    """
    dataset = dataset or get_dataset_config()
    brands = ", ".join(dataset.platform_display_names)
    context = dataset.domain.reviewer_context.strip()
    reviewer_note = f"\n\n{context}" if context else ""

    return f"""You are a product analyst classifying customer reviews of
{dataset.domain.description} ({brands}) for a Product Manager.{reviewer_note}

Your output feeds a product intelligence tool. A PM will make roadmap decisions
from it, so a confident wrong label is far more damaging than an omitted one.

# Your task

For each review, identify every product area it touches, the specific issue or
strength within each, and review-level attributes.

# Core rules

1. **MULTI-LABEL.** Reviews average 2.19 product areas. Most touch more than
   one. Label every area genuinely present — do not stop at the first.

2. **EVIDENCE IS MANDATORY.** Every area label needs an `evidence_span`: a
   VERBATIM, CONTIGUOUS substring copied exactly from the review. Do not
   paraphrase, do not join fragments with "...", do not fix spelling. Spans are
   checked automatically against the review text; an unverifiable span makes the
   whole label untrustworthy.

3. **NEVER INFER BEYOND THE TEXT.** If a review says delivery was late, do not
   assume the customer contacted support. Label only what is stated. When a
   review is vague, use fewer labels and lower confidence — that is the correct
   answer, not a failure.

4. **AREAS ARE POLARITY-NEUTRAL.** Each area label carries EITHER an
   `issue_type` (something went wrong) OR a `strength_type` (something went
   well), never both and never neither. A single review may praise one area and
   criticise another.

5. **SENTIMENT COMES FROM THE TEXT, NOT THE STAR RATING.** You are not given the
   rating, deliberately. Sarcasm exists in this corpus.

6. **NO ASSIGNABLE AREA.** If a review is generic with no identifiable product
   surface ("worst app ever", "very good"), return a single area label with
   product_area `{taxonomy.fallback_area.id}`, a strength_type or issue_type of
   null, and the evidence span. Use this sparingly — most vague-seeming reviews
   still name something concrete.

7. **CONFIDENCE.** Per-label `confidence` reflects certainty in THAT label.
   `overall_confidence` reflects certainty in the whole enrichment. Be honest:
   a low score on an ambiguous review is more useful than false precision.

8. **SEVERITY** applies only when sentiment is negative or mixed. Otherwise null.

# Product area taxonomy
{_format_areas(taxonomy)}

# Review-level attributes
{_format_attributes(taxonomy)}

**support_escalation** — true when the customer contacted, or tried to contact,
support about the issue, regardless of which area the issue belongs to. This is
independent of whether `{taxonomy.special_area("support_area")}` is one of the areas.

**pain_point** — one sentence naming the specific problem in the customer's own
framing. Null when the review is purely positive.

# Borderline cases

These are the distinctions that are actually hard. Follow them exactly:

{chr(10).join(f"{index}. {rule.strip()}" for index, rule in enumerate(taxonomy.borderline_rules, 1))}

# Output

Return JSON matching the provided schema. Echo each `review_id` back exactly as
given. Return exactly one result per review, in the order supplied."""


def build_user_message(reviews: pd.DataFrame) -> str:
    """Render a group of reviews as the user turn.

    Reviews go here rather than in the system block so the cached prefix stays
    identical across every request.
    """
    parts = [f"Classify the following {len(reviews)} review(s).\n"]
    for _, row in reviews.iterrows():
        parts.append(
            f"\n<review>\n"
            f"review_id: {row['review_id']}\n"
            f"platform: {row['platform']}\n"
            f"text: {row['review_text']}\n"
            f"</review>"
        )
    return "\n".join(parts)
