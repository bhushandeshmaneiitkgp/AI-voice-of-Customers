"""
Layer 9 -- root-cause hypotheses, grounded in retrieved evidence.

Phase 4 says *what* hurts and Phase 5 says *where it is worse*. Neither says
*why*. This asks a model for candidate mechanisms behind a pain point, given
only the reviews the retriever supplied.

The whole layer is built around one failure mode. A model asked "why is this
happening" will always produce a fluent answer, and a fluent answer citing
plausible-looking review ids is indistinguishable from a correct one at a
glance. So every citation is checked against the evidence actually supplied,
and a hypothesis citing anything else is rejected rather than repaired -- the
same rule Phase 3 applies to evidence spans, for the same reason.

A hypothesis is explicitly **not a finding**. Each one carries the check that
would confirm or kill it, because the useful output of this layer is a short
list of things worth instrumenting, not a narrative.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from voc.providers import LLMProvider, ProviderError
from voc.retrieval import Evidence, format_evidence_block

logger = logging.getLogger(__name__)

#: Bump when the prompt or schema changes in a way that invalidates stored
#: hypotheses. Mirrors PROMPT_VERSION in enrich.py.
ROOTCAUSE_PROMPT_VERSION = "v1"

#: Hypotheses requested per pain point. More than a handful and the model pads
#: with restatements of the pain point itself.
DEFAULT_HYPOTHESES = 3

#: Output budget per pain point. Sized for the verbose case; undersizing
#: truncates mid-JSON and loses the whole response.
MAX_OUTPUT_TOKENS = 2000


class RootCauseHypothesis(BaseModel):
    """One candidate mechanism behind a pain point."""

    model_config = {"extra": "forbid"}

    hypothesis: str = Field(min_length=10, description="The proposed cause, one sentence.")
    mechanism: str = Field(
        min_length=10,
        description="How that cause produces this complaint, concretely.",
    )
    #: Review ids from the supplied evidence that support it. Validated against
    #: what was actually supplied -- a model citing an id it was never given is
    #: inventing corroboration.
    supporting_review_ids: list[str] = Field(default_factory=list)
    #: What in the evidence argues against it. Required, because a model that
    #: never finds counter-evidence is not reading the evidence.
    disconfirming_evidence: str = Field(
        default="",
        description="Anything in the supplied reviews that weakens the hypothesis.",
    )
    #: The observation that would settle it. This is the actionable part.
    proposed_check: str = Field(
        min_length=10,
        description="A specific check -- a metric, log, or query -- that confirms or kills it.",
    )
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("supporting_review_ids")
    @classmethod
    def _dedupe(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class RootCauseResponse(BaseModel):
    model_config = {"extra": "forbid"}

    hypotheses: list[RootCauseHypothesis] = Field(default_factory=list)


@dataclass
class RootCauseIssue:
    """Something wrong with a returned hypothesis."""

    pain_point: str
    kind: str
    detail: str


@dataclass
class RootCauseResult:
    """Hypotheses for one pain point, plus what was rejected."""

    pain_point: str
    product_area: str
    issue_type: str
    hypotheses: list[RootCauseHypothesis] = field(default_factory=list)
    issues: list[RootCauseIssue] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    requests_made: int = 0


SYSTEM_PROMPT = """\
You are a product analyst examining customer reviews for a quick-commerce app.

You will be given ONE pain point and a numbered set of customer reviews that
mention it. Propose up to {n} candidate ROOT CAUSES.

Rules, all of them binding:

1. Use ONLY the reviews supplied. Do not draw on general knowledge about
   delivery apps to invent a cause the evidence does not point to.
2. Cite supporting reviews by their exact bracketed id, e.g. a1b2c3d4e5f6a7b8.
   Never cite an id that is not in the supplied set.
3. A root cause is a MECHANISM, not a restatement. "Customers are unhappy with
   support" restates the pain point. "Agents close tickets on first reply
   because the queue is measured on response time, not resolution" is a cause.
4. Report disconfirming evidence. If some reviews cut against your hypothesis,
   say which and how. A hypothesis with no counter-evidence noted will be
   treated as under-examined.
5. Every hypothesis must carry a proposed_check: one concrete observation --
   a metric, a log query, a support-ticket field -- that would confirm or kill
   it. If you cannot name one, the hypothesis is too vague to include.
6. Set confidence honestly. Evidence here is self-reported customer text; it
   supports hypotheses, it does not establish causes.

Return JSON only, matching this shape:
{{"hypotheses": [{{"hypothesis": "...", "mechanism": "...",
  "supporting_review_ids": ["..."], "disconfirming_evidence": "...",
  "proposed_check": "...", "confidence": 0.0}}]}}
"""


def build_system_prompt(n: int = DEFAULT_HYPOTHESES) -> str:
    return SYSTEM_PROMPT.format(n=n)


def build_user_message(
    product_area: str,
    issue_type: str,
    stats: dict[str, Any],
    hits: Sequence[Evidence],
) -> str:
    """Render one pain point and its evidence.

    The aggregate statistics go in alongside the quotes because a cause has to
    explain the size of the thing: a mechanism that would affect a handful of
    orders does not explain 954 reviews.
    """
    lines = [
        f"PAIN POINT: {product_area} / {issue_type}",
        "",
        "Aggregate signal across the corpus:",
        f"  reviews raising it : {stats.get('volume', 'unknown')}",
        f"  mean severity      : {stats.get('mean_severity', 'unknown')} (low=1, critical=4)",
        f"  drove support      : {_percent(stats.get('escalation_rate'))}",
        f"  stated intent to leave : {_percent(stats.get('churn_rate'))}",
        "",
        f"CUSTOMER REVIEWS ({len(hits)}):",
        "",
        format_evidence_block(hits),
    ]
    return "\n".join(lines)


def _percent(value: Any) -> str:
    return f"{float(value) * 100:.0f}%" if isinstance(value, (int, float)) else "unknown"


def build_response_schema(n: int = DEFAULT_HYPOTHESES) -> dict[str, Any]:
    """JSON schema for providers that enforce one."""
    return {
        "type": "object",
        "properties": {
            "hypotheses": {
                "type": "array",
                "maxItems": n,
                "items": {
                    "type": "object",
                    "properties": {
                        "hypothesis": {"type": "string"},
                        "mechanism": {"type": "string"},
                        "supporting_review_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "disconfirming_evidence": {"type": "string"},
                        "proposed_check": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": [
                        "hypothesis", "mechanism", "supporting_review_ids",
                        "disconfirming_evidence", "proposed_check", "confidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["hypotheses"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_citations(
    hypotheses: Sequence[RootCauseHypothesis],
    supplied_ids: Sequence[str],
    pain_point: str,
) -> tuple[list[RootCauseHypothesis], list[RootCauseIssue]]:
    """Drop hypotheses citing evidence that was never supplied.

    Rejected rather than repaired. Stripping the bad id and keeping the text
    would leave a claim whose stated support does not exist, which is worse
    than no claim: it reads as evidence-backed and is not.

    A hypothesis citing nothing at all is also rejected. The prompt asks for
    causes grounded in the supplied reviews, and an uncited one is the model
    reasoning from general knowledge about delivery apps.
    """
    allowed = set(supplied_ids)
    kept: list[RootCauseHypothesis] = []
    issues: list[RootCauseIssue] = []

    for item in hypotheses:
        invented = [rid for rid in item.supporting_review_ids if rid not in allowed]
        if invented:
            issues.append(
                RootCauseIssue(
                    pain_point=pain_point,
                    kind="invented_citation",
                    detail=f"cited {invented[:3]} which were not supplied: "
                           f"{item.hypothesis[:80]}",
                )
            )
            continue
        if not item.supporting_review_ids:
            issues.append(
                RootCauseIssue(
                    pain_point=pain_point,
                    kind="uncited_hypothesis",
                    detail=item.hypothesis[:120],
                )
            )
            continue
        if not item.disconfirming_evidence.strip():
            # Kept, but flagged: an unexamined hypothesis is still a lead.
            issues.append(
                RootCauseIssue(
                    pain_point=pain_point,
                    kind="no_disconfirming_evidence",
                    detail=item.hypothesis[:120],
                )
            )
        kept.append(item)

    return kept, issues


def parse_response(text: str, supplied_ids: Sequence[str], pain_point: str) -> RootCauseResult:
    """Parse, schema-check, and citation-check one response."""
    area, _, issue = pain_point.partition("/")
    result = RootCauseResult(
        pain_point=pain_point,
        product_area=area.strip(),
        issue_type=issue.strip(),
        evidence_ids=list(supplied_ids),
    )

    payload = _extract_json(text)
    if payload is None:
        result.issues.append(
            RootCauseIssue(pain_point, "unparseable_response", text[:200])
        )
        return result

    # Some models return the array bare rather than under "hypotheses".
    if isinstance(payload, list):
        payload = {"hypotheses": payload}

    try:
        parsed = RootCauseResponse(**payload)
    except ValidationError as exc:
        result.issues.append(
            RootCauseIssue(pain_point, "schema_violation", str(exc)[:200])
        )
        return result

    kept, issues = validate_citations(parsed.hypotheses, supplied_ids, pain_point)
    result.hypotheses = kept
    result.issues.extend(issues)
    return result


def _extract_json(text: str) -> Any | None:
    """Tolerate fenced blocks and leading prose; reject anything else."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1]
        if candidate.startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost brace or bracket span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_hypotheses(
    product_area: str,
    issue_type: str,
    stats: dict[str, Any],
    hits: Sequence[Evidence],
    profile: Any,
    provider: LLMProvider,
    n: int = DEFAULT_HYPOTHESES,
) -> RootCauseResult:
    """Ask for root causes for one pain point. Never raises on a bad response."""
    pain_point = f"{product_area} / {issue_type}"
    supplied_ids = [hit.review_id for hit in hits]

    if not hits:
        return RootCauseResult(
            pain_point=pain_point,
            product_area=product_area,
            issue_type=issue_type,
            issues=[RootCauseIssue(pain_point, "no_evidence",
                                   "retrieval returned nothing to reason over")],
        )

    try:
        completion = provider.complete(
            profile,
            build_system_prompt(n),
            build_user_message(product_area, issue_type, stats, hits),
            build_response_schema(n),
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    except (ProviderError, Exception) as exc:  # noqa: BLE001 - recorded, run continues
        logger.error("Root-cause request failed for %s: %s", pain_point, exc)
        return RootCauseResult(
            pain_point=pain_point,
            product_area=product_area,
            issue_type=issue_type,
            evidence_ids=supplied_ids,
            issues=[RootCauseIssue(pain_point, "api_error", str(exc)[:200])],
        )

    result = parse_response(completion.text, supplied_ids, pain_point)
    result.product_area = product_area
    result.issue_type = issue_type
    result.requests_made = 1
    result.usage = dict(completion.usage or {})
    return result
