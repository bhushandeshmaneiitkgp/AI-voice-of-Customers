"""
The root-cause graph: retrieve -> hypothesise -> validate -> (retry | done).

Why a graph and not a function call. Everything before Phase 6 is a straight
line, and a line is the right shape for it. This stage is the first with a real
decision in it: when the model cites reviews it was never given, the run should
fetch *different* evidence and try again rather than accept the answer or drop
the pain point. That is a cycle, and expressing a cycle as nested retry logic
inside one function is how retry budgets get lost and silent infinite loops
appear.

Every node is a plain function over a TypedDict. LangGraph wires them together
and nothing else, so the orchestration is fully testable without it -- which
matters because the interesting behaviour here is the retry decision, not the
framework.

The retry deliberately widens and *shifts* the evidence rather than repeating
the same request at a higher temperature. A model that invented a citation from
one set of reviews will often invent another from the same set; giving it more
and different material is a real change of input, not a reroll.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypedDict

from voc.rootcause import RootCauseResult, generate_hypotheses
from voc.retrieval import Evidence, Retriever

logger = logging.getLogger(__name__)

#: Attempts per pain point, including the first. Two retries is enough to
#: survive a bad sample; beyond that the problem is the evidence, not the draw,
#: and spending more requests on it is buying noise.
MAX_ATTEMPTS = 3

#: Extra evidence pulled in on each retry.
RETRY_EVIDENCE_STEP = 4


class RootCauseState(TypedDict, total=False):
    """State threaded through the graph for one pain point."""

    product_area: str
    issue_type: str
    stats: dict[str, Any]
    platform: str | None

    k: int
    attempt: int
    evidence: list[Evidence]
    result: RootCauseResult | None
    #: Accumulated across attempts, so a run that eventually succeeds still
    #: records what the earlier attempts got wrong.
    rejected: list[str]
    done: bool


def retrieve_node(state: RootCauseState, retriever: Retriever) -> RootCauseState:
    """Fetch evidence for the pain point at the current breadth."""
    k = state.get("k", 8)
    retrieved = retriever.evidence_for_pain_point(
        state["product_area"],
        state["issue_type"],
        k=k,
        platform=state.get("platform"),
    )
    if retrieved.truncated_by_filters:
        logger.info(
            "%s/%s: only %d review(s) available at k=%d",
            state["product_area"], state["issue_type"], len(retrieved.hits), k,
        )
    return {**state, "evidence": retrieved.hits}


def hypothesise_node(
    state: RootCauseState, profile: Any, provider: Any, n: int = 3
) -> RootCauseState:
    """Ask the model for causes over the retrieved evidence."""
    result = generate_hypotheses(
        state["product_area"],
        state["issue_type"],
        state.get("stats", {}),
        state.get("evidence", []),
        profile,
        provider,
        n=n,
    )
    return {**state, "result": result, "attempt": state.get("attempt", 0) + 1}


def validate_node(state: RootCauseState) -> RootCauseState:
    """Decide whether this attempt stands.

    Citation checking already happened inside ``parse_response``; this node
    reads the outcome and decides between accepting, retrying, and giving up.
    An attempt is acceptable the moment it yields one surviving hypothesis --
    partial success is still a lead, and discarding it because a sibling
    hypothesis was rejected would throw away good work.
    """
    result = state.get("result")
    attempt = state.get("attempt", 1)
    rejected = list(state.get("rejected", []))

    if result is None:
        return {**state, "done": True}

    fatal = {"invented_citation", "uncited_hypothesis", "unparseable_response",
             "schema_violation", "api_error", "no_evidence"}
    rejected.extend(
        f"attempt {attempt}: {issue.kind} — {issue.detail[:100]}"
        for issue in result.issues
        if issue.kind in fatal
    )

    if result.hypotheses:
        return {**state, "done": True, "rejected": rejected}

    if attempt >= MAX_ATTEMPTS:
        logger.warning(
            "%s: no grounded hypothesis after %d attempt(s); giving up",
            result.pain_point, attempt,
        )
        return {**state, "done": True, "rejected": rejected}

    # Widen the evidence rather than rerolling the same request.
    logger.info(
        "%s: attempt %d produced nothing grounded; retrying with more evidence",
        result.pain_point, attempt,
    )
    return {
        **state,
        "done": False,
        "rejected": rejected,
        "k": state.get("k", 8) + RETRY_EVIDENCE_STEP,
    }


def _should_continue(state: RootCauseState) -> str:
    return "done" if state.get("done") else "retry"


def build_rootcause_graph(
    retriever: Retriever, profile: Any, provider: Any, n: int = 3
) -> Any:
    """Compile the graph. Imported lazily so the rest of Phase 6 runs without it."""
    from langgraph.graph import END, StateGraph

    graph = StateGraph(RootCauseState)
    graph.add_node("retrieve", lambda s: retrieve_node(s, retriever))
    graph.add_node("hypothesise", lambda s: hypothesise_node(s, profile, provider, n))
    graph.add_node("validate", lambda s: validate_node(s))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "hypothesise")
    graph.add_edge("hypothesise", "validate")
    graph.add_conditional_edges(
        "validate", _should_continue, {"done": END, "retry": "retrieve"}
    )
    return graph.compile()


def run_pain_point(
    product_area: str,
    issue_type: str,
    stats: dict[str, Any],
    retriever: Retriever,
    profile: Any,
    provider: Any,
    k: int = 8,
    n: int = 3,
    platform: str | None = None,
    runner: Callable[[RootCauseState], RootCauseState] | None = None,
) -> RootCauseState:
    """Run one pain point through the graph.

    ``runner`` is injectable so the orchestration can be exercised without
    compiling a LangGraph -- the retry decision is the part worth testing, and
    it lives in ``validate_node``, not in the framework.
    """
    initial: RootCauseState = {
        "product_area": product_area,
        "issue_type": issue_type,
        "stats": stats,
        "platform": platform,
        "k": k,
        "attempt": 0,
        "evidence": [],
        "result": None,
        "rejected": [],
        "done": False,
    }

    if runner is not None:
        return runner(initial)

    graph = build_rootcause_graph(retriever, profile, provider, n)
    # recursion_limit bounds the retry cycle at the framework level too, so a
    # bug in the conditional edge cannot spin forever burning API calls.
    return graph.invoke(initial, {"recursion_limit": MAX_ATTEMPTS * 3 + 5})


def run_sequential(
    pain_points: list[dict[str, Any]],
    retriever: Retriever,
    profile: Any,
    provider: Any,
    k: int = 8,
    n: int = 3,
    progress: Callable[[int, int], None] | None = None,
) -> list[RootCauseResult]:
    """Run several pain points, one at a time.

    Serial rather than concurrent: this is tens of requests, not hundreds, and
    a graph carrying a retry cycle per item is not worth parallelising until
    the volume justifies the failure modes that come with it.
    """
    results: list[RootCauseResult] = []
    for index, pain_point in enumerate(pain_points, start=1):
        state = run_pain_point(
            pain_point["product_area"],
            pain_point["issue_type"],
            pain_point,
            retriever,
            profile,
            provider,
            k=k,
            n=n,
        )
        result = state.get("result")
        if result is not None:
            results.append(result)
        if progress:
            progress(index, len(pain_points))
    return results
