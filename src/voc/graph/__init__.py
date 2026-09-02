"""
Layer 13 -- LangGraph orchestration.

The pipeline up to Phase 5 is a straight line: each script reads the previous
one's parquet and writes its own. Phase 6 is the first stage with a *decision*
in it -- retrieve, hypothesise, check the citations, and retry with different
evidence when the check fails -- which is what a graph buys over a function.

Exposed as ``build_rootcause_graph`` so the orchestration is testable without
LangGraph in the loop: every node is a plain function over a TypedDict, and the
graph only wires them together.
"""

from voc.graph.rootcause_graph import (
    RootCauseState,
    build_rootcause_graph,
    hypothesise_node,
    retrieve_node,
    run_pain_point,
    validate_node,
)

__all__ = [
    "RootCauseState",
    "build_rootcause_graph",
    "hypothesise_node",
    "retrieve_node",
    "run_pain_point",
    "validate_node",
]
