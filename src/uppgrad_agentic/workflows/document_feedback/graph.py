from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState
from uppgrad_agentic.workflows.document_feedback.nodes.load_document import load_document
from uppgrad_agentic.workflows.document_feedback.nodes.detect_doc_type import detect_doc_type_and_relevance
from uppgrad_agentic.workflows.document_feedback.nodes.end_with_error import end_with_error
from uppgrad_agentic.workflows.document_feedback.nodes.fetch_profile_snapshot import fetch_profile_snapshot
from uppgrad_agentic.workflows.document_feedback.nodes.extract_doc_sections import extract_doc_sections
from uppgrad_agentic.workflows.document_feedback.nodes.parse_user_instructions import parse_user_instructions
from uppgrad_agentic.workflows.document_feedback.nodes.get_opportunity_context import get_opportunity_context
from uppgrad_agentic.workflows.document_feedback.nodes.build_context_pack import build_context_pack
from uppgrad_agentic.workflows.document_feedback.nodes.analyze_structure import analyze_structure
from uppgrad_agentic.workflows.document_feedback.nodes.analyze_style import analyze_style
from uppgrad_agentic.workflows.document_feedback.nodes.analyze_content_gaps import analyze_content_gaps
from uppgrad_agentic.workflows.document_feedback.nodes.analyze_ats import analyze_ats
from uppgrad_agentic.workflows.document_feedback.nodes.analyze_opportunity_alignment import analyze_opportunity_alignment
from uppgrad_agentic.workflows.document_feedback.nodes.synthesize_feedback import synthesize_feedback
from uppgrad_agentic.workflows.document_feedback.nodes.evaluate_output import evaluate_output
from uppgrad_agentic.workflows.document_feedback.nodes.human_gate import human_gate
from uppgrad_agentic.workflows.document_feedback.nodes.finalize import finalize


REJECT_CONFIDENCE = 0.70

_ANALYSIS_NODES = [
    "analyze_structure",
    "analyze_style",
    "analyze_content_gaps",
    "analyze_ats",
    "analyze_opportunity_alignment",
]


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _route_after_detect(state: DocFeedbackState) -> str:
    # If any earlier node produced an error, end.
    if state.get("result", {}).get("status") == "error":
        return "end_with_error"

    cls = state.get("doc_classification") or {}
    relevant = bool(cls.get("relevant", False))
    conf = float(cls.get("confidence", 0.0))
    doc_type = cls.get("doc_type", "UNKNOWN")

    # Conservative rejection rule
    if (not relevant) and (conf >= REJECT_CONFIDENCE):
        return "end_with_error"

    if doc_type == "CV":
        return "cv_route"
    if doc_type == "SOP":
        return "sop_route"
    if doc_type == "COVER_LETTER":
        return "cover_route"

    # v1 choice: reject UNKNOWN (you can later route to generic feedback or ask user)
    return "end_with_error"


MAX_EVAL_ITERATIONS = 2


def _route_after_evaluate(state: DocFeedbackState) -> str:
    if state.get("result", {}).get("status") == "error":
        return "end_with_error"

    evaluation_result = state.get("evaluation_result") or {}
    passed = evaluation_result.get("passed", True)
    # iteration_count has already been incremented by evaluate_output.
    iteration_count = state.get("iteration_count", 0)

    if passed or iteration_count >= MAX_EVAL_ITERATIONS:
        return "human_gate"
    return "synthesize_feedback"


def _dispatch_analysis(state: DocFeedbackState) -> list[Send]:
    """Fan out to all parallel analysis nodes, passing context_pack as payload."""
    if state.get("result", {}).get("status") == "error":
        return []

    context_pack = state.get("context_pack") or {}
    return [Send(node, context_pack) for node in _ANALYSIS_NODES]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    g = StateGraph(DocFeedbackState)

    # Phase 0
    g.add_node("load_document", load_document)
    g.add_node("detect_doc_type", detect_doc_type_and_relevance)
    g.add_node("end_with_error", end_with_error)

    # Phase 1 — context assembly (shared by all doc types)
    g.add_node("fetch_profile_snapshot", fetch_profile_snapshot)
    g.add_node("extract_doc_sections", extract_doc_sections)
    g.add_node("parse_user_instructions", parse_user_instructions)
    g.add_node("get_opportunity_context", get_opportunity_context)
    g.add_node("build_context_pack", build_context_pack)

    # Doc-type routing entry-points (no-op pass-throughs)
    g.add_node("cv_route", lambda state: {})
    g.add_node("sop_route", lambda state: {})
    g.add_node("cover_route", lambda state: {})

    # Phase 2 — parallel analysis
    g.add_node("analyze_structure", analyze_structure)
    g.add_node("analyze_style", analyze_style)
    g.add_node("analyze_content_gaps", analyze_content_gaps)
    g.add_node("analyze_ats", analyze_ats)
    g.add_node("analyze_opportunity_alignment", analyze_opportunity_alignment)

    # Phase 3 — synthesis and planning
    g.add_node("synthesize_feedback", synthesize_feedback)

    # Phase 4 — evaluation loop
    g.add_node("evaluate_output", evaluate_output)

    # Phase 5 — human gate (interrupt/resume)
    g.add_node("human_gate", human_gate)

    # Phase 6 — rewrite
    g.add_node("finalize", finalize)

    # -----------------------------------------------------------------------
    # Edges — Phase 0
    # -----------------------------------------------------------------------
    g.add_edge(START, "load_document")
    g.add_edge("load_document", "detect_doc_type")

    g.add_conditional_edges(
        "detect_doc_type",
        _route_after_detect,
        {
            "cv_route": "cv_route",
            "sop_route": "sop_route",
            "cover_route": "cover_route",
            "end_with_error": "end_with_error",
        },
    )

    # All three doc-type routes converge into the shared Phase 1 sequence.
    for route in ("cv_route", "sop_route", "cover_route"):
        g.add_edge(route, "fetch_profile_snapshot")

    # -----------------------------------------------------------------------
    # Edges — Phase 1 (linear sequence)
    # -----------------------------------------------------------------------
    g.add_edge("fetch_profile_snapshot", "extract_doc_sections")
    g.add_edge("extract_doc_sections", "parse_user_instructions")
    g.add_edge("parse_user_instructions", "get_opportunity_context")
    g.add_edge("get_opportunity_context", "build_context_pack")

    # -----------------------------------------------------------------------
    # Edges — Phase 2 (fan-out via Send, fan-in at synthesize_feedback)
    # -----------------------------------------------------------------------
    g.add_conditional_edges("build_context_pack", _dispatch_analysis, _ANALYSIS_NODES)

    # All parallel branches converge at synthesize_feedback.
    for node in _ANALYSIS_NODES:
        g.add_edge(node, "synthesize_feedback")

    # -----------------------------------------------------------------------
    # Edges — Phase 4 (evaluation loop)
    # -----------------------------------------------------------------------
    g.add_edge("synthesize_feedback", "evaluate_output")

    # If evaluation fails and retries remain, loop back to synthesize_feedback.
    # If passed or retry cap (MAX_EVAL_ITERATIONS) reached, proceed forward.
    g.add_conditional_edges(
        "evaluate_output",
        _route_after_evaluate,
        {
            "synthesize_feedback": "synthesize_feedback",
            "human_gate": "human_gate",
            "end_with_error": "end_with_error",
        },
    )

    # -----------------------------------------------------------------------
    # Edges — Phase 5 (human gate)
    # -----------------------------------------------------------------------
    g.add_edge("human_gate", "finalize")

    # -----------------------------------------------------------------------
    # Edges — Phase 6 (rewrite)
    # -----------------------------------------------------------------------
    g.add_edge("finalize", END)

    # -----------------------------------------------------------------------
    # Edges — error path
    # -----------------------------------------------------------------------
    g.add_edge("end_with_error", END)

    # Use the provided checkpointer, or fall back to MemorySaver for
    # standalone/CLI usage. Production callers (e.g., Django) should pass
    # a PostgresSaver for durable interrupt/resume across requests.
    return g.compile(checkpointer=checkpointer or MemorySaver())
