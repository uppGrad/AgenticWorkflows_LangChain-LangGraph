from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState
from uppgrad_agentic.workflows.document_feedback.nodes.load_document import load_document
from uppgrad_agentic.workflows.document_feedback.nodes.detect_doc_type import detect_doc_type_and_relevance
from uppgrad_agentic.workflows.document_feedback.nodes.end_with_error import end_with_error
from uppgrad_agentic.workflows.document_feedback.nodes.fetch_profile_snapshot import fetch_profile_snapshot
from uppgrad_agentic.workflows.document_feedback.nodes.extract_doc_sections import extract_doc_sections
from uppgrad_agentic.workflows.document_feedback.nodes.parse_user_instructions import parse_user_instructions
from uppgrad_agentic.workflows.document_feedback.nodes.get_opportunity_context import get_opportunity_context
from uppgrad_agentic.workflows.document_feedback.nodes.build_context_pack import build_context_pack


REJECT_CONFIDENCE = 0.70


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


def build_graph():
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

    # Routing entry-points (no-op pass-throughs; kept so _route_after_detect
    # can name them separately if per-type branching is needed in a later phase)
    g.add_node("cv_route", lambda state: {})
    g.add_node("sop_route", lambda state: {})
    g.add_node("cover_route", lambda state: {})

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
    g.add_edge("build_context_pack", END)

    g.add_edge("end_with_error", END)

    return g.compile()
