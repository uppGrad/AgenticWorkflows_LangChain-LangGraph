from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState
from uppgrad_agentic.workflows.document_feedback.nodes.load_document import load_document
from uppgrad_agentic.workflows.document_feedback.nodes.detect_doc_type import detect_doc_type_and_relevance
from uppgrad_agentic.workflows.document_feedback.nodes.end_with_error import end_with_error


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
        return "cv_flow_start"
    if doc_type == "SOP":
        return "sop_flow_start"
    if doc_type == "COVER_LETTER":
        return "cover_flow_start"

    # v1 choice: reject UNKNOWN (you can later route to generic feedback or ask user)
    return "end_with_error"


def build_graph():
    g = StateGraph(DocFeedbackState)

    g.add_node("load_document", load_document)
    g.add_node("detect_doc_type", detect_doc_type_and_relevance)

    # placeholders for later
    g.add_node("cv_flow_start", lambda state: {"result": {"status": "ok", "user_message": "CV flow: TODO"}})
    g.add_node("sop_flow_start", lambda state: {"result": {"status": "ok", "user_message": "SOP flow: TODO"}})
    g.add_node("cover_flow_start", lambda state: {"result": {"status": "ok", "user_message": "Cover letter flow: TODO"}})

    g.add_node("end_with_error", end_with_error)

    g.add_edge(START, "load_document")
    g.add_edge("load_document", "detect_doc_type")

    g.add_conditional_edges(
        "detect_doc_type",
        _route_after_detect,
        {
            "cv_flow_start": "cv_flow_start",
            "sop_flow_start": "sop_flow_start",
            "cover_flow_start": "cover_flow_start",
            "end_with_error": "end_with_error",
        },
    )

    g.add_edge("cv_flow_start", END)
    g.add_edge("sop_flow_start", END)
    g.add_edge("cover_flow_start", END)
    g.add_edge("end_with_error", END)

    return g.compile()
