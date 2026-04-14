from __future__ import annotations

from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


def build_context_pack(state: DocFeedbackState) -> dict:
    if state.get("result", {}).get("status") == "error":
        return {}

    doc_classification = state.get("doc_classification") or {}

    context_pack: dict = {
        # Document identity
        "doc_type": doc_classification.get("doc_type", "UNKNOWN"),
        "doc_meta": state.get("doc_meta") or {},
        # Structured sections extracted from the raw document
        "doc_sections": state.get("doc_sections") or {},
        # User's goals and constraints parsed from their free-text instructions
        "parsed_instructions": state.get("parsed_instructions") or {},
        # User profile fetched from the database (stub for now)
        "profile_snapshot": state.get("profile_snapshot") or {},
        # Opportunity the user is targeting, or empty dict if none provided
        "opportunity_context": state.get("opportunity_context") or {},
        # Convenience flag so analysis nodes can branch without re-checking
        "has_opportunity": bool(state.get("opportunity_context")),
    }

    return {"context_pack": context_pack}
