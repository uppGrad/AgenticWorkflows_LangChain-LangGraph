from __future__ import annotations

from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


def end_with_error(state: DocFeedbackState) -> dict:
    # No-op node: ensures we always have a result for the frontend.
    # If already set, keep it; else set generic.
    if state.get("result", {}).get("status") == "error":
        return {}
    return {
        "result": {
            "status": "error",
            "error_code": "UNKNOWN_ERROR",
            "user_message": "Something went wrong. Please try again.",
        }
    }
