from __future__ import annotations

import logging
from typing import Any, Dict

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)


def submit_internal(state: AutoApplyState) -> dict:
    """Record submission intent for internal jobs.

    The actual Application row is created server-side by the backend adapter
    after the graph terminates (Spec A3). This node is a marker, not a writer.
    """
    updates = {"current_step": "submit_internal", "step_history": ["submit_internal"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    opportunity_data = state.get("opportunity_data") or {}
    tailored: Dict[str, Any] = state.get("tailored_documents") or {}

    cv_content = (tailored.get("CV") or {}).get("content", "")
    cl_content = (tailored.get("Cover Letter") or {}).get("content", "")

    package: Dict[str, Any] = {
        "CV": cv_content,
        "Cover Letter": cl_content,
        "submission_type": "internal",
        "warnings": list(state.get("compatibility_warnings") or []),
    }

    logger.info(
        "submit_internal: recorded internal submission intent for opportunity_id=%s",
        state.get("opportunity_id", "unknown"),
    )

    return {
        **updates,
        "application_package": package,
        "result": {
            "status": "ok",
            "user_message": (
                f"Your application for {opportunity_data.get('title', 'the role')} "
                f"at {opportunity_data.get('company', 'the company')} is ready for submission."
            ),
            "details": {"submission_type": "internal"},
        },
    }
