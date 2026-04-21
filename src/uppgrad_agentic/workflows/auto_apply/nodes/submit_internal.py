from __future__ import annotations

# Stub — submits CV and Cover Letter to the UppGrad platform backend.
# Replace the _post_to_backend call with a real HTTP/RPC call during integration.

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)

_REQUIRED_DOCS = ("CV", "Cover Letter")


def _post_to_backend(
    opportunity_id: str,
    cv_content: str,
    cover_letter_content: str,
) -> Dict[str, Any]:
    """Stub backend call. Replace with real HTTP POST during integration.

    Real implementation:
      POST /api/applications/internal
      {
        "opportunity_id": ...,
        "cv": ...,
        "cover_letter": ...
      }
    Returns the created application record from the platform.
    """
    logger.info(
        "submit_internal: [STUB] posting to backend — opportunity_id=%s cv_chars=%d cl_chars=%d",
        opportunity_id, len(cv_content), len(cover_letter_content),
    )
    return {
        "platform_application_id": f"app-{opportunity_id}-stub",
        "status": "submitted",
        "submitted_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def submit_internal(state: AutoApplyState) -> dict:
    if state.get("result", {}).get("status") == "error":
        return {}

    opportunity_id = state.get("opportunity_id", "unknown")
    opportunity_data = state.get("opportunity_data") or {}
    tailored_documents: Dict[str, Any] = state.get("tailored_documents") or {}

    cv_content = (tailored_documents.get("CV") or {}).get("content", "")
    cover_letter_content = (tailored_documents.get("Cover Letter") or {}).get("content", "")

    backend_response = _post_to_backend(opportunity_id, cv_content, cover_letter_content)

    application_package: Dict[str, Any] = {
        "CV": cv_content,
        "Cover Letter": cover_letter_content,
        "submission_type": "internal",
        "platform_application_id": backend_response.get("platform_application_id"),
    }

    return {
        "application_package": application_package,
        "result": {
            "status": "ok",
            "user_message": (
                f"Your application for {opportunity_data.get('title', 'the role')} "
                f"at {opportunity_data.get('company', 'the company')} has been submitted successfully."
            ),
            "details": {
                "submission_type": "internal",
                "platform_application_id": backend_response.get("platform_application_id"),
                "submitted_at": backend_response.get("submitted_at"),
            },
        },
    }
