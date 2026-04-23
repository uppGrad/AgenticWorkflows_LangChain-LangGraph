from __future__ import annotations

from typing import Any, Dict

from langgraph.types import interrupt

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

# ---------------------------------------------------------------------------
# Resume value contract
#
# When the graph is resumed the caller must pass Command(resume=approval)
# where `approval` is:
#
#   {
#       "approved": true,              # required — true to proceed, false to cancel
#       "feedback": {                  # optional per-document comments
#           "CV": "Looks great",
#           "Cover Letter": "Make it more formal",
#       }
#   }
#
# - IMPORTANT: always send a non-empty dict — LangGraph treats falsy resume values
#   (None, empty dict, empty string) as "no resume" and will re-interrupt the node.
# - approved=true proceeds to route_by_source → submission or handoff.
# - approved=false cancels the workflow gracefully; no documents are submitted.
# - feedback is stored in human_review_2 for audit purposes.
#
# Interrupt payload (what the frontend receives):
#
#   {
#       "tailored_documents": {
#           "CV": {
#               "id": 0,
#               "content_preview": "...(first 400 chars)...",
#               "tailoring_depth": "light",
#               "llm_used": false,
#               "char_count": 838
#           },
#           ...
#       },
#       "evaluation_result": {passed, issues, iteration},
#       "opportunity_title": "Software Engineer at Acme Corp",
#       "opportunity_type": "job",
#   }
# ---------------------------------------------------------------------------

_PREVIEW_CHARS = 400


def human_gate_2(state: AutoApplyState) -> dict:
    updates = {"current_step": "human_gate_2", "step_history": ["human_gate_2"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    tailored_documents: Dict[str, Any] = state.get("tailored_documents") or {}
    opportunity_data = state.get("opportunity_data") or {}
    opportunity_type = state.get("opportunity_type", "")
    evaluation_result = state.get("evaluation_result") or {}

    title = opportunity_data.get("title") or "this opportunity"
    company = (
        opportunity_data.get("company")
        or opportunity_data.get("university")
        or opportunity_data.get("provider_name")
        or ""
    )
    opportunity_title = f"{title} at {company}" if company else title

    # Build previews for the frontend — expose first 400 chars of each document
    doc_previews: Dict[str, Any] = {}
    for idx, (doc_type, info) in enumerate(tailored_documents.items()):
        content = info.get("content") or ""
        doc_previews[doc_type] = {
            "id": idx,
            "content_preview": content[:_PREVIEW_CHARS],
            "tailoring_depth": info.get("tailoring_depth", ""),
            "llm_used": info.get("llm_used", False),
            "char_count": len(content),
        }

    # -----------------------------------------------------------------------
    # Suspend. Resumes via Command(resume=approval).
    # -----------------------------------------------------------------------
    approval: Dict[str, Any] = interrupt(
        {
            "tailored_documents": doc_previews,
            "evaluation_result": evaluation_result,
            "opportunity_title": opportunity_title,
            "opportunity_type": opportunity_type,
        }
    )

    # -----------------------------------------------------------------------
    # Validate and normalise the resume value
    # -----------------------------------------------------------------------
    if not isinstance(approval, dict):
        approval = {}

    approved: bool = bool(approval.get("approved", False))
    feedback: Dict[str, str] = {
        k: str(v) for k, v in (approval.get("feedback") or {}).items()
    }

    return {
        **updates,
        "human_review_2": {
            "approved": approved,
            "feedback": feedback,
        },
    }
