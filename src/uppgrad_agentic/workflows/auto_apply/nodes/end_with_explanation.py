from __future__ import annotations

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState


def end_with_explanation(state: AutoApplyState) -> dict:
    eligibility = state.get("eligibility_result") or {}
    reasons = eligibility.get("reasons") or []
    reason_text = " ".join(reasons) if reasons else "You are not eligible for this opportunity."

    return {
        "result": {
            "status": "error",
            "error_code": "INELIGIBLE",
            "user_message": (
                "Unfortunately, we cannot proceed with this application. "
                f"{reason_text}"
            ),
            "details": {"eligibility_result": eligibility},
        }
    }
