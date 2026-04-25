from __future__ import annotations

from langgraph.types import interrupt

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

MAX_GATE_0_ITERATIONS = 2


def human_gate_0(state: AutoApplyState) -> dict:
    updates = {"current_step": "human_gate_0", "step_history": ["human_gate_0"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    iteration = state.get("gate_0_iteration_count", 0)
    if iteration >= MAX_GATE_0_ITERATIONS:
        return {
            **updates,
            "result": {
                "status": "error",
                "error_code": "PROFILE_INCOMPLETE_AFTER_RETRIES",
                "user_message": (
                    "Profile is still incomplete after the maximum number of completion attempts. "
                    "Please update your profile and start a new application session."
                ),
            },
        }

    eligibility = state.get("eligibility_result") or {}
    payload = {
        "type": "profile_completion",
        "missing_fields": eligibility.get("missing_fields", []),
        "reasons": eligibility.get("reasons", []),
        "iteration": iteration,
    }
    response = interrupt(payload)

    return {
        **updates,
        "human_review_0": response or {},
        "gate_0_iteration_count": iteration + 1,
    }
