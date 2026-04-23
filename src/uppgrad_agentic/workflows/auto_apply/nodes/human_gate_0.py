from __future__ import annotations

# Stub — will be implemented with LangGraph interrupt() during human-in-the-loop phase.
# Triggered when eligibility_and_readiness returns decision="pending" due to missing
# profile fields or documents. Presents missing_fields to the user and suspends the
# graph until they complete their profile and re-submit.

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState


def human_gate_0(state: AutoApplyState) -> dict:
    updates = {"current_step": "human_gate_0", "step_history": ["human_gate_0"]}
    eligibility = state.get("eligibility_result") or {}
    missing = eligibility.get("missing_fields") or []
    return {
        **updates,
        "result": {
            "status": "ok",
            "user_message": (
                "[STUB] human_gate_0: workflow suspended pending profile completion. "
                f"Missing: {', '.join(missing)}"
            ),
        },
    }
