from unittest.mock import patch

from uppgrad_agentic.workflows.auto_apply.nodes.human_gate_0 import human_gate_0


def test_calls_interrupt_with_missing_fields():
    state = {
        "eligibility_result": {"decision": "pending", "missing_fields": ["email", "document:CV"]},
        "gate_0_iteration_count": 0,
    }
    with patch("uppgrad_agentic.workflows.auto_apply.nodes.human_gate_0.interrupt") as fake_int:
        fake_int.return_value = {"profile_completed": True}
        out = human_gate_0(state)
    fake_int.assert_called_once()
    payload = fake_int.call_args.args[0]
    assert payload["missing_fields"] == ["email", "document:CV"]
    assert out["gate_0_iteration_count"] == 1
    assert out.get("human_review_0") == {"profile_completed": True}


def test_returns_error_when_iteration_cap_exceeded():
    state = {
        "eligibility_result": {"decision": "pending", "missing_fields": ["email"]},
        "gate_0_iteration_count": 2,
    }
    out = human_gate_0(state)
    assert out["result"]["status"] == "error"
    assert out["result"]["error_code"] == "PROFILE_INCOMPLETE_AFTER_RETRIES"


def test_short_circuits_on_upstream_error():
    state = {"result": {"status": "error", "error_code": "X"}}
    out = human_gate_0(state)
    assert out == {"current_step": "human_gate_0", "step_history": ["human_gate_0"]}
