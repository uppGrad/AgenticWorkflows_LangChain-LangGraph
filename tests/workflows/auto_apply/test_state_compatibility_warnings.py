from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState


def test_state_declares_compatibility_warnings():
    keys = AutoApplyState.__annotations__
    assert "compatibility_warnings" in keys
