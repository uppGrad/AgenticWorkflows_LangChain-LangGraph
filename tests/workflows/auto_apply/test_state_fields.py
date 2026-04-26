from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState


def test_state_declares_new_keys():
    keys = AutoApplyState.__annotations__
    assert "profile_snapshot" in keys
    assert "discovered_apply_url" in keys
    assert "discovery_method" in keys
    assert "discovery_confidence" in keys
    assert "gate_0_iteration_count" in keys


def test_state_accepts_profile_snapshot():
    s: AutoApplyState = {"profile_snapshot": {"name": "X"}}
    assert s["profile_snapshot"]["name"] == "X"


def test_state_accepts_discovery_fields():
    s: AutoApplyState = {
        "discovered_apply_url": "https://x.com/job/1",
        "discovery_method": "ats",
        "discovery_confidence": 0.9,
    }
    assert s["discovered_apply_url"] == "https://x.com/job/1"
    assert s["discovery_method"] == "ats"
    assert s["discovery_confidence"] == 0.9


def test_state_accepts_gate_0_iteration_count():
    s: AutoApplyState = {"gate_0_iteration_count": 1}
    assert s["gate_0_iteration_count"] == 1
