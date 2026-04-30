from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState


def test_state_declares_new_keys():
    keys = AutoApplyState.__annotations__
    assert "profile_snapshot" in keys
    assert "discovered_apply_url" in keys
    assert "discovery_method" in keys
    assert "discovery_confidence" in keys
    assert "requirement_items" in keys
    assert "tailored_answers" in keys
    assert "auto_submit_feasible_at_gate_1" in keys


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


def test_state_accepts_new_gate1_fields():
    s: AutoApplyState = {
        "requirement_items": [{"id": 0, "category": "document"}],
        "tailored_answers": {"3": {"content": "..."}},
        "auto_submit_feasible_at_gate_1": True,
    }
    assert s["requirement_items"][0]["id"] == 0
    assert s["tailored_answers"]["3"]["content"] == "..."
    assert s["auto_submit_feasible_at_gate_1"] is True
