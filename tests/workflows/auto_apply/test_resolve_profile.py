from uppgrad_agentic.workflows.auto_apply._profile import resolve_profile


def test_returns_profile_snapshot_when_present():
    state = {"profile_snapshot": {"name": "Real User", "email": "real@x.com"}}
    profile = resolve_profile(state)
    assert profile["name"] == "Real User"


def test_falls_back_to_stub_when_snapshot_absent():
    profile = resolve_profile({})
    assert profile["name"] == "Alex Johnson"


def test_falls_back_to_stub_when_snapshot_empty_dict():
    profile = resolve_profile({"profile_snapshot": {}})
    assert profile["name"] == "Alex Johnson"
