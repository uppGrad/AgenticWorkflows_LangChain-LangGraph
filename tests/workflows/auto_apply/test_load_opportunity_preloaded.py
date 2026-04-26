from uppgrad_agentic.workflows.auto_apply.nodes.load_opportunity import load_opportunity


def test_short_circuits_when_opportunity_data_preloaded():
    state = {
        "opportunity_type": "job",
        "opportunity_id": "real-123",
        "opportunity_data": {"id": 999, "title": "Real Job", "company": "RealCorp"},
    }
    out = load_opportunity(state)
    # Must NOT overwrite the pre-loaded data with the stub
    assert "opportunity_data" not in out
    assert out["current_step"] == "load_opportunity"
    assert out["step_history"] == ["load_opportunity"]


def test_falls_back_to_stub_when_no_opportunity_data():
    state = {"opportunity_type": "job", "opportunity_id": "job-001"}
    out = load_opportunity(state)
    # CLI / stub mode — node loads from _STUB_RECORDS
    assert out["opportunity_data"]["title"] == "Software Engineer"
