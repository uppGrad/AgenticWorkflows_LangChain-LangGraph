from uppgrad_agentic.workflows.auto_apply.nodes.submit_internal import submit_internal


def test_records_submission_intent_without_fake_id():
    state = {
        "opportunity_id": "job-42",
        "opportunity_data": {"id": 42, "title": "SWE", "company": "Acme"},
        "tailored_documents": {
            "CV": {"content": "the CV"},
            "Cover Letter": {"content": "the CL"},
        },
    }
    out = submit_internal(state)
    pkg = out["application_package"]
    assert pkg["submission_type"] == "internal"
    assert pkg["CV"] == "the CV"
    assert pkg["Cover Letter"] == "the CL"
    # No fake platform_application_id — adapter creates the real Application row
    assert "platform_application_id" not in pkg
    assert out["result"]["status"] == "ok"


def test_short_circuits_on_upstream_error():
    state = {"result": {"status": "error", "error_code": "X"}}
    out = submit_internal(state)
    assert out.get("current_step") == "submit_internal"
    assert "application_package" not in out
