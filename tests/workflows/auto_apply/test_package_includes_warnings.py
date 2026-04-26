from uppgrad_agentic.workflows.auto_apply.nodes.package_and_handoff import package_and_handoff
from uppgrad_agentic.workflows.auto_apply.nodes.submit_internal import submit_internal


def _state(warnings):
    return {
        "opportunity_type": "job",
        "opportunity_id": "1",
        "opportunity_data": {"id": 1, "title": "X", "company": "Y"},
        "compatibility_warnings": warnings,
        "tailored_documents": {"CV": {"content": "...", "tailoring_depth": "light"}},
        "scraped_requirements": {"status": "failed", "confidence": 0.0, "source": ""},
    }


def test_handoff_package_carries_warnings():
    out = package_and_handoff(_state(["Job is on-site in Ankara, you're in Istanbul"]))
    pkg = out["application_package"]
    assert "warnings" in pkg
    assert pkg["warnings"] == ["Job is on-site in Ankara, you're in Istanbul"]


def test_handoff_package_warnings_empty_list_when_no_issues():
    out = package_and_handoff(_state([]))
    assert out["application_package"]["warnings"] == []


def test_internal_submit_package_carries_warnings():
    state = _state(["Test warning"])
    state["tailored_documents"] = {
        "CV": {"content": "cv content"},
        "Cover Letter": {"content": "cl content"},
    }
    out = submit_internal(state)
    pkg = out["application_package"]
    assert pkg["warnings"] == ["Test warning"]
