from unittest.mock import patch

from uppgrad_agentic.workflows.auto_apply.nodes.discover_apply_url import (
    discover_apply_url_node,
)
from uppgrad_agentic.tools.url_discovery import DiscoveryResult


def _state(opp_type="job", url_direct=None, employer_id=None,
           preset_url="", preset_method=""):
    return {
        "opportunity_type": opp_type,
        "opportunity_id": "1",
        "opportunity_data": {
            "id": 42, "title": "SWE", "company": "Acme",
            "url": "https://linkedin.com/jobs/view/42",
            "url_direct": url_direct,
            "employer_id": employer_id,
            "company_url": None, "posted_time": None, "location": "",
        },
        "discovered_apply_url": preset_url or None,
        "discovery_method": preset_method or None,
    }


def test_skips_for_non_jobs():
    out = discover_apply_url_node(_state(opp_type="masters"))
    assert out["current_step"] == "discover_apply_url"
    assert "discovery_method" not in out


def test_skips_for_internal_jobs():
    out = discover_apply_url_node(_state(employer_id=1))
    assert out["discovery_method"] == "skipped_internal"
    assert out["discovered_apply_url"] is None


def test_uses_cached_when_already_in_state():
    state = _state(preset_url="https://cached.com/job/1", preset_method="ats")
    out = discover_apply_url_node(state)
    assert out["discovered_apply_url"] == "https://cached.com/job/1"
    assert out["discovery_method"] == "ats"


def test_url_direct_short_circuit_via_orchestrator():
    state = _state(url_direct="https://acme.com/apply/1")
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.discover_apply_url.discover_apply_url",
        return_value=DiscoveryResult(url="https://acme.com/apply/1",
                                     method="url_direct", confidence=1.0),
    ):
        out = discover_apply_url_node(state)
    assert out["discovered_apply_url"] == "https://acme.com/apply/1"
    assert out["discovery_method"] == "url_direct"


def test_failed_path_does_not_set_error():
    state = _state(url_direct=None)
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.discover_apply_url.discover_apply_url",
        return_value=DiscoveryResult(url="", method="failed", confidence=0.0),
    ):
        out = discover_apply_url_node(state)
    assert out["discovered_apply_url"] is None
    assert out["discovery_method"] == "failed"
    assert "result" not in out


def test_short_circuits_on_upstream_error():
    state = _state()
    state["result"] = {"status": "error", "error_code": "X"}
    out = discover_apply_url_node(state)
    assert out == {"current_step": "discover_apply_url", "step_history": ["discover_apply_url"]}
