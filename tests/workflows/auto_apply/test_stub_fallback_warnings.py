"""Stub-fallback warnings (Spec/Decision 2 follow-up).

Production runs must always provide profile_snapshot and opportunity_data via
the backend adapter. If either is missing and we degrade to stubs, the system
emits a WARNING-level log so we can detect the regression in prod.
"""
import logging

from uppgrad_agentic.workflows.auto_apply._profile import resolve_profile
from uppgrad_agentic.workflows.auto_apply.nodes.load_opportunity import load_opportunity


def test_resolve_profile_warns_when_falling_back_to_stub(caplog):
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="uppgrad_agentic.workflows.auto_apply._profile"):
        resolve_profile({})
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("falling back to in-repo stub profile" in r.message for r in warnings), (
        f"expected stub-fallback warning, got: {[r.message for r in warnings]}"
    )


def test_resolve_profile_does_not_warn_when_snapshot_provided(caplog):
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="uppgrad_agentic.workflows.auto_apply._profile"):
        resolve_profile({"profile_snapshot": {"name": "Real"}})
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("falling back" in r.message for r in warnings), (
        "must not warn when a real snapshot is provided"
    )


def test_load_opportunity_warns_when_falling_back_to_stub(caplog):
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="uppgrad_agentic.workflows.auto_apply.nodes.load_opportunity"):
        load_opportunity({"opportunity_type": "job", "opportunity_id": "job-001"})
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("falling back to" in r.message and "stub records" in r.message for r in warnings), (
        f"expected stub-fallback warning, got: {[r.message for r in warnings]}"
    )


def test_load_opportunity_does_not_warn_when_preloaded(caplog):
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="uppgrad_agentic.workflows.auto_apply.nodes.load_opportunity"):
        load_opportunity({
            "opportunity_type": "job",
            "opportunity_id": "real-1",
            "opportunity_data": {"id": 1, "title": "Real Job"},
        })
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("falling back" in r.message for r in warnings), (
        "must not warn when opportunity_data is pre-loaded"
    )
