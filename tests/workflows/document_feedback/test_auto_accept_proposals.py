"""auto_accept_proposals — drop-in replacement for human_gate when the
doc-feedback graph is invoked from auto-apply tailoring."""
from uppgrad_agentic.workflows.document_feedback.nodes.auto_accept_proposals import (
    auto_accept_proposals,
)


def _proposal(idx: int, *, requires_confirmation: bool = False, **kw):
    """Minimal proposal shape — only the fields auto_accept_proposals reads."""
    return {
        "section": kw.get("section", "Experience"),
        "rationale": kw.get("rationale", "Add quantified outcome"),
        "before_text": kw.get("before_text", f"original sentence {idx}"),
        "after_text": kw.get("after_text", f"rewritten sentence {idx}"),
        "confidence": kw.get("confidence", 0.85),
        "requires_confirmation": requires_confirmation,
        "action": kw.get("action", "rewrite"),
    }


def test_accepts_every_proposal_when_no_requires_confirmation_flag():
    state = {
        "proposals": [_proposal(0), _proposal(1), _proposal(2)],
    }
    out = auto_accept_proposals(state)
    review = out["human_review"]
    assert len(review["approved_proposals"]) == 3
    for pid in ("0", "1", "2"):
        assert review["decisions"][pid]["action"] == "accept"


def test_rejects_proposals_with_requires_confirmation_true():
    """`requires_confirmation=True` is the model's own caution flag (PII
    removals, References-on-request, photo removal). Auto-apply must not
    override it — those need explicit human review."""
    state = {
        "proposals": [
            _proposal(0, requires_confirmation=False),
            _proposal(1, requires_confirmation=True),
            _proposal(2, requires_confirmation=False),
        ],
    }
    out = auto_accept_proposals(state)
    review = out["human_review"]
    assert len(review["approved_proposals"]) == 2
    assert review["decisions"]["0"]["action"] == "accept"
    assert review["decisions"]["1"]["action"] == "reject"
    assert review["decisions"]["1"]["comment"] == "auto-rejected: requires_confirmation"
    assert review["decisions"]["2"]["action"] == "accept"


def test_empty_proposals_produces_empty_review():
    out = auto_accept_proposals({"proposals": []})
    assert out["human_review"]["approved_proposals"] == []
    assert out["human_review"]["decisions"] == {}


def test_short_circuits_on_upstream_error():
    """If a prior node set result.status='error', skip the work — same
    convention as every other doc-feedback node."""
    state = {
        "proposals": [_proposal(0)],
        "result": {"status": "error", "error_code": "FOO"},
    }
    out = auto_accept_proposals(state)
    # No human_review produced when short-circuited.
    assert "human_review" not in out
    assert out["current_step"] == "auto_accept_proposals"


def test_approved_proposals_strip_synthetic_id_field():
    """The synthetic `id` field added during numbering must not leak into
    the approved_proposals list — finalize doesn't expect it."""
    state = {"proposals": [_proposal(0), _proposal(1)]}
    out = auto_accept_proposals(state)
    for p in out["human_review"]["approved_proposals"]:
        assert "id" not in p
        assert "section" in p  # actual fields preserved
