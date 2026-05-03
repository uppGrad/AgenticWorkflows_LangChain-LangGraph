"""auto_accept_proposals — drop-in replacement for human_gate when the
doc-feedback graph is invoked from auto-apply tailoring.

Acceptance rule is doc-type-aware because `requires_confirmation` has
DIFFERENT semantics on each synth path:

  - CV: `requires_confirmation=True` is set ONLY for PII removals (DOB,
    marital status, photo) — the rest of the flag's CV usage maps to
    structural changes the user might still want to OK. We treat True
    as a hard reject on CV to stay safe with PII.

  - SOP / COVER_LETTER: `requires_confirmation=True` is set for EVERY
    substance rewrite + every delete + every merge per the substance
    prompt — that's the highest-value work doc-feedback does. Filtering
    on the flag would throw out ~70% of the pipeline's output. Trust
    the evaluator (grounding + AI-tells + preserve_sentences) instead.
"""
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


def _state(proposals, doc_type="CV"):
    """Helper: build a minimal state with doc_type set for routing."""
    return {
        "proposals": proposals,
        "doc_classification": {"doc_type": doc_type},
    }


# ─── CV path: requires_confirmation maps to PII risk → reject ───────────────

def test_cv_accepts_every_proposal_when_no_requires_confirmation_flag():
    out = auto_accept_proposals(_state([_proposal(0), _proposal(1), _proposal(2)]))
    review = out["human_review"]
    assert len(review["approved_proposals"]) == 3
    for pid in ("0", "1", "2"):
        assert review["decisions"][pid]["action"] == "accept"


def test_cv_rejects_proposals_with_requires_confirmation_true():
    """On the CV path, `requires_confirmation=True` flags PII removals
    (DOB, marital status, photo). Visa context can justify keeping them
    — auto-apply must not override that. Reject."""
    out = auto_accept_proposals(_state([
        _proposal(0, requires_confirmation=False),
        _proposal(1, requires_confirmation=True),
        _proposal(2, requires_confirmation=False),
    ]))
    review = out["human_review"]
    assert len(review["approved_proposals"]) == 2
    assert review["decisions"]["0"]["action"] == "accept"
    assert review["decisions"]["1"]["action"] == "reject"
    assert "CV" in review["decisions"]["1"]["comment"]
    assert review["decisions"]["2"]["action"] == "accept"


# ─── SOP / COVER_LETTER path: requires_confirmation = substance rewrite → accept ────

def test_sop_accepts_proposals_with_requires_confirmation_true():
    """On SOP path, `requires_confirmation=True` is set for EVERY
    substance rewrite — that's the whole point of running the pipeline.
    Filtering on the flag would discard the highest-value output. We
    trust the evaluator to have bounded safety and accept."""
    out = auto_accept_proposals(_state([
        _proposal(0, requires_confirmation=True),
        _proposal(1, requires_confirmation=True),
        _proposal(2, requires_confirmation=False),
    ], doc_type="SOP"))
    review = out["human_review"]
    assert len(review["approved_proposals"]) == 3
    for pid in ("0", "1", "2"):
        assert review["decisions"][pid]["action"] == "accept"


def test_cover_letter_accepts_proposals_with_requires_confirmation_true():
    """Same rule as SOP — substance rewrites on COVER_LETTER are the
    whole point. Accept."""
    out = auto_accept_proposals(_state([
        _proposal(0, requires_confirmation=True, action="rewrite"),
        _proposal(1, requires_confirmation=True, action="delete"),
        _proposal(2, requires_confirmation=True, action="merge"),
    ], doc_type="COVER_LETTER"))
    review = out["human_review"]
    assert len(review["approved_proposals"]) == 3


def test_unknown_doc_type_falls_back_to_pii_aware_filter():
    """Default-deny stance: an unmapped doc_type behaves like CV (filter
    on requires_confirmation) so we don't accidentally accept risky
    proposals from a path we haven't reasoned about."""
    out = auto_accept_proposals(_state([
        _proposal(0, requires_confirmation=True),
        _proposal(1, requires_confirmation=False),
    ], doc_type="UNKNOWN"))
    review = out["human_review"]
    assert len(review["approved_proposals"]) == 1
    assert review["decisions"]["0"]["action"] == "reject"


# ─── Common shape contracts ─────────────────────────────────────────────────

def test_empty_proposals_produces_empty_review():
    out = auto_accept_proposals(_state([]))
    assert out["human_review"]["approved_proposals"] == []
    assert out["human_review"]["decisions"] == {}


def test_short_circuits_on_upstream_error():
    """If a prior node set result.status='error', skip the work — same
    convention as every other doc-feedback node."""
    state = {
        "proposals": [_proposal(0)],
        "doc_classification": {"doc_type": "CV"},
        "result": {"status": "error", "error_code": "FOO"},
    }
    out = auto_accept_proposals(state)
    assert "human_review" not in out
    assert out["current_step"] == "auto_accept_proposals"


def test_approved_proposals_strip_synthetic_id_field():
    """The synthetic `id` field added during numbering must not leak into
    the approved_proposals list — finalize doesn't expect it."""
    out = auto_accept_proposals(_state([_proposal(0), _proposal(1)]))
    for p in out["human_review"]["approved_proposals"]:
        assert "id" not in p
        assert "section" in p  # actual fields preserved
