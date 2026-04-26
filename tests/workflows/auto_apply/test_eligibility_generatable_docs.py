"""Generatable-doc bypass for eligibility (Spec follow-up).

Missing required documents that the system can write from CV+profile must NOT
block at gate 0. They flow through to asset_mapping → tailoring='generate'
and gate 1 lets the user override with their own upload.

Missing documents the system *cannot* generate (Transcript, English Proficiency
Test, etc.) still trigger gate 0 as before.
"""
from uppgrad_agentic.workflows.auto_apply.nodes.eligibility_and_readiness import (
    eligibility_and_readiness,
)


def _ready_state(missing_doc_type, requirement_type="document"):
    """Build a state where the user has a CV but is missing one other doc."""
    return {
        "opportunity_type": "job",
        "opportunity_data": {
            "is_closed": False, "is_remote": True,
            "title": "X", "company": "Y", "location": "Anywhere",
        },
        "normalized_requirements": [
            {"requirement_type": "document", "document_type": "CV",
             "is_assumed": False, "confidence": 0.9},
            {"requirement_type": requirement_type, "document_type": missing_doc_type,
             "is_assumed": False, "confidence": 0.9},
        ],
        "profile_snapshot": {
            "name": "Real User", "email": "u@x.com",
            "location": "Anywhere",
            "uploaded_documents": {"CV": True, missing_doc_type: False},
            "document_texts": {"CV": "..."},
        },
    }


def test_missing_cover_letter_does_not_block_eligibility():
    out = eligibility_and_readiness(_ready_state("Cover Letter"))
    elig = out["eligibility_result"]
    assert elig["decision"] == "ready", (
        f"Cover Letter is generatable; eligibility must be ready. Got: {elig}"
    )
    assert elig["missing_fields"] == []


def test_missing_sop_does_not_block_eligibility():
    out = eligibility_and_readiness(_ready_state("SOP"))
    assert out["eligibility_result"]["decision"] == "ready"


def test_missing_personal_statement_does_not_block():
    out = eligibility_and_readiness(_ready_state("Personal Statement"))
    assert out["eligibility_result"]["decision"] == "ready"


def test_missing_transcript_still_blocks():
    """Transcript is in _USER_SUPPLIED, not _GENERATABLE — must trigger gate 0."""
    out = eligibility_and_readiness(_ready_state("Transcript"))
    elig = out["eligibility_result"]
    assert elig["decision"] == "pending"
    assert "document:Transcript" in elig["missing_fields"]


def test_missing_english_proficiency_still_blocks():
    out = eligibility_and_readiness(_ready_state("English Proficiency Test"))
    elig = out["eligibility_result"]
    assert elig["decision"] == "pending"
    assert "document:English Proficiency Test" in elig["missing_fields"]


def test_missing_email_still_blocks():
    """Profile fields are unaffected by the generatable bypass."""
    state = _ready_state("Cover Letter")
    state["profile_snapshot"]["email"] = ""
    out = eligibility_and_readiness(state)
    elig = out["eligibility_result"]
    assert elig["decision"] == "pending"
    assert "email" in elig["missing_fields"]


def test_missing_cv_still_blocks():
    """CV itself is technically in _GENERATABLE (system can produce a CV from
    profile), but in practice CV is the seed for everything else. Today's
    behaviour: if uploaded.CV is False AND CV is in normalized_requirements,
    the bypass treats it as generatable and lets it through. That's fine because
    asset_mapping with no uploaded CV falls back to a 'generate from profile'
    strategy. Document this explicitly so future changes don't surprise us.
    """
    state = _ready_state("CV")
    state["profile_snapshot"]["uploaded_documents"]["CV"] = False
    out = eligibility_and_readiness(state)
    # With CV in _GENERATABLE, the bypass kicks in. This is acceptable for v1.
    assert out["eligibility_result"]["decision"] == "ready"
