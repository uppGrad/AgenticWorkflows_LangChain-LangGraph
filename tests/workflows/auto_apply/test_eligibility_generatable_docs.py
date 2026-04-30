"""Eligibility no longer hard-blocks on missing documents — gate 1 collects
per-requirement choices (Step 4 of the gate-0 removal). The only hard-block
left is a passed deadline.
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
    assert elig["decision"] == "ready"
    assert elig["missing_fields"] == []


def test_missing_sop_does_not_block_eligibility():
    out = eligibility_and_readiness(_ready_state("SOP"))
    assert out["eligibility_result"]["decision"] == "ready"


def test_missing_personal_statement_does_not_block():
    out = eligibility_and_readiness(_ready_state("Personal Statement"))
    assert out["eligibility_result"]["decision"] == "ready"


def test_missing_transcript_no_longer_blocks():
    """Gate 0 is removed; Transcript collection now happens at gate 1."""
    out = eligibility_and_readiness(_ready_state("Transcript"))
    assert out["eligibility_result"]["decision"] == "ready"


def test_missing_english_proficiency_no_longer_blocks():
    out = eligibility_and_readiness(_ready_state("English Proficiency Test"))
    assert out["eligibility_result"]["decision"] == "ready"


def test_missing_email_no_longer_blocks():
    """Profile completeness is no longer checked at the eligibility step."""
    state = _ready_state("Cover Letter")
    state["profile_snapshot"]["email"] = ""
    out = eligibility_and_readiness(state)
    assert out["eligibility_result"]["decision"] == "ready"


def test_passed_deadline_still_blocks():
    """The single remaining hard-block reason."""
    state = _ready_state("Cover Letter")
    state["opportunity_data"]["deadline"] = "2000-01-01"
    out = eligibility_and_readiness(state)
    assert out["eligibility_result"]["decision"] == "ineligible"
