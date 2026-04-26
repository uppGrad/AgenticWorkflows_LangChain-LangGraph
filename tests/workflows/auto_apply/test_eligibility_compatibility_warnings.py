from uppgrad_agentic.workflows.auto_apply.nodes.eligibility_and_readiness import (
    eligibility_and_readiness,
)


def _job_state(is_remote=False, job_location="Ankara, TR", user_location="Istanbul, TR",
               deadline=None, has_cv=True):
    return {
        "opportunity_type": "job",
        "opportunity_data": {
            "is_closed": False, "is_remote": is_remote,
            "title": "X", "company": "Y", "location": job_location,
            "deadline": deadline,
        },
        "normalized_requirements": [
            {"requirement_type": "document", "document_type": "CV",
             "is_assumed": False, "confidence": 0.9},
        ],
        "profile_snapshot": {
            "name": "U", "email": "u@x.com",
            "location": user_location,
            "uploaded_documents": {"CV": has_cv},
            "document_texts": {"CV": "..."} if has_cv else {},
        },
    }


def test_location_mismatch_emits_warning_not_block():
    out = eligibility_and_readiness(_job_state(
        is_remote=False, job_location="Ankara, TR", user_location="Istanbul, TR",
    ))
    assert out["eligibility_result"]["decision"] == "ready"
    assert any("Ankara" in w for w in out["compatibility_warnings"])


def test_remote_job_no_warning():
    out = eligibility_and_readiness(_job_state(
        is_remote=True, user_location="Istanbul, TR",
    ))
    assert out["eligibility_result"]["decision"] == "ready"
    assert out["compatibility_warnings"] == []


def test_deadline_passed_still_hard_blocks():
    out = eligibility_and_readiness(_job_state(deadline="2020-01-01"))
    assert out["eligibility_result"]["decision"] == "ineligible"
    assert "deadline" in out["eligibility_result"]["reasons"][0].lower()


def test_scholarship_age_cap_emits_warning_not_block():
    state = {
        "opportunity_type": "scholarship",
        "opportunity_data": {
            "title": "S", "req_age": "Under 35",
            "data": {},
        },
        "normalized_requirements": [
            {"requirement_type": "document", "document_type": "CV",
             "is_assumed": False, "confidence": 0.9},
        ],
        "profile_snapshot": {
            "name": "U", "email": "u@x.com", "age": 36,
            "uploaded_documents": {"CV": True},
            "document_texts": {"CV": "..."},
        },
    }
    out = eligibility_and_readiness(state)
    assert out["eligibility_result"]["decision"] == "ready"
    assert any("under 35" in w.lower() for w in out["compatibility_warnings"])


def test_phd_degree_mismatch_emits_warning_not_block():
    state = {
        "opportunity_type": "phd",
        "opportunity_data": {
            "title": "PhD CS", "degree_type": "PhD",
            "data": {"requirements": {"academic": "MSc required"}},
        },
        "normalized_requirements": [
            {"requirement_type": "document", "document_type": "CV",
             "is_assumed": False, "confidence": 0.9},
        ],
        "profile_snapshot": {
            "name": "U", "email": "u@x.com", "degree_level": "BSc",
            "uploaded_documents": {"CV": True},
            "document_texts": {"CV": "..."},
        },
    }
    out = eligibility_and_readiness(state)
    assert out["eligibility_result"]["decision"] == "ready"
    assert any("masters" in w.lower() for w in out["compatibility_warnings"])
