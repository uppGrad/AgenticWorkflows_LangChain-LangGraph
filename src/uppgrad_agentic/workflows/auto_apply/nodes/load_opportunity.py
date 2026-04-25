from __future__ import annotations

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

# ---------------------------------------------------------------------------
# Stub DB records — replaced with real DB queries during backend integration
# ---------------------------------------------------------------------------

_STUB_JOB: dict = {
    "id": "job-001",
    "title": "Software Engineer",
    "company": "Acme Corp",
    "location": "London, UK",
    "description": (
        "We are looking for a talented Software Engineer to join our team. "
        "You will design and build scalable backend services using Python and Go. "
        "3+ years of experience required."
    ),
    "job_type": "Full-time",
    "job_level": "Mid-level",
    "job_function": "Engineering",
    "company_industry": "Technology",
    "is_remote": True,
    "is_closed": False,
    "url": "https://www.linkedin.com/jobs/view/job-001",
    "url_direct": "https://acmecorp.com/careers/software-engineer",
    "site": "linkedin",
    "employer_id": None,  # NULL = external
    "posted_time": "2026-04-01T10:00:00Z",
    "salary": "£60,000 – £80,000",
}

_STUB_MASTERS: dict = {
    "id": "prog-001",
    "url": "https://example-university.ac.uk/msc-cs",
    "title": "MSc Computer Science",
    "university": "Example University",
    "location": "Manchester, UK",
    "duration": "1 year",
    "degree_type": "MSc",
    "study_mode": "Full-time",
    "program_type": "masters",
    "tuition_fee": "£22,000/year",
    "venue": "On-campus",
    "data": {
        "description": "A rigorous MSc in Computer Science covering algorithms, ML, and distributed systems.",
        "requirements": {
            "academic": "2:1 BSc in Computer Science or related field",
            "english": "IELTS 6.5 overall, no band below 6.0",
            "other": "Personal statement, two references, CV",
        },
        "curriculum": ["Algorithms", "Machine Learning", "Distributed Systems", "Research Methods"],
        "funding": "Scholarship opportunities available for high-achieving students",
        "living_costs": "£12,000/year estimated",
        "start_dates": ["September 2026"],
    },
}

_STUB_PHD: dict = {
    "id": "prog-002",
    "url": "https://example-university.ac.uk/phd-ml",
    "title": "PhD Machine Learning",
    "university": "Example University",
    "location": "Manchester, UK",
    "duration": "3–4 years",
    "degree_type": "PhD",
    "study_mode": "Full-time",
    "program_type": "phd",
    "tuition_fee": "Fully funded (stipend £18,000/year)",
    "venue": "On-campus",
    "data": {
        "description": "Doctoral research in machine learning, deep learning, and AI safety.",
        "requirements": {
            "academic": "MSc or first-class BSc in relevant field",
            "english": "IELTS 7.0",
            "other": "Research proposal, CV, two academic references, SOP",
        },
        "curriculum": [],
        "funding": "Fully funded studentship",
        "living_costs": "£12,000/year estimated",
        "start_dates": ["October 2026", "January 2027"],
    },
}

_STUB_SCHOLARSHIP: dict = {
    "id": "sch-001",
    "url": "https://example-foundation.org/scholarship",
    "title": "Global Excellence Scholarship",
    "provider_name": "Example Foundation",
    "disciplines": ["Engineering", "Computer Science", "Mathematics"],
    "grant_display": "Full tuition + £15,000 living allowance",
    "location": "UK",
    "deadline": "2026-06-30",
    "scholarship_type": "Merit-based",
    "coverage": "Full tuition and living expenses",
    "description": "Awarded to outstanding students pursuing postgraduate study in STEM fields.",
    "benefits": "Full tuition, annual stipend, travel grant",
    "eligibility_text": (
        "Open to international students. Must hold an offer from a UK university. "
        "GPA 3.5+ or equivalent. Under 35 years of age."
    ),
    "req_disciplines": ["Engineering", "Computer Science", "Mathematics"],
    "req_locations": ["International"],
    "req_nationality": [],
    "req_age": "Under 35",
    "req_study_experience": "Undergraduate degree required",
    "application_info": "Submit CV, personal statement, and two references via online portal.",
    "data": {
        "description": "Awarded to outstanding students pursuing postgraduate study in STEM fields.",
        "eligibility": {
            "nationality": "International students",
            "academic": "GPA 3.5+ or equivalent first-class/2:1 degree",
            "age": "Under 35",
            "other": "Must hold a conditional or unconditional offer from a UK university",
        },
        "required_documents": ["CV", "Personal Statement / Cover Letter", "Two academic references"],
        "deadline": "2026-06-30",
    },
}

_STUB_RECORDS: dict = {
    "job": _STUB_JOB,
    "masters": _STUB_MASTERS,
    "phd": _STUB_PHD,
    "scholarship": _STUB_SCHOLARSHIP,
}


def _fetch_opportunity(opportunity_type: str, opportunity_id: str) -> dict | None:
    """Stub DB lookup. Replace with real table queries during integration.

    Real implementation:
      - "job"         → SELECT * FROM linkedin_jobs WHERE id = %s
      - "masters"     → SELECT * FROM programs WHERE id = %s AND program_type = 'masters'
      - "phd"         → SELECT * FROM programs WHERE id = %s AND program_type = 'phd'
      - "scholarship" → SELECT * FROM scholarships WHERE id = %s
    """
    return _STUB_RECORDS.get(opportunity_type)


def load_opportunity(state: AutoApplyState) -> dict:
    updates = {"current_step": "load_opportunity", "step_history": ["load_opportunity"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    # Pre-loaded by backend adapter (Spec A1) — pass through with no DB hit.
    if state.get("opportunity_data"):
        return updates

    opportunity_type = state.get("opportunity_type")
    opportunity_id = state.get("opportunity_id")

    if not opportunity_type:
        return {
            **updates,
            "result": {
                "status": "error",
                "error_code": "MISSING_OPPORTUNITY_TYPE",
                "user_message": "No opportunity type was provided.",
            },
        }

    if not opportunity_id:
        return {
            **updates,
            "result": {
                "status": "error",
                "error_code": "MISSING_OPPORTUNITY_ID",
                "user_message": "No opportunity ID was provided.",
            },
        }

    valid_types = ("job", "masters", "phd", "scholarship")
    if opportunity_type not in valid_types:
        return {
            **updates,
            "result": {
                "status": "error",
                "error_code": "INVALID_OPPORTUNITY_TYPE",
                "user_message": f"Unknown opportunity type '{opportunity_type}'. Must be one of: {', '.join(valid_types)}.",
            },
        }

    record = _fetch_opportunity(opportunity_type, opportunity_id)

    if record is None:
        return {
            **updates,
            "result": {
                "status": "error",
                "error_code": "OPPORTUNITY_NOT_FOUND",
                "user_message": f"No {opportunity_type} opportunity found with ID '{opportunity_id}'.",
            },
        }

    return {**updates, "opportunity_data": record}
