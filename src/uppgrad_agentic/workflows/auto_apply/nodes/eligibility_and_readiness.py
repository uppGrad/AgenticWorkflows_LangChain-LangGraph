from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any, Dict, List, Tuple

from uppgrad_agentic.workflows.auto_apply.schemas import EligibilityResult
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stub user profile — replaced with real DB lookup keyed on state["user_id"]
# during backend integration.
# ---------------------------------------------------------------------------

_STUB_PROFILE: Dict[str, Any] = {
    "name": "Alex Johnson",
    "email": "alex.johnson@email.com",
    "age": 27,
    "nationality": "International",
    "location": "London, UK",
    "degree_level": "BSc",          # highest completed degree
    "disciplines": ["Computer Science", "Engineering"],
    "gpa": 3.7,
    "uploaded_documents": {         # document_type → True/False
        "CV": True,
        "Cover Letter": False,
        "SOP": False,
        "Personal Statement": False,
        "Research Proposal": False,
        "Transcript": False,
        "References": False,
        "English Proficiency Test": False,
        "Portfolio": False,
        "Writing Sample": False,
    },
    # Stub document texts — replaced with real content from storage during integration
    "document_texts": {
        "CV": (
            "Alex Johnson\n"
            "alex.johnson@email.com | London, UK\n\n"
            "EDUCATION\n"
            "BSc Computer Science, University of London, 2021\n"
            "GPA: 3.7/4.0\n\n"
            "EXPERIENCE\n"
            "Junior Software Engineer, TechStart Ltd, 2021–2023\n"
            "- Developed and maintained REST APIs using Python and Flask\n"
            "- Improved database query performance by 40% through indexing and query optimisation\n"
            "- Collaborated in an agile team of 8 engineers across two product squads\n\n"
            "Software Engineering Intern, DataCo, Summer 2020\n"
            "- Built data ingestion pipelines processing 500K daily records\n"
            "- Wrote unit and integration tests, achieving 85% coverage\n\n"
            "SKILLS\n"
            "Languages: Python, JavaScript, SQL, Go (basic)\n"
            "Tools: Git, Docker, PostgreSQL, Redis, AWS (EC2, S3, Lambda)\n"
            "Practices: REST APIs, Agile/Scrum, CI/CD, test-driven development\n"
        ),
    },
}


def _get_stub_profile() -> Dict[str, Any]:
    """Stub profile fetch. Replace with real DB lookup during integration."""
    return _STUB_PROFILE


# ---------------------------------------------------------------------------
# Date parsing helpers
# ---------------------------------------------------------------------------

_DATE_FORMATS = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"]


def _parse_date(value: str) -> date | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _today() -> date:
    return date.today()


# ---------------------------------------------------------------------------
# Per-type eligibility checks
# ---------------------------------------------------------------------------

def _check_deadline(opportunity_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (passed, reason). passed=True means deadline has already passed."""
    deadline_str = (
        opportunity_data.get("deadline")
        or (opportunity_data.get("data") or {}).get("deadline")
        or ""
    )
    if not deadline_str:
        return False, ""

    deadline = _parse_date(str(deadline_str))
    if deadline is None:
        logger.warning("eligibility_and_readiness: could not parse deadline '%s'", deadline_str)
        return False, ""

    today = _today()
    if today > deadline:
        return True, f"Application deadline was {deadline.isoformat()} — it has already passed (today is {today.isoformat()})."
    return False, ""


def _check_job_eligibility(opportunity_data: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    if opportunity_data.get("is_closed"):
        issues.append("This job posting is closed and no longer accepting applications.")

    if not opportunity_data.get("is_remote", False):
        job_location = (opportunity_data.get("location") or "").lower()
        user_location = (profile.get("location") or "").lower()
        if job_location and user_location and job_location not in user_location and user_location not in job_location:
            issues.append(
                f"Job is on-site in '{opportunity_data.get('location')}' but your location is '{profile.get('location')}'. "
                "Remote work is not offered."
            )

    return issues


def _check_program_eligibility(opportunity_data: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    data = opportunity_data.get("data") or {}
    academic_req = (data.get("requirements") or {}).get("academic", "").lower()

    degree_level = (profile.get("degree_level") or "").lower()

    # PhD programs typically require at least a masters or first-class BSc
    degree_type = (opportunity_data.get("degree_type") or "").lower()
    if "phd" in degree_type:
        if degree_level not in ("msc", "ma", "meng", "mres", "phd", "masters"):
            issues.append(
                f"A PhD program typically requires a Masters degree or a first-class BSc. "
                f"Your highest recorded degree is '{profile.get('degree_level', 'unknown')}'."
            )
    elif "msc" in degree_type or "masters" in degree_type or "ma" in degree_type:
        if degree_level not in ("bsc", "ba", "beng", "msc", "ma", "meng", "phd", "masters", "undergraduate"):
            issues.append(
                f"A Masters program requires an undergraduate degree. "
                f"Your highest recorded degree is '{profile.get('degree_level', 'unknown')}'."
            )

    return issues


def _check_scholarship_eligibility(opportunity_data: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    data = opportunity_data.get("data") or {}
    eligibility = data.get("eligibility") or {}

    # Age check
    req_age_str = (
        opportunity_data.get("req_age")
        or eligibility.get("age")
        or ""
    )
    if req_age_str:
        match = re.search(r"(\d+)", req_age_str)
        if match:
            age_limit = int(match.group(1))
            user_age = profile.get("age")
            if user_age is not None and user_age >= age_limit:
                issues.append(
                    f"Scholarship requires applicants to be under {age_limit} years old. "
                    f"Your recorded age is {user_age}."
                )

    # Discipline check
    req_disciplines = opportunity_data.get("req_disciplines") or []
    if req_disciplines:
        user_disciplines = [d.lower() for d in (profile.get("disciplines") or [])]
        req_lower = [d.lower() for d in req_disciplines]
        if not any(d in req_lower or any(r in d for r in req_lower) for d in user_disciplines):
            issues.append(
                f"Scholarship is open to disciplines: {', '.join(req_disciplines)}. "
                f"Your recorded disciplines ({', '.join(profile.get('disciplines') or [])}) may not qualify."
            )

    # Nationality check (only if whitelist is non-empty)
    req_nationality = opportunity_data.get("req_nationality") or []
    if req_nationality:
        user_nationality = (profile.get("nationality") or "").lower()
        req_lower = [n.lower() for n in req_nationality]
        if user_nationality not in req_lower:
            issues.append(
                f"Scholarship requires one of the following nationalities: {', '.join(req_nationality)}. "
                f"Your recorded nationality is '{profile.get('nationality', 'unknown')}'."
            )

    return issues


# ---------------------------------------------------------------------------
# Profile completeness check
# ---------------------------------------------------------------------------

def _check_profile_completeness(
    profile: Dict[str, Any],
    normalized_requirements: List[Dict[str, Any]],
) -> List[str]:
    """Return a list of missing fields/documents that BLOCK auto-apply.

    A document is only "missing" in the blocking sense if the system cannot
    write it from CV + profile data. Documents in `_GENERATABLE` (Cover Letter,
    SOP, Personal Statement, etc.) are *not* flagged as missing here — they
    flow through to asset_mapping which assigns tailoring_depth='generate',
    and gate 1 lets the user override with their own upload if desired.
    """
    # Lazy import to avoid a circular dependency between the eligibility node
    # (graph entry) and the asset_mapping node.
    from uppgrad_agentic.workflows.auto_apply.nodes.asset_mapping import _GENERATABLE

    missing: List[str] = []

    if not profile.get("name"):
        missing.append("name")
    if not profile.get("email"):
        missing.append("email")

    uploaded = profile.get("uploaded_documents") or {}
    for req in normalized_requirements:
        doc_type = req.get("document_type", "")
        req_type = req.get("requirement_type", "")
        if req_type != "document":
            continue
        if uploaded.get(doc_type):
            continue
        # Skip docs the system can generate from CV + profile.
        if doc_type in _GENERATABLE:
            continue
        missing.append(f"document:{doc_type}")

    return missing


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def eligibility_and_readiness(state: AutoApplyState) -> dict:
    updates = {"current_step": "eligibility_and_readiness", "step_history": ["eligibility_and_readiness"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    opportunity_type = state.get("opportunity_type", "")
    opportunity_data = state.get("opportunity_data") or {}
    normalized_requirements = state.get("normalized_requirements") or []
    from uppgrad_agentic.workflows.auto_apply._profile import resolve_profile
    profile = resolve_profile(state)

    reasons: List[str] = []
    missing_fields: List[str] = []

    # ------------------------------------------------------------------
    # 1. Deadline check — applies to all opportunity types
    # ------------------------------------------------------------------
    deadline_passed, deadline_reason = _check_deadline(opportunity_data)
    if deadline_passed:
        result = EligibilityResult(
            decision="ineligible",
            reasons=[deadline_reason],
            missing_fields=[],
        )
        return {**updates, "eligibility_result": result.model_dump()}

    # ------------------------------------------------------------------
    # 2. Hard eligibility constraints — per opportunity type
    # ------------------------------------------------------------------
    if opportunity_type == "job":
        issues = _check_job_eligibility(opportunity_data, profile)
    elif opportunity_type in ("masters", "phd"):
        issues = _check_program_eligibility(opportunity_data, profile)
    elif opportunity_type == "scholarship":
        issues = _check_scholarship_eligibility(opportunity_data, profile)
    else:
        issues = []

    if issues:
        # Hard eligibility failures → ineligible (not pending, user cannot fix these)
        result = EligibilityResult(
            decision="ineligible",
            reasons=issues,
            missing_fields=[],
        )
        return {**updates, "eligibility_result": result.model_dump()}

    # ------------------------------------------------------------------
    # 3. Profile completeness check
    # ------------------------------------------------------------------
    missing_fields = _check_profile_completeness(profile, normalized_requirements)

    if missing_fields:
        # Document uploads are fixable by the user → pending
        doc_missing = [f for f in missing_fields if f.startswith("document:")]
        profile_missing = [f for f in missing_fields if not f.startswith("document:")]

        pending_reasons: List[str] = []
        if profile_missing:
            pending_reasons.append(
                f"Your profile is missing required fields: {', '.join(profile_missing)}."
            )
        if doc_missing:
            doc_names = [f.removeprefix("document:") for f in doc_missing]
            pending_reasons.append(
                f"The following documents have not been uploaded yet: {', '.join(doc_names)}."
            )

        result = EligibilityResult(
            decision="pending",
            reasons=pending_reasons,
            missing_fields=missing_fields,
        )
        return {**updates, "eligibility_result": result.model_dump()}

    # ------------------------------------------------------------------
    # 4. Ready
    # ------------------------------------------------------------------
    result = EligibilityResult(
        decision="ready",
        reasons=["All eligibility checks passed and required documents are present."],
        missing_fields=[],
    )
    return {**updates, "eligibility_result": result.model_dump()}
