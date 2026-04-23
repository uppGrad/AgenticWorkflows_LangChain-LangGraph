from __future__ import annotations

from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


# Stub profile: hardcoded until the user database exists.
_STUB_PROFILE: dict = {
    "user_id": "stub-user-001",
    "name": "Alex Johnson",
    "email": "alex.johnson@example.com",
    "education": [
        {
            "degree": "BSc Computer Science",
            "institution": "State University",
            "year": 2022,
            "gpa": 3.7,
        }
    ],
    "experience": [
        {
            "title": "Software Engineer Intern",
            "company": "TechCorp",
            "duration_months": 6,
            "description": "Built REST APIs and wrote unit tests for a microservices backend.",
        },
        {
            "title": "Junior Developer",
            "company": "StartupXYZ",
            "duration_months": 14,
            "description": "Full-stack development with React and FastAPI; led migration to Docker.",
        },
    ],
    "skills": ["Python", "JavaScript", "React", "FastAPI", "Docker", "PostgreSQL", "Git"],
    "languages": ["English (native)", "Spanish (intermediate)"],
    "target_roles": ["Software Engineer", "Backend Developer"],
    "target_programs": [],
}


def fetch_profile_snapshot(state: DocFeedbackState) -> dict:
    updates = {"current_step": "fetch_profile_snapshot", "step_history": ["fetch_profile_snapshot"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    # If the caller pre-injected profile_snapshot (e.g., Django adapter),
    # skip the stub and use the real data already in state.
    if state.get("profile_snapshot"):
        return updates

    # Fallback: return the hardcoded stub for standalone/CLI usage so
    # downstream nodes have realistic data to work with.
    return {**updates, "profile_snapshot": _STUB_PROFILE}
