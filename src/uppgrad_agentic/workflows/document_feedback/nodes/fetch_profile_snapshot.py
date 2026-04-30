from __future__ import annotations

from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


# Stub profile: hardcoded for CLI / standalone runs. Schema mirrors the
# Django adapter's build_profile_snapshot so node code can rely on the same
# keys whether running locally or in production.
_STUB_PROFILE: dict = {
    "user_id": "stub-user-001",
    "name": "Alex Johnson",
    "email": "alex.johnson@example.com",
    "location": "Berlin, Germany",
    "bio": "Recent CS grad focused on distributed systems and developer tooling.",
    "linkedin_url": None,
    "github_url": None,
    "portfolio_url": None,
    "education": [
        {
            "degree": "BSc Computer Science",
            "institution": "State University",
            "major": "Computer Science",
            "start_year": 2018,
            "year": 2022,
            "gpa": 3.7,
        }
    ],
    "experience": [
        {
            "title": "Software Engineer Intern",
            "company": "TechCorp",
            "location": "Remote",
            "start_date": "2021-06-01",
            "end_date": "2021-12-01",
            "duration_months": 6,
            "description": "Built REST APIs and wrote unit tests for a microservices backend.",
        },
        {
            "title": "Junior Developer",
            "company": "StartupXYZ",
            "location": "Berlin",
            "start_date": "2022-08-01",
            "end_date": "2023-10-01",
            "duration_months": 14,
            "description": "Full-stack development with React and FastAPI; led migration to Docker.",
        },
    ],
    "projects": [
        {
            "title": "Distributed log aggregator",
            "description": "Side project: Go + Kafka log fan-in service used by a small open-source community.",
        },
    ],
    "publications": [],
    "achievements": [
        {"title": "Dean's List", "given_by": "State University", "date": "2021-12-15"},
    ],
    "skills": ["Python", "JavaScript", "React", "FastAPI", "Docker", "PostgreSQL", "Git"],
    "languages": ["English (native)", "Spanish (intermediate)"],
    "target_roles": ["Software Engineer", "Backend Developer"],
    "target_programs": [],
    "interests": ["Distributed Systems", "Developer Experience"],
    "work_style": "REMOTE",
    "work_type": "FULL_TIME",
    "experience_level": "ENTRY_LEVEL",
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
