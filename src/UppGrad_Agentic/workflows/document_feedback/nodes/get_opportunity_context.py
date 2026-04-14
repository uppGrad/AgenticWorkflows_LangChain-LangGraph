from __future__ import annotations

import re

from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


# Keywords in user_instructions that suggest the user has a specific opportunity
# in mind. When any are found we return a mock opportunity so downstream nodes
# can exercise the opportunity-alignment analysis path.
_OPPORTUNITY_SIGNALS = re.compile(
    r"\b(job|position|role|vacancy|opening|internship|fellowship|"
    r"program|programme|applying\s+to|applying\s+for|msc|phd|mba|master|"
    r"company|organisation|organization|university|school)\b",
    re.IGNORECASE,
)

# Hardcoded mock returned when the user appears to have a specific opportunity.
# Real implementation will look up the opportunity from the database or a URL
# the user provided.
_MOCK_OPPORTUNITY: dict = {
    "title": "Software Engineer",
    "organization": "Acme Technologies",
    "description": (
        "We are looking for a Software Engineer to join our platform team. "
        "You will design and build scalable backend services, collaborate with "
        "product managers, and contribute to our open-source tooling."
    ),
    "requirements": [
        "Bachelor's degree in Computer Science or related field",
        "2+ years of experience with Python or Go",
        "Experience with cloud infrastructure (AWS / GCP)",
        "Strong understanding of REST API design",
        "Familiarity with Docker and Kubernetes is a plus",
    ],
    "keywords": [
        "Python", "Go", "REST API", "microservices", "Docker",
        "Kubernetes", "AWS", "GCP", "scalable systems",
    ],
    "source": "mock",
}


def get_opportunity_context(state: DocFeedbackState) -> dict:
    if state.get("result", {}).get("status") == "error":
        return {}

    instructions = (state.get("user_instructions") or "").strip()

    if instructions and _OPPORTUNITY_SIGNALS.search(instructions):
        # User appears to be targeting a specific opportunity — return the mock
        # so the opportunity-alignment analysis path has data to work with.
        return {"opportunity_context": _MOCK_OPPORTUNITY}

    # No opportunity context provided: return an empty dict so downstream nodes
    # can skip the opportunity-alignment analysis branch cleanly.
    return {"opportunity_context": {}}
