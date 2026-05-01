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
#
# Schema note for backend integration: the document feedback substance pipeline
# treats the optional fields below as a *menu* of company signals the LLM may
# selectively reference — not a checklist. Populate any subset that is
# verifiably true; leave fields out entirely if you don't have confident data.
# The substance prompt explicitly forbids invention of company facts not
# present here, and caps the synthesizer to ~one signal per rewritten paragraph
# so the doc doesn't read as name-dropping.
#
# Optional menu fields (all lists unless noted):
#   - mission (str): one-sentence company mission, in the company's own words
#                    when possible.
#   - products: notable products / surfaces the candidate could plausibly engage
#               with. Short labels, not marketing copy.
#   - values: 2-4 stated company values. Same rule.
#   - distinctive_responsibilities: bullets that make THIS role distinct from
#               a sibling role at the same company (e.g. "owns the inference
#               console UI" vs. generic "frontend work"). Often the most
#               useful field for role-specificity rewrites.
#   - recent_signals: recent papers, launches, blog posts, or org moves the
#               candidate could legitimately reference. Each item should be a
#               concrete short label, not a paragraph.
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
    # Optional substance menu — see schema note above.
    "mission": "Help small businesses run their operations on a single platform.",
    "products": [
        "Acme Platform (multi-tenant SaaS)",
        "Acme Open (open-source CLI for platform extensions)",
    ],
    "values": ["Customer obsession", "Build in the open", "Ship to learn"],
    "distinctive_responsibilities": [
        "Owns the platform's billing service end-to-end (design through on-call).",
        "Drives the migration from monolith to event-driven services.",
    ],
    "recent_signals": [
        "Open-sourced Acme Open earlier this year",
        "Recent engineering blog post on cutting p99 latency by 40% via async batching",
    ],
    "source": "mock",
}


def get_opportunity_context(state: DocFeedbackState) -> dict:
    updates = {"current_step": "get_opportunity_context", "step_history": ["get_opportunity_context"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    # If the caller pre-injected opportunity_context (e.g., Django adapter),
    # skip the stub/mock logic and use the real data already in state.
    if state.get("opportunity_context"):
        return updates

    instructions = (state.get("user_instructions") or "").strip()

    if instructions and _OPPORTUNITY_SIGNALS.search(instructions):
        # User appears to be targeting a specific opportunity — return the mock
        # so the opportunity-alignment analysis path has data to work with.
        return {**updates, "opportunity_context": _MOCK_OPPORTUNITY}

    # No opportunity context provided: return an empty dict so downstream nodes
    # can skip the opportunity-alignment analysis branch cleanly.
    return {**updates, "opportunity_context": {}}
