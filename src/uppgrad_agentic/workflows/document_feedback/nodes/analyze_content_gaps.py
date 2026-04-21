from __future__ import annotations

import re
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from uppgrad_agentic.common.llm import get_llm


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ContentGapsAnalysis(BaseModel):
    gaps: List[str] = Field(
        default_factory=list,
        description="Content that should be present given the user's profile but is missing.",
    )
    unexploited_strengths: List[str] = Field(
        default_factory=list,
        description="Strengths from the user profile that are not mentioned or are underplayed.",
    )
    weak_claims: List[str] = Field(
        default_factory=list,
        description="Vague or unsupported statements that would benefit from specifics.",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Concrete suggestions for filling the identified gaps.",
    )


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

_VAGUE_PATTERNS = [
    re.compile(r"\b(various|several|many|numerous|a number of)\b", re.IGNORECASE),
    re.compile(r"\b(good|great|excellent|strong|extensive)\s+(knowledge|experience|skills)\b", re.IGNORECASE),
    re.compile(r"\bresponsible\s+for\b", re.IGNORECASE),
    re.compile(r"\bhelped\s+(to\s+)?\w+", re.IGNORECASE),
    re.compile(r"\bworked\s+on\b", re.IGNORECASE),
    re.compile(r"\binvolved\s+in\b", re.IGNORECASE),
]


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def _heuristic(doc_sections: dict[str, str], profile_snapshot: dict) -> ContentGapsAnalysis:
    doc_text_lower = _normalise(" ".join(doc_sections.values()))

    gaps: List[str] = []
    unexploited: List[str] = []
    weak_claims: List[str] = []
    recommendations: List[str] = []

    # Check if profile skills appear in the document
    skills = profile_snapshot.get("skills") or []
    missing_skills = [s for s in skills if s.lower() not in doc_text_lower]
    if missing_skills:
        unexploited.append(f"Skills from profile not mentioned: {', '.join(missing_skills)}.")
        recommendations.append(
            f"Add a Skills section or weave in: {', '.join(missing_skills[:5])}."
        )

    # Check if work experience entries are reflected
    for exp in profile_snapshot.get("experience") or []:
        company = (exp.get("company") or "").lower()
        title = (exp.get("title") or "").lower()
        if company and company not in doc_text_lower:
            gaps.append(f"Work experience at '{exp.get('company')}' not found in document.")
            recommendations.append(f"Include your role as {exp.get('title')} at {exp.get('company')}.")
        elif title and title not in doc_text_lower:
            unexploited.append(f"Job title '{exp.get('title')}' not explicitly stated.")

    # Check education
    for edu in profile_snapshot.get("education") or []:
        institution = (edu.get("institution") or "").lower()
        if institution and institution not in doc_text_lower:
            gaps.append(f"Education at '{edu.get('institution')}' not mentioned.")

    # Detect vague language
    full_text = " ".join(doc_sections.values())
    seen_patterns: set[str] = set()
    for pattern in _VAGUE_PATTERNS:
        for m in pattern.finditer(full_text):
            phrase = m.group(0).lower()
            if phrase not in seen_patterns:
                seen_patterns.add(phrase)
                # Find the enclosing sentence for context
                start = max(0, m.start() - 60)
                snippet = full_text[start: m.end() + 60].replace("\n", " ").strip()
                weak_claims.append(f"Vague phrasing: '...{snippet}...'")

    if weak_claims:
        recommendations.append(
            "Replace vague phrases with quantified achievements "
            "(e.g. 'Led migration of X, reducing deploy time by 40%')."
        )

    return ContentGapsAnalysis(
        gaps=gaps,
        unexploited_strengths=unexploited,
        weak_claims=weak_claims[:8],  # cap to avoid noise
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM = """You are comparing an application document against the user's known profile.

Identify:
- gaps: important content absent from the document that the profile supports
- unexploited_strengths: profile strengths not mentioned or underplayed
- weak_claims: vague statements lacking specifics or metrics
- recommendations: concrete, actionable suggestions to fill each gap

Be specific. Reference actual profile details (skills, roles, companies) when pointing out gaps.
"""

_MAX_DOC_CHARS = 5000
_MAX_PROFILE_CHARS = 2000


# ---------------------------------------------------------------------------
# Node — receives context_pack via Send
# ---------------------------------------------------------------------------

def analyze_content_gaps(context_pack: dict) -> dict:
    doc_type = context_pack.get("doc_type", "UNKNOWN")
    doc_sections = context_pack.get("doc_sections") or {}
    profile_snapshot = context_pack.get("profile_snapshot") or {}

    llm = get_llm()
    if llm is None:
        result = _heuristic(doc_sections, profile_snapshot)
        return {"analysis_results": {"content_gaps": result.model_dump()}}

    doc_text = " ".join(doc_sections.values())[:_MAX_DOC_CHARS]
    profile_text = str(profile_snapshot)[:_MAX_PROFILE_CHARS]

    structured = llm.with_structured_output(ContentGapsAnalysis)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content=(
                f"Document type: {doc_type}\n\n"
                f"User profile:\n{profile_text}\n\n"
                f"Document text (truncated):\n{doc_text}"
            )
        ),
    ]

    try:
        result: ContentGapsAnalysis = structured.invoke(msgs)
        return {"analysis_results": {"content_gaps": result.model_dump()}}
    except Exception as e:
        result = _heuristic(doc_sections, profile_snapshot)
        out = result.model_dump()
        out["recommendations"] = out.get("recommendations", []) + [f"[LLM failed, used heuristic: {e}]"]
        return {"analysis_results": {"content_gaps": out}}
