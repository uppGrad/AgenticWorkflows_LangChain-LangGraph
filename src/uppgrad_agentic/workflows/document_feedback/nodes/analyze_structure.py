from __future__ import annotations

import re
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.common.prompt_context import format_profile_brief, format_user_focus


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class StructureAnalysis(BaseModel):
    missing_sections: List[str] = Field(
        default_factory=list,
        description="Sections expected for this doc type that are absent.",
    )
    ordering_issues: List[str] = Field(
        default_factory=list,
        description="Problems with section order (e.g. 'Education appears after Skills').",
    )
    layout_issues: List[str] = Field(
        default_factory=list,
        description="Layout or formatting problems (e.g. 'No clear section headers detected').",
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Overall structure quality (0 = poor, 1 = excellent).")
    summary: str = Field(..., description="One-sentence overall assessment of document structure.")


# ---------------------------------------------------------------------------
# Per-doc-type expected sections and their canonical order
# ---------------------------------------------------------------------------

_EXPECTED: dict[str, list[str]] = {
    "CV": [
        "Summary", "Objective", "Profile",
        "Experience", "Work Experience", "Professional Experience",
        "Education",
        "Skills", "Technical Skills",
        "Projects",
        "Certifications",
    ],
    "SOP": [
        "Introduction",
        "Background", "Academic Background",
        "Research Interests", "Research Experience",
        "Goals", "Career Goals",
        "Conclusion",
    ],
    "COVER_LETTER": [
        "Opening",
        "Body",
        "Closing",
    ],
}

# Minimum required sections (subset of expected)
_REQUIRED: dict[str, list[list[str]]] = {
    "CV": [
        ["Experience", "Work Experience", "Professional Experience", "Employment"],
        ["Education", "Academic Background"],
        ["Skills", "Technical Skills", "Core Competencies"],
    ],
    "SOP": [
        ["Introduction", "Opening"],
        ["Goals", "Career Goals", "Academic Goals", "Future Plans"],
        ["Conclusion", "Closing"],
    ],
    "COVER_LETTER": [
        ["Opening", "Introduction"],
        ["Body", "Main Body"],
        ["Closing", "Conclusion"],
    ],
}

SYSTEM = """You are evaluating the structure of an application document.

Assess:
- missing_sections: required sections absent from the document for its type
- ordering_issues: sections that appear in a suboptimal order
- layout_issues: formatting problems (missing headers, wall-of-text, etc.)
- score: 0.0–1.0 overall structure quality
- summary: one sentence overall assessment

Be specific and actionable. Only flag real problems, not stylistic preferences.
If a "User focus" block is provided, prioritise structural findings that serve those goals.
If an "Applicant profile" block is provided (e.g. target roles, education stage), let it inform
expectations about what sections matter most — never invent details not present in it.

CV-specific rules — read carefully, these correct common LLM defaults:
- A "Summary" / "Objective" / "Profile" section is OPTIONAL, not required. Only flag
  its absence as a missing_section when one of these is true:
  (a) the applicant has 5+ years of experience and the CV reads as a long, complex
      narrative the reader would benefit from being oriented to;
  (b) the applicant is a clear career-changer and the CV's fit for the target role
      isn't obvious from the Experience section;
  (c) the user's instructions explicitly asked for one.
  For early-career CVs, a generic summary is filler that displaces real content
  ("motivated team player seeking opportunities" makes the CV WORSE). Default to
  NOT recommending a Summary section.
- Required CV sections are: Experience (or equivalent), Education, Skills.
  Everything else is contextual.
- Skills CATEGORISATION (Languages | Frameworks | Tools) is also contextual.
  Recommend it only when the existing Skills section is long enough (12+ entries)
  that grouping aids scanning. A short skills list is fine flat.
"""


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def _heuristic(doc_type: str, doc_sections: dict[str, str]) -> StructureAnalysis:
    present = {k.lower() for k in doc_sections}
    required_groups = _REQUIRED.get(doc_type, [])

    missing: List[str] = []
    for group in required_groups:
        if not any(variant.lower() in present for variant in group):
            missing.append(group[0])  # report canonical name

    layout_issues: List[str] = []
    if not doc_sections or list(doc_sections.keys()) == ["Body"]:
        layout_issues.append("No distinct section headers detected — document may lack clear structure.")

    # Check for very short sections (< 30 chars) which suggest incomplete content
    for name, text in doc_sections.items():
        if name not in ("Preamble",) and len(text.strip()) < 30:
            layout_issues.append(f"Section '{name}' appears nearly empty.")

    # Simple ordering check for CV: Education before Experience is unusual
    ordering_issues: List[str] = []
    if doc_type == "CV":
        keys_lower = [k.lower() for k in doc_sections]
        edu_idx = next((i for i, k in enumerate(keys_lower) if "education" in k), None)
        exp_idx = next((i for i, k in enumerate(keys_lower) if "experience" in k), None)
        if edu_idx is not None and exp_idx is not None and edu_idx > exp_idx:
            ordering_issues.append("Education appears after Experience; reverse order is more standard for most CV formats.")

    total = len(required_groups)
    present_count = total - len(missing)
    score = round((present_count / max(total, 1)) * (1.0 - 0.05 * len(layout_issues)), 2)
    score = max(0.0, min(1.0, score))

    summary = (
        "Document structure looks solid." if score >= 0.8
        else f"Structure needs improvement: {len(missing)} required section(s) missing."
    )

    return StructureAnalysis(
        missing_sections=missing,
        ordering_issues=ordering_issues,
        layout_issues=layout_issues,
        score=score,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Node — receives context_pack via Send
# ---------------------------------------------------------------------------

def analyze_structure(context_pack: dict) -> dict:
    updates = {"step_history": ["analyze_structure"]}
    doc_type = context_pack.get("doc_type", "UNKNOWN")
    doc_sections = context_pack.get("doc_sections") or {}
    sections_summary = "\n".join(
        f"[{name}]: {text[:300]}{'...' if len(text) > 300 else ''}"
        for name, text in doc_sections.items()
    )

    llm = get_llm()
    if llm is None:
        result = _heuristic(doc_type, doc_sections)
        return {**updates, "analysis_results": {"structure": result.model_dump()}}

    user_focus = format_user_focus(context_pack.get("parsed_instructions"))
    profile_brief = format_profile_brief(context_pack.get("profile_snapshot"))

    body = (
        f"Document type: {doc_type}\n\n"
        f"Sections present (name + first 300 chars):\n{sections_summary or '(none detected)'}"
    )
    if user_focus:
        body += f"\n\n{user_focus}"
    if profile_brief:
        body += f"\n\n{profile_brief}"

    structured = llm.with_structured_output(StructureAnalysis)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=body),
    ]

    try:
        result: StructureAnalysis = structured.invoke(msgs)
        return {**updates, "analysis_results": {"structure": result.model_dump()}}
    except Exception as e:
        result = _heuristic(doc_type, doc_sections)
        out = result.model_dump()
        out["layout_issues"] = out.get("layout_issues", []) + [f"[LLM failed, used heuristic: {e}]"]
        return {**updates, "analysis_results": {"structure": out}}
