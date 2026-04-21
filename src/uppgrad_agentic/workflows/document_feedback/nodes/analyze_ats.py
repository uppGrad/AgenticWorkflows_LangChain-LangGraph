from __future__ import annotations

import re
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from uppgrad_agentic.common.llm import get_llm


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ATSAnalysis(BaseModel):
    keyword_hits: List[str] = Field(
        default_factory=list,
        description="Standard professional keywords found in the CV.",
    )
    missing_keywords: List[str] = Field(
        default_factory=list,
        description="Common ATS keywords absent from the CV.",
    )
    formatting_issues: List[str] = Field(
        default_factory=list,
        description="ATS-unfriendly formatting patterns detected.",
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Overall ATS friendliness (0 = poor, 1 = excellent).")
    recommendations: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Common ATS keywords across professional roles
# ---------------------------------------------------------------------------

_ATS_KEYWORDS = [
    # Technical
    "python", "javascript", "java", "sql", "git", "docker", "kubernetes",
    "aws", "gcp", "azure", "rest", "api", "linux", "ci/cd", "agile", "scrum",
    # Leadership / soft skills
    "led", "managed", "developed", "designed", "implemented", "delivered",
    "improved", "reduced", "increased", "built", "architected",
    # Business
    "stakeholder", "cross-functional", "revenue", "roadmap", "strategy",
]

# Patterns that ATS systems often struggle with
_ATS_UNFRIENDLY = [
    (re.compile(r"[│┃|]{3,}"), "Table-like character borders detected — ATS may misparse columns."),
    (re.compile(r"[●•▪▸◆✓✗★]"), "Non-standard bullet characters detected — use plain hyphens or asterisks."),
    (re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE), "Page-number text detected — remove for ATS."),
    (re.compile(r"[\u2018\u2019\u201c\u201d]"), "Curly/smart quotes detected — replace with straight quotes."),
    (
        re.compile(r"(header|footer|table of contents)", re.IGNORECASE),
        "References to headers/footers/TOC detected — these may not parse correctly.",
    ),
]


def _heuristic(doc_sections: dict[str, str], opportunity_context: dict | None = None) -> ATSAnalysis:
    full_text = " ".join(doc_sections.values()).lower()

    # Build keyword list: start with generic, then add opportunity-specific
    keywords_to_check = list(_ATS_KEYWORDS)
    if opportunity_context:
        # Extract keywords from opportunity title and description
        opp_text = " ".join([
            opportunity_context.get("title", ""),
            opportunity_context.get("description", ""),
            " ".join(opportunity_context.get("keywords", [])),
        ])
        opp_tokens = set(re.findall(r"\b[a-z][a-z0-9+#/.]{2,}\b", opp_text.lower()))
        # Filter out common stopwords
        opp_keywords = [kw for kw in opp_tokens if kw not in {
            "the", "and", "for", "are", "with", "that", "this", "will", "have",
            "from", "they", "your", "our", "their", "its", "you", "not", "but",
            "all", "can", "may", "was", "has", "been", "more", "any", "who",
            "about", "experience", "work", "looking", "team", "join", "role",
            "company", "ideal", "candidate", "requirements", "responsibilities",
            "ability", "strong", "must", "including", "working", "position",
        }]
        keywords_to_check = list(set(keywords_to_check + opp_keywords))

    hits = [kw for kw in keywords_to_check if re.search(rf"\b{re.escape(kw)}\b", full_text)]
    missing = [kw for kw in keywords_to_check if kw not in hits]

    formatting_issues: List[str] = []
    original_text = " ".join(doc_sections.values())
    for pattern, message in _ATS_UNFRIENDLY:
        if pattern.search(original_text):
            formatting_issues.append(message)

    # Check for contact info (basic ATS expectation)
    if not re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", original_text):
        formatting_issues.append("No email address detected — include contact details.")
    if not re.search(r"\b(\+?\d[\d\s\-().]{7,})\b", original_text):
        formatting_issues.append("No phone number detected — include contact details.")

    # Score: keyword coverage minus formatting penalty
    coverage = len(hits) / max(len(_ATS_KEYWORDS), 1)
    formatting_penalty = 0.08 * len(formatting_issues)
    score = round(max(0.0, min(1.0, coverage - formatting_penalty)), 2)

    recommendations: List[str] = []
    if missing[:5]:
        recommendations.append(f"Consider adding relevant keywords: {', '.join(missing[:5])}.")
    if formatting_issues:
        recommendations.append("Fix formatting issues listed above to improve ATS parse rate.")

    return ATSAnalysis(
        keyword_hits=hits,
        missing_keywords=missing[:15],  # top missing only
        formatting_issues=formatting_issues,
        score=score,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM = """You are evaluating a CV for ATS (Applicant Tracking System) compatibility.

Assess:
- keyword_hits: standard professional / technical keywords present in the CV
- missing_keywords: important ATS keywords absent (based on common roles and the document's domain)
- formatting_issues: ATS-unfriendly patterns (special characters, tables, non-standard bullets, etc.)
- score: 0.0–1.0 ATS friendliness
- recommendations: specific fixes

Focus on machine-parseability, not subjective quality.
When an opportunity description is provided, focus keyword analysis on keywords
relevant to THAT SPECIFIC role/opportunity, not generic tech keywords.
"""

_MAX_CHARS = 6000
_MAX_OPP_CHARS = 2000


# ---------------------------------------------------------------------------
# Node — receives context_pack via Send
# ---------------------------------------------------------------------------

def analyze_ats(context_pack: dict) -> dict:
    doc_type = context_pack.get("doc_type", "UNKNOWN")

    # ATS analysis is only meaningful for CVs
    if doc_type != "CV":
        return {"analysis_results": {"ats": {}}}

    doc_sections = context_pack.get("doc_sections") or {}
    opportunity_context = context_pack.get("opportunity_context") or {}

    llm = get_llm()
    if llm is None:
        result = _heuristic(doc_sections, opportunity_context)
        return {"analysis_results": {"ats": result.model_dump()}}

    doc_text = " ".join(doc_sections.values())[:_MAX_CHARS]

    # Include opportunity context in the prompt when available
    opp_section = ""
    if opportunity_context and opportunity_context.get("title"):
        opp_text = str(opportunity_context)[:_MAX_OPP_CHARS]
        opp_section = (
            f"\n\nTARGET OPPORTUNITY (tailor keyword analysis to this role):\n{opp_text}\n"
            "Focus your missing_keywords on keywords that are relevant to THIS specific "
            "opportunity, not generic tech keywords."
        )

    structured = llm.with_structured_output(ATSAnalysis)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"CV text (truncated):\n{doc_text}{opp_section}"),
    ]

    try:
        result: ATSAnalysis = structured.invoke(msgs)
        return {"analysis_results": {"ats": result.model_dump()}}
    except Exception as e:
        result = _heuristic(doc_sections, opportunity_context)
        out = result.model_dump()
        out["recommendations"] = out.get("recommendations", []) + [f"[LLM failed, used heuristic: {e}]"]
        return {"analysis_results": {"ats": out}}
