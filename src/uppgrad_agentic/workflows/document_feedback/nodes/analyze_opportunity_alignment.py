from __future__ import annotations

import re
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from uppgrad_agentic.common.llm import get_llm


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class OpportunityAlignmentAnalysis(BaseModel):
    aligned_requirements: List[str] = Field(
        default_factory=list,
        description="Opportunity requirements that the document addresses.",
    )
    missing_requirements: List[str] = Field(
        default_factory=list,
        description="Opportunity requirements not addressed in the document.",
    )
    keyword_matches: List[str] = Field(
        default_factory=list,
        description="Keywords from the opportunity description found in the document.",
    )
    missing_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords from the opportunity description absent from the document.",
    )
    alignment_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Overall alignment between document and opportunity (0 = poor, 1 = strong).",
    )
    recommendations: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set[str]:
    """Lower-case word tokens, stripping punctuation."""
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


_STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "will", "have",
    "from", "they", "your", "our", "their", "its", "you", "not", "but",
    "all", "can", "may", "was", "has", "been", "more", "any", "who",
}


def _heuristic(
    doc_sections: dict[str, str],
    opportunity_context: dict,
) -> OpportunityAlignmentAnalysis:
    doc_text = " ".join(doc_sections.values())
    doc_tokens = _tokenise(doc_text)

    # Extract keywords from opportunity
    opp_text = " ".join([
        opportunity_context.get("title", ""),
        opportunity_context.get("description", ""),
        " ".join(opportunity_context.get("keywords", [])),
    ])
    opp_tokens = _tokenise(opp_text) - _STOPWORDS
    explicit_keywords = [kw.lower() for kw in opportunity_context.get("keywords", [])]

    kw_matches = [kw for kw in explicit_keywords if kw.lower() in doc_text.lower()]
    kw_missing = [kw for kw in explicit_keywords if kw.lower() not in doc_text.lower()]

    # Check requirements
    requirements: List[str] = opportunity_context.get("requirements") or []
    aligned: List[str] = []
    missing: List[str] = []
    for req in requirements:
        req_tokens = _tokenise(req) - _STOPWORDS
        overlap = req_tokens & doc_tokens
        # Require at least 40 % of requirement tokens to be present
        if req_tokens and len(overlap) / len(req_tokens) >= 0.4:
            aligned.append(req)
        else:
            missing.append(req)

    total = len(aligned) + len(missing)
    req_score = len(aligned) / max(total, 1)
    kw_score = len(kw_matches) / max(len(explicit_keywords), 1) if explicit_keywords else 0.5
    alignment_score = round((req_score + kw_score) / 2, 2)

    recommendations: List[str] = []
    if kw_missing:
        recommendations.append(
            f"Weave in missing opportunity keywords: {', '.join(kw_missing[:6])}."
        )
    if missing:
        recommendations.append(
            f"Address unmet requirements, e.g.: '{missing[0]}'."
        )
    if alignment_score < 0.5:
        recommendations.append(
            "Significant alignment gap — consider tailoring the document more closely to this opportunity."
        )

    return OpportunityAlignmentAnalysis(
        aligned_requirements=aligned,
        missing_requirements=missing,
        keyword_matches=kw_matches,
        missing_keywords=kw_missing,
        alignment_score=alignment_score,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM = """You are assessing how well an application document aligns with a specific opportunity.

Assess:
- aligned_requirements: opportunity requirements addressed in the document
- missing_requirements: opportunity requirements not addressed
- keyword_matches: opportunity keywords present in the document
- missing_keywords: opportunity keywords absent from the document
- alignment_score: 0.0–1.0 overall alignment
- recommendations: specific actions to improve alignment

Be precise — quote requirement text and keyword text directly from the opportunity where possible.
"""

_MAX_DOC_CHARS = 5000
_MAX_OPP_CHARS = 2000


# ---------------------------------------------------------------------------
# Node — receives context_pack via Send
# ---------------------------------------------------------------------------

def analyze_opportunity_alignment(context_pack: dict) -> dict:
    updates = {"step_history": ["analyze_opportunity_alignment"]}
    opportunity_context = context_pack.get("opportunity_context") or {}

    # No opportunity provided — nothing to align against
    if not opportunity_context:
        return {**updates, "analysis_results": {"opportunity_alignment": {}}}

    doc_sections = context_pack.get("doc_sections") or {}
    doc_type = context_pack.get("doc_type", "UNKNOWN")

    llm = get_llm()
    if llm is None:
        result = _heuristic(doc_sections, opportunity_context)
        return {**updates, "analysis_results": {"opportunity_alignment": result.model_dump()}}

    doc_text = " ".join(doc_sections.values())[:_MAX_DOC_CHARS]
    opp_text = str(opportunity_context)[:_MAX_OPP_CHARS]

    structured = llm.with_structured_output(OpportunityAlignmentAnalysis)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content=(
                f"Document type: {doc_type}\n\n"
                f"Opportunity details:\n{opp_text}\n\n"
                f"Document text (truncated):\n{doc_text}"
            )
        ),
    ]

    try:
        result: OpportunityAlignmentAnalysis = structured.invoke(msgs)
        return {**updates, "analysis_results": {"opportunity_alignment": result.model_dump()}}
    except Exception as e:
        result = _heuristic(doc_sections, opportunity_context)
        out = result.model_dump()
        out["recommendations"] = out.get("recommendations", []) + [f"[LLM failed, used heuristic: {e}]"]
        return {**updates, "analysis_results": {"opportunity_alignment": out}}
