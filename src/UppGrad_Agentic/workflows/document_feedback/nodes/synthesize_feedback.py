from __future__ import annotations

import json
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.workflows.document_feedback.schemas import ChangeProposal
from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------

class SynthesisOutput(BaseModel):
    proposals: List[ChangeProposal] = Field(
        default_factory=list,
        description="Prioritized list of change proposals, most impactful first.",
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM = """You are a document improvement advisor synthesizing multiple analysis reports \
into concrete, actionable change proposals.

Each proposal must:
- target a specific section of the document
- include the exact original text (before_text) and your proposed replacement (after_text)
- provide a clear rationale explaining why the change improves the document
- have a confidence score (0.0–1.0)
- set requires_confirmation=true for structural or substantive content changes; \
  false for minor style/formatting fixes

Prioritize proposals by impact: structural gaps first, then content gaps, \
then style and ATS improvements, then opportunity alignment.

Merge overlapping findings into a single proposal. Avoid duplicates.
Return at most 15 proposals.
"""

_MAX_ANALYSIS_CHARS = 6000
_MAX_DOC_CHARS = 4000


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _heuristic_proposals(
    analysis_results: dict,
    doc_sections: dict,
) -> List[ChangeProposal]:
    proposals: List[ChangeProposal] = []

    # --- Structure ---
    structure = analysis_results.get("structure") or {}
    for section in structure.get("missing_sections", []):
        proposals.append(ChangeProposal(
            section=section,
            rationale=f"Required section '{section}' is missing from the document.",
            before_text="",
            after_text=f"[Add a '{section}' section with relevant content]",
            confidence=0.85,
            requires_confirmation=True,
        ))
    for issue in structure.get("ordering_issues", []):
        proposals.append(ChangeProposal(
            section="Document Structure",
            rationale=issue,
            before_text="[Current section order]",
            after_text="[Reorder sections as recommended]",
            confidence=0.75,
            requires_confirmation=True,
        ))
    for issue in structure.get("layout_issues", []):
        proposals.append(ChangeProposal(
            section="Formatting",
            rationale=issue,
            before_text="[Current layout]",
            after_text="[Apply clear section headers and consistent formatting]",
            confidence=0.70,
            requires_confirmation=False,
        ))

    # --- Content gaps ---
    content_gaps = analysis_results.get("content_gaps") or {}
    for gap in content_gaps.get("gaps", []):
        proposals.append(ChangeProposal(
            section="Content",
            rationale=gap,
            before_text="",
            after_text="[Add the missing content as identified]",
            confidence=0.80,
            requires_confirmation=True,
        ))
    for strength in content_gaps.get("unexploited_strengths", []):
        proposals.append(ChangeProposal(
            section="Content",
            rationale=strength,
            before_text="[Underplayed or absent content]",
            after_text="[Highlight this strength explicitly]",
            confidence=0.75,
            requires_confirmation=False,
        ))
    for claim in (content_gaps.get("weak_claims") or [])[:4]:
        proposals.append(ChangeProposal(
            section="Content",
            rationale="Vague or unsupported claim found in document.",
            before_text=claim,
            after_text="[Replace with a quantified achievement or specific detail]",
            confidence=0.70,
            requires_confirmation=False,
        ))

    # --- Style ---
    style = analysis_results.get("style") or {}
    for issue in (style.get("issues") or [])[:4]:
        proposals.append(ChangeProposal(
            section="Style",
            rationale=issue,
            before_text="[See issue description above]",
            after_text="[Apply suggested correction]",
            confidence=0.65,
            requires_confirmation=False,
        ))
    for passive in (style.get("passive_voice_instances") or [])[:3]:
        proposals.append(ChangeProposal(
            section="Style",
            rationale="Passive voice weakens impact; convert to active voice.",
            before_text=passive,
            after_text="[Rewrite starting with a strong action verb]",
            confidence=0.70,
            requires_confirmation=False,
        ))

    # --- ATS ---
    ats = analysis_results.get("ats") or {}
    missing_kw = ats.get("missing_keywords") or []
    if missing_kw:
        kw_list = ", ".join(missing_kw[:10])
        proposals.append(ChangeProposal(
            section="Skills / Keywords",
            rationale=f"ATS keywords absent from document: {kw_list}.",
            before_text="[Current Skills section]",
            after_text=f"[Add relevant keywords: {kw_list}]",
            confidence=0.75,
            requires_confirmation=False,
        ))
    for fmt_issue in (ats.get("formatting_issues") or [])[:2]:
        proposals.append(ChangeProposal(
            section="Formatting",
            rationale=fmt_issue,
            before_text="[Current formatting]",
            after_text="[Apply ATS-friendly formatting]",
            confidence=0.70,
            requires_confirmation=False,
        ))

    # --- Opportunity alignment ---
    alignment = analysis_results.get("opportunity_alignment") or {}
    for req in (alignment.get("missing_requirements") or [])[:4]:
        proposals.append(ChangeProposal(
            section="Opportunity Alignment",
            rationale=f"Opportunity requirement not addressed in document: {req}",
            before_text="",
            after_text=f"[Address requirement: {req}]",
            confidence=0.80,
            requires_confirmation=True,
        ))
    missing_opp_kw = alignment.get("missing_keywords") or []
    if missing_opp_kw:
        kw_list = ", ".join(missing_opp_kw[:8])
        proposals.append(ChangeProposal(
            section="Opportunity Alignment",
            rationale=f"Keywords from the opportunity description are absent: {kw_list}.",
            before_text="[Current document text]",
            after_text=f"[Incorporate opportunity keywords: {kw_list}]",
            confidence=0.75,
            requires_confirmation=False,
        ))

    # Deduplicate on rationale prefix and cap at 15
    seen: set[str] = set()
    unique: List[ChangeProposal] = []
    for p in proposals:
        key = p.rationale[:80]
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique[:15]


# ---------------------------------------------------------------------------
# Node — convergence point after all parallel analysis nodes
# ---------------------------------------------------------------------------

def synthesize_feedback(state: DocFeedbackState) -> dict:
    if state.get("result", {}).get("status") == "error":
        return {}

    analysis_results = state.get("analysis_results") or {}
    context_pack = state.get("context_pack") or {}
    doc_sections = context_pack.get("doc_sections") or state.get("doc_sections") or {}
    doc_type = (state.get("doc_classification") or {}).get("doc_type", "UNKNOWN")
    parsed_instructions = state.get("parsed_instructions") or {}

    # On retry: include evaluator feedback so the LLM can address specific issues.
    prior_eval = state.get("evaluation_result") or {}
    prior_issues = prior_eval.get("issues") or []

    llm = get_llm()
    if llm is None:
        proposals = _heuristic_proposals(analysis_results, doc_sections)
        return {"proposals": [p.model_dump() for p in proposals]}

    analysis_text = json.dumps(analysis_results, indent=2)[:_MAX_ANALYSIS_CHARS]
    doc_text = " ".join(doc_sections.values())[:_MAX_DOC_CHARS]
    focus = parsed_instructions.get("focus_areas") or []
    focus_line = f"User focus areas: {', '.join(focus)}\n" if focus else ""
    retry_section = (
        "\n\nPREVIOUS ATTEMPT WAS REJECTED. Issues to fix:\n"
        + "\n".join(f"- {iss}" for iss in prior_issues)
        if prior_issues else ""
    )

    structured = llm.with_structured_output(SynthesisOutput)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content=(
                f"Document type: {doc_type}\n"
                f"{focus_line}"
                f"{retry_section}"
                f"\nAnalysis results (JSON):\n{analysis_text}"
                f"\n\nDocument text (first {_MAX_DOC_CHARS} chars):\n{doc_text}"
            )
        ),
    ]

    try:
        output: SynthesisOutput = structured.invoke(msgs)
        return {"proposals": [p.model_dump() for p in output.proposals]}
    except Exception as e:
        proposals = _heuristic_proposals(analysis_results, doc_sections)
        result = [p.model_dump() for p in proposals]
        if result:
            result[0]["rationale"] += f" [LLM failed, used heuristic: {e}]"
        return {"proposals": result}
