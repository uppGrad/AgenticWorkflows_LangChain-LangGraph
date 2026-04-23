from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.workflows.document_feedback.schemas import ChangeProposal
from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------

class SynthesisOutput(BaseModel):
    proposals: List[ChangeProposal] = Field(
        default_factory=list,
        description="Prioritized list of change proposals, most impactful first.",
    )


# ---------------------------------------------------------------------------
# Prompts — with strong anti-hallucination guardrails
# ---------------------------------------------------------------------------

SYSTEM = """\
You are an elite career document advisor — a former Big Tech recruiter and \
professional resume writer synthesizing multiple analysis reports into \
concrete, high-impact change proposals.

═══════════════════════ RULES ═══════════════════════

1. **GROUND EVERYTHING IN THE DOCUMENT.** You must NEVER invent:
   - Skills, tools, or technologies the candidate does not mention
   - Awards, certifications, or honors that do not appear
   - Job titles, company names, or experiences not present
   However, you SHOULD:
   - Strengthen weak descriptions with powerful action verbs (e.g. \
     "Worked on" → "Architected and delivered")
   - Suggest quantification prompts where the candidate likely has data \
     (e.g. "Built API" → "Built REST API serving [X] requests/day — \
     add your actual number")
   - Restructure bullet points for maximum recruiter impact (lead with \
     result, then method, then tech)
   - Add industry-standard keywords that are SYNONYMS of existing skills \
     (e.g. if they mention "React" you can add "React.js")

2. **before_text MUST be a VERBATIM QUOTE** copied exactly from the document text \
   provided below. It must appear character-for-character in the document. \
   If you cannot find an exact quote, set before_text to an empty string "" \
   and explain in the rationale that this is a new addition suggestion.

3. **after_text should be polished, recruiter-ready text.** Don't just rephrase — \
   make it genuinely better:
   - Use the XYZ formula: "Accomplished [X] as measured by [Y], by doing [Z]"
   - Use strong action verbs: Led, Architected, Optimized, Spearheaded, Delivered
   - Remove filler words and weak phrases
   - Ensure consistent tense (past for previous roles, present for current)
   - For new sections (before_text=""), provide a realistic template the user \
     can fill in, not just "[Add X here]"

4. **One proposal per change.** Do not bundle multiple unrelated changes into \
   one proposal. Each proposal should target a single, specific edit.

═══════════════════════════════════════════════════════════════

Each proposal must:
- target a specific section of the document
- include the exact original text (before_text) and your proposed replacement (after_text)
- provide a clear, specific rationale explaining the improvement (not generic advice)
- have a confidence score (0.0–1.0)
- set requires_confirmation=true for structural or substantive content changes; \
  false for minor style/formatting fixes

Prioritize proposals by recruiter impact:
1. Weak/vague bullet points that can be made quantifiable and action-oriented
2. Missing high-impact sections (Summary, Skills categorization)
3. Poor structure or ordering that hurts scannability
4. ATS keyword gaps (using SYNONYMS of existing skills only)
5. Opportunity alignment (tailoring language to the target role)

Good proposal examples:
  ✅ "Worked on backend" → "Developed and maintained 3 microservices handling 10K+ daily requests using Django and PostgreSQL"
  ✅ "Helped with testing" → "Implemented comprehensive test suite achieving [X]% code coverage, reducing production bugs by [X]%"
  ✅ Adding a professional summary section with a draft the user can customize
  ✅ Reorganizing skills into categorized groups (Languages | Frameworks | Tools)

Bad proposal examples (NEVER do these):
  ❌ Adding "AWS, Azure, Kubernetes" when candidate only mentions "Docker"
  ❌ Inventing "Dean's List (2022-2023)" when no awards section exists
  ❌ Completely rewriting experiences with fabricated responsibilities

Merge overlapping findings into a single proposal. Avoid duplicates.
Return 8-15 high-impact proposals. Quality over quantity.
"""

_MAX_ANALYSIS_CHARS = 12000
_MAX_DOC_CHARS = 8000

# Minimum fuzzy-match ratio for before_text to be considered "grounded"
_MIN_MATCH_RATIO = 0.55


# ---------------------------------------------------------------------------
# Post-processing: validate proposals against the actual document
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase for fuzzy matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _before_text_is_grounded(before_text: str, full_doc_text: str) -> bool:
    """Check if before_text exists (or nearly exists) in the document."""
    if not before_text or before_text.startswith("["):
        # Empty or placeholder before_text is fine (new section suggestions)
        return True

    norm_before = _normalize(before_text)
    norm_doc = _normalize(full_doc_text)

    # Exact substring match (fast path)
    if norm_before in norm_doc:
        return True

    # Sliding-window fuzzy match — check if any window of similar length
    # in the document is a close match
    window_len = len(norm_before)
    if window_len < 10:
        # Very short text — require exact match
        return norm_before in norm_doc

    best_ratio = 0.0
    step = max(1, window_len // 4)
    for i in range(0, max(1, len(norm_doc) - window_len + 1), step):
        window = norm_doc[i : i + window_len + 20]  # slight oversize for flexibility
        ratio = SequenceMatcher(None, norm_before, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            if best_ratio >= _MIN_MATCH_RATIO:
                return True

    return best_ratio >= _MIN_MATCH_RATIO


def _validate_proposals(
    proposals: List[dict],
    doc_sections: dict,
) -> List[dict]:
    """Filter out proposals with hallucinated before_text."""
    full_text = " ".join(doc_sections.values())
    validated = []
    dropped = 0

    for p in proposals:
        before = p.get("before_text", "")
        if _before_text_is_grounded(before, full_text):
            validated.append(p)
        else:
            dropped += 1
            logger.warning(
                "Dropped hallucinated proposal (before_text not found in document): "
                "section=%s, before_text=%.80s…",
                p.get("section", "?"),
                before,
            )

    if dropped:
        logger.info(
            "Validation: kept %d / %d proposals (%d dropped as ungrounded)",
            len(validated), len(proposals), dropped,
        )

    return validated


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
    updates = {"current_step": "synthesize_feedback", "step_history": ["synthesize_feedback"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    analysis_results = state.get("analysis_results") or {}
    context_pack = state.get("context_pack") or {}
    doc_sections = context_pack.get("doc_sections") or state.get("doc_sections") or {}
    doc_type = (state.get("doc_classification") or {}).get("doc_type", "UNKNOWN")
    parsed_instructions = state.get("parsed_instructions") or {}
    opportunity_context = context_pack.get("opportunity_context") or state.get("opportunity_context") or {}

    # On retry: include evaluator feedback so the LLM can address specific issues.
    prior_eval = state.get("evaluation_result") or {}
    prior_issues = prior_eval.get("issues") or []

    llm = get_llm()
    if llm is None:
        proposals = _heuristic_proposals(analysis_results, doc_sections)
        return {**updates, "proposals": [p.model_dump() for p in proposals]}

    analysis_text = json.dumps(analysis_results, indent=2)[:_MAX_ANALYSIS_CHARS]
    doc_text = " ".join(doc_sections.values())[:_MAX_DOC_CHARS]
    focus = parsed_instructions.get("focus_areas") or []
    focus_line = f"User focus areas: {', '.join(focus)}\n" if focus else ""
    retry_section = (
        "\n\nPREVIOUS ATTEMPT WAS REJECTED. Issues to fix:\n"
        + "\n".join(f"- {iss}" for iss in prior_issues)
        if prior_issues else ""
    )

    # Include opportunity context so proposals are tailored to the target role
    opp_section = ""
    if opportunity_context and opportunity_context.get("title"):
        opp_text = json.dumps(opportunity_context, indent=2)[:2000]
        opp_section = (
            f"\n\nTARGET OPPORTUNITY (tailor proposals to this role):\n{opp_text}\n"
            "Prioritize changes that align the document with THIS specific opportunity.\n"
            "IMPORTANT: Only suggest rephrasing EXISTING content to better match the "
            "opportunity. Do NOT add skills or experiences the candidate does not have.\n"
        )

    structured = llm.with_structured_output(SynthesisOutput)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content=(
                f"Document type: {doc_type}\n"
                f"{focus_line}"
                f"{retry_section}"
                f"{opp_section}"
                f"\nAnalysis results (JSON):\n{analysis_text}"
                f"\n\nDocument text (first {_MAX_DOC_CHARS} chars):\n{doc_text}"
            )
        ),
    ]

    try:
        output: SynthesisOutput = structured.invoke(msgs)
        raw_proposals = [p.model_dump() for p in output.proposals]

        # Post-process: drop proposals with hallucinated before_text
        validated = _validate_proposals(raw_proposals, doc_sections)
        return {**updates, "proposals": validated}
    except Exception as e:
        proposals = _heuristic_proposals(analysis_results, doc_sections)
        result = [p.model_dump() for p in proposals]
        if result:
            result[0]["rationale"] += f" [LLM failed, used heuristic: {e}]"
        return {**updates, "proposals": result}
