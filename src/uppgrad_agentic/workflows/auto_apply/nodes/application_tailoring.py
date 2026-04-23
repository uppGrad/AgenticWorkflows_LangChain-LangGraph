from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.workflows.auto_apply.nodes.eligibility_and_readiness import _get_stub_profile
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)

_MAX_SOURCE_CHARS = 6_000
_MAX_OPP_CHARS = 3_000

# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM = """You are a professional application document writer working in apply-mode.
Your job is to tailor or generate a specific document for a job application.
Changes are applied directly — there is no proposal review step.

Tailoring depth:
  light    — keep ~80% of the content intact; adjust phrasing, highlight relevant experience,
             mirror keywords from the opportunity without fabricating new facts.
  deep     — substantially rewrite to match the opportunity's language and requirements;
             reorder sections for impact, expand relevant achievements, compress less relevant ones.
  generate — create the document from scratch using only facts from the provided source material
             and profile; do not invent credentials, dates, or experiences.

Hard rules (always apply):
- Never fabricate facts, qualifications, roles, or achievements not present in the source.
- Mirror language and keywords from the opportunity description where truthful.
- Return ONLY the document text — no explanations, no markdown fences, no headings like "Here is your CV:".
- Preserve all factual details (dates, employer names, grades) exactly as given."""

_DEPTH_INSTRUCTIONS: Dict[str, str] = {
    "light": (
        "Make light, targeted adjustments only. "
        "Focus on the summary/objective line and skill keywords. "
        "Do not restructure sections."
    ),
    "deep": (
        "Substantially rewrite for this specific opportunity. "
        "Reorder bullets for relevance, expand achievements that align with the role, "
        "compress sections that are less relevant. "
        "Rephrase throughout using the opportunity's language."
    ),
    "generate": (
        "Write this document from scratch using ONLY the facts in the source material provided. "
        "Structure it appropriately for the document type. "
        "Do not invent any details not present in the source."
    ),
    "none": "",  # handled before reaching LLM
}


def _build_prompt(
    doc_type: str,
    tailoring_depth: str,
    source_text: str,
    opportunity_data: Dict[str, Any],
    opportunity_type: str,
    normalized_requirements: List[Dict[str, Any]],
) -> str:
    title = opportunity_data.get("title", "Unknown Role")
    company = (
        opportunity_data.get("company")
        or opportunity_data.get("university")
        or opportunity_data.get("provider_name")
        or "Unknown Organisation"
    )
    description = (
        opportunity_data.get("description")
        or str((opportunity_data.get("data") or {}).get("description", ""))
        or ""
    )
    reqs_text = "\n".join(
        f"- {r.get('document_type', '')} ({r.get('requirement_type', '')})"
        for r in normalized_requirements
        if r.get("requirement_type") == "document"
    )

    depth_instruction = _DEPTH_INSTRUCTIONS.get(tailoring_depth, "")

    return (
        f"Document to produce: {doc_type}\n"
        f"Tailoring depth: {tailoring_depth}\n"
        f"Depth instruction: {depth_instruction}\n\n"
        f"=== OPPORTUNITY ===\n"
        f"Title: {title}\n"
        f"Organisation: {company}\n"
        f"Type: {opportunity_type}\n"
        f"Description:\n{description[:_MAX_OPP_CHARS]}\n\n"
        f"Required documents for this application:\n{reqs_text}\n\n"
        f"=== SOURCE MATERIAL ===\n"
        f"{source_text[:_MAX_SOURCE_CHARS]}\n\n"
        f"Produce the tailored {doc_type} now."
    )


# ---------------------------------------------------------------------------
# LLM tailoring
# ---------------------------------------------------------------------------

def _llm_tailor(
    doc_type: str,
    tailoring_depth: str,
    source_text: str,
    opportunity_data: Dict[str, Any],
    opportunity_type: str,
    normalized_requirements: List[Dict[str, Any]],
    llm,
) -> Optional[str]:
    prompt = _build_prompt(
        doc_type, tailoring_depth, source_text,
        opportunity_data, opportunity_type, normalized_requirements,
    )
    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        text = (response.content or "").strip()
        # Strip any markdown fences the model may have added despite instructions
        text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        return text if text else None
    except Exception as exc:
        logger.warning("application_tailoring: LLM call failed for %s — %s", doc_type, exc)
        return None


# ---------------------------------------------------------------------------
# Heuristic tailoring (no LLM)
# ---------------------------------------------------------------------------

def _extract_keywords(opportunity_data: Dict[str, Any]) -> List[str]:
    """Pull salient keywords from the opportunity record."""
    text = " ".join([
        opportunity_data.get("title", ""),
        opportunity_data.get("description", ""),
        str((opportunity_data.get("data") or {}).get("description", "")),
        str((opportunity_data.get("data") or {}).get("requirements", "")),
    ]).lower()

    # Simple noun/skill extraction — pick capitalised tokens and known tech terms
    tech_terms = re.findall(
        r"\b(python|java|javascript|typescript|go|rust|sql|nosql|docker|kubernetes|"
        r"aws|gcp|azure|machine learning|deep learning|nlp|react|django|fastapi|"
        r"postgresql|redis|microservices|rest api|ci\/cd|agile|scrum|leadership|"
        r"communication|research|analysis|data|backend|frontend|full.?stack)\b",
        text,
    )
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: List[str] = []
    for t in tech_terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:12]


def _heuristic_tailor(
    doc_type: str,
    tailoring_depth: str,
    source_text: str,
    opportunity_data: Dict[str, Any],
    profile: Dict[str, Any],
) -> str:
    title = opportunity_data.get("title", "the role")
    company = (
        opportunity_data.get("company")
        or opportunity_data.get("university")
        or opportunity_data.get("provider_name")
        or "the organisation"
    )
    keywords = _extract_keywords(opportunity_data)
    kw_phrase = ", ".join(keywords[:6]) if keywords else ""

    if tailoring_depth == "generate":
        name = profile.get("name", "Applicant")
        email = profile.get("email", "")
        degree = profile.get("degree_level", "degree")
        disciplines = ", ".join(profile.get("disciplines") or [])

        if doc_type == "Cover Letter":
            return (
                f"{name}\n{email}\n\n"
                f"Dear Hiring Manager,\n\n"
                f"I am writing to apply for the position of {title} at {company}. "
                f"With a {degree} in {disciplines} and hands-on experience in {kw_phrase or 'relevant technologies'}, "
                f"I am confident in my ability to contribute effectively to your team.\n\n"
                f"My background has equipped me with strong skills in {kw_phrase or 'the relevant areas'}. "
                f"I am particularly drawn to this opportunity because of the alignment between my experience "
                f"and the requirements of the role.\n\n"
                f"I would welcome the chance to discuss how my background can benefit {company}. "
                f"Please find my CV attached for your consideration.\n\n"
                f"Yours sincerely,\n{name}"
            )

        if doc_type in ("SOP", "Personal Statement"):
            return (
                f"Statement of Purpose\n\n"
                f"My interest in {title} at {company} stems from a deep commitment to {disciplines}. "
                f"During my {degree} studies, I developed expertise in {kw_phrase or 'core areas of the field'}, "
                f"which I am eager to apply in a graduate context.\n\n"
                f"[Source material summary — real generation requires LLM]\n\n"
                f"{source_text[:1500]}"
            )

        # Generic generate: prepend header to source
        return (
            f"{doc_type} — Generated for {title} at {company}\n"
            f"[Heuristic generation — LLM unavailable]\n\n"
            f"{source_text}"
        )

    if tailoring_depth in ("light", "deep"):
        intro = (
            f"[Tailored for: {title} at {company}"
            + (f" | Key areas: {kw_phrase}" if kw_phrase else "")
            + "]\n\n"
        )
        if tailoring_depth == "deep":
            # Append a targeted skills addendum
            addendum = (
                f"\n\n--- Relevance to {title} ---\n"
                f"This application highlights skills in {kw_phrase or 'the required areas'} "
                f"directly relevant to the requirements at {company}."
            )
            return intro + source_text + addendum
        return intro + source_text

    # tailoring_depth == "none" — pass through unchanged
    return source_text


# ---------------------------------------------------------------------------
# Source document resolver
# ---------------------------------------------------------------------------

def _resolve_source_text(
    doc_type: str,
    source_document: str,
    additional_uploads: Dict[str, str],
    profile: Dict[str, Any],
) -> str:
    """Return the text to tailor from, in priority order:
    1. User upload from this gate session
    2. Stored document text from profile
    3. CV text as fallback base
    4. Empty string (generate will create from profile fields)
    """
    # User uploaded a fresh copy at gate 1
    if additional_uploads.get(doc_type):
        return additional_uploads[doc_type]

    doc_texts = profile.get("document_texts") or {}

    # Exact source document match
    if source_document and doc_texts.get(source_document):
        return doc_texts[source_document]

    # Fall back to CV as a generation base
    if doc_texts.get("CV"):
        return doc_texts["CV"]

    return ""


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def application_tailoring(state: AutoApplyState) -> dict:
    updates = {"current_step": "application_tailoring", "step_history": ["application_tailoring"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    opportunity_type = state.get("opportunity_type", "")
    opportunity_data = state.get("opportunity_data") or {}
    normalized_requirements = state.get("normalized_requirements") or []
    human_review_1 = state.get("human_review_1") or {}
    confirmed_mappings: Dict[str, Any] = human_review_1.get("confirmed_mappings") or {}
    additional_uploads: Dict[str, str] = human_review_1.get("additional_uploads") or {}

    if not confirmed_mappings:
        logger.warning("application_tailoring: no confirmed_mappings in human_review_1 — skipping")
        return {**updates, "tailored_documents": {}}

    profile = _get_stub_profile()
    llm = get_llm()
    tailored: Dict[str, Any] = {}

    for doc_type, mapping in confirmed_mappings.items():
        if mapping.get("skip"):
            logger.info("application_tailoring: skipping %s (user opted out)", doc_type)
            continue

        tailoring_depth: str = mapping.get("tailoring_depth", "light")
        source_document: str = mapping.get("source_document", "")

        # "none" means the user must supply it — no tailoring possible
        if tailoring_depth == "none":
            logger.info(
                "application_tailoring: %s has depth=none — recording as user-supplied", doc_type
            )
            tailored[doc_type] = {
                "content": additional_uploads.get(doc_type, ""),
                "tailoring_depth": "none",
                "source": source_document,
                "llm_used": False,
                "note": f"{doc_type} must be supplied by the user; no generation attempted.",
            }
            continue

        source_text = _resolve_source_text(doc_type, source_document, additional_uploads, profile)

        llm_used = False
        if llm is not None:
            result = _llm_tailor(
                doc_type, tailoring_depth, source_text,
                opportunity_data, opportunity_type, normalized_requirements, llm,
            )
            if result:
                content = result
                llm_used = True
            else:
                content = _heuristic_tailor(
                    doc_type, tailoring_depth, source_text, opportunity_data, profile
                )
        else:
            content = _heuristic_tailor(
                doc_type, tailoring_depth, source_text, opportunity_data, profile
            )

        tailored[doc_type] = {
            "content": content,
            "tailoring_depth": tailoring_depth,
            "source": source_document,
            "llm_used": llm_used,
        }

        logger.info(
            "application_tailoring: %s — depth=%s llm=%s chars=%d",
            doc_type, tailoring_depth, llm_used, len(content),
        )

    return {**updates, "tailored_documents": tailored}
