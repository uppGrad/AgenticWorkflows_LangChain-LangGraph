"""Pre-tailoring analysis of a user-uploaded document.

Called by `application_tailoring` (Step 6) at the start of the upload path:
  PreA  → T1 → LA → T2
  ^^^^

LLM-only — emits an empty/safe analysis when no LLM is configured or the
call fails. The caller (`application_tailoring`) feeds the analysis into the
T1 prompt so the model knows where to focus the first tailoring pass.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.workflows.auto_apply.schemas import UploadedDocPreAnalysis

logger = logging.getLogger(__name__)

_MAX_DOC_CHARS = 6_000
_MAX_OPP_CHARS = 3_000

_SYSTEM = """You are reviewing a user-uploaded application document before
it is tailored for a specific opportunity. Produce a short, structured
analysis the tailoring step will use as a focus list.

Return JSON only — do not include any commentary.

Fields:
- completeness: 1-3 sentences naming what core sections / signals are present
  vs missing for the document type and opportunity.
- relevance: 1-3 sentences on how well the document matches the opportunity's
  requirements and language.
- correctness: 1-3 sentences on factual / structural / formatting issues, if
  any. Say "no issues identified" if the document is clean.
- overall_quality: one of "needs_major_work" | "needs_revision" |
  "ready_for_polish".
- top_priorities: up to 3 prioritised changes the tailoring pass should
  address. Be concrete (e.g. "lead with the systems-design role",
  "tighten the summary to 2 lines"). Empty list if the document is already
  in good shape.
"""


def _safe_default() -> UploadedDocPreAnalysis:
    return UploadedDocPreAnalysis(
        completeness="No analysis produced — LLM unavailable.",
        relevance="No analysis produced — LLM unavailable.",
        correctness="No analysis produced — LLM unavailable.",
        overall_quality="needs_revision",
        top_priorities=[],
    )


def _build_prompt(
    opportunity_data: Dict[str, Any],
    profile: Dict[str, Any],
    uploaded_text: str,
    doc_type: str,
    user_prompt: Optional[str],
) -> str:
    title = opportunity_data.get("title", "Unknown role")
    org = (
        opportunity_data.get("company")
        or opportunity_data.get("university")
        or opportunity_data.get("provider_name")
        or "Unknown organisation"
    )
    description = (
        opportunity_data.get("description")
        or str((opportunity_data.get("data") or {}).get("description", ""))
        or ""
    )
    profile_lines = [
        f"Name: {profile.get('name', '')}",
        f"Location: {profile.get('location', '')}",
        f"Highest degree: {profile.get('degree_level', '')}",
        f"Disciplines: {', '.join(profile.get('disciplines') or [])}",
    ]
    if profile.get("bio"):
        profile_lines.append(f"Bio: {profile['bio']}")

    return (
        f"Document type: {doc_type}\n\n"
        f"=== OPPORTUNITY ===\n"
        f"Title: {title}\n"
        f"Organisation: {org}\n"
        f"Description:\n{description[:_MAX_OPP_CHARS]}\n\n"
        f"=== USER PROFILE (summary) ===\n"
        + "\n".join(profile_lines)
        + "\n\n"
        f"=== USER GUIDANCE FOR THIS DOCUMENT ===\n"
        f"{user_prompt or '(none)'}\n\n"
        f"=== UPLOADED {doc_type.upper()} ===\n"
        f"{uploaded_text[:_MAX_DOC_CHARS]}\n\n"
        "Produce the structured analysis now."
    )


def analyze_upload_pre(
    opportunity_data: Dict[str, Any],
    profile: Dict[str, Any],
    uploaded_text: str,
    doc_type: str,
    user_prompt: Optional[str] = None,
) -> UploadedDocPreAnalysis:
    """Run the LLM pre-analysis. Returns a safe default on any failure."""
    llm = get_llm()
    if llm is None:
        logger.warning("upload_pre_analysis: no LLM configured — returning safe default")
        return _safe_default()

    prompt = _build_prompt(opportunity_data, profile, uploaded_text, doc_type, user_prompt)
    structured = llm.with_structured_output(UploadedDocPreAnalysis)
    try:
        result: UploadedDocPreAnalysis = structured.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        # Cap top_priorities at 3 in case the model exceeds the soft cap
        if len(result.top_priorities) > 3:
            result = result.model_copy(update={"top_priorities": result.top_priorities[:3]})
        return result
    except Exception as exc:
        logger.warning("upload_pre_analysis: LLM call failed for %s — %s", doc_type, exc)
        return _safe_default()
