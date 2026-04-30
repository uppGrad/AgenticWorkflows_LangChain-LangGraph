"""Post-T1 light analysis of an uploaded document.

Called by `application_tailoring` (Step 6) between T1 and T2 in the upload
path:
  PreA → T1 → LA → T2
              ^^

The output flags remaining gaps so T2 can polish without re-doing T1's
work. LLM-only — emits an empty/safe analysis when no LLM is configured or
the call fails.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.workflows.auto_apply.schemas import UploadedDocLightPostAnalysis

logger = logging.getLogger(__name__)

_MAX_DOC_CHARS = 6_000
_MAX_OPP_CHARS = 3_000

_SYSTEM = """You are reviewing a tailored document AFTER a first tailoring
pass and BEFORE a final polishing pass. Identify only the gaps that remain
so the polishing pass can fix them.

Return JSON only — do not include any commentary.

Fields (each list capped at 3 items, empty when nothing applies):
- structure_issues: ordering / formatting / structural problems still
  present (e.g. "skills section comes before experience").
- content_gap_vs_opportunity: missing elements the opportunity description
  explicitly asks for that the document does not yet address.
- content_gap_vs_profile: strengths in the user's profile that the T1 pass
  failed to surface (publications, projects, achievements that should be
  there but are missing).
"""


def _safe_default() -> UploadedDocLightPostAnalysis:
    return UploadedDocLightPostAnalysis(
        structure_issues=[],
        content_gap_vs_opportunity=[],
        content_gap_vs_profile=[],
    )


def _summarise_profile(profile: Dict[str, Any]) -> str:
    parts: List[str] = []
    if profile.get("name"):
        parts.append(f"Name: {profile['name']}")
    if profile.get("location"):
        parts.append(f"Location: {profile['location']}")
    if profile.get("degree_level"):
        parts.append(f"Highest degree: {profile['degree_level']}")
    if profile.get("disciplines"):
        parts.append(f"Disciplines: {', '.join(profile['disciplines'])}")
    if profile.get("bio"):
        parts.append(f"Bio: {profile['bio']}")
    if profile.get("projects"):
        parts.append(f"Projects: {profile['projects']}")
    if profile.get("publications"):
        parts.append(f"Publications: {profile['publications']}")
    if profile.get("achievements"):
        parts.append(f"Achievements: {profile['achievements']}")
    return "\n".join(parts) if parts else "(no profile details available)"


def _build_prompt(
    opportunity_data: Dict[str, Any],
    profile: Dict[str, Any],
    t1_output: str,
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

    return (
        f"Document type: {doc_type}\n\n"
        f"=== OPPORTUNITY ===\n"
        f"Title: {title}\n"
        f"Organisation: {org}\n"
        f"Description:\n{description[:_MAX_OPP_CHARS]}\n\n"
        f"=== USER PROFILE (summary) ===\n{_summarise_profile(profile)}\n\n"
        f"=== USER GUIDANCE FOR THIS DOCUMENT ===\n{user_prompt or '(none)'}\n\n"
        f"=== T1 OUTPUT ({doc_type}) ===\n{t1_output[:_MAX_DOC_CHARS]}\n\n"
        "Produce the structured post-T1 analysis now."
    )


def _cap(items: List[str], n: int = 3) -> List[str]:
    return items[:n] if len(items) > n else items


def analyze_upload_light_post(
    opportunity_data: Dict[str, Any],
    profile: Dict[str, Any],
    t1_output: str,
    doc_type: str,
    user_prompt: Optional[str] = None,
) -> UploadedDocLightPostAnalysis:
    """Run the LLM post-T1 analysis. Returns a safe default on any failure."""
    llm = get_llm()
    if llm is None:
        logger.warning("upload_light_post_analysis: no LLM configured — returning safe default")
        return _safe_default()

    prompt = _build_prompt(opportunity_data, profile, t1_output, doc_type, user_prompt)
    structured = llm.with_structured_output(UploadedDocLightPostAnalysis)
    try:
        result: UploadedDocLightPostAnalysis = structured.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return result.model_copy(
            update={
                "structure_issues": _cap(result.structure_issues),
                "content_gap_vs_opportunity": _cap(result.content_gap_vs_opportunity),
                "content_gap_vs_profile": _cap(result.content_gap_vs_profile),
            }
        )
    except Exception as exc:
        logger.warning("upload_light_post_analysis: LLM call failed for %s — %s", doc_type, exc)
        return _safe_default()
