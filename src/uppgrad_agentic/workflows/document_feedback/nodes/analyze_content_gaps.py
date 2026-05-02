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

class CVAntiPattern(BaseModel):
    pattern: str = Field(
        ...,
        description=(
            "Short label for the anti-pattern, one of: "
            "'references_on_request', 'generic_hobbies', 'cv_title', "
            "'first_person_pronouns', 'photo', 'pii_personal_details', "
            "'multiple_objectives', 'other'."
        ),
    )
    excerpt: str = Field(
        ...,
        description="Verbatim excerpt from the document showing the anti-pattern.",
    )
    section: str = Field(default="", description="Section the excerpt was found in.")
    recommendation: str = Field(
        ...,
        description="Concrete action: 'delete', 'remove unless visa-required', 'rewrite without first-person', etc.",
    )


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
    well_constructed_bullets: List[str] = Field(
        default_factory=list,
        description=(
            "CV-only. Bullets that already do the right thing — past-tense action "
            "verb at the start AND a numeric outcome OR a named technology/scope. "
            "Surfaced verbatim so the synthesizer leaves them alone (or only "
            "touches them for ATS-keyword injection / tense fixes)."
        ),
    )
    cv_antipatterns: List[CVAntiPattern] = Field(
        default_factory=list,
        description=(
            "CV-only. Universal CV anti-patterns the synthesizer should propose "
            "removing: 'References available upon request', generic hobbies "
            "lists, 'Curriculum Vitae' as a title, first-person pronouns on "
            "Experience bullets, photos, PII (DOB / marital status / "
            "nationality unless visa-required)."
        ),
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


# CV-specific patterns
_BULLET_LINE_RE = re.compile(r"^\s*[-•*●▪◆–]\s*(.+)", re.MULTILINE)
_STRONG_ACTION_VERBS = (
    r"led|built|designed|launched|shipped|reduced|increased|delivered|"
    r"architected|implemented|developed|migrated|refactored|scaled|optimi[sz]ed|"
    r"owned|drove|grew|cut|eliminated|automated|deployed|spearheaded|"
    r"engineered|published|presented|negotiated|founded|raised|recruited"
)
_BULLET_ACTION_RE = re.compile(rf"^\s*({_STRONG_ACTION_VERBS})\b", re.IGNORECASE)
_NUMERIC_OUTCOME_RE = re.compile(
    # Either a number followed by a unit (with NO trailing \b — `%` is
    # non-word so \b would fail before whitespace), OR a number followed by
    # a unit word with the trailing \b enforcing word-boundary.
    r"\b\d+(?:[.,]\d+)?\s*(?:%|x\b|"
    r"(?:percent|users?|customers?|requests?|hours?|days?|weeks?|months?|years?|"
    r"k|m|b|million|billion|seconds?|ms|fps|qps|rps)\b)",
    re.IGNORECASE,
)
_NAMED_TECH_RE = re.compile(r"(?<=\s)[A-Z][a-zA-Z0-9+#.]{2,}")

_REFERENCES_ON_REQUEST_RE = re.compile(
    r"references?\s+(?:are\s+)?available\s+(?:up)?on\s+request",
    re.IGNORECASE,
)
_HOBBIES_LINE_RE = re.compile(
    r"^\s*(hobbies|interests)\s*[:\-]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_GENERIC_HOBBY_TOKENS = {
    "travelling", "traveling", "reading", "music", "movies", "films",
    "sports", "cooking", "photography", "gaming", "hiking",
}
_CV_TITLE_RE = re.compile(
    r"^\s*(curriculum\s+vitae|c\.?v\.?|resume|résumé)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_FIRST_PERSON_BULLET_RE = re.compile(
    r"^\s*[-•*●▪◆–]?\s*(I\s+(?:am|was|have|had|did|do|led|built|designed|implemented|developed|worked|managed))",
    re.MULTILINE,
)
_PII_DOB_RE = re.compile(
    r"\b(date\s+of\s+birth|d\.?o\.?b\.?|born\s+on)\b",
    re.IGNORECASE,
)
_PII_MARITAL_RE = re.compile(
    r"\b(marital\s+status|single|married|divorced|widowed)\b",
    re.IGNORECASE,
)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def _identify_well_constructed_bullets(doc_sections: dict[str, str]) -> List[str]:
    """CV bullets that already deliver: action verb + (numeric outcome OR named tech)."""
    keep: List[str] = []
    for name, text in doc_sections.items():
        if name == "Preamble":
            continue
        for match in _BULLET_LINE_RE.finditer(text):
            bullet = match.group(1).strip()
            if not _BULLET_ACTION_RE.match(bullet):
                continue
            has_number = bool(_NUMERIC_OUTCOME_RE.search(bullet))
            # Named tech: e.g. "Django", "PostgreSQL", "GraphQL". Strip common
            # sentence-start capitalised words by requiring length ≥3 tokens.
            tech_hits = [m for m in _NAMED_TECH_RE.findall(bullet) if len(m) >= 3]
            if has_number or len(tech_hits) >= 2:
                keep.append(bullet)
    # Cap to avoid overwhelming the synthesis prompt; the strongest signal
    # is "we found some" — the synthesizer already won't recommend changes
    # when the bullet matches its own quality bar.
    return keep[:12]


def _identify_cv_antipatterns(doc_sections: dict[str, str]) -> List[CVAntiPattern]:
    """Detect universal CV anti-patterns. Heuristic only — the LLM path can do better."""
    findings: List[CVAntiPattern] = []
    full_text = "\n".join(doc_sections.values())

    # References on request — universal anti-pattern.
    m = _REFERENCES_ON_REQUEST_RE.search(full_text)
    if m:
        # Surface a slice for context; preserve the matched phrase verbatim.
        findings.append(CVAntiPattern(
            pattern="references_on_request",
            excerpt=m.group(0),
            section="",
            recommendation="Delete this line. Recruiters assume references can be requested.",
        ))

    # Generic hobbies.
    for hm in _HOBBIES_LINE_RE.finditer(full_text):
        items = hm.group(2)
        item_tokens = {tok.strip(", .").lower() for tok in re.split(r"[,/&]", items) if tok.strip()}
        if item_tokens and item_tokens.issubset(_GENERIC_HOBBY_TOKENS | {""}):
            findings.append(CVAntiPattern(
                pattern="generic_hobbies",
                excerpt=hm.group(0).strip(),
                section="",
                recommendation=(
                    "Delete the section, or replace with specific notable interests "
                    "(e.g. an open-source project you contribute to, a competitive "
                    "athletic record). Generic hobbies waste prime real estate."
                ),
            ))

    # "Curriculum Vitae" / "Resume" title.
    if _CV_TITLE_RE.search(full_text):
        findings.append(CVAntiPattern(
            pattern="cv_title",
            excerpt=_CV_TITLE_RE.search(full_text).group(0).strip(),
            section="Header",
            recommendation="Delete the title — the format already signals what this is. Use the space for your name + contact line.",
        ))

    # First-person pronouns on Experience bullets — only flag when count >=2.
    fp_matches = _FIRST_PERSON_BULLET_RE.findall(full_text)
    if len(fp_matches) >= 2:
        findings.append(CVAntiPattern(
            pattern="first_person_pronouns",
            excerpt=fp_matches[0],
            section="Experience",
            recommendation="Rewrite Experience bullets without 'I' — start with the action verb directly ('Led X', not 'I led X').",
        ))

    # PII: DOB / marital status — flag for review (visa contexts may justify).
    if _PII_DOB_RE.search(full_text):
        findings.append(CVAntiPattern(
            pattern="pii_personal_details",
            excerpt=_PII_DOB_RE.search(full_text).group(0),
            section="",
            recommendation="Remove date of birth unless required for visa/work-permit purposes in your target market. In most markets it invites bias.",
        ))
    if _PII_MARITAL_RE.search(full_text):
        # Match has to be on its own line / header context to avoid false
        # positives like "I am single-handed responsible for…".
        for m in _PII_MARITAL_RE.finditer(full_text):
            line_start = full_text.rfind("\n", 0, m.start()) + 1
            line_end = full_text.find("\n", m.end())
            line = full_text[line_start: line_end if line_end != -1 else len(full_text)].strip()
            if len(line) <= 40 and "marital" in line.lower():
                findings.append(CVAntiPattern(
                    pattern="pii_personal_details",
                    excerpt=line,
                    section="",
                    recommendation="Remove marital status — it has no professional relevance and invites bias.",
                ))
                break

    return findings


def _heuristic(
    doc_sections: dict[str, str],
    profile_snapshot: dict,
    doc_type: str = "",
) -> ContentGapsAnalysis:
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

    well_constructed: List[str] = []
    antipatterns: List[CVAntiPattern] = []
    if doc_type == "CV":
        well_constructed = _identify_well_constructed_bullets(doc_sections)
        antipatterns = _identify_cv_antipatterns(doc_sections)
        if antipatterns:
            recommendations.append(
                "Remove CV anti-patterns: "
                + ", ".join(p.pattern for p in antipatterns)
                + "."
            )

    return ContentGapsAnalysis(
        gaps=gaps,
        unexploited_strengths=unexploited,
        weak_claims=weak_claims[:8],  # cap to avoid noise
        well_constructed_bullets=well_constructed,
        cv_antipatterns=antipatterns,
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

CV-only fields (populate ONLY when document type is CV):
- well_constructed_bullets: list bullets from the document that ALREADY do the
  right thing — past-tense action verb at the start AND a numeric outcome OR a
  concrete named technology/scope. Copy the bullet VERBATIM. The synthesizer
  uses this list to know what NOT to propose changes to (a bullet that says
  "Reduced p99 latency by 40% via async batching" is finished work; rewriting
  it to fit a formula would produce something different but no better).
- cv_antipatterns: universal CV anti-patterns the synthesizer should propose
  removing. Use the labels: 'references_on_request' (e.g. "References available
  upon request"); 'generic_hobbies' (a Hobbies/Interests section listing only
  generic items like "travelling, reading, music"); 'cv_title' (literally
  "Curriculum Vitae" or "Resume" as a heading at the top — wastes prime real
  estate); 'first_person_pronouns' (Experience bullets starting with "I led /
  I designed" — CVs should drop the I); 'photo' (a photo on the CV — ATS-
  unfriendly + bias risk in many markets); 'pii_personal_details' (DOB,
  marital status, nationality unless visa-required); 'multiple_objectives';
  'other'. For each entry: copy the offending excerpt VERBATIM and write a
  one-line recommendation. Be conservative — flag clear instances only.

Be specific. Reference actual profile details (skills, roles, companies) when pointing out gaps.
Never invent profile details not present in the "Applicant profile" block.
If a "User focus" block is provided, prioritise gaps and recommendations that serve those goals.
"""

_MAX_DOC_CHARS = 5000


# ---------------------------------------------------------------------------
# Node — receives context_pack via Send
# ---------------------------------------------------------------------------

def analyze_content_gaps(context_pack: dict) -> dict:
    updates = {"step_history": ["analyze_content_gaps"]}
    doc_type = context_pack.get("doc_type", "UNKNOWN")
    doc_sections = context_pack.get("doc_sections") or {}
    profile_snapshot = context_pack.get("profile_snapshot") or {}

    llm = get_llm()
    if llm is None:
        result = _heuristic(doc_sections, profile_snapshot, doc_type=doc_type)
        return {**updates, "analysis_results": {"content_gaps": result.model_dump()}}

    doc_text = " ".join(doc_sections.values())[:_MAX_DOC_CHARS]
    user_focus = format_user_focus(context_pack.get("parsed_instructions"))
    profile_brief = format_profile_brief(profile_snapshot)
    # Fall back to a raw dump if formatting yielded nothing — content_gaps
    # depends on profile data being visible to the LLM.
    if not profile_brief and profile_snapshot:
        profile_brief = "Applicant profile:\n" + str(profile_snapshot)[:1500]

    body = f"Document type: {doc_type}\n\n"
    if profile_brief:
        body += f"{profile_brief}\n\n"
    body += f"Document text (truncated):\n{doc_text}"
    if user_focus:
        body += f"\n\n{user_focus}"

    structured = llm.with_structured_output(ContentGapsAnalysis)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=body),
    ]

    try:
        result: ContentGapsAnalysis = structured.invoke(msgs)
        return {**updates, "analysis_results": {"content_gaps": result.model_dump()}}
    except Exception as e:
        result = _heuristic(doc_sections, profile_snapshot, doc_type=doc_type)
        out = result.model_dump()
        out["recommendations"] = out.get("recommendations", []) + [f"[LLM failed, used heuristic: {e}]"]
        return {**updates, "analysis_results": {"content_gaps": out}}
