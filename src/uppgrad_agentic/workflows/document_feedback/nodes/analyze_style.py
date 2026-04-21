from __future__ import annotations

import re
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from uppgrad_agentic.common.llm import get_llm


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class StyleAnalysis(BaseModel):
    tone: str = Field(..., description="Detected overall tone (e.g. 'formal', 'informal', 'inconsistent').")
    clarity_score: float = Field(..., ge=0.0, le=1.0, description="0 = very unclear, 1 = very clear.")
    issues: List[str] = Field(default_factory=list, description="Specific style or grammar problems found.")
    passive_voice_instances: List[str] = Field(
        default_factory=list,
        description="Example phrases using passive voice that could be rewritten actively.",
    )
    suggestions: List[str] = Field(default_factory=list, description="Actionable improvement suggestions.")


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

_PASSIVE_RE = re.compile(
    r"\b(was|were|been|being|is|are)\s+\w+ed\b", re.IGNORECASE
)
_FILLER_WORDS = [
    "very", "really", "quite", "basically", "actually", "literally",
    "just", "simply", "clearly", "obviously",
]
_INFORMAL_SIGNALS = ["gonna", "wanna", "kind of", "sort of", "you know", "stuff", "things"]
_FORMAL_SIGNALS = ["therefore", "furthermore", "consequently", "hereby", "respective"]
_FIRST_PERSON_RE = re.compile(r"\bI\b")

# Long sentence threshold (words)
_LONG_SENTENCE_WORDS = 35


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _heuristic(doc_type: str, doc_sections: dict[str, str]) -> StyleAnalysis:
    full_text = " ".join(doc_sections.values())
    lower = full_text.lower()
    sentences = _sentences(full_text)

    issues: List[str] = []
    suggestions: List[str] = []

    # Passive voice
    passive_matches = _PASSIVE_RE.findall(full_text)
    passive_instances = [
        m.group(0) for m in _PASSIVE_RE.finditer(full_text)
    ][:5]  # cap at 5 examples

    if len(passive_matches) > 5:
        issues.append(f"High passive voice usage ({len(passive_matches)} instances detected).")
        suggestions.append("Rewrite passive constructions with active verbs to sound more assertive.")

    # Long sentences
    long_sentences = [s for s in sentences if len(s.split()) > _LONG_SENTENCE_WORDS]
    if len(long_sentences) > 3:
        issues.append(f"{len(long_sentences)} sentences exceed {_LONG_SENTENCE_WORDS} words — hard to scan.")
        suggestions.append("Break long sentences into shorter, punchy statements.")

    # Filler words
    filler_hits = [w for w in _FILLER_WORDS if re.search(rf"\b{w}\b", lower)]
    if filler_hits:
        issues.append(f"Filler words detected: {', '.join(filler_hits)}.")
        suggestions.append("Remove filler words to tighten the prose.")

    # Tone
    informal_hits = sum(1 for s in _INFORMAL_SIGNALS if s in lower)
    formal_hits = sum(1 for s in _FORMAL_SIGNALS if s in lower)
    first_person_count = len(_FIRST_PERSON_RE.findall(full_text))

    if informal_hits >= 2:
        tone = "informal"
        issues.append("Informal language detected — not appropriate for application documents.")
    elif formal_hits >= 2 or (doc_type == "SOP" and first_person_count > 0):
        tone = "formal"
    elif doc_type == "CV" and first_person_count > 5:
        tone = "informal"
        issues.append("CV uses first-person ('I') frequently; consider removing personal pronouns.")
    else:
        tone = "neutral"

    # Clarity score: penalise for each issue category
    penalty = 0.1 * len(issues)
    clarity_score = round(max(0.3, 1.0 - penalty), 2)

    return StyleAnalysis(
        tone=tone,
        clarity_score=clarity_score,
        issues=issues,
        passive_voice_instances=passive_instances,
        suggestions=suggestions,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM = """You are evaluating the writing style of an application document.

Assess:
- tone: overall tone detected (formal / informal / inconsistent / neutral)
- clarity_score: 0.0–1.0 (1 = very clear and well-written)
- issues: specific style or grammar problems (passive overuse, filler words, inconsistent tense, etc.)
- passive_voice_instances: up to 5 example phrases using passive voice
- suggestions: actionable rewrites or improvements

Focus on patterns, not one-off typos. Be specific about section or phrase location where helpful.
"""

_MAX_CHARS = 6000


# ---------------------------------------------------------------------------
# Node — receives context_pack via Send
# ---------------------------------------------------------------------------

def analyze_style(context_pack: dict) -> dict:
    doc_type = context_pack.get("doc_type", "UNKNOWN")
    doc_sections = context_pack.get("doc_sections") or {}
    full_text = " ".join(doc_sections.values())[:_MAX_CHARS]

    llm = get_llm()
    if llm is None:
        result = _heuristic(doc_type, doc_sections)
        return {"analysis_results": {"style": result.model_dump()}}

    structured = llm.with_structured_output(StyleAnalysis)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content=(
                f"Document type: {doc_type}\n\n"
                f"Document text (truncated to {_MAX_CHARS} chars):\n{full_text}"
            )
        ),
    ]

    try:
        result: StyleAnalysis = structured.invoke(msgs)
        return {"analysis_results": {"style": result.model_dump()}}
    except Exception as e:
        result = _heuristic(doc_type, doc_sections)
        out = result.model_dump()
        out["issues"] = out.get("issues", []) + [f"[LLM failed, used heuristic: {e}]"]
        return {"analysis_results": {"style": out}}
