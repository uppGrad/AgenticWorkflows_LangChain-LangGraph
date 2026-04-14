from __future__ import annotations

import re
from typing import List, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output
# ---------------------------------------------------------------------------

class ParsedInstructions(BaseModel):
    intent: str = Field(
        ...,
        description=(
            "Short phrase summarising the user's primary goal "
            "(e.g. 'improve clarity', 'tailor for software-engineering role', 'strengthen SOP narrative')."
        ),
    )
    tone_preferences: List[str] = Field(
        default_factory=list,
        description="Tone or style preferences explicitly or implicitly stated (e.g. 'formal', 'concise', 'technical').",
    )
    target_role: Optional[str] = Field(
        default=None,
        description="Job title or role the user is targeting, if mentioned.",
    )
    target_program: Optional[str] = Field(
        default=None,
        description="Graduate program or institution the user is targeting, if mentioned.",
    )
    explicit_constraints: List[str] = Field(
        default_factory=list,
        description=(
            "Specific constraints the user stated, such as 'keep it to one page', "
            "'do not change the opening paragraph', 'use British English'."
        ),
    )


SYSTEM = """You parse a user's free-text instructions about their application document.

Extract:
- intent: the user's primary goal in one short phrase
- tone_preferences: any tone or style requirements (can be empty list)
- target_role: specific job title or role mentioned, or null
- target_program: graduate program / institution mentioned, or null
- explicit_constraints: hard rules the user stated (can be empty list)

Be conservative. Only populate fields from information actually present in the instructions.
"""


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

_TONE_KEYWORDS = {
    "formal": ["formal", "professional", "polished"],
    "concise": ["concise", "brief", "short", "succinct"],
    "technical": ["technical", "detailed", "in-depth"],
    "casual": ["casual", "friendly", "conversational"],
    "academic": ["academic", "scholarly"],
}

_ROLE_PATTERN = re.compile(
    r"\b(?:applying\s+(?:for|to)|role\s+(?:of|as)|position\s+(?:of|as)|job\s+(?:as|title)|for\s+(?:a\s+)?(?:the\s+)?)([A-Za-z][A-Za-z\s\-/]+?)(?:\s+(?:role|position|job|at|in|with)|[,.]|$)",
    re.IGNORECASE,
)

_PROGRAM_PATTERN = re.compile(
    r"\b(?:program|course|msc|ms\b|phd|mba|master|bachelor|applying\s+to)\s+(?:in\s+|of\s+|at\s+)?([A-Za-z][A-Za-z\s\-]+?)(?:\s+at\s+([A-Za-z][A-Za-z\s]+?))?(?:[,.]|$)",
    re.IGNORECASE,
)

_CONSTRAINT_PATTERNS = [
    re.compile(r"(?:keep|limit|one\s+page|single\s+page|no\s+more\s+than\s+\d+\s+pages?)", re.IGNORECASE),
    re.compile(r"do\s+not\s+(?:change|alter|remove|modify)[^.]+", re.IGNORECASE),
    re.compile(r"use\s+(?:British|American|US|UK)\s+English", re.IGNORECASE),
    re.compile(r"avoid\s+[^.]+", re.IGNORECASE),
    re.compile(r"must\s+(?:include|contain|have)[^.]+", re.IGNORECASE),
]


def _heuristic_parse(instructions: str) -> ParsedInstructions:
    text = instructions.strip()
    lower = text.lower()

    # Intent: try to summarise from key verbs
    if any(kw in lower for kw in ["improve", "enhance", "strengthen"]):
        intent = "improve document quality"
    elif any(kw in lower for kw in ["tailor", "customise", "customize", "target"]):
        intent = "tailor document for a specific opportunity"
    elif any(kw in lower for kw in ["shorten", "brief", "concise", "cut"]):
        intent = "make document more concise"
    elif any(kw in lower for kw in ["expand", "elaborate", "detail"]):
        intent = "expand document content"
    elif any(kw in lower for kw in ["grammar", "spelling", "proofread"]):
        intent = "fix grammar and spelling"
    else:
        intent = "general document feedback"

    # Tone
    tone_preferences: List[str] = []
    for label, keywords in _TONE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            tone_preferences.append(label)

    # Target role
    target_role: Optional[str] = None
    role_match = _ROLE_PATTERN.search(text)
    if role_match:
        target_role = role_match.group(1).strip().title()

    # Target program
    target_program: Optional[str] = None
    prog_match = _PROGRAM_PATTERN.search(text)
    if prog_match:
        parts = [p for p in prog_match.groups() if p]
        target_program = " at ".join(p.strip().title() for p in parts)

    # Explicit constraints: collect short matched snippets
    constraints: List[str] = []
    for pattern in _CONSTRAINT_PATTERNS:
        for m in pattern.finditer(text):
            snippet = m.group(0).strip().rstrip(",.")
            if snippet and snippet not in constraints:
                constraints.append(snippet)

    return ParsedInstructions(
        intent=intent,
        tone_preferences=tone_preferences,
        target_role=target_role,
        target_program=target_program,
        explicit_constraints=constraints,
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def parse_user_instructions(state: DocFeedbackState) -> dict:
    if state.get("result", {}).get("status") == "error":
        return {}

    instructions = (state.get("user_instructions") or "").strip()

    # No instructions provided: return an empty-but-valid structure.
    if not instructions:
        return {
            "parsed_instructions": ParsedInstructions(
                intent="general document feedback",
                tone_preferences=[],
                target_role=None,
                target_program=None,
                explicit_constraints=[],
            ).model_dump()
        }

    llm = get_llm()
    if llm is None:
        parsed = _heuristic_parse(instructions)
        return {"parsed_instructions": parsed.model_dump()}

    structured = llm.with_structured_output(ParsedInstructions)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"User instructions:\n{instructions}"),
    ]

    try:
        parsed: ParsedInstructions = structured.invoke(msgs)
        return {"parsed_instructions": parsed.model_dump()}
    except Exception as e:
        parsed = _heuristic_parse(instructions)
        out = parsed.model_dump()
        out.setdefault("explicit_constraints", [])
        out["explicit_constraints"].append(f"[parse fallback: LLM failed — {e}]")
        return {"parsed_instructions": out}
