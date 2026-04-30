"""Application evaluation (Step 6 rewrite — informational only).

The previous retry loop is gone. This node now produces a list of warnings
that gate 2 surfaces alongside the tailored materials. Checks span both
`tailored_documents` and the new `tailored_answers`:
  - minimum length
  - placeholder text
  - keyword coverage against the opportunity
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)

MIN_DOC_CHARS = 200
MIN_ANSWER_CHARS = 60

# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------

_PLACEHOLDER_PATTERNS = [
    re.compile(
        r"\[(?:TODO|INSERT|YOUR\s|COMPANY|DATE\b|ROLE\b|POSITION|PLACEHOLDER|TBD\b|NAME\b)[^\]]{0,60}\]",
        re.IGNORECASE,
    ),
    re.compile(r"\{[a-z_]{2,30}\}", re.IGNORECASE),
    re.compile(r"<(?:your|name|company|position|role|insert)[^>]{0,40}>", re.IGNORECASE),
    re.compile(r"\[Source material summary.*?requires LLM\]", re.IGNORECASE),
]


def _find_placeholders(text: str) -> List[str]:
    found: List[str] = []
    for pattern in _PLACEHOLDER_PATTERNS:
        for match in pattern.finditer(text):
            found.append(match.group(0)[:80])
    return found


# ---------------------------------------------------------------------------
# Keyword coverage
# ---------------------------------------------------------------------------

def _extract_opportunity_keywords(opportunity_data: Dict[str, Any]) -> List[str]:
    blob = " ".join([
        opportunity_data.get("title", ""),
        opportunity_data.get("description", ""),
        str((opportunity_data.get("data") or {}).get("description", "")),
        str((opportunity_data.get("data") or {}).get("requirements", "")),
    ]).lower()

    tech = re.findall(
        r"\b(python|java|javascript|typescript|go|rust|sql|nosql|docker|kubernetes|"
        r"aws|gcp|azure|machine learning|deep learning|nlp|react|django|fastapi|"
        r"postgresql|redis|microservices|rest api|ci\/cd|agile|scrum|"
        r"research|analysis|data|backend|frontend)\b",
        blob,
    )
    seen: set[str] = set()
    unique: List[str] = []
    for t in tech:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:10]


# ---------------------------------------------------------------------------
# Per-output checks
# ---------------------------------------------------------------------------

def _check_document(
    doc_type: str,
    info: Dict[str, Any],
    opportunity_keywords: List[str],
) -> List[str]:
    warnings: List[str] = []
    content: str = info.get("content") or ""

    if len(content) < MIN_DOC_CHARS:
        warnings.append(
            f"{doc_type}: content is short ({len(content)} chars, expected ≥{MIN_DOC_CHARS})."
        )

    placeholders = _find_placeholders(content)
    if placeholders:
        sample = ", ".join(f'"{p}"' for p in placeholders[:3])
        warnings.append(f"{doc_type}: contains unfilled placeholder text: {sample}.")

    if opportunity_keywords and len(content) >= MIN_DOC_CHARS:
        text_lower = content.lower()
        missing = [kw for kw in opportunity_keywords if kw not in text_lower]
        if len(missing) == len(opportunity_keywords):
            warnings.append(
                f"{doc_type}: no opportunity-specific keywords detected "
                f"(expected at least one of: {', '.join(opportunity_keywords[:5])})."
            )
    return warnings


def _check_answer(
    key: str,
    info: Dict[str, Any],
) -> List[str]:
    warnings: List[str] = []
    content: str = info.get("content") or ""
    question = info.get("question") or key

    if len(content) < MIN_ANSWER_CHARS:
        warnings.append(
            f"Answer to '{question}': content is short ({len(content)} chars, expected ≥{MIN_ANSWER_CHARS})."
        )
    placeholders = _find_placeholders(content)
    if placeholders:
        sample = ", ".join(f'"{p}"' for p in placeholders[:3])
        warnings.append(f"Answer to '{question}': contains placeholder text: {sample}.")
    return warnings


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def application_evaluation(state: AutoApplyState) -> dict:
    updates = {"current_step": "application_evaluation", "step_history": ["application_evaluation"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    tailored_documents: Dict[str, Any] = state.get("tailored_documents") or {}
    tailored_answers: Dict[str, Any] = state.get("tailored_answers") or {}
    opportunity_data: Dict[str, Any] = state.get("opportunity_data") or {}

    opportunity_keywords = _extract_opportunity_keywords(opportunity_data)
    warnings: List[str] = []

    if not tailored_documents and not tailored_answers:
        warnings.append("No tailored documents or answers were produced.")
    else:
        for doc_type, info in tailored_documents.items():
            warnings.extend(_check_document(doc_type, info, opportunity_keywords))
        for key, info in tailored_answers.items():
            warnings.extend(_check_answer(key, info))

    logger.info(
        "application_evaluation: %d warnings across %d docs and %d answers",
        len(warnings), len(tailored_documents), len(tailored_answers),
    )

    return {
        **updates,
        "evaluation_result": {
            "warnings": warnings,
        },
    }
