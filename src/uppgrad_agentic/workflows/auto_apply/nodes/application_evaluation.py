from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)

MAX_EVAL_ITERATIONS = 2
MIN_DOC_CHARS = 200

# ---------------------------------------------------------------------------
# Placeholder detection
# Catches unfilled template slots but intentionally ignores the heuristic
# metadata headers like "[Tailored for: ...]" that the tailoring node writes.
# ---------------------------------------------------------------------------

_PLACEHOLDER_PATTERNS = [
    # [TODO], [INSERT NAME], [YOUR COMPANY], [DATE], [TBD] — square-bracket slots
    re.compile(
        r"\[(?:TODO|INSERT|YOUR\s|COMPANY|DATE\b|ROLE\b|POSITION|PLACEHOLDER|TBD\b|NAME\b)[^\]]{0,60}\]",
        re.IGNORECASE,
    ),
    # {name}, {company_name}, {role} — curly-brace template variables
    re.compile(r"\{[a-z_]{2,30}\}", re.IGNORECASE),
    # <NAME>, <COMPANY NAME> — angle-bracket slots
    re.compile(r"<(?:your|name|company|position|role|insert)[^>]{0,40}>", re.IGNORECASE),
    # Explicit stubs left by heuristic generate path for SOP
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


def _check_keyword_coverage(text: str, keywords: List[str], min_hits: int = 1) -> List[str]:
    """Return keywords from the required list that are absent from the document."""
    text_lower = text.lower()
    return [kw for kw in keywords if kw not in text_lower]


# ---------------------------------------------------------------------------
# Per-document checks
# ---------------------------------------------------------------------------

def _evaluate_document(
    doc_type: str,
    info: Dict[str, Any],
    opportunity_keywords: List[str],
) -> List[str]:
    issues: List[str] = []
    content: str = info.get("content") or ""
    tailoring_depth: str = info.get("tailoring_depth", "light")

    # Skip user-supplied documents (depth=none) — we can't evaluate content we didn't write
    if tailoring_depth == "none":
        return []

    # 1. Minimum length
    if len(content) < MIN_DOC_CHARS:
        issues.append(
            f"{doc_type}: content is too short ({len(content)} chars, minimum {MIN_DOC_CHARS})."
        )

    # 2. Placeholder text
    placeholders = _find_placeholders(content)
    if placeholders:
        sample = ", ".join(f'"{p}"' for p in placeholders[:3])
        issues.append(
            f"{doc_type}: contains unfilled placeholder text: {sample}."
        )

    # 3. Keyword coverage — require at least 1 keyword from the opportunity
    if opportunity_keywords and len(content) >= MIN_DOC_CHARS:
        missing_kws = _check_keyword_coverage(content, opportunity_keywords, min_hits=1)
        if len(missing_kws) == len(opportunity_keywords):
            # None of the extracted keywords appear — flag as low relevance
            issues.append(
                f"{doc_type}: no opportunity-specific keywords detected "
                f"(expected at least one of: {', '.join(opportunity_keywords[:5])})."
            )

    return issues


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def application_evaluation(state: AutoApplyState) -> dict:
    if state.get("result", {}).get("status") == "error":
        return {}

    tailored_documents: Dict[str, Any] = state.get("tailored_documents") or {}
    opportunity_data: Dict[str, Any] = state.get("opportunity_data") or {}
    current_iteration: int = state.get("iteration_count") or 0

    new_iteration = current_iteration + 1
    opportunity_keywords = _extract_opportunity_keywords(opportunity_data)

    all_issues: List[str] = []

    if not tailored_documents:
        all_issues.append("No tailored documents were produced.")
    else:
        for doc_type, info in tailored_documents.items():
            doc_issues = _evaluate_document(doc_type, info, opportunity_keywords)
            all_issues.extend(doc_issues)

    passed = len(all_issues) == 0

    logger.info(
        "application_evaluation: iteration=%d passed=%s issues=%d",
        new_iteration, passed, len(all_issues),
    )
    for issue in all_issues:
        logger.info("  issue: %s", issue)

    return {
        "evaluation_result": {
            "passed": passed,
            "issues": all_issues,
            "iteration": new_iteration,
        },
        "iteration_count": new_iteration,
    }
