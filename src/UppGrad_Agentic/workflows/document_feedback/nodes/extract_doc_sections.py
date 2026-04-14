from __future__ import annotations

import re
from typing import Dict, List

from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


# Ordered header lists per doc type. Earlier entries take priority when two
# headers are close together. Matching is case-insensitive on word boundaries.
_CV_HEADERS: List[str] = [
    "summary", "objective", "profile",
    "education", "academic background",
    "experience", "work experience", "professional experience", "employment",
    "skills", "technical skills", "core competencies",
    "projects", "personal projects", "selected projects",
    "certifications", "certificates", "licenses",
    "publications", "research",
    "awards", "honors", "achievements",
    "volunteer", "volunteering",
    "languages",
    "interests", "hobbies",
    "references",
]

_SOP_HEADERS: List[str] = [
    "introduction", "opening",
    "background", "academic background", "professional background",
    "research interests", "research experience", "research",
    "motivation", "why this program", "why this university",
    "goals", "career goals", "academic goals", "future plans",
    "contributions", "fit",
    "conclusion", "closing",
]

_COVER_HEADERS: List[str] = [
    "opening", "introduction",
    "body", "main body", "why i am a strong candidate", "why this role",
    "relevant experience", "skills and experience",
    "closing", "conclusion",
]

_FALLBACK_SECTION = "Body"


def _header_pattern(headers: List[str]) -> re.Pattern:
    escaped = [re.escape(h) for h in headers]
    # Match a header that appears on its own line (possibly followed by a colon),
    # optionally surrounded by whitespace / decorators like ===, ---, ***.
    alternatives = "|".join(escaped)
    return re.compile(
        r"^\s*(?:[-=*#]{0,4}\s*)?(" + alternatives + r")\s*[-=:*#]{0,4}\s*$",
        re.IGNORECASE | re.MULTILINE,
    )


def _split_by_headers(text: str, pattern: re.Pattern) -> Dict[str, str]:
    matches = list(pattern.finditer(text))
    if not matches:
        return {_FALLBACK_SECTION: text.strip()}

    sections: Dict[str, str] = {}

    # Text before the first header goes into a "Preamble" bucket (often empty).
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections["Preamble"] = preamble

    for i, match in enumerate(matches):
        header = match.group(1).strip().title()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        # Merge duplicate headers by appending content.
        if header in sections:
            sections[header] = sections[header] + "\n\n" + content
        else:
            sections[header] = content

    return sections


def extract_doc_sections(state: DocFeedbackState) -> dict:
    if state.get("result", {}).get("status") == "error":
        return {}

    raw_text = state.get("raw_text", "") or ""
    doc_type = (state.get("doc_classification") or {}).get("doc_type", "UNKNOWN")

    if doc_type == "CV":
        pattern = _header_pattern(_CV_HEADERS)
    elif doc_type == "SOP":
        pattern = _header_pattern(_SOP_HEADERS)
    elif doc_type == "COVER_LETTER":
        pattern = _header_pattern(_COVER_HEADERS)
    else:
        # UNKNOWN: try CV headers as a best-effort fallback.
        pattern = _header_pattern(_CV_HEADERS)

    sections = _split_by_headers(raw_text, pattern)
    return {"doc_sections": sections}
