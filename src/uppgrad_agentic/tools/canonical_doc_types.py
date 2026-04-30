"""Heuristic classification of form-field labels to canonical document types.

Used by `determine_requirements` (jobs path) to tag each `field_type='file'`
FormField with a `canonical_document_type`. The result drives gate-1
auto-generate-button visibility (USER_SUPPLIED types can only be uploaded;
GENERATABLE types can be auto-generated) and gate-1 deduplication.

The set of canonical types must match the union of `_GENERATABLE` and
`_USER_SUPPLIED` defined in
`uppgrad_agentic.workflows.auto_apply.nodes.asset_mapping` (kept here as
literals to avoid importing the heavy node module).
"""
from __future__ import annotations

from typing import List, Optional


# Canonical doc types — keep in lockstep with asset_mapping._GENERATABLE /
# asset_mapping._USER_SUPPLIED.
GENERATABLE: List[str] = [
    "CV",
    "Cover Letter",
    "Motivation Letter",
    "SOP",
    "Personal Statement",
    "Research Proposal",
    "Writing Sample",
    "References",
]
USER_SUPPLIED: List[str] = [
    "Transcript",
    "English Proficiency Test",
    "Portfolio",
    "Certificate",
    "Passport",
    "Birth Certificate",
]
ALL_TYPES: List[str] = GENERATABLE + USER_SUPPLIED


# Heuristic keyword table. Order matters — earlier entries win when a label
# matches multiple keywords (e.g. "cover letter" matches before "letter").
# Keys are lowercase substrings; values are the canonical doc type string.
_KEYWORD_TABLE: List[tuple[str, str]] = [
    # Cover Letter / Motivation Letter (more specific than "letter")
    ("cover letter", "Cover Letter"),
    ("motivation letter", "Motivation Letter"),
    ("letter of motivation", "Motivation Letter"),
    # CV / Resume
    ("curriculum vitae", "CV"),
    ("resume", "CV"),
    ("résumé", "CV"),
    (" cv ", "CV"),
    ("cv/", "CV"),
    ("/cv", "CV"),
    # Statement of Purpose / Personal Statement
    ("statement of purpose", "SOP"),
    ("sop ", "SOP"),
    (" sop", "SOP"),
    ("personal statement", "Personal Statement"),
    # Research Proposal
    ("research proposal", "Research Proposal"),
    # Writing Sample
    ("writing sample", "Writing Sample"),
    ("work sample", "Writing Sample"),
    # References / Recommendations
    ("reference letter", "References"),
    ("letter of reference", "References"),
    ("recommendation letter", "References"),
    ("letter of recommendation", "References"),
    ("references", "References"),
    # Transcripts
    ("transcript", "Transcript"),
    ("academic record", "Transcript"),
    ("grade report", "Transcript"),
    # English proficiency
    ("english proficiency", "English Proficiency Test"),
    ("toefl", "English Proficiency Test"),
    ("ielts", "English Proficiency Test"),
    ("duolingo english", "English Proficiency Test"),
    # Portfolio
    ("portfolio", "Portfolio"),
    # Certificates / passports / birth certificates
    ("birth certificate", "Birth Certificate"),
    ("passport", "Passport"),
    ("certificate", "Certificate"),
]


def classify_label(label: str) -> Optional[str]:
    """Return a canonical doc type for the given file-field label, or None
    when no heuristic matches. The caller is expected to fall back to an
    LLM classifier for unmatched labels.

    Matching is case-insensitive and uses substring containment with a
    small amount of padding to avoid spurious matches on " cv " inside
    longer words.
    """
    if not label:
        return None
    padded = f" {label.lower().strip()} "
    for keyword, canonical in _KEYWORD_TABLE:
        if keyword in padded:
            return canonical
    return None
