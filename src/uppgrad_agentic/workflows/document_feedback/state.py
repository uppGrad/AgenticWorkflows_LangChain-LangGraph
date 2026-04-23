from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


DocType = Literal["CV", "SOP", "COVER_LETTER", "UNKNOWN"]


def _merge_analysis(
    left: Dict[str, Any] | None,
    right: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Reducer for analysis_results: merges parallel analysis outputs by key."""
    return {**(left or {}), **(right or {})}


class FileInput(TypedDict, total=False):
    name: str
    mime: str
    path: str
    bytes: bytes


class DocMeta(TypedDict, total=False):
    file_name: str
    mime: str
    char_count: int
    page_count: Optional[int]
    extraction_warnings: List[str]


class DocClassification(TypedDict, total=False):
    doc_type: DocType
    relevant: bool
    confidence: float
    reasons: List[str]
    language: Optional[str]


class WorkflowResult(TypedDict, total=False):
    status: Literal["ok", "error"]
    error_code: Optional[str]
    user_message: Optional[str]
    details: Optional[Dict[str, Any]]


class DocFeedbackState(TypedDict, total=False):
    # inputs
    file: FileInput
    user_instructions: str

    # derived
    raw_text: str
    doc_meta: DocMeta
    doc_classification: DocClassification

    # phase 1: context assembly
    profile_snapshot: Dict[str, Any]
    doc_sections: Dict[str, str]
    parsed_instructions: Dict[str, Any]
    opportunity_context: Dict[str, Any]
    context_pack: Dict[str, Any]

    # phase 2: parallel analysis (merged via _merge_analysis reducer)
    analysis_results: Annotated[Dict[str, Any], _merge_analysis]

    # phase 3: synthesis output
    proposals: List[Dict[str, Any]]  # list of ChangeProposal dicts

    # phase 4: evaluation loop
    evaluation_result: Dict[str, Any]  # EvaluationResult dict
    iteration_count: int

    # phase 5: human review decisions
    human_review: Dict[str, Any]  # {"approved_proposals": [...], "decisions": {...}}

    # phase 6: rewrite output
    final_document: str  # LaTeX source code
    final_pdf_bytes: Optional[bytes]  # compiled PDF binary
    diff: Dict[str, Any]  # {"applied": [...], "rejected": [...], "conflicts": [...], ...}

    # frontend progress tracking
    current_step: Optional[str]
    step_history: Annotated[List[str], operator.add]

    # final response for frontend
    result: WorkflowResult
