from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


DocType = Literal["CV", "SOP", "COVER_LETTER", "UNKNOWN"]


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

    # final response for frontend
    result: WorkflowResult
