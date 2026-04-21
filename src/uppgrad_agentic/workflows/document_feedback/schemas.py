# src/uppgrad_agentic/workflows/document_feedback/schemas.py
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


DocType = Literal["CV", "SOP", "COVER_LETTER", "UNKNOWN"]


class DocTypeClassification(BaseModel):
    doc_type: DocType = Field(..., description="Document type classification")
    relevant: bool = Field(..., description="Whether this looks like an application-related document")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list, description="Short reasons/signals found")
    language: Optional[str] = Field(default=None, description="Detected language, if confident")


class ChangeProposal(BaseModel):
    section: str = Field(..., description="Document section the change applies to (e.g. 'Experience', 'Introduction')")
    rationale: str = Field(..., description="Why this change is recommended")
    before_text: str = Field(..., description="Original text to be replaced")
    after_text: str = Field(..., description="Proposed replacement text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in this proposal")
    requires_confirmation: bool = Field(..., description="Whether user must explicitly approve before applying")


class EvaluationResult(BaseModel):
    passed: bool = Field(..., description="Whether the proposals passed quality checks")
    issues: List[str] = Field(default_factory=list, description="Descriptions of any groundedness or format problems found")
    iteration: int = Field(..., description="Which evaluation iteration this result belongs to (0-indexed)")
