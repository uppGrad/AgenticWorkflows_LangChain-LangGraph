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
