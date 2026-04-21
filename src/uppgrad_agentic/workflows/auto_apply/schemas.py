from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


OpportunityType = Literal["job", "masters", "phd", "scholarship"]
TailoringDepth = Literal["none", "light", "deep", "generate"]
EligibilityDecision = Literal["ready", "pending", "ineligible", "manual_review"]
ScrapeStatus = Literal["full", "partial", "failed"]


class NormalizedRequirement(BaseModel):
    requirement_type: str = Field(..., description="Category of requirement (e.g. 'document', 'eligibility', 'language')")
    document_type: str = Field(..., description="Document type required (e.g. 'CV', 'Cover Letter', 'SOP', 'Transcript')")
    is_assumed: bool = Field(..., description="True if this requirement is a default assumption, not scraped or parsed from source")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence that this requirement is real and accurate")


class ScrapeResult(BaseModel):
    status: ScrapeStatus = Field(..., description="Quality of the scrape: full, partial, or failed")
    requirements: List[NormalizedRequirement] = Field(default_factory=list, description="Normalized requirements extracted from the scraped page")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in the scraped requirements")
    source: str = Field(..., description="URL or source that was scraped")


class EligibilityResult(BaseModel):
    decision: EligibilityDecision = Field(..., description="Eligibility outcome: ready | pending | ineligible | manual_review")
    reasons: List[str] = Field(default_factory=list, description="Reasons supporting the decision")
    missing_fields: List[str] = Field(default_factory=list, description="Profile fields that are missing and needed to continue")


class AssetMap(BaseModel):
    requirement_type: str = Field(..., description="Document type being mapped (e.g. 'CV', 'Cover Letter', 'SOP')")
    source_document: str = Field(..., description="Which user document is used as the base (e.g. 'CV', 'profile', or empty if none)")
    tailoring_depth: TailoringDepth = Field(..., description="How much work is needed: none | light | deep | generate")
    available: bool = Field(..., description="True if the user already has this exact document on file")
    notes: str = Field(default="", description="Explains the mapping decision or flags issues the user should know about")


class AssetMappingOutput(BaseModel):
    """Container used for LLM structured output — wraps the full list of mappings."""
    mappings: List[AssetMap] = Field(..., description="One AssetMap entry per normalized requirement")
