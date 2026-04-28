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


# ─── Form-field extraction (Phase 2 of auto-submit foundation) ──────────────

FormFieldType = Literal["file", "text", "textarea", "select", "checkbox", "radio", "number", "email", "url", "date", "tel"]
FormFieldValueSource = Literal["user_profile", "user_document", "user_answer", "computed", "unknown"]


class FormField(BaseModel):
    """One input on the application form — file upload, text input, dropdown,
    etc. Captured from the rendered DOM at scrape time so a future auto-submit
    step can fill it without re-extracting structure.
    """
    label: str = Field(..., description="Human-readable label shown next to the field (e.g. 'Resume', 'Country', 'How did you hear about us?')")
    field_type: FormFieldType = Field(..., description="Input element type from the DOM")
    name: str = Field(default="", description="Form input `name` attribute, used when actually submitting; '' when not visible in markup")
    required: bool = Field(default=False, description="True when the field is marked required (HTML `required` or labeled with an asterisk)")
    options: List[str] = Field(default_factory=list, description="For select/radio: list of option labels in order. Empty for non-choice fields.")
    accepts_file: List[str] = Field(default_factory=list, description="For file inputs: accepted file extensions or MIME types (e.g. ['.pdf', '.docx']). Empty for non-file fields.")
    expected_source: FormFieldValueSource = Field(
        default="unknown",
        description=(
            "Where the value should come from when auto-filling: "
            "user_profile (Student row), user_document (CV/Cover Letter/etc. from asset_mapping), "
            "user_answer (free-text question we LLM-draft), computed (e.g. 'today's date'), "
            "or unknown when the LLM can't classify."
        ),
    )


class FormSchema(BaseModel):
    """Container used for LLM structured output — the full set of fields
    found on the application form."""
    fields: List[FormField] = Field(default_factory=list, description="One FormField per input on the form, in document order")
    form_action: str = Field(default="", description="The form's `action` attribute when present — useful for direct POST submission")
    form_method: str = Field(default="POST", description="The form's `method` attribute (POST/GET); defaults to POST when not specified")
