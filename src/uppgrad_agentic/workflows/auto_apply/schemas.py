from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


OpportunityType = Literal["job", "masters", "phd", "scholarship"]
TailoringDepth = Literal["light", "deep", "generate"]
EligibilityDecision = Literal["ready", "ineligible"]
ScrapeStatus = Literal["full", "partial", "failed"]
RequirementCategory = Literal["document", "text", "misc"]


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
    decision: EligibilityDecision = Field(..., description="Eligibility outcome: ready | ineligible")
    reasons: List[str] = Field(default_factory=list, description="Reasons supporting the decision")
    missing_fields: List[str] = Field(default_factory=list, description="Profile fields that are missing and needed to continue")


# DEPRECATED — replaced by RequirementItem in Step 6 of the gate-1 remodel.
# Kept defined for one cycle to ease the deprecation cliff for any external
# consumers; remove in a follow-up once nothing references it.
class AssetMap(BaseModel):
    requirement_type: str = Field(..., description="Document type being mapped (e.g. 'CV', 'Cover Letter', 'SOP')")
    source_document: str = Field(..., description="Which user document is used as the base (e.g. 'CV', 'profile', or empty if none)")
    tailoring_depth: TailoringDepth = Field(..., description="How much work is needed: light | deep | generate")
    available: bool = Field(..., description="True if the user already has this exact document on file")
    notes: str = Field(default="", description="Explains the mapping decision or flags issues the user should know about")


# DEPRECATED — superseded by RequirementItem.
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
    canonical_document_type: str = Field(
        default="",
        description=(
            "Canonical document-type label for file inputs (e.g. 'CV', 'Cover Letter', "
            "'Transcript', 'Portfolio'). Populated only for field_type='file' by "
            "determine_requirements via heuristic + LLM classification. Empty for "
            "non-file fields."
        ),
    )


class FormSchema(BaseModel):
    """Container used for LLM structured output — the full set of fields
    found on the application form."""
    fields: List[FormField] = Field(default_factory=list, description="One FormField per input on the form, in document order")
    form_action: str = Field(default="", description="The form's `action` attribute when present — useful for direct POST submission")
    form_method: str = Field(default="POST", description="The form's `method` attribute (POST/GET); defaults to POST when not specified")


# ─── Requirement model (replaces AssetMap in Step 6) ──────────────────────────

class RequirementItem(BaseModel):
    """One actionable item the user reviews at gate 1.

    Built by asset_mapping from either form_fields (jobs with extraction) or
    normalized_requirements (everything else). Documents map to user uploads
    or auto-generation; texts map to free-form questions; misc collapses
    profile/identity fields into a single virtual line.
    """
    id: int = Field(..., description="Stable index used by the gate-1 resume payload to key per-item choices")
    category: RequirementCategory = Field(..., description="document | text | misc")
    label: str = Field(..., description="Human-readable label shown to the user")
    description: str = Field(default="", description="Longer explanation of what this requirement is")
    field_type: Optional[str] = Field(
        default=None,
        description="FormFieldType for items derived from form_fields; None when derived from normalized_requirements",
    )
    required: bool = Field(default=False, description="True when the source FormField is required or the requirement is hard-blocking")
    document_type: Optional[str] = Field(
        default=None,
        description="Canonical document type when category='document' (e.g. 'CV', 'Cover Letter', 'Transcript')",
    )
    question: Optional[str] = Field(
        default=None,
        description="For category='text': the FormField label, used as the prompt for auto-generation",
    )
    form_field_index: Optional[int] = Field(
        default=None,
        description="Back-pointer into state['form_fields']; None for items derived from normalized_requirements",
    )


# ─── Upload analysis schemas (Step 6 two-pass tailoring inputs) ───────────────

class UploadedDocPreAnalysis(BaseModel):
    """Pre-tailoring analysis of a user-uploaded document."""
    completeness: str = Field(..., description="Short prose: what core sections / signals are present vs missing")
    relevance: str = Field(..., description="Short prose: how well the document matches the opportunity's requirements")
    correctness: str = Field(..., description="Short prose: factual / structural / formatting issues, if any")
    overall_quality: Literal["needs_major_work", "needs_revision", "ready_for_polish"] = Field(
        ...,
        description="Bucket verdict driving downstream tailoring effort",
    )
    top_priorities: List[str] = Field(
        default_factory=list,
        description="Up to 3 prioritised changes the tailoring pass should address",
    )


class UploadedDocLightPostAnalysis(BaseModel):
    """Post-T1 light analysis flagging remaining gaps before T2."""
    structure_issues: List[str] = Field(
        default_factory=list,
        description="Up to 3 ordering/formatting/structural problems remaining after T1",
    )
    content_gap_vs_opportunity: List[str] = Field(
        default_factory=list,
        description="Up to 3 missing elements that the opportunity explicitly asks for",
    )
    content_gap_vs_profile: List[str] = Field(
        default_factory=list,
        description="Up to 3 strengths from the user's profile that T1 failed to surface",
    )


# ─── Auto-fill schemas (consumed by tools/playwright_filler.py) ──────────────
#
# These describe the value-planning + fill-result contract between the
# adapter (which orchestrates) and the agentic-side helpers (which compute
# values + drive Playwright). NO references to AutoApplyState or LangGraph
# here — this layer is intentionally graph-agnostic so node ordering and
# gate semantics can change without breaking auto-fill.

FormFieldFillStatus = Literal["filled", "skipped"]
FormFieldFillSource = Literal[
    "user_profile",       # value pulled from the student's profile snapshot
    "user_document",      # path to a tailored/uploaded document file
    "user_answer",        # LLM-drafted free-text answer for the question
    "computed",           # derived (e.g. today's date)
    "mock",               # placeholder used in dry-run / test mode
    "llm_inferred",       # Tier 4b: LLM derived a value from profile+CV for
                          # a field value_planner couldn't fill deterministically.
                          # Bounded budget; only fires for skipped+no_value entries
                          # whose label survives the deny-list (salary/SSN/DOB).
    "no_value",           # no value could be planned (skipped)
]


class FormFieldFillPlan(BaseModel):
    """One row in the fill plan — what value to set for one FormField."""
    field: FormField = Field(..., description="The form field this plan targets")
    value: str = Field(default="", description="String value to set; for file fields this is a filesystem path")
    status: FormFieldFillStatus = Field(default="filled")
    source: FormFieldFillSource = Field(default="no_value")
    reason: str = Field(default="", description="Short note on why this value was chosen (or why skipped)")
    # Post-fill state probe results — populated by `_probe_field_state` after
    # tier 1-4 actions run. `verified` distinguishes "filled and DOM agrees"
    # from "filled but DOM state diverged" (e.g. text typed into a combobox
    # with no option selected, or value swallowed by a read-only field).
    # `observed_value` is the string we read back from the DOM after the fill
    # action, normalised per field_type (see `_probe_field_state`). When the
    # probe couldn't read state at all, both stay defaults (verified=False,
    # observed_value="") and we treat the row as needing correction.
    verified: bool = Field(default=False, description="True when DOM state matches intended value after fill")
    observed_value: str = Field(default="", description="Value read back from the DOM after the fill action")
    correction_attempts: int = Field(default=0, description="Number of LLM-driven drift corrections attempted on this field")


FillFieldOutcome = Literal[
    "ok",                  # filled deterministically (Tier 1-3) AND DOM state verified
    "ok_llm",              # filled via Tier 4 LLM-picked selector AND DOM state verified
    "ok_corrected",        # filled, DOM drifted, drift-corrector recovered (Tier 5)
    "drift_unresolved",    # filled but DOM diverged AND correction attempts exhausted
    "plan_skip",           # planner produced status=skipped; nothing attempted
    "no_locator",          # all tiers (incl. LLM) couldn't locate the input
    "fill_error",          # locator found but action failed (timeout, etc.)
    "select_error",
    "checkbox_error",
    "radio_error",
    "file_error",
    "llm_refused_submit",  # LLM picker tried to point at a submit button — refused
    "llm_skipped",         # LLM tier skipped (budget exhausted / no LLM)
    "llm_exec_error",
]


class FormFieldFillReport(BaseModel):
    """Per-field outcome of attempting to fill the form."""
    label: str
    field_type: str
    outcome: FillFieldOutcome
    detail: str = Field(default="")


class FormFillResult(BaseModel):
    """Final result of a form-fill attempt."""
    form_url: str
    success: bool = Field(..., description="True when at least one field filled and no submit-button click occurred")
    fields_total: int = 0
    fields_filled_native: int = 0
    fields_filled_llm: int = 0
    fields_skipped: int = 0
    fields_failed: int = 0
    llm_picker_calls: int = 0
    # Post-fill verification counters (Tier 5 — added 2026-05). The
    # original counters above only count "the action executed without
    # exception"; these distinguish "the DOM actually reflects the
    # intended value" from "we set it but observed state diverged".
    # `fields_verified` ⊆ `fields_filled_native + fields_filled_llm`.
    # `fields_drift_corrected` is the subset that needed an LLM-driven
    # correction. `fields_drift_unresolved` is "filled, drifted,
    # correction couldn't recover" — the user-visible failure mode this
    # tier was added to surface.
    fields_verified: int = 0
    fields_drift_corrected: int = 0
    fields_drift_unresolved: int = 0
    drift_correction_calls: int = 0
    captcha_detected: bool = False
    submit_clicked: bool = Field(default=False, description="MUST be False unless explicit submission is authorized in a future feature")
    reports: List[FormFieldFillReport] = Field(default_factory=list)
    error: str = Field(default="")
