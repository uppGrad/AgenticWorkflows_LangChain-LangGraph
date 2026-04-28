from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


OpportunityType = Literal["job", "masters", "phd", "scholarship"]


class WorkflowResult(TypedDict, total=False):
    status: Literal["ok", "error"]
    error_code: Optional[str]
    user_message: Optional[str]
    details: Optional[Dict[str, Any]]


class AutoApplyState(TypedDict, total=False):
    # inputs (from frontend)
    opportunity_type: OpportunityType
    opportunity_id: str

    # opportunity intelligence
    opportunity_data: Dict[str, Any]          # raw DB record
    scraped_requirements: Dict[str, Any]      # ScrapeResult dict: status, requirements, confidence, source
    normalized_requirements: List[Dict[str, Any]]  # list of NormalizedRequirement dicts

    # injected by backend adapter (Spec A1) — replaces _get_stub_profile lookups
    profile_snapshot: Dict[str, Any]

    # apply-URL discovery (Spec A6) — populated when discovery feature ships
    discovered_apply_url: Optional[str]
    discovery_method: Optional[str]
    discovery_confidence: Optional[float]

    # Verified page content from discovery — propagated to scrape_application_page
    # so we don't re-fetch the same URL twice (Phase 2 of v2.1 follow-up).
    # `discovered_page_content` is markdown for the browser path / HTML for httpx
    # (good for prose extraction). `discovered_raw_html` is always actual HTML
    # (used by form-field extraction).
    discovered_page_content: Optional[str]
    discovered_raw_html: Optional[str]
    discovered_http_status: Optional[int]

    # Apply-form URL resolved by per-ATS rules. Equals discovered_apply_url for
    # ATSes that keep the form on the same URL (Greenhouse, Workable). Differs
    # for split-URL ATSes (Ashby /application, Lever /apply). None when the
    # form is not reachable via simple URL navigation (Workday auth wall).
    discovered_form_url: Optional[str]

    # True when discovery found a real listing page that says the posting is
    # closed ("no longer accepting applications", etc.). Workflow surfaces
    # this in the handoff package so the user knows alongside their materials.
    posting_closed: bool

    # Structured application-form fields extracted from the rendered form HTML.
    # One entry per <input>/<select>/<textarea> visible on the form, with type,
    # label, options, and value-source classification. Consumed by future
    # auto-submit step; surfaced in the handoff package today.
    form_fields: List[Dict[str, Any]]

    # human_gate_0 retry counter (Spec §6.2) — caps the eligibility re-check loop
    gate_0_iteration_count: int

    # compatibility warnings (Spec follow-up — deadline-passed + missing
    # user-supplied docs are the only hard-block reasons; everything else
    # like location mismatch / age cap / degree level becomes a non-blocking
    # warning the UI surfaces on the apply screen and the handoff package).
    compatibility_warnings: List[str]

    # eligibility
    eligibility_result: Dict[str, Any]        # EligibilityResult dict: decision, reasons, missing_fields

    # asset mapping
    asset_mapping: List[Dict[str, Any]]       # list of AssetMap dicts, one per normalized requirement

    # human gate 1 — user reviews document mapping
    human_review_1: Dict[str, Any]            # user selections from gate 1

    # application tailoring
    tailored_documents: Dict[str, Any]        # document_type → tailored content

    # evaluation loop
    evaluation_result: Dict[str, Any]
    iteration_count: int

    # human gate 2 — user approves final package
    human_review_2: Dict[str, Any]

    # submission
    application_package: Dict[str, Any]       # final documents ready for handoff or submission
    application_record: Dict[str, Any]        # logged outcome

    # frontend progress tracking
    current_step: Optional[str]
    step_history: Annotated[List[str], operator.add]

    # final response for frontend
    result: WorkflowResult
