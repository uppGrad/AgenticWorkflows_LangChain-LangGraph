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
