from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)


def record_application(state: AutoApplyState) -> dict:
    # Always runs, even after an error path, to capture the outcome.
    opportunity_id = state.get("opportunity_id", "unknown")
    opportunity_type = state.get("opportunity_type", "unknown")
    application_package: Dict[str, Any] = state.get("application_package") or {}
    scraped_requirements = state.get("scraped_requirements") or {}
    result = state.get("result") or {}

    # Determine outcome from the result set by submit_internal or package_and_handoff
    outcome: str
    if result.get("status") == "error":
        outcome = "failed"
    elif (application_package.get("submission_type") == "internal"
          or (application_package.get("documents") is None
              and application_package.get("CV"))):
        outcome = "submitted"
    else:
        outcome = "packaged_for_handoff"

    # Collect the document types that ended up in the package
    docs_in_package: list[str]
    if "documents" in application_package:
        docs_in_package = list(application_package["documents"].keys())
    else:
        docs_in_package = [k for k in application_package if k not in (
            "submission_type", "platform_application_id", "opportunity",
            "scrape_status", "scrape_confidence", "scrape_source",
        )]

    record: Dict[str, Any] = {
        "outcome": outcome,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "opportunity_id": opportunity_id,
        "opportunity_type": opportunity_type,
        "document_types_submitted": docs_in_package,
    }

    # Carry scrape provenance for jobs (useful for future external submission automation)
    if opportunity_type == "job":
        record["scrape_status"] = scraped_requirements.get("status", "failed")
        record["scrape_confidence"] = scraped_requirements.get("confidence", 0.0)

    # Carry the platform application ID for internal submissions
    if application_package.get("platform_application_id"):
        record["platform_application_id"] = application_package["platform_application_id"]

    logger.info(
        "record_application: outcome=%s opportunity_type=%s docs=%s",
        outcome, opportunity_type, docs_in_package,
    )

    return {"application_record": record}
