from __future__ import annotations

import logging
from typing import Any, Dict

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)


def package_and_handoff(state: AutoApplyState) -> dict:
    updates = {"current_step": "package_and_handoff", "step_history": ["package_and_handoff"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    opportunity_type = state.get("opportunity_type", "")
    opportunity_data = state.get("opportunity_data") or {}
    tailored_documents: Dict[str, Any] = state.get("tailored_documents") or {}
    scraped_requirements = state.get("scraped_requirements") or {}

    title = opportunity_data.get("title", "this opportunity")
    org = (
        opportunity_data.get("company")
        or opportunity_data.get("university")
        or opportunity_data.get("provider_name")
        or "the organisation"
    )
    url = (
        opportunity_data.get("url_direct")
        or opportunity_data.get("url")
        or opportunity_data.get("application_url")
        or ""
    )

    # Assemble the package — include full content for each tailored document
    package: Dict[str, Any] = {
        "documents": {
            doc_type: {
                "content": info.get("content", ""),
                "tailoring_depth": info.get("tailoring_depth", ""),
                "char_count": len(info.get("content") or ""),
            }
            for doc_type, info in tailored_documents.items()
            if not info.get("skip") and info.get("tailoring_depth") != "none"
        },
        "opportunity": {
            "type": opportunity_type,
            "id": state.get("opportunity_id", ""),
            "title": title,
            "organisation": org,
            "application_url": url,
        },
        "submission_type": "handoff",
    }

    # For jobs: include scrape provenance so the user knows whether requirements
    # were scraped or assumed, and for future external submission automation.
    if opportunity_type == "job":
        package["scrape_status"] = scraped_requirements.get("status", "failed")
        package["scrape_confidence"] = scraped_requirements.get("confidence", 0.0)
        package["scrape_source"] = scraped_requirements.get("source", "")

    doc_names = list(package["documents"].keys())

    logger.info(
        "package_and_handoff: assembled package for %s — docs=%s scrape_status=%s",
        opportunity_type,
        doc_names,
        package.get("scrape_status", "n/a"),
    )

    return {
        **updates,
        "application_package": package,
        "result": {
            "status": "ok",
            "user_message": (
                f"Your application package for {title} at {org} is ready. "
                f"Documents included: {', '.join(doc_names)}."
                + (f" Apply at: {url}" if url else "")
            ),
            "details": {
                "submission_type": "handoff",
                "documents": doc_names,
                "application_url": url,
            },
        },
    }
