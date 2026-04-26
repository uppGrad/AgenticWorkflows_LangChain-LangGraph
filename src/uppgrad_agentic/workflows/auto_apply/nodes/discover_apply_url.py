from __future__ import annotations

import logging

from uppgrad_agentic.common.llm import get_search_provider
from uppgrad_agentic.tools.url_discovery import discover_apply_url
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)


def discover_apply_url_node(state: AutoApplyState) -> dict:
    updates = {"current_step": "discover_apply_url", "step_history": ["discover_apply_url"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    if state.get("opportunity_type") != "job":
        return updates

    opportunity_data = state.get("opportunity_data") or {}

    # Internal jobs (employer_id == 1) submit through the platform — discovery N/A.
    if opportunity_data.get("employer_id") == 1:
        return {
            **updates,
            "discovered_apply_url": None,
            "discovery_method": "skipped_internal",
            "discovery_confidence": 0.0,
        }

    # Cache hit — adapter pre-loaded a known-good URL.
    cached_url = state.get("discovered_apply_url")
    cached_method = state.get("discovery_method")
    if cached_url and cached_method and cached_method != "failed":
        return {
            **updates,
            "discovered_apply_url": cached_url,
            "discovery_method": cached_method,
            "discovery_confidence": state.get("discovery_confidence") or 0.0,
        }

    search_provider = get_search_provider()
    result = discover_apply_url(opportunity_data, search_provider=search_provider)

    logger.info(
        "discover_apply_url: method=%s confidence=%.2f url=%s",
        result.method, result.confidence, result.url,
    )

    return {
        **updates,
        "discovered_apply_url": result.url or None,
        "discovery_method": result.method,
        "discovery_confidence": result.confidence,
    }
