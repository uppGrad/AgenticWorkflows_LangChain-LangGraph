from __future__ import annotations

import logging

from uppgrad_agentic.tools.web_fetcher import fetch_url_with_fallback
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)


def scrape_application_page(state: AutoApplyState) -> dict:
    updates = {"current_step": "scrape_application_page", "step_history": ["scrape_application_page"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    if state.get("opportunity_type") != "job":
        return updates

    target_url = (state.get("discovered_apply_url") or "").strip()

    if not target_url:
        logger.info("scrape_application_page: no discovered URL — recording failed scrape")
        return {
            **updates,
            "scraped_requirements": {
                "status": "failed",
                "requirements": [],
                "confidence": 0.0,
                "source": "",
                "raw_content": "",
                "http_status": 0,
                "error": "no apply URL discovered",
            },
        }

    # Fast path: discovery already verified + fetched the URL. Use that content
    # instead of re-fetching (avoids the second-look thin-detection conflict
    # AND halves the request count, reducing ban risk).
    pre_fetched = (state.get("discovered_page_content") or "").strip()
    if pre_fetched:
        pre_status = state.get("discovered_http_status") or 200
        logger.info(
            "scrape_application_page: using pre-fetched content from discovery (%d chars, status=%s) for %s",
            len(pre_fetched), pre_status, target_url,
        )
        return {
            **updates,
            "scraped_requirements": {
                "status": "partial",
                "requirements": [],
                "confidence": 0.0,
                "source": target_url,
                "raw_content": pre_fetched,
                "http_status": pre_status,
            },
        }

    # Slow path: discovery didn't fetch (url_direct, cache hit without text).
    # Fetch fresh via the tiered fetcher.
    fetch = fetch_url_with_fallback(target_url)

    if not fetch.success or fetch.thin:
        logger.warning(
            "scrape_application_page: fetch thin/failed for %s (status=%s, signals=%s)",
            target_url, fetch.http_status, fetch.thin_signals,
        )
        return {
            **updates,
            "scraped_requirements": {
                "status": "failed",
                "requirements": [],
                "confidence": 0.0,
                "source": target_url,
                "raw_content": "",
                "http_status": fetch.http_status,
                "error": fetch.error or f"thin: {','.join(fetch.thin_signals)}",
            },
        }

    logger.info("scrape_application_page: fetched %d chars from %s (browser=%s)",
                len(fetch.text), target_url, fetch.used_browser)
    return {
        **updates,
        "scraped_requirements": {
            "status": "partial",
            "requirements": [],
            "confidence": 0.0,
            "source": target_url,
            "raw_content": fetch.text,
            "http_status": fetch.http_status,
        },
    }
