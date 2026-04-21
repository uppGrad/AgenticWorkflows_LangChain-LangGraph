from __future__ import annotations

import logging

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 10  # seconds
_MAX_CONTENT_BYTES = 500_000


def _fetch_url(url: str) -> tuple[int, str]:
    """Fetch a URL and return (status_code, text). Never raises."""
    try:
        import requests  # type: ignore

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; UppGrad-Bot/1.0; +https://uppgrad.com)"
            )
        }
        resp = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT, allow_redirects=True)
        content = resp.text[:_MAX_CONTENT_BYTES]
        return resp.status_code, content
    except Exception as exc:
        logger.warning("scrape_application_page: fetch failed for %s — %s", url, exc)
        return 0, ""


def scrape_application_page(state: AutoApplyState) -> dict:
    if state.get("result", {}).get("status") == "error":
        return {}

    # Only runs for job opportunities
    if state.get("opportunity_type") != "job":
        return {}

    opportunity_data = state.get("opportunity_data") or {}
    url_direct = opportunity_data.get("url_direct") or ""
    url_fallback = opportunity_data.get("url") or ""
    target_url = url_direct if url_direct else url_fallback

    if not target_url:
        logger.warning("scrape_application_page: no URL available in opportunity_data")
        return {
            "scraped_requirements": {
                "status": "failed",
                "requirements": [],
                "confidence": 0.0,
                "source": "",
                "raw_content": "",
                "http_status": 0,
                "error": "No URL available",
            }
        }

    http_status, raw_content = _fetch_url(target_url)

    if http_status == 0 or not raw_content:
        logger.warning("scrape_application_page: empty response from %s (http %s)", target_url, http_status)
        return {
            "scraped_requirements": {
                "status": "failed",
                "requirements": [],
                "confidence": 0.0,
                "source": target_url,
                "raw_content": "",
                "http_status": http_status,
                "error": "Empty or unreachable response",
            }
        }

    if http_status >= 400:
        logger.warning("scrape_application_page: HTTP %s from %s", http_status, target_url)
        return {
            "scraped_requirements": {
                "status": "failed",
                "requirements": [],
                "confidence": 0.0,
                "source": target_url,
                "raw_content": "",
                "http_status": http_status,
                "error": f"HTTP {http_status}",
            }
        }

    logger.info("scrape_application_page: fetched %d chars from %s", len(raw_content), target_url)
    # Store raw content; evaluate_scrape will assess quality and normalize requirements
    return {
        "scraped_requirements": {
            "status": "partial",   # evaluate_scrape will set the final status
            "requirements": [],
            "confidence": 0.0,
            "source": target_url,
            "raw_content": raw_content,
            "http_status": http_status,
        }
    }
