from unittest.mock import patch

from uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page import (
    scrape_application_page,
)
from uppgrad_agentic.tools.web_fetcher import FetchResult


def _state(discovered=None, method="ats"):
    return {
        "opportunity_type": "job",
        "opportunity_id": "1",
        "opportunity_data": {"id": 42, "title": "SWE", "company": "Acme",
                             "url": "https://linkedin.com/x", "url_direct": None},
        "discovered_apply_url": discovered,
        "discovery_method": method,
    }


def test_no_discovered_url_records_failed():
    out = scrape_application_page(_state(discovered=None, method="failed"))
    sr = out["scraped_requirements"]
    assert sr["status"] == "failed"
    assert sr["raw_content"] == ""


def test_skips_for_non_jobs():
    state = _state(discovered="https://acme.com/job/1")
    state["opportunity_type"] = "masters"
    out = scrape_application_page(state)
    assert "scraped_requirements" not in out


def test_uses_discovered_url_records_content():
    state = _state(discovered="https://boards.greenhouse.io/acme/jobs/1", method="ats")
    fake_fetch = FetchResult(
        success=True, thin=False,
        text="Apply now. Upload CV and Cover Letter.",
        http_status=200,
    )
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page.fetch_url_with_fallback",
        return_value=fake_fetch,
    ):
        out = scrape_application_page(state)
    sr = out["scraped_requirements"]
    assert sr["status"] == "partial"
    assert sr["source"] == "https://boards.greenhouse.io/acme/jobs/1"
    assert "Upload CV" in sr["raw_content"]


def test_thin_response_records_failed():
    state = _state(discovered="https://x.com/job/1", method="ats")
    fake_fetch = FetchResult(
        success=True, thin=True,
        text="Cloudflare. Captcha.",
        http_status=200,
        thin_signals=["cloudflare", "captcha"],
    )
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page.fetch_url_with_fallback",
        return_value=fake_fetch,
    ):
        out = scrape_application_page(state)
    sr = out["scraped_requirements"]
    assert sr["status"] == "failed"


def test_short_circuits_on_upstream_error():
    state = _state(discovered="https://x.com/1")
    state["result"] = {"status": "error"}
    out = scrape_application_page(state)
    assert "scraped_requirements" not in out


def test_uses_pre_fetched_content_when_present():
    """When discovery already verified + fetched, scrape uses that content
    instead of re-fetching the same URL."""
    state = _state(discovered="https://boards.greenhouse.io/acme/jobs/1", method="ats")
    state["discovered_page_content"] = "Pre-fetched apply page content from discovery."
    state["discovered_http_status"] = 200

    # Patch fetch_url_with_fallback so we can detect if it was called
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page.fetch_url_with_fallback",
    ) as fake_fetch:
        out = scrape_application_page(state)

    # Fast path: no network call
    fake_fetch.assert_not_called()
    sr = out["scraped_requirements"]
    assert sr["status"] == "partial"
    assert sr["raw_content"] == "Pre-fetched apply page content from discovery."
    assert sr["http_status"] == 200


def test_falls_back_to_fetch_when_no_pre_fetched_content():
    """url_direct path doesn't pre-fetch — scrape still fetches via web_fetcher."""
    from uppgrad_agentic.tools.web_fetcher import FetchResult
    state = _state(discovered="https://acme.com/apply/1", method="url_direct")
    # discovered_page_content not set
    fake_fetch_result = FetchResult(
        success=True, thin=False,
        text="Fresh fetch content for url_direct path.",
        http_status=200,
    )
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page.fetch_url_with_fallback",
        return_value=fake_fetch_result,
    ) as fake_fetch:
        out = scrape_application_page(state)

    fake_fetch.assert_called_once_with("https://acme.com/apply/1")
    sr = out["scraped_requirements"]
    assert sr["status"] == "partial"
    assert sr["raw_content"] == "Fresh fetch content for url_direct path."
