from unittest.mock import MagicMock

from uppgrad_agentic.tools.url_discovery import (
    discover_apply_url, _build_ats_query, _build_careers_query, _build_generic_query,
    DiscoveryResult,
)
from uppgrad_agentic.tools.search import SearchResult
from uppgrad_agentic.tools.web_fetcher import FetchResult


def _job(title="Senior Backend Engineer", company="Acme Corp",
         url_direct=None, company_url=None):
    return {
        "id": 42, "title": title, "company": company,
        "url": "https://www.linkedin.com/jobs/view/42",
        "url_direct": url_direct,
        "company_url": company_url,
        "posted_time": "2026-04-20T00:00:00Z",
        "location": "London, UK",
    }


def test_short_circuit_when_url_direct_present():
    job = _job(url_direct="https://acme.com/apply/1")
    result = discover_apply_url(job, search_provider=None)
    assert result.method == "url_direct"
    assert result.url == "https://acme.com/apply/1"
    assert result.confidence == 1.0


def test_failed_when_no_search_provider_and_no_url_direct():
    result = discover_apply_url(_job(), search_provider=None)
    assert result.method == "failed"
    assert result.url == ""


def test_ats_query_format():
    q = _build_ats_query("Senior Backend Engineer", "Acme Corp")
    assert '"Senior Backend Engineer"' in q
    assert '"Acme Corp"' in q
    assert "site:greenhouse.io" in q
    assert "site:lever.co" in q
    assert "site:myworkdayjobs.com" in q


def test_careers_query_format():
    q = _build_careers_query("Senior Backend Engineer", "https://acme.com/about")
    assert '"Senior Backend Engineer"' in q
    assert "site:acme.com" in q


def test_careers_query_returns_none_without_company_url():
    q = _build_careers_query("Senior Backend Engineer", None)
    assert q is None


def test_ats_tier_returns_first_verified(monkeypatch):
    job = _job()
    fake_search = MagicMock()
    fake_search.search.return_value = [
        SearchResult(url="https://boards.greenhouse.io/acme/jobs/1",
                     title="Senior Backend Engineer at Acme Corp", snippet="Apply now"),
    ]
    fake_fetch = MagicMock(return_value=FetchResult(
        success=True, thin=False,
        text="Acme Corp is hiring Senior Backend Engineer in London",
        http_status=200,
    ))
    monkeypatch.setattr("uppgrad_agentic.tools.url_discovery.fetch_url_with_fallback", fake_fetch)

    result = discover_apply_url(job, search_provider=fake_search)
    assert result.method == "ats"
    assert result.url.startswith("https://boards.greenhouse.io/")


def test_falls_through_to_careers_when_ats_fails(monkeypatch):
    job = _job(company_url="https://acmecorp.com")
    fake_search = MagicMock()
    fake_search.search.side_effect = [
        [SearchResult(url="https://boards.greenhouse.io/acme/jobs/1",
                      title="Marketing Manager", snippet="")],
        [SearchResult(url="https://acmecorp.com/careers/role",
                      title="Senior Backend Engineer", snippet="")],
    ]

    def fake_fetch(url):
        if "greenhouse" in url:
            return FetchResult(success=True, thin=False, text="Marketing role.", http_status=200)
        return FetchResult(success=True, thin=False,
                           text="Senior Backend Engineer position. Apply.",
                           http_status=200)

    monkeypatch.setattr("uppgrad_agentic.tools.url_discovery.fetch_url_with_fallback", fake_fetch)

    result = discover_apply_url(job, search_provider=fake_search)
    assert result.method == "careers"
    assert result.url == "https://acmecorp.com/careers/role"


def test_returns_failed_when_all_tiers_miss(monkeypatch):
    job = _job(company_url="https://acmecorp.com")
    fake_search = MagicMock()
    fake_search.search.return_value = []
    monkeypatch.setattr("uppgrad_agentic.tools.url_discovery.fetch_url_with_fallback", MagicMock())

    result = discover_apply_url(job, search_provider=fake_search)
    assert result.method == "failed"


def test_successful_discovery_propagates_verified_text(monkeypatch):
    """The orchestrator returns the verified page content + http_status so the
    downstream scrape step can use it without re-fetching the same URL."""
    job = _job()
    fake_search = MagicMock()
    fake_search.search.return_value = [
        SearchResult(url="https://boards.greenhouse.io/acme/jobs/1",
                     title="Senior Backend Engineer at Acme Corp", snippet=""),
    ]
    fake_fetch = MagicMock(return_value=FetchResult(
        success=True, thin=False,
        text="Acme Corp is hiring Senior Backend Engineer in London, UK.",
        http_status=200,
    ))
    monkeypatch.setattr("uppgrad_agentic.tools.url_discovery.fetch_url_with_fallback", fake_fetch)

    result = discover_apply_url(job, search_provider=fake_search)
    assert result.method == "ats"
    assert "Senior Backend Engineer" in result.text
    assert result.http_status == 200


def test_thin_candidate_skipped_before_scoring(monkeypatch):
    """A thin httpx response (404 / captcha wall / JS shell) is rejected at the
    fetcher gate, never reaches score_candidate. Discovery moves on."""
    job = _job(company_url="https://acmecorp.com")
    fake_search = MagicMock()
    fake_search.search.side_effect = [
        # Tier 1: thin response — should be rejected without scoring
        [SearchResult(url="https://boards.greenhouse.io/acme/jobs/1",
                      title="Senior Backend Engineer", snippet="")],
        # Tier 2: substantive response — should win
        [SearchResult(url="https://acmecorp.com/careers/role",
                      title="Senior Backend Engineer", snippet="")],
    ]

    def fake_fetch(url):
        if "greenhouse" in url:
            return FetchResult(
                success=True, thin=True, text="Cloudflare. Captcha challenge.",
                http_status=200, thin_signals=["cloudflare", "captcha"],
            )
        return FetchResult(success=True, thin=False,
                           text="Senior Backend Engineer in London, UK. Apply.",
                           http_status=200)

    monkeypatch.setattr("uppgrad_agentic.tools.url_discovery.fetch_url_with_fallback", fake_fetch)

    result = discover_apply_url(job, search_provider=fake_search)
    # Greenhouse thin → skipped; careers tier picked up
    assert result.method == "careers"
    assert result.url == "https://acmecorp.com/careers/role"


def test_url_direct_path_returns_no_text(monkeypatch):
    """url_direct short-circuit doesn't fetch — text stays empty."""
    job = _job(url_direct="https://acme.com/apply/1")
    result = discover_apply_url(job, search_provider=None)
    assert result.method == "url_direct"
    assert result.text == ""
    assert result.http_status == 0


def test_careers_tier_skipped_for_linkedin_company_url(monkeypatch):
    """linkedin_jobs.company_url is the LinkedIn company page URL — not a real
    career site. Careers tier must skip it (otherwise we waste a Brave call on
    `site:linkedin.com`)."""
    from uppgrad_agentic.tools.url_discovery import _build_careers_query
    assert _build_careers_query("X", "https://www.linkedin.com/company/celonis") is None
    assert _build_careers_query("X", "https://www.indeed.com/cmp/acme") is None
    assert _build_careers_query("X", "https://glassdoor.com/Overview/...") is None
    # Real company domain still works
    assert _build_careers_query("X", "https://celonis.com") == '"X" site:celonis.com'


def test_strict_verification_rejects_wrong_location_match(monkeypatch):
    """Regression guard: a Greenhouse URL for the right title+company but a
    DIFFERENT location must not pass verification. Live test caught a Schwyz,
    Switzerland linkedin_jobs row falsely matching a Cleveland, Ohio Greenhouse
    page on title+company alone."""
    job = {
        "id": 42,
        "title": "Client Value Partner (CVP)",
        "company": "Celonis",
        "location": "Schwyz, Switzerland",
        "posted_time": "2026-04-19",
        "description": "Process intelligence platform Celonis is hiring in Schwyz.",
        "url_direct": None, "company_url": None,
    }
    fake_search = MagicMock()
    fake_search.search.return_value = [
        SearchResult(
            url="https://job-boards.greenhouse.io/celonis/jobs/7539033003",
            title="Client Value Partner (CVP)", snippet="",
        ),
    ]
    # Page mentions Cleveland and Ohio, NOT Schwyz/Switzerland.
    cleveland_page = (
        "Client Value Partner (CVP) at Celonis. Cleveland, Ohio. "
        "Drive customer success in the Midwest. Apply now via this form. "
    ) * 10
    fake_fetch = MagicMock(return_value=FetchResult(
        success=True, thin=False, text=cleveland_page, http_status=200,
    ))
    monkeypatch.setattr("uppgrad_agentic.tools.url_discovery.fetch_url_with_fallback", fake_fetch)

    result = discover_apply_url(job, search_provider=fake_search)
    # Title fuzz passes (title is on page) and company-in-URL passes (boards.greenhouse.io/celonis),
    # but neither location nor description-keyword corroborates → only 1 corroborator → ATS bar of 2 fails.
    assert result.method == "failed", (
        f"Expected rejection of wrong-location match; got {result.method} "
        f"with confidence {result.confidence}"
    )
