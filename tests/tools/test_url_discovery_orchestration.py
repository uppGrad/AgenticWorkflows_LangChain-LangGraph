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
