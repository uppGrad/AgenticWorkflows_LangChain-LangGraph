from unittest.mock import patch, AsyncMock, MagicMock

import httpx
import respx

from uppgrad_agentic.tools.web_fetcher import fetch_url_with_fallback


@respx.mock
def test_returns_httpx_result_when_fallback_disabled(monkeypatch):
    monkeypatch.delenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", raising=False)
    body = "<html><body>" + ("real content " * 200) + "</body></html>"
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url_with_fallback("https://acme.com/jobs/1")
    assert result.used_browser is False
    assert result.thin is False


@respx.mock
def test_no_fallback_when_httpx_succeeds_with_substantial_content(monkeypatch):
    monkeypatch.setenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "true")
    body = "<html><body>" + ("real content " * 200) + "</body></html>"
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url_with_fallback("https://acme.com/jobs/1")
    assert result.used_browser is False


@respx.mock
def test_fallback_fires_when_httpx_thin_and_browser_enabled(monkeypatch):
    monkeypatch.setenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "true")
    body = "Cloudflare. JavaScript required."
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(200, text=body))

    fake_result = MagicMock(
        success=True, markdown="Real apply page content from Playwright. " * 50,
        html="", status_code=200, metadata={"title": "Apply"},
    )
    fake_crawler = AsyncMock()
    fake_crawler.__aenter__.return_value = fake_crawler
    fake_crawler.__aexit__.return_value = False
    fake_crawler.arun = AsyncMock(return_value=fake_result)

    with patch("uppgrad_agentic.tools.web_fetcher._build_async_crawler", return_value=fake_crawler):
        result = fetch_url_with_fallback("https://acme.com/jobs/1")
    assert result.used_browser is True
    assert result.success is True
    assert "Real apply page content" in result.text


@respx.mock
def test_fallback_silently_skipped_when_crawl4ai_not_installed(monkeypatch):
    monkeypatch.setenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "true")
    respx.get("https://acme.com/jobs/1").mock(
        return_value=httpx.Response(200, text="Cloudflare. JavaScript required."))
    with patch("uppgrad_agentic.tools.web_fetcher._build_async_crawler",
               side_effect=ImportError("crawl4ai not installed")):
        result = fetch_url_with_fallback("https://acme.com/jobs/1")
    # Returns the httpx (thin) result, browser flag stays False
    assert result.used_browser is False
    assert result.thin is True
