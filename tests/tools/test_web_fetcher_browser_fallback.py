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


# ─── Bug A — escalation policy by status ──────────────────────────────────────

@respx.mock
def test_no_browser_escalation_when_httpx_returns_4xx(monkeypatch):
    """Browser can't fix a real server-side 4xx — the page genuinely doesn't
    exist (or we're forbidden). Wasting a browser launch on a 404 is pure
    overhead. Surfaced live on GitHub 199838: boards→job-boards Greenhouse 301
    resolves to a 404, browser was launched anyway and returned a thin
    intermediate response."""
    monkeypatch.setenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "true")
    respx.get("https://dead.com/jobs/1").mock(return_value=httpx.Response(404, text="Not found"))
    sentinel = MagicMock()
    with patch("uppgrad_agentic.tools.web_fetcher._build_async_crawler",
               return_value=sentinel) as crawler_factory:
        result = fetch_url_with_fallback("https://dead.com/jobs/1")
    crawler_factory.assert_not_called()
    assert result.used_browser is False
    assert result.http_status == 404
    assert result.thin is True


@respx.mock
def test_no_browser_escalation_when_httpx_returns_5xx(monkeypatch):
    """5xx is a server problem; browser also won't help."""
    monkeypatch.setenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "true")
    respx.get("https://dead.com/jobs/2").mock(return_value=httpx.Response(503, text="Down"))
    with patch("uppgrad_agentic.tools.web_fetcher._build_async_crawler") as crawler_factory:
        result = fetch_url_with_fallback("https://dead.com/jobs/2")
    crawler_factory.assert_not_called()
    assert result.used_browser is False


@respx.mock
def test_browser_result_with_301_status_and_tiny_body_is_flagged_thin(monkeypatch):
    """Crawl4AI surfaces the FIRST status in a redirect chain (verified live:
    boards.greenhouse.io 301 → job-boards.greenhouse.io → final = Greenhouse
    502 error page with ~246 bytes of markdown). We need the wrapper to flag
    this as thin via body-length even though status=301 < 400."""
    monkeypatch.setenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "true")
    body = "<html><body>" + ("real content " * 200) + "<form><input/></form></body></html>"
    body_thin = "<html><body>JavaScript required to view this page.</body></html>"
    respx.get("https://x.com/jobs/dead").mock(return_value=httpx.Response(200, text=body_thin))
    fake_result = MagicMock(
        success=True,
        markdown="# Error  \n502\n## Bad Gateway\n### a little lost in the weeds",
        html="", status_code=301,
        redirected_url="https://job-boards.example.com/x/jobs/4554047",
    )
    fake_crawler = AsyncMock()
    fake_crawler.__aenter__.return_value = fake_crawler
    fake_crawler.__aexit__.return_value = False
    fake_crawler.arun = AsyncMock(return_value=fake_result)
    with patch("uppgrad_agentic.tools.web_fetcher._build_async_crawler", return_value=fake_crawler):
        result = fetch_url_with_fallback("https://x.com/jobs/dead")
    assert result.used_browser is True
    assert result.thin is True
    assert any("body_len" in s for s in result.thin_signals)


@respx.mock
def test_browser_fallback_passes_wait_config_to_crawl4ai(monkeypatch):
    """Bug D — Crawl4AI on Ashby React SPA returned text_len=1 because the
    default extractor doesn't wait for client-side hydration. We must pass a
    CrawlerRunConfig with a JS wait condition that defers extraction until
    the body has substantial visible text."""
    monkeypatch.setenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "true")
    body = "<html><body><noscript>You need to enable JavaScript to run this app.</noscript></body></html>"
    respx.get("https://ashby.com/x/1").mock(return_value=httpx.Response(200, text=body))
    fake_result = MagicMock(
        success=True, markdown="Ashby SPA hydrated content. " * 200,
        html="", status_code=200, redirected_url="https://ashby.com/x/1",
    )
    fake_crawler = AsyncMock()
    fake_crawler.__aenter__.return_value = fake_crawler
    fake_crawler.__aexit__.return_value = False
    fake_crawler.arun = AsyncMock(return_value=fake_result)
    with patch("uppgrad_agentic.tools.web_fetcher._build_async_crawler", return_value=fake_crawler):
        fetch_url_with_fallback("https://ashby.com/x/1")
    args, kwargs = fake_crawler.arun.call_args
    config = kwargs.get("config")
    assert config is not None, "expected CrawlerRunConfig kwarg passed to crawler.arun"
    # The wait_for needs to be SOMETHING (JS expression or selector). The exact
    # form is implementation detail — we just verify it's configured.
    wait_for = getattr(config, "wait_for", None)
    assert wait_for, f"expected wait_for to be set on CrawlerRunConfig, got {wait_for!r}"


@respx.mock
def test_browser_fallback_uses_final_url_after_httpx_resolved_redirects(monkeypatch):
    """When httpx already followed a 301/302 chain to the final URL, we must
    pass THAT URL to Crawl4AI on escalation — not the original input. The
    browser shouldn't re-run the redirect chain we already paid to resolve."""
    monkeypatch.setenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "true")
    respx.get("https://old.example.com/r/1").mock(
        return_value=httpx.Response(301, headers={"Location": "https://new.example.com/r/1"})
    )
    # Final URL serves a thin SPA shell so httpx escalates.
    body = "<html><body><noscript>You need to enable JavaScript to run this app.</noscript>" + (
        "<script>" + ("var a = 1; " * 1000) + "</script>") + "</body></html>"
    respx.get("https://new.example.com/r/1").mock(return_value=httpx.Response(200, text=body))

    fake_result = MagicMock(
        success=True,
        markdown="Real apply page rendered by Playwright. " * 50,
        html="", status_code=200, redirected_url="https://new.example.com/r/1",
    )
    fake_crawler = AsyncMock()
    fake_crawler.__aenter__.return_value = fake_crawler
    fake_crawler.__aexit__.return_value = False
    fake_crawler.arun = AsyncMock(return_value=fake_result)

    with patch("uppgrad_agentic.tools.web_fetcher._build_async_crawler", return_value=fake_crawler):
        result = fetch_url_with_fallback("https://old.example.com/r/1")

    # Verify Crawl4AI was called with the FINAL URL, not the original input.
    args, kwargs = fake_crawler.arun.call_args
    assert kwargs.get("url") == "https://new.example.com/r/1", (
        f"expected browser to fetch resolved URL, got {kwargs.get('url')!r}"
    )
    assert result.used_browser is True


@respx.mock
def test_browser_escalation_still_fires_for_200_spa_shell(monkeypatch):
    """Regression guard for the fix above: 200 + SPA-shell body must still
    escalate to browser fallback (this is what the env-gated path is for)."""
    monkeypatch.setenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "true")
    body = (
        "<html><body><noscript>You need to enable JavaScript to run this app.</noscript>"
        + ("<script>" + ("var a = 1; " * 1000) + "</script>") + "</body></html>"
    )
    respx.get("https://spa.com/jobs/1").mock(return_value=httpx.Response(200, text=body))
    fake_result = MagicMock(
        success=True, markdown="Real apply page rendered by browser. " * 50,
        html="", status_code=200, metadata={"title": "Apply"},
    )
    fake_crawler = AsyncMock()
    fake_crawler.__aenter__.return_value = fake_crawler
    fake_crawler.__aexit__.return_value = False
    fake_crawler.arun = AsyncMock(return_value=fake_result)
    with patch("uppgrad_agentic.tools.web_fetcher._build_async_crawler", return_value=fake_crawler):
        result = fetch_url_with_fallback("https://spa.com/jobs/1")
    assert result.used_browser is True
    assert result.success is True
