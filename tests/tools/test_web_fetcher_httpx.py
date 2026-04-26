import httpx
import respx

from uppgrad_agentic.tools.web_fetcher import fetch_url, FetchResult


@respx.mock
def test_returns_success_for_substantial_html():
    body = "<html><body>" + ("Real apply page content. " * 200) + "<form><input type='file' name='resume'></form></body></html>"
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url("https://acme.com/jobs/1")
    assert isinstance(result, FetchResult)
    assert result.success is True
    assert result.thin is False
    assert result.http_status == 200
    assert "apply page content" in result.text


@respx.mock
def test_returns_thin_for_404():
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(404, text="Page not found"))
    result = fetch_url("https://acme.com/jobs/1")
    assert result.success is False
    assert result.thin is True
    assert result.http_status == 404


@respx.mock
def test_returns_thin_for_anti_bot_keywords():
    # Substantial body (>500 bytes) with anti-bot signals — should still trip thin detection
    body = ("<html><body>Cloudflare protection active. " * 30) + (
        "Please complete the captcha to continue. JavaScript required to view this page. "
    )
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url("https://acme.com/jobs/1")
    assert result.success is True
    assert result.thin is True
    assert any(s in ("cloudflare", "captcha", "javascript required") for s in result.thin_signals)


@respx.mock
def test_returns_thin_for_short_body():
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(200, text="short"))
    result = fetch_url("https://acme.com/jobs/1")
    assert result.thin is True


@respx.mock
def test_returns_failure_on_network_error():
    respx.get("https://acme.com/jobs/1").mock(side_effect=httpx.ConnectError("boom"))
    result = fetch_url("https://acme.com/jobs/1")
    assert result.success is False
    assert result.http_status == 0
    assert "boom" in result.error
