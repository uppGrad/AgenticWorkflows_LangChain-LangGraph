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


# ─── Thin-detector false positives (Bug #6 — surfaced live on Anthropic 228527) ─

@respx.mock
def test_anthropic_greenhouse_shape_with_svg_404_and_recaptcha_keys_is_not_thin():
    """The exact shape of the live Anthropic 228527 false positive: a 73KB
    Greenhouse page whose body contains BOTH an embedded SVG with `20.1404` in
    path-data AND a `GOOGLE_RECAPTCHA_INVISIBLE_KEY` config token. With the old
    substring-only rule, the two false-positive matches together flagged the
    page as thin. Must now be evaluated with word boundaries → not thin."""
    body = (
        "<html><body>" + ("Detailed Greenhouse job description with real content. " * 800)
        + '<svg><path d="M14.9088 19.7733 14.978 19.9538 14.9747 20.1404C14.9714 20.3269"/></svg>'
        + '<script>var x = {"GOOGLE_RECAPTCHA_INVISIBLE_KEY":"a","GOOGLE_RECAPTCHA_KEY":"b"};</script>'
        + "</body></html>"
    )
    respx.get("https://gh.com/jobs/1").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url("https://gh.com/jobs/1")
    assert result.thin is False, f"unexpected thin signals: {result.thin_signals}"


@respx.mock
def test_real_anti_bot_wall_with_word_boundary_keywords_still_thin():
    """We must still catch genuine anti-bot walls. A page whose body contains
    explicit anti-bot phrases as standalone tokens (word boundaries respected,
    not embedded inside identifiers) should still be flagged thin."""
    body = (
        "<html><body><h1>Please verify you are human</h1>"
        "<p>Cloudflare protection is checking your browser.</p>"
        "<p>Solve the captcha to continue.</p>" + ("placeholder text " * 30) + "</body></html>"
    )
    respx.get("https://wall.com/jobs/1").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url("https://wall.com/jobs/1")
    assert result.thin is True


@respx.mock
def test_spa_shell_with_explicit_enable_javascript_phrase_is_thin():
    """Ashby/SPA shells render with phrases like `You need to enable JavaScript
    to run this app`. These pages may have large raw HTML (inline JS bundles)
    but the explicit `enable javascript` phrase is itself a strong, unambiguous
    SPA-shell signal — one strong signal must be enough to flag as thin."""
    body = (
        "<html><body>"
        + '<noscript>You need to enable JavaScript to run this app.</noscript>'
        + ("<script>" + ("var a = 1; " * 5000) + "</script>")
        + "</body></html>"
    )
    respx.get("https://ashby.com/jobs/1").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url("https://ashby.com/jobs/1")
    assert result.thin is True


@respx.mock
def test_word_boundary_404_inside_numeric_string_does_not_trigger_alone():
    """Defensive: a body with `20.1404` substring but NO other thin signal
    must not be flagged thin (the digit chunk is not a standalone `404`)."""
    body = "<html><body>" + ("Real role description content. " * 200) + (
        '<svg><path d="20.1404 30.4042"/></svg>') + "</body></html>"
    respx.get("https://gh.com/jobs/2").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url("https://gh.com/jobs/2")
    assert result.thin is False


@respx.mock
def test_word_boundary_captcha_inside_identifier_does_not_trigger_alone():
    """Defensive: `captcha` inside `RECAPTCHA_INVISIBLE_KEY` must not match a
    standalone `captcha` token."""
    body = "<html><body>" + ("Real role description content. " * 200) + (
        '<script>const x = "GOOGLE_RECAPTCHA_INVISIBLE_KEY";</script>') + "</body></html>"
    respx.get("https://gh.com/jobs/3").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url("https://gh.com/jobs/3")
    assert result.thin is False


# ─── final_url propagation (Round 4 — redirect chain resolved by httpx) ──────

@respx.mock
def test_final_url_equals_original_when_no_redirect():
    body = "<html><body>" + ("Real content. " * 200) + "</body></html>"
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url("https://acme.com/jobs/1")
    assert result.final_url == "https://acme.com/jobs/1"


@respx.mock
def test_final_url_reflects_followed_redirect():
    """httpx with follow_redirects=True chases 301/302 and lands at the final
    URL. We must surface that final URL so downstream callers (browser
    fallback, scrape, cache) operate on the resolved address rather than the
    original input."""
    respx.get("https://old.example.com/r/1").mock(
        return_value=httpx.Response(301, headers={"Location": "https://new.example.com/r/1"})
    )
    body = "<html><body>" + ("Real content. " * 200) + "</body></html>"
    respx.get("https://new.example.com/r/1").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url("https://old.example.com/r/1")
    assert result.final_url == "https://new.example.com/r/1"
    assert result.http_status == 200
