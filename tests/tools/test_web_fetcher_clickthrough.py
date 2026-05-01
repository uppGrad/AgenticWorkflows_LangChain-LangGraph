"""`force_browser_fetch(click_apply_cta=True)` — Tier-2b click-through pass.

Some ATSes (Workable's `/j/<slug>/` listing, SmartRecruiters listings, some
careers sites) gate the form behind an "Apply for this job" button. The
no-click browser pass renders the listing but finds no form; the click-through
pass dispatches a click on the first apply-style CTA and waits for a
form/input to appear.

These tests pin the wiring (the env-disabled / ImportError edge cases) and
the run-config shape (CTA matcher present, form-visible wait predicate set)
without invoking a real browser.
"""
from unittest.mock import patch, AsyncMock, MagicMock

from uppgrad_agentic.tools.web_fetcher import (
    _build_crawler_run_config,
    force_browser_fetch,
)


def test_run_config_default_waits_for_body_text_only():
    """Without `click_apply_cta`, the config doesn't ship js_code and waits
    on body innerText length — same shape we've used since the SPA-fallback
    landed (Anthropic / Greenhouse hydration)."""
    fake_config_cls = MagicMock()
    with patch(
        "crawl4ai.CrawlerRunConfig", fake_config_cls,
    ):
        _build_crawler_run_config(timeout_seconds=20.0)
    kwargs = fake_config_cls.call_args.kwargs
    assert "innerText" in kwargs["wait_for"]
    assert "js_code" not in kwargs


def test_run_config_clickthrough_emits_click_js_and_form_wait():
    """When `click_apply_cta=True`, the config carries js_code that
    matches an apply-style CTA and the wait predicate flips to "form/
    non-hidden input/textarea/select on the page". This is the contract
    the click-through path relies on."""
    fake_config_cls = MagicMock()
    with patch(
        "crawl4ai.CrawlerRunConfig", fake_config_cls,
    ):
        _build_crawler_run_config(timeout_seconds=25.0, click_apply_cta=True)
    kwargs = fake_config_cls.call_args.kwargs
    js = "\n".join(kwargs["js_code"])
    assert "Apply for this job" in js or "apply for this job" in js.lower()
    assert "Continue" in js or "continue" in js.lower()
    # Wait predicate now keys off form/input visibility, not body length
    assert "form" in kwargs["wait_for"]
    assert "input" in kwargs["wait_for"] or "textarea" in kwargs["wait_for"]


def test_force_browser_fetch_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", raising=False)
    assert force_browser_fetch("https://x", click_apply_cta=True) is None


def test_force_browser_fetch_clickthrough_invokes_crawler(monkeypatch):
    """End-to-end wiring: `click_apply_cta=True` is forwarded all the way
    into `_crawl_with_browser`, which pulls a config built with
    `click_apply_cta=True` (verified by capturing the build args)."""
    monkeypatch.setenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "true")

    fake_result = MagicMock(
        success=True, markdown="text after click " * 60,
        html="<form><input name=email/></form>",
        status_code=200,
    )
    fake_crawler = AsyncMock()
    fake_crawler.__aenter__.return_value = fake_crawler
    fake_crawler.__aexit__.return_value = False
    fake_crawler.arun = AsyncMock(return_value=fake_result)

    captured = {}

    def _spy_build_config(timeout_seconds, click_apply_cta=False):
        captured["click_apply_cta"] = click_apply_cta
        return MagicMock(name="RunConfig")

    with patch(
        "uppgrad_agentic.tools.web_fetcher._build_async_crawler",
        return_value=fake_crawler,
    ), patch(
        "uppgrad_agentic.tools.web_fetcher._build_crawler_run_config",
        side_effect=_spy_build_config,
    ):
        out = force_browser_fetch(
            "https://apply.workable.com/x/j/abc/", click_apply_cta=True,
        )

    assert captured["click_apply_cta"] is True
    assert out is not None
    assert out.success is True
    assert "<form>" in out.raw_html
