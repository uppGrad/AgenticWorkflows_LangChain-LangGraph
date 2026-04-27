from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0
_MAX_BYTES = 500_000

# Strong keywords: specific multi-word phrases that effectively never appear on
# a legitimate apply page. A single hit flags the page thin.
_STRONG_THIN_PHRASES = [
    "page not found",
    "access denied",
    "javascript required",
    "enable javascript",
    "challenge-platform",
    "please verify you are human",
]

# Weak keywords: short tokens (often substrings of unrelated content) that only
# count when found as standalone tokens (word-boundary regex) AND when ≥2 are
# present. Substring matching used to false-positive on `20.1404` (SVG path
# data) → `404` and `RECAPTCHA_INVISIBLE_KEY` → `captcha` together flagging
# legitimate Greenhouse pages as thin.
_WEAK_THIN_TOKENS = ["404", "cloudflare", "robot", "captcha"]

# Pre-compiled word-boundary regexes for weak tokens.
_WEAK_TOKEN_PATTERNS = [re.compile(rf"\b{re.escape(t)}\b") for t in _WEAK_THIN_TOKENS]

_MIN_BODY_BYTES = 500


@dataclass
class FetchResult:
    success: bool                    # HTTP fetch returned 2xx
    thin: bool                       # Content looks like anti-bot, JS shell, short, or 4xx
    text: str                        # Body (HTML), truncated to _MAX_BYTES
    http_status: int                 # Final response status (after httpx-followed redirects)
    final_url: str = ""              # Final URL after redirect chain (httpx resolves redirects natively)
    error: str = ""
    thin_signals: List[str] = field(default_factory=list)
    used_browser: bool = False       # True when we escalated to Playwright/Crawl4AI


def _detect_thin(text: str, status: int) -> tuple[bool, List[str]]:
    if status >= 400:
        return True, [f"http_status={status}"]
    if len(text.strip()) < _MIN_BODY_BYTES:
        return True, [f"body_len={len(text)}"]
    lowered = text.lower()
    strong_hits = [kw for kw in _STRONG_THIN_PHRASES if kw in lowered]
    if strong_hits:
        return True, strong_hits
    weak_hits = [
        tok for tok, pat in zip(_WEAK_THIN_TOKENS, _WEAK_TOKEN_PATTERNS)
        if pat.search(lowered)
    ]
    if len(weak_hits) >= 2:
        return True, weak_hits
    return False, []


def fetch_url(url: str) -> FetchResult:
    """Fetch a URL using httpx. Always returns a FetchResult (never raises).

    Caller can inspect `.thin` to decide whether to escalate to a browser.
    The browser escalation lives in `fetch_url_with_fallback` (Task 2.2).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; UppGrad-Bot/1.0; +https://uppgrad.com)"
        ),
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=_TIMEOUT, follow_redirects=True)
    except httpx.HTTPError as exc:
        logger.warning("fetch_url: network error for %s — %s", url, exc)
        return FetchResult(
            success=False, thin=True, text="",
            http_status=0, final_url=url, error=str(exc),
            thin_signals=["network_error"],
        )

    text = resp.text[:_MAX_BYTES]
    thin, signals = _detect_thin(text, resp.status_code)

    return FetchResult(
        success=resp.status_code < 400,
        thin=thin,
        text=text,
        http_status=resp.status_code,
        final_url=str(resp.url),
        thin_signals=signals,
    )


import asyncio


def _browser_fallback_enabled() -> bool:
    return os.getenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "").lower() in ("true", "1", "yes")


def _build_async_crawler():
    """Construct a Crawl4AI crawler. Raises ImportError if crawl4ai missing.

    Patched in tests to inject a fake crawler.
    """
    from crawl4ai import AsyncWebCrawler  # noqa: lazy import — heavy
    return AsyncWebCrawler(verbose=False)


def _build_crawler_run_config(timeout_seconds: float):
    """Construct a CrawlerRunConfig that defers extraction until the React/SPA
    tree has hydrated. The JS wait expression checks for substantial visible
    text on the page — domain-agnostic, matches Ashby/Workday/etc. that render
    everything client-side after initial HTML.

    Lazy-imported so importing `web_fetcher` doesn't pull in crawl4ai when the
    browser fallback isn't enabled. Patched in tests."""
    from crawl4ai import CrawlerRunConfig  # noqa: lazy import — heavy
    return CrawlerRunConfig(
        page_timeout=int(timeout_seconds * 1000),
        wait_for="js:() => document.body && document.body.innerText.length > 1000",
    )


async def _crawl_with_browser(url: str, timeout_seconds: float = 25.0) -> FetchResult:
    """Use Crawl4AI / Playwright to fetch a URL when httpx returned thin content.

    httpx already resolved any redirect chain; `url` is the final destination.
    """
    crawler = _build_async_crawler()  # may raise ImportError; caller handles
    config = _build_crawler_run_config(timeout_seconds)

    async with crawler:
        try:
            result = await crawler.arun(url=url, config=config)
        except Exception as exc:
            logger.warning("web_fetcher: crawl4ai error for %s — %s", url, exc)
            return FetchResult(
                success=False, thin=True, text="",
                http_status=0, final_url=url, error=str(exc),
                thin_signals=["browser_error"], used_browser=True,
            )

    final_url = getattr(result, "redirected_url", None) or url
    if not getattr(result, "success", False):
        return FetchResult(
            success=False, thin=True, text="",
            http_status=getattr(result, "status_code", 0) or 0,
            final_url=final_url,
            error=getattr(result, "error_message", "") or "crawl unsuccessful",
            thin_signals=["crawl_unsuccessful"], used_browser=True,
        )

    md = (getattr(result, "markdown", "") or "")[:_MAX_BYTES]
    thin, signals = _detect_thin(md, getattr(result, "status_code", 200))
    return FetchResult(
        success=True, thin=thin, text=md,
        http_status=getattr(result, "status_code", 200) or 200,
        final_url=final_url,
        thin_signals=signals, used_browser=True,
    )


def fetch_url_with_fallback(url: str) -> FetchResult:
    """Fetch with httpx; escalate to Playwright/Crawl4AI when configured AND
    httpx is thin AND the thin signal is one a browser can plausibly fix
    (anti-bot wall, JS shell, short body). Real 4xx/5xx responses mean the URL
    is wrong or the server is down — browser launch can't help and burns time."""
    httpx_result = fetch_url(url)
    if not httpx_result.thin:
        return httpx_result
    if httpx_result.http_status >= 400:
        return httpx_result
    if not _browser_fallback_enabled():
        return httpx_result
    # Use the URL httpx already resolved through any redirect chain — saves
    # Crawl4AI from re-running the same redirects (and from any intermittent
    # response variance between httpx and the headless browser).
    target = httpx_result.final_url or url
    try:
        return asyncio.run(_crawl_with_browser(target))
    except ImportError:
        logger.warning("web_fetcher: crawl4ai not installed — returning httpx result")
        return httpx_result
