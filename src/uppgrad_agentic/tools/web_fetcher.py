from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0
_MAX_BYTES = 500_000

# Heuristics for "this httpx response is not a real apply page"
_THIN_KEYWORDS = [
    "404", "page not found", "access denied",
    "javascript required", "enable javascript",
    "cloudflare", "robot", "captcha", "challenge-platform",
    "please verify you are human",
]
_MIN_BODY_BYTES = 500


@dataclass
class FetchResult:
    success: bool                    # HTTP fetch returned 2xx
    thin: bool                       # Content looks like anti-bot, JS shell, short, or 4xx
    text: str                        # Body (HTML), truncated to _MAX_BYTES
    http_status: int
    error: str = ""
    thin_signals: List[str] = field(default_factory=list)
    used_browser: bool = False       # True when we escalated to Playwright/Crawl4AI


def _detect_thin(text: str, status: int) -> tuple[bool, List[str]]:
    if status >= 400:
        return True, [f"http_status={status}"]
    if len(text.strip()) < _MIN_BODY_BYTES:
        return True, [f"body_len={len(text)}"]
    lowered = text.lower()
    hits = [kw for kw in _THIN_KEYWORDS if kw in lowered]
    if len(hits) >= 2:
        return True, hits
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
            http_status=0, error=str(exc),
            thin_signals=["network_error"],
        )

    text = resp.text[:_MAX_BYTES]
    thin, signals = _detect_thin(text, resp.status_code)

    return FetchResult(
        success=resp.status_code < 400,
        thin=thin,
        text=text,
        http_status=resp.status_code,
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


async def _crawl_with_browser(url: str, timeout_seconds: float = 25.0) -> FetchResult:
    """Use Crawl4AI / Playwright to fetch a URL when httpx returned thin content."""
    crawler = _build_async_crawler()  # may raise ImportError; caller handles

    async with crawler:
        try:
            result = await crawler.arun(url=url, page_timeout=int(timeout_seconds * 1000))
        except Exception as exc:
            logger.warning("web_fetcher: crawl4ai error for %s — %s", url, exc)
            return FetchResult(
                success=False, thin=True, text="",
                http_status=0, error=str(exc),
                thin_signals=["browser_error"], used_browser=True,
            )

    if not getattr(result, "success", False):
        return FetchResult(
            success=False, thin=True, text="",
            http_status=getattr(result, "status_code", 0) or 0,
            error=getattr(result, "error_message", "") or "crawl unsuccessful",
            thin_signals=["crawl_unsuccessful"], used_browser=True,
        )

    md = (getattr(result, "markdown", "") or "")[:_MAX_BYTES]
    thin, signals = _detect_thin(md, getattr(result, "status_code", 200))
    return FetchResult(
        success=True, thin=thin, text=md,
        http_status=getattr(result, "status_code", 200) or 200,
        thin_signals=signals, used_browser=True,
    )


def fetch_url_with_fallback(url: str) -> FetchResult:
    """Fetch with httpx; escalate to Playwright/Crawl4AI when configured AND httpx is thin."""
    httpx_result = fetch_url(url)
    if not httpx_result.thin:
        return httpx_result
    if not _browser_fallback_enabled():
        return httpx_result
    try:
        return asyncio.run(_crawl_with_browser(url))
    except ImportError:
        logger.warning("web_fetcher: crawl4ai not installed — returning httpx result")
        return httpx_result
