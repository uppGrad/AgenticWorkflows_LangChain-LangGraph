"""Pull the application form area out of rendered HTML and strip noise so
an LLM can extract structured form-field info without burning tokens on
scripts, styles, analytics, or page chrome.

Strategy:
1. If a `<form>` element exists with at least one input descendant, return
   that subtree. Cheap, works for Greenhouse / Lever / SmartRecruiters /
   anything that uses native form markup.
2. Otherwise, fall back to the entire `<body>` with `<script>`, `<style>`,
   `<meta>`, `<link>`, `<head>` stripped. Modern React-driven ATSes (Ashby,
   newer Workday) don't use native `<form>` — they put the inputs in plain
   `<div>`s and submit via fetch. The body still contains the
   `<input>/<select>/<textarea>` tags the LLM needs.

Used by the form-extraction node downstream of `_crawl_with_browser`. The
output is HTML (not markdown) — input/select/textarea/option tags are
preserved so the LLM can read field types, names, required flags, and
dropdown option lists.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag

# Hosts of third-party ATSes that companies sometimes embed via iframe on
# their own careers pages (MongoDB→Greenhouse, Stripe→Greenhouse, etc.). When
# the parent page has no inline form, we follow the first iframe whose src
# points at one of these — the form lives there.
_ATS_IFRAME_HOSTS = (
    "greenhouse.io",
    "job-boards.greenhouse.io",
    "boards.greenhouse.io",
    "jobs.lever.co",
    "lever.co",
    "jobs.ashbyhq.com",
    "ashbyhq.com",
    "apply.workable.com",
    "workable.com",
    "smartrecruiters.com",
    "myworkdayjobs.com",
    "bamboohr.com",
    "jobvite.com",
    "recruitee.com",
)

# Tags that add noise without structural value for form extraction.
_STRIP_TAGS = ("script", "style", "meta", "link", "noscript")


def _strip_noise(node: Tag) -> None:
    """Remove `<script>/<style>/<meta>/<link>/<noscript>` and hidden inputs
    in-place. These add tokens without structural value for form extraction."""
    for tag_name in _STRIP_TAGS:
        for el in node.find_all(tag_name):
            el.decompose()
    for hidden in node.find_all("input", attrs={"type": "hidden"}):
        hidden.decompose()


def extract_form_html(html: str) -> str:
    """Return the application form's HTML, cleaned. '' when neither a `<form>`
    nor any form-like input cluster is found in the page.

    When multiple `<form>` elements exist (search bar + apply form, etc.),
    pick the one with the most input descendants — empirically the
    application form.
    """
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return ""

    # Strategy 1: a real <form> element with inputs.
    forms = soup.find_all("form")

    def _score(node: Tag) -> int:
        return len(node.find_all(("input", "select", "textarea")))

    if forms:
        form = max(forms, key=_score)
        if _score(form) > 0:
            _strip_noise(form)
            return str(form)

    # Strategy 2: no <form> tag (modern React-driven ATSes). Fall back to
    # the body with noise stripped. The body still contains the input/
    # select/textarea tags the LLM needs to identify the form.
    body = soup.body
    if body is None or _score(body) == 0:
        return ""
    _strip_noise(body)
    return str(body)


def extract_ats_iframe_src(html: str) -> Optional[str]:
    """Find the src of the first iframe that points at a known ATS host.

    Some careers pages (mongodb.com/careers/<id>, similar company-direct
    sites) host the apply form inside a cross-origin iframe served by an
    ATS like Greenhouse. Our normal extraction can't see inside cross-origin
    iframes; we instead follow them by fetching the iframe's src directly.

    Returns None when no ATS iframe is present.
    """
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return None
    for iframe in soup.find_all("iframe"):
        src = (iframe.get("src") or "").strip()
        if not src or src.startswith("about:") or src.startswith("javascript:"):
            continue
        try:
            host = (urlparse(src).netloc or "").lower()
        except ValueError:
            continue
        if not host:
            continue
        # Match against known ATS hosts (with optional subdomain prefix).
        if any(host == ats or host.endswith("." + ats) for ats in _ATS_IFRAME_HOSTS):
            return src
    return None
