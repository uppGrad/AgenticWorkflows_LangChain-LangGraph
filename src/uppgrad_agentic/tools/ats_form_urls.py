"""Per-ATS rules for resolving the apply-form URL from a posting's overview URL.

Some ATSes split each posting across two URLs:
  - Overview: job description, marketing, "Apply" button
  - Application form: actual <form> with input fields, file uploads

Discovery surfaces the overview URL (that's what Brave indexes). To
extract form structure we need the form URL. Patterns are deterministic
per-ATS; this module encodes them.

When None is returned, the caller should treat the form as not reachable
via simple URL navigation — typically Workday-style auth walls.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse, urlunparse


# Query parameters added by upstream scrapers (LinkedIn, Indeed, etc.)
# that are pure tracker noise. Stripping them keeps fetched URLs clean
# AND prevents `_ensure_suffix` from appending the path suffix INSIDE
# the query string when the input was a path-less URL (e.g.
# `https://jobs.lever.co/.../apply?source=LinkedIn&urlHash=...`). For
# anything in this set, the value is irrelevant — we drop it entirely.
_TRACKER_QUERY_PARAMS = frozenset({
    "urlhash",       # LinkedIn
    "trid",          # SmartRecruiters/LinkedIn referrer trail
    "gh_src",        # Greenhouse source param
    "utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term",
    "source",        # generic scraper-injected
    "ref",
})


def _strip_tracker_query(parsed) -> str:
    """Drop tracker params from a parsed URL's query string. Returns the
    remaining query string (empty when all params were trackers)."""
    if not parsed.query:
        return ""
    pairs = []
    for chunk in parsed.query.split("&"):
        if not chunk:
            continue
        key, _, _ = chunk.partition("=")
        if key.lower() in _TRACKER_QUERY_PARAMS:
            continue
        pairs.append(chunk)
    return "&".join(pairs)


def _strip_tracker_from_path(path: str) -> str:
    """Strip scrape-artifact tracker params that got glued to the path
    instead of cleanly attached as a query string. Two real-world shapes:

      `/apply&urlHash=ZjNc`              (LinkedIn-scraped Lever)
      `/j/41ED129B39&urlHash=iziN`       (LinkedIn-scraped Workable)

    These can't be parsed as proper queries (no `?`) so urllib treats
    them as part of the path. Standard URL paths never contain `&`
    unencoded, so anything after the first `&` is scrape junk.

    We split on the first `&`, keep the head, and drop the tail when
    every subsequent segment matches `<tracker_key>=<value>`. If the
    tail contains anything else (defensive — `&` in a path is already
    weird), keep the original to avoid breaking unfamiliar URL shapes.
    """
    if "&" not in path:
        return path
    head, _, tail = path.partition("&")
    # tail is now `urlHash=ZjNc` or `urlHash=ZjNc&trid=...`
    for chunk in tail.split("&"):
        if not chunk:
            continue
        key, sep, _ = chunk.partition("=")
        if not sep or key.lower() not in _TRACKER_QUERY_PARAMS:
            return path  # something non-trackery in there — leave untouched
    return head


def _ensure_suffix(url: str, suffix: str) -> str:
    """Append `suffix` to `url`'s PATH (not its full string) if not already
    present. Strips upstream tracker query params (`urlHash`, `trid`,
    `utm_*`, `source`, `gh_src`, `ref`) because they:

      1. Are pure noise — no ATS form URL meaningfully depends on them.
      2. Poison naive string-append: an input like
         `https://jobs.lever.co/<co>/<id>/apply?source=LinkedIn&urlHash=zlzS`
         would otherwise become
         `https://jobs.lever.co/<co>/<id>/apply?source=LinkedIn&urlHash=zlzS/apply`
         — the suffix lands inside the query string and the ATS returns
         a blank shell.

    Naive `url.rstrip("/")` survives only when the input is a
    query-free, fragment-free URL ending at a path component — true
    only for some scraped rows. This parses the URL, works on the
    `path` field exclusively, drops tracker params from the query, and
    re-assembles. `suffix` should start with '/'.
    """
    parsed = urlparse(url)
    path = (parsed.path or "/").rstrip("/")
    path = _strip_tracker_from_path(path)
    if not path.endswith(suffix):
        path = path + suffix
    cleaned_query = _strip_tracker_query(parsed)
    return urlunparse((
        parsed.scheme, parsed.netloc, path, parsed.params,
        cleaned_query, "",  # drop fragments too
    ))


def resolve_application_form_url(overview_url: str) -> Optional[str]:
    """Return the apply-form URL for an ATS overview URL.

    Returns:
      - The form URL when there's a deterministic per-ATS rule for it.
      - The original overview URL when the ATS keeps the form on the same
        URL as the description (Greenhouse, Workable, company-direct careers).
      - None when no public form URL is reachable (Workday auth wall) or the
        input is malformed.
    """
    if not overview_url:
        return None
    try:
        parsed = urlparse(overview_url)
    except ValueError:
        return None
    host = (parsed.netloc or "").lower()
    if not host:
        return None

    # Ashby: <overview>/application
    if host == "jobs.ashbyhq.com":
        return _ensure_suffix(overview_url, "/application")

    # Lever: <overview>/apply. Lever serves regional subdomains
    # (`jobs.eu.lever.co`, `jobs.lever.co`) — match the suffix.
    if host.endswith(".lever.co") or host == "lever.co":
        return _ensure_suffix(overview_url, "/apply")

    # SmartRecruiters: <overview>/apply
    if host.endswith(".smartrecruiters.com") or host == "smartrecruiters.com":
        return _ensure_suffix(overview_url, "/apply")

    # Workable: `<listing>/apply/`. The listing URL renders metadata + an
    # "Apply for this job" button that NAVIGATES to the apply form (a
    # different route, NOT a same-page progressive disclosure). For
    # single-page Workable apply forms, every input lives on `/apply/`
    # server-rendered after hydration. Pre-resolving here lets the
    # standard fetch flow grab the form HTML directly without clicking
    # any button — passive extraction only, no submission risk. Multi-
    # step Workable variants (page-2 file upload + Q&A behind a
    # "Continue" button) are still out of scope; see ats_coverage.md.
    if host == "apply.workable.com":
        return _ensure_suffix(overview_url, "/apply") + "/"

    # Workday: auth wall. No deterministic public form URL.
    if host.endswith(".myworkdayjobs.com"):
        return None

    # Greenhouse and unknown hosts (e.g. company-direct careers pages
    # like mongodb.com/careers/jobs/...): form is on the same URL as
    # the overview, possibly rendered below the description.
    return overview_url
