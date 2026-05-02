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
from urllib.parse import urlparse


def _ensure_suffix(url: str, suffix: str) -> str:
    """Append `suffix` to `url`'s path if not already present. `suffix` should
    start with '/'."""
    stripped = url.rstrip("/")
    if stripped.endswith(suffix):
        return stripped
    return stripped + suffix


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

    # Lever: <overview>/apply
    if host == "jobs.lever.co":
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
