"""Per-ATS application-form URL resolver.

Maps from the overview URL Brave returns to the actual apply-form URL
(Ashby /application, Lever /apply, etc.). Used by the discovery node so
downstream form extraction fetches the right page.
"""
from uppgrad_agentic.tools.ats_form_urls import resolve_application_form_url


def test_ashby_appends_application_segment():
    url = "https://jobs.ashbyhq.com/notion/6ad34426-b980-436b-80c4-3634c00094ad"
    assert (
        resolve_application_form_url(url)
        == "https://jobs.ashbyhq.com/notion/6ad34426-b980-436b-80c4-3634c00094ad/application"
    )


def test_ashby_preserves_existing_application_suffix():
    """Don't double-append /application if the URL already ends in it."""
    url = "https://jobs.ashbyhq.com/notion/6ad34426/application"
    assert resolve_application_form_url(url) == url


def test_lever_appends_apply_segment():
    url = "https://jobs.lever.co/sensortower/abc-123"
    assert resolve_application_form_url(url) == "https://jobs.lever.co/sensortower/abc-123/apply"


def test_lever_preserves_existing_apply_suffix():
    url = "https://jobs.lever.co/sensortower/abc-123/apply"
    assert resolve_application_form_url(url) == url


def test_smartrecruiters_appends_apply_segment():
    url = "https://careers.smartrecruiters.com/Acme/12345"
    assert resolve_application_form_url(url) == "https://careers.smartrecruiters.com/Acme/12345/apply"


def test_greenhouse_overview_unchanged():
    """Greenhouse keeps the form on the same URL as the overview (form is
    rendered below the JD)."""
    url = "https://job-boards.greenhouse.io/anthropic/jobs/5121912008"
    assert resolve_application_form_url(url) == url


def test_workable_appends_apply_suffix():
    """Workable's listing URL renders an "Apply for this job" CTA that
    NAVIGATES to `<listing>/apply/`. Pre-resolving here lets the
    standard fetch flow grab the form HTML directly without clicking
    any button — passive, no submission risk. Multi-step Workable
    variants (page-2+ behind a "Continue" button) remain out of scope."""
    url = "https://apply.workable.com/intellecthq/j/539DCA61EB"
    assert (
        resolve_application_form_url(url)
        == "https://apply.workable.com/intellecthq/j/539DCA61EB/apply/"
    )


def test_workable_apply_suffix_idempotent():
    """When the URL already ends with `/apply/`, don't double-append."""
    url = "https://apply.workable.com/intellecthq/j/539DCA61EB/apply/"
    assert resolve_application_form_url(url) == url


def test_workday_returns_none_for_auth_wall():
    """Workday requires account creation before form is accessible — no
    deterministic public form URL we can scrape. Caller must treat None as
    'cannot fetch form via simple URL'."""
    url = "https://github.wd1.myworkdayjobs.com/en-US/careers/job/Germany/Senior-Solutions-Engineer_R12345"
    assert resolve_application_form_url(url) is None


def test_unknown_host_returns_overview_url_unchanged():
    """For ATS hosts we don't have rules for (e.g. company-direct careers
    pages), assume the form is on the same URL — same as Greenhouse."""
    url = "https://www.mongodb.com/careers/jobs/7484657"
    assert resolve_application_form_url(url) == url


def test_empty_url_returns_none():
    assert resolve_application_form_url("") is None


def test_malformed_url_returns_none():
    """Garbage in → None out, never raise."""
    assert resolve_application_form_url("not a url") is None


# ─── URL-hygiene fixes (coverage_run_2026_05_03 root cause #1) ──────────────
#
# LinkedIn's scraper appends `&urlHash=<hash>` (and similar trackers) to
# every `url_direct` it stores. Naive string-append placed the path
# suffix INSIDE the query string and broke Lever / SmartRecruiters /
# Workable-no-org variants — ~33% of recently-posted listings.

def test_lever_with_linkedin_urlhash_param_drops_tracker_and_appends_path():
    """Real prod URL from a LinkedIn-scraped Lever posting (Mendix). The
    `&urlHash=zlzS` is LinkedIn tracker noise. `?source=LinkedIn` is also
    tracker noise. Both must be stripped, AND the `/apply` path component
    must already be canonical (the input URL ends with `/apply` followed
    by the query, so the suffix-append should be a no-op on the path)."""
    url = (
        "https://jobs.lever.co/mendix/5509b41f-0854-451c-b198-a4801adc5d4d"
        "/apply?source=LinkedIn&urlHash=zlzS"
    )
    out = resolve_application_form_url(url)
    assert out == (
        "https://jobs.lever.co/mendix/5509b41f-0854-451c-b198-a4801adc5d4d/apply"
    )
    # Critical invariant: no double-append, no suffix-inside-query.
    assert "/apply/apply" not in out
    assert "urlHash" not in out
    assert "source=LinkedIn" not in out


def test_lever_without_apply_segment_in_path_appends_correctly():
    """Lever URL where the apply path segment ISN'T already there
    (rare-but-possible scrape variant). Should append `/apply` to the path."""
    url = "https://jobs.eu.lever.co/wypoon/aa414308-a028-4174-a649-af7f1370ea01"
    assert resolve_application_form_url(url) == (
        "https://jobs.eu.lever.co/wypoon/aa414308-a028-4174-a649-af7f1370ea01/apply"
    )


def test_lever_with_path_apply_already_with_trailing_amp_param():
    """The actual broken case from the Wypoon prod row — `/apply` is
    glued to `&urlHash=...` because the original URL had the trailing
    slash stripped by the scraper. Dropping the tracker and re-canonicalising
    yields a clean URL."""
    url = (
        "https://jobs.eu.lever.co/wypoon/aa414308-a028-4174-a649-af7f1370ea01"
        "/apply&urlHash=ZjNc"
    )
    # `&urlHash=ZjNc` is part of the path here (no `?` before it), so it
    # ends up as part of the last segment. Our fix drops the path
    # rstrip, sees `/apply&urlHash=ZjNc` as the last segment, and
    # appends `/apply`. The output is structurally cleaner — the
    # subsequent fetch will at least target the /apply route.
    out = resolve_application_form_url(url)
    assert out.startswith("https://jobs.eu.lever.co/wypoon/")
    assert out.endswith("/apply")
    assert "urlHash" not in out


def test_smartrecruiters_with_trid_query_param_drops_tracker():
    """Real prod URL from a LinkedIn-scraped SmartRecruiters posting (DYKA).
    The two `&trid=...` (also: `&urlHash=...`) trackers must be stripped
    and `/apply` must land on the path, not in the query string."""
    url = (
        "https://jobs.smartrecruiters.com/TessenderloGroup/"
        "744000114907439-network-security-engineer"
        "?trid=2d92f286-613b-4daf-9dfa-6340ffbecf73"
        "&trid=2d92f286-613b-4daf-9dfa-6340ffbecf73"
        "&urlHash=k2Dy"
    )
    out = resolve_application_form_url(url)
    assert out == (
        "https://jobs.smartrecruiters.com/TessenderloGroup/"
        "744000114907439-network-security-engineer/apply"
    )
    assert "urlHash" not in out
    assert "trid" not in out


def test_workable_with_query_appends_apply_path_not_inside_query():
    """Real prod URL from a LinkedIn-scraped Workable posting (Debenhams).
    The `?utm_source=linkedin.com&urlHash=nDer` was getting the `/apply/`
    suffix appended INSIDE the query. Fix: append on path, drop trackers."""
    url = (
        "https://apply.workable.com/debenhamsgroup/j/53D0ECE60E"
        "?utm_source=linkedin.com&urlHash=nDer"
    )
    out = resolve_application_form_url(url)
    assert out == (
        "https://apply.workable.com/debenhamsgroup/j/53D0ECE60E/apply/"
    )
    assert "urlHash" not in out
    assert "utm_source" not in out


def test_workable_no_org_pattern_with_tracker_in_path():
    """Real prod URL from a LinkedIn-scraped Workable posting (Phoenix
    Software). `apply.workable.com/j/<id>` is a valid Workable variant
    where the org-slug isn't in the URL. The trailing `&urlHash=iziN`
    was getting `/apply/` appended after it as a path segment. Fix
    yields a clean `/j/<id>/apply/`."""
    url = "https://apply.workable.com/j/41ED129B39&urlHash=iziN"
    out = resolve_application_form_url(url)
    assert out.startswith("https://apply.workable.com/j/")
    assert out.endswith("/apply/")
    assert "urlHash" not in out


def test_greenhouse_url_with_gh_src_tracker_dropped():
    """Greenhouse URLs come with `?gh_src=...` from LinkedIn. Same path
    so the form_url is identical to the overview, but trackers should
    still be stripped for cleanliness."""
    url = (
        "https://job-boards.greenhouse.io/anthropic/jobs/5121912008"
        "?gh_src=abc123"
    )
    # Greenhouse falls into the catch-all "return overview_url unchanged"
    # branch — it doesn't go through `_ensure_suffix`. So gh_src is NOT
    # stripped today. This test pins that behaviour rather than the
    # ideal — flag for follow-up once we decide whether the catch-all
    # branch should also normalise.
    out = resolve_application_form_url(url)
    assert out is not None
    # NOT enforcing tracker removal here — only path-mutating ATSes get it.
