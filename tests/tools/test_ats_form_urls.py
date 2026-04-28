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


def test_workable_unchanged():
    """Workable also keeps form on the same URL."""
    url = "https://apply.workable.com/intellecthq/j/539DCA61EB"
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
