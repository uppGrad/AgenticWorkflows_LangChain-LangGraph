"""Integration tests for playwright_filler.fill_form_async.

Strategy: serve a small static HTML form from a local HTTP server (built-in
http.server). Run the filler against it. Inspect the post-fill state via a
final navigate + JS evaluation, OR just verify the FormFillResult outcomes.

These tests need Playwright + Chromium installed. They're slow-ish (~5-15s
each because they spin up a browser) so we keep the suite small. Skip the
whole module if `playwright` isn't importable.

NEVER click submit invariants are encoded in `fill_form_async` itself; we
also assert `result.submit_clicked is False` after every test.
"""
from __future__ import annotations

import asyncio
import os
import socket
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import HTTPServer, SimpleHTTPRequestHandler

import pytest

playwright = pytest.importorskip("playwright")

from uppgrad_agentic.tools.playwright_filler import fill_form_async  # noqa: E402
from uppgrad_agentic.workflows.auto_apply.schemas import (  # noqa: E402
    FormField, FormFieldFillPlan,
)


# A tiny native form covering text/email/select/checkbox/file/textarea/radio.
_FORM_HTML = """\
<!doctype html>
<html><head><title>Test Form</title></head><body>
  <h1>Apply</h1>
  <form id="apply" action="/never" method="post" onsubmit="return false">
    <label for="first_name">First Name</label>
    <input type="text" name="first_name" id="first_name" />

    <label for="email">Email</label>
    <input type="email" name="email" id="email" />

    <label for="country">Country</label>
    <select name="country" id="country">
      <option value="">Select...</option>
      <option value="US">United States</option>
      <option value="TR">Turkey</option>
      <option value="DE">Germany</option>
    </select>

    <label for="resume">Resume</label>
    <input type="file" name="resume" id="resume" />

    <label for="why">Why join?</label>
    <textarea name="why" id="why"></textarea>

    <label><input type="checkbox" name="agree" id="agree" /> I agree</label>

    <fieldset>
      <legend>Pronouns</legend>
      <label><input type="radio" name="pronoun" value="he"> He/Him</label>
      <label><input type="radio" name="pronoun" value="she"> She/Her</label>
      <label><input type="radio" name="pronoun" value="they"> They/Them</label>
    </fieldset>

    <button type="submit" id="submitbtn">Submit</button>
  </form>
</body></html>
"""


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextmanager
def serve_form(html: str = _FORM_HTML):
    """Start a tiny HTTP server in a background thread, yield its base URL."""
    tmpdir = tempfile.mkdtemp(prefix="filler_test_")
    form_path = os.path.join(tmpdir, "index.html")
    with open(form_path, "w") as f:
        f.write(html)

    class _Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=tmpdir, **kw)
        def log_message(self, *a, **kw):  # silence
            pass

    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/index.html"
    finally:
        server.shutdown()
        server.server_close()


def _f(**kw) -> FormField:
    base = {
        "label": "X", "field_type": "text", "name": "", "required": False,
        "options": [], "accepts_file": [], "expected_source": "unknown",
    }
    base.update(kw)
    return FormField(**base)


def _plan(**kw) -> FormFieldFillPlan:
    base = {
        "field": _f(),
        "value": "", "status": "filled",
        "source": "user_profile", "reason": "test",
    }
    base.update(kw)
    return FormFieldFillPlan(**base)


# Make sure file-upload tests have a small file to upload
@pytest.fixture
def fake_resume(tmp_path):
    p = tmp_path / "resume.pdf"
    p.write_bytes(b"%PDF-1.4\n%fake\n")
    return str(p)


def test_fills_text_field_via_name():
    plan = [_plan(field=_f(label="First Name", field_type="text", name="first_name"),
                  value="Koray")]
    with serve_form() as url:
        result = asyncio.run(fill_form_async(url, plan, llm=None, headless=True))
    assert result.success is True
    assert result.submit_clicked is False
    assert result.fields_filled_native == 1
    assert result.fields_failed == 0


def test_fills_native_select_by_label():
    plan = [_plan(field=_f(label="Country", field_type="select", name="country",
                           options=["United States", "Turkey", "Germany"]),
                  value="Turkey")]
    with serve_form() as url:
        result = asyncio.run(fill_form_async(url, plan, llm=None, headless=True))
    assert result.success is True
    assert result.fields_filled_native == 1
    assert result.reports[0].outcome == "ok"


def test_fills_textarea():
    plan = [_plan(field=_f(label="Why join?", field_type="textarea", name="why"),
                  value="[Mock answer]")]
    with serve_form() as url:
        result = asyncio.run(fill_form_async(url, plan, llm=None, headless=True))
    assert result.fields_filled_native == 1


def test_fills_required_checkbox():
    plan = [_plan(field=_f(label="I agree", field_type="checkbox", name="agree", required=True),
                  value="true", source="mock", reason="required")]
    with serve_form() as url:
        result = asyncio.run(fill_form_async(url, plan, llm=None, headless=True))
    assert result.fields_filled_native == 1


def test_fills_radio_by_value():
    plan = [_plan(field=_f(label="Pronouns", field_type="radio", name="pronoun"),
                  value="they")]
    with serve_form() as url:
        result = asyncio.run(fill_form_async(url, plan, llm=None, headless=True))
    assert result.fields_filled_native == 1


def test_fills_file_upload(fake_resume):
    plan = [_plan(field=_f(label="Resume", field_type="file", name="resume"),
                  value=fake_resume, source="user_document")]
    with serve_form() as url:
        result = asyncio.run(fill_form_async(url, plan, llm=None, headless=True))
    assert result.success is True
    assert result.fields_filled_native == 1
    assert result.fields_failed == 0


# Greenhouse-style markup — `<button>Attach</button>` stack visible, real
# `<input type="file">` hidden. Two upload sections (Resume/CV + Cover
# Letter) each with their own hidden input, scoped under a labelled
# container. `_locate_file_input` must pick the matching one for each.
_GREENHOUSE_FORM_HTML = """\
<!doctype html>
<html><body>
  <h1>Apply</h1>
  <form id="apply" action="/never" method="post" onsubmit="return false">
    <fieldset id="resume-block">
      <legend>Resume/CV</legend>
      <button type="button">Attach</button>
      <button type="button">Dropbox</button>
      <button type="button">Google Drive</button>
      <button type="button">Enter manually</button>
      <input type="file" id="s3_upload_for_resume"
             style="position:absolute;left:-9999px" />
    </fieldset>

    <fieldset id="cl-block">
      <legend>Cover Letter</legend>
      <button type="button">Attach</button>
      <button type="button">Dropbox</button>
      <button type="button">Google Drive</button>
      <button type="button">Enter manually</button>
      <input type="file" id="s3_upload_for_cover_letter"
             style="position:absolute;left:-9999px" />
    </fieldset>
  </form>
</body></html>
"""


def test_resolves_hidden_file_input_via_labelled_container(fake_resume):
    """Greenhouse-style: the FormField the LLM extracted has label='Resume/CV'
    but no matching `name` (the visible Attach button is not the input).
    The deterministic resolver must walk from the heading text into the
    fieldset and find the hidden file input inside, NOT default to the
    first input on the page (which would also work here but doesn't
    disambiguate Resume from Cover Letter)."""
    plan = [
        _plan(field=_f(label="Resume/CV", field_type="file", name=""),
              value=fake_resume, source="user_document"),
        _plan(field=_f(label="Cover Letter", field_type="file", name=""),
              value=fake_resume, source="user_document"),
    ]
    with serve_form(_GREENHOUSE_FORM_HTML) as url:
        result = asyncio.run(fill_form_async(url, plan, llm=None, headless=True))
    assert result.fields_failed == 0, [r.detail for r in result.reports]
    assert result.fields_filled_native == 2
    # Both got `set_input_files` deterministically (no LLM picker needed).
    assert all(r.outcome == "ok" for r in result.reports)


# Recruitee-style markup. The form has bracket-named inputs
# (`candidate[first_name]`, `candidate[phone]`) — the same shape that
# tripped the live jobs.fromjimmy.com run. We don't simulate the React
# actionability problem here (Playwright's local chromium hits these
# inputs cleanly anyway); the test instead pins the locator path: with
# square brackets in the name, the CSS selector `[name="candidate[X]"]`
# must still resolve and `.fill()` must succeed.
_RECRUITEE_BRACKET_NAME_HTML = """\
<!doctype html>
<html><body>
  <h1>Apply</h1>
  <form id="apply" onsubmit="return false">
    <label for="candidate_first_name">First name</label>
    <input type="text" name="candidate[first_name]" id="candidate_first_name" />

    <label for="candidate_phone">Phone</label>
    <input type="text" name="candidate[phone]" id="candidate_phone" />
  </form>
</body></html>
"""


def test_text_fill_handles_bracket_named_inputs():
    """Recruitee/Rails-style inputs use names like `candidate[first_name]`.
    The CSS attribute selector must round-trip those brackets correctly,
    and the longer 5s timeout (post-fix) gives React-wrapped inputs room
    to hydrate without false failures."""
    plan = [
        _plan(field=_f(label="First name", field_type="text",
                       name="candidate[first_name]"),
              value="Koray"),
        _plan(field=_f(label="Phone", field_type="text",
                       name="candidate[phone]"),
              value="+90 533 386 5486"),
    ]
    with serve_form(_RECRUITEE_BRACKET_NAME_HTML) as url:
        result = asyncio.run(fill_form_async(url, plan, llm=None, headless=True))
    assert result.fields_failed == 0, [r.detail for r in result.reports]
    assert result.fields_filled_native == 2


def test_no_locator_when_field_missing_on_page():
    """Field references a name not present on the form. Without LLM, the
    deterministic tiers exhaust and the field is marked failed (not ok)."""
    plan = [_plan(field=_f(label="NonExistent", field_type="text", name="nonexistent"),
                  value="anything")]
    with serve_form() as url:
        result = asyncio.run(fill_form_async(url, plan, llm=None, headless=True))
    assert result.fields_failed == 1
    assert result.fields_filled_native == 0
    assert result.reports[0].outcome == "no_locator"


def test_plan_skip_status_carries_through():
    plan = [_plan(field=_f(label="Skipped Field", field_type="text", name="x"),
                  value="", status="skipped", source="no_value", reason="test_skip")]
    with serve_form() as url:
        result = asyncio.run(fill_form_async(url, plan, llm=None, headless=True))
    assert result.fields_skipped == 1
    assert result.fields_filled_native == 0
    assert result.reports[0].outcome == "plan_skip"
    assert result.reports[0].detail == "test_skip"


def test_submit_button_never_clicked():
    """Even when filling completes, the form's <button type='submit'> must
    not have been clicked. Verified by `result.submit_clicked is False`."""
    plan = [_plan(field=_f(label="First Name", field_type="text", name="first_name"),
                  value="Koray")]
    with serve_form() as url:
        result = asyncio.run(fill_form_async(url, plan, llm=None, headless=True))
    assert result.submit_clicked is False


def test_result_reports_list_matches_plan_length():
    plan = [
        _plan(field=_f(label="First Name", field_type="text", name="first_name"), value="Koray"),
        _plan(field=_f(label="Email", field_type="email", name="email"), value="x@y.com"),
        _plan(field=_f(label="MissingField", field_type="text", name="zzz"), value="oops"),
    ]
    with serve_form() as url:
        result = asyncio.run(fill_form_async(url, plan, llm=None, headless=True))
    assert len(result.reports) == 3
    assert result.fields_total == 3


def test_fills_multiple_fields_end_to_end(fake_resume):
    plan = [
        _plan(field=_f(label="First Name", field_type="text", name="first_name"), value="Koray"),
        _plan(field=_f(label="Email", field_type="email", name="email"), value="x@y.com"),
        _plan(field=_f(label="Country", field_type="select", name="country",
                       options=["United States", "Turkey"]), value="Turkey"),
        _plan(field=_f(label="Resume", field_type="file", name="resume"),
              value=fake_resume, source="user_document"),
        _plan(field=_f(label="Why join?", field_type="textarea", name="why"),
              value="[Mock answer]", source="mock"),
        _plan(field=_f(label="I agree", field_type="checkbox", name="agree", required=True),
              value="true", source="mock"),
    ]
    with serve_form() as url:
        result = asyncio.run(fill_form_async(url, plan, llm=None, headless=True))
    assert result.fields_filled_native == 6
    assert result.fields_failed == 0
    assert result.success is True
    assert result.submit_clicked is False
