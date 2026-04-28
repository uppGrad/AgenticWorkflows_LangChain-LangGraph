"""Form-field extraction node — feeds the application-form HTML to an LLM
and emits a structured `List[FormField]` capturing every input on the form.
"""
from unittest.mock import MagicMock, patch

from uppgrad_agentic.workflows.auto_apply.nodes.extract_form_fields import (
    extract_form_fields,
)
from uppgrad_agentic.workflows.auto_apply.schemas import FormField, FormSchema


def _state_with_html(
    form_url: str | None, raw_html: str, overview_url: str | None = None,
) -> dict:
    """Build a minimal state where the previous nodes already populated the
    form URL + raw HTML. By default overview_url == form_url (Greenhouse case)
    so the node uses in-state raw_html instead of re-fetching."""
    return {
        "opportunity_type": "job",
        "opportunity_id": "1",
        "opportunity_data": {"id": 1},
        "discovered_apply_url": overview_url if overview_url is not None else form_url,
        "discovered_form_url": form_url,
        "discovered_raw_html": raw_html,
        "scraped_requirements": {"source": form_url or ""},
        "result": {},
    }


def test_skips_non_job_opportunity_types():
    """Form extraction is a job-only concern — programs/scholarships have no
    apply form to drive."""
    state = {"opportunity_type": "masters", "opportunity_id": "1",
             "discovered_form_url": "https://x", "result": {}}
    out = extract_form_fields(state)
    # Should produce step tracking but no form_fields write.
    assert out.get("current_step") == "extract_form_fields"
    assert "form_fields" not in out


def test_skips_when_no_form_url():
    """When discovery couldn't resolve a form URL (Workday auth wall, failed
    discovery), the node short-circuits with empty fields list."""
    state = _state_with_html(form_url=None, raw_html="<html></html>")
    out = extract_form_fields(state)
    assert out.get("form_fields") == []


def test_extracts_fields_when_form_present(monkeypatch):
    """Happy path: state has rendered HTML containing a <form>, the LLM is
    configured, and the node returns a structured FormField list."""
    html = """
    <html><body><form>
      <label>Resume</label><input type="file" name="resume" required>
      <label>Cover Letter</label><input type="file" name="cover">
      <label>Why us?</label><textarea name="why" required></textarea>
    </form></body></html>
    """
    state = _state_with_html(form_url="https://x.example.com/apply", raw_html=html)

    fake_schema = FormSchema(
        fields=[
            FormField(label="Resume", field_type="file", name="resume",
                      required=True, accepts_file=[".pdf", ".docx"],
                      expected_source="user_document"),
            FormField(label="Cover Letter", field_type="file", name="cover",
                      required=False, expected_source="user_document"),
            FormField(label="Why us?", field_type="textarea", name="why",
                      required=True, expected_source="user_answer"),
        ],
    )
    fake_structured = MagicMock()
    fake_structured.invoke.return_value = fake_schema
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured

    with patch("uppgrad_agentic.workflows.auto_apply.nodes.extract_form_fields.get_llm",
               return_value=fake_llm):
        out = extract_form_fields(state)

    fields = out.get("form_fields") or []
    assert len(fields) == 3
    labels = [f["label"] for f in fields]
    assert labels == ["Resume", "Cover Letter", "Why us?"]
    types = {f["field_type"] for f in fields}
    assert types == {"file", "textarea"}


def test_skips_when_html_has_no_form(monkeypatch):
    """raw_html present but no <form> element (description-only page) → empty
    list, no LLM call."""
    state = _state_with_html(
        form_url="https://x.example.com/apply",
        raw_html="<html><body><h1>Apply via email</h1><p>Send your CV.</p></body></html>",
    )
    fake_llm = MagicMock()
    with patch("uppgrad_agentic.workflows.auto_apply.nodes.extract_form_fields.get_llm",
               return_value=fake_llm):
        out = extract_form_fields(state)
    assert out.get("form_fields") == []
    fake_llm.with_structured_output.assert_not_called()


def test_returns_empty_list_when_llm_unavailable():
    """No LLM configured → can't classify field types reliably → return [].
    Form extraction has no useful heuristic fallback."""
    html = "<form><input type='file' name='resume' required></form>"
    state = _state_with_html(form_url="https://x", raw_html=html)
    with patch("uppgrad_agentic.workflows.auto_apply.nodes.extract_form_fields.get_llm",
               return_value=None):
        out = extract_form_fields(state)
    assert out.get("form_fields") == []


def test_short_circuits_on_prior_error():
    state = _state_with_html(form_url="https://x", raw_html="<form></form>")
    state["result"] = {"status": "error", "error_code": "WHATEVER"}
    out = extract_form_fields(state)
    assert "form_fields" not in out


def test_follows_ats_iframe_when_form_not_in_parent_page(monkeypatch):
    """Tier 3: company-direct careers pages (e.g. mongodb.com/careers/<id>)
    sometimes embed the apply form in a cross-origin Greenhouse iframe. After
    our other tiers fail to find a form, we follow the iframe src and extract
    from there."""
    overview_url = "https://www.mongodb.com/careers/jobs/7484657"
    parent_html_no_form = """
    <html><body>
      <h1>Senior Software Engineer</h1>
      <p>Job description</p>
      <iframe id="grnhse_iframe"
              src="https://job-boards.greenhouse.io/embed/job_app?for=mongodb&token=7484657">
      </iframe>
    </body></html>
    """
    iframe_html_with_form = """
    <html><body><form>
      <label>First Name</label><input type="text" name="first_name" required>
      <label>Resume</label><input type="file" name="resume" required>
      <label>Why MongoDB?</label><textarea name="why" required></textarea>
    </form></body></html>
    """
    state = _state_with_html(
        form_url=overview_url, overview_url=overview_url,
        raw_html=parent_html_no_form,
    )

    from uppgrad_agentic.tools.web_fetcher import FetchResult

    def fake_fetch(url: str) -> FetchResult:
        # Should be called for the iframe src.
        assert "greenhouse.io/embed" in url, f"unexpected fetch URL: {url}"
        return FetchResult(
            success=True, thin=False, text="md",
            http_status=200, raw_html=iframe_html_with_form,
        )

    fake_schema = FormSchema(fields=[
        FormField(label="First Name", field_type="text", name="first_name",
                  required=True, expected_source="user_profile"),
        FormField(label="Resume", field_type="file", name="resume",
                  required=True, expected_source="user_document"),
        FormField(label="Why MongoDB?", field_type="textarea", name="why",
                  required=True, expected_source="user_answer"),
    ])
    fake_structured = MagicMock()
    fake_structured.invoke.return_value = fake_schema
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured

    with patch("uppgrad_agentic.workflows.auto_apply.nodes.extract_form_fields.fetch_url_with_fallback",
               side_effect=fake_fetch), \
         patch("uppgrad_agentic.workflows.auto_apply.nodes.extract_form_fields.force_browser_fetch",
               return_value=None), \
         patch("uppgrad_agentic.workflows.auto_apply.nodes.extract_form_fields.get_llm",
               return_value=fake_llm):
        out = extract_form_fields(state)

    fields = out.get("form_fields") or []
    assert len(fields) == 3
    assert {f["name"] for f in fields} == {"first_name", "resume", "why"}


def test_fetches_form_url_when_it_differs_from_overview(monkeypatch):
    """Ashby/Lever case: overview URL has the JD only; form lives on a
    sibling URL. Node must fetch the form URL with the browser fallback
    and use ITS rendered HTML for extraction."""
    overview_url = "https://jobs.ashbyhq.com/notion/abc-123"
    form_url = "https://jobs.ashbyhq.com/notion/abc-123/application"

    # State: overview HTML (no form), form_url set to a different URL.
    state = _state_with_html(
        form_url=form_url,
        raw_html="<html><body><h1>Solutions Engineer</h1><p>JD only.</p></body></html>",
        overview_url=overview_url,
    )

    # Patch the fetcher to return form-page HTML when the node fetches form_url.
    form_page_html = """
    <html><body><form>
      <label>Resume</label><input type="file" name="resume" required>
      <label>LinkedIn URL</label><input type="url" name="linkedin">
    </form></body></html>
    """
    from uppgrad_agentic.tools.web_fetcher import FetchResult

    def fake_fetch(url: str) -> FetchResult:
        assert url == form_url, f"expected form URL, got {url}"
        return FetchResult(
            success=True, thin=False, text="markdown",
            http_status=200, raw_html=form_page_html,
        )

    fake_schema = FormSchema(
        fields=[
            FormField(label="Resume", field_type="file", name="resume",
                      required=True, expected_source="user_document"),
            FormField(label="LinkedIn URL", field_type="url", name="linkedin",
                      required=False, expected_source="user_profile"),
        ],
    )
    fake_structured = MagicMock()
    fake_structured.invoke.return_value = fake_schema
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured

    with patch("uppgrad_agentic.workflows.auto_apply.nodes.extract_form_fields.fetch_url_with_fallback",
               side_effect=fake_fetch), \
         patch("uppgrad_agentic.workflows.auto_apply.nodes.extract_form_fields.get_llm",
               return_value=fake_llm):
        out = extract_form_fields(state)

    fields = out.get("form_fields") or []
    assert len(fields) == 2
    assert fields[0]["name"] == "resume"
    assert fields[1]["name"] == "linkedin"
