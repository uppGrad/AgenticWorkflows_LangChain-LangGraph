"""Unit tests for value_planner.compute_form_values — pure function."""
from datetime import date

from uppgrad_agentic.tools.value_planner import (
    compute_form_values, plan_field_value, _first_real_option,
)
from uppgrad_agentic.workflows.auto_apply.schemas import FormField


_PROFILE = {
    "first_name": "Koray", "last_name": "Sevil", "full_name": "Koray Sevil",
    "email": "koraysevil@gmail.com", "phone": "+90 555 1234567",
    "country": "Turkey", "city": "Istanbul", "location": "Istanbul, Turkey",
    "linkedin": "https://www.linkedin.com/in/koraysevil",
    "github": "https://github.com/koraysevil",
    "website": "https://koraysevil.com",
}


def _f(**kwargs) -> FormField:
    base = {
        "label": "X", "field_type": "text", "name": "", "required": False,
        "options": [], "accepts_file": [], "expected_source": "unknown",
    }
    base.update(kwargs)
    return FormField(**base)


# ─── Profile lookup paths ────────────────────────────────────────────────────

def test_first_name_filled_from_profile():
    p = plan_field_value(_f(label="First Name", field_type="text"), _PROFILE, {}, {})
    assert p.status == "filled"
    assert p.source == "user_profile"
    assert p.value == "Koray"


def test_email_filled_from_profile():
    p = plan_field_value(_f(label="Email", field_type="email"), _PROFILE, {}, {})
    assert p.value == "koraysevil@gmail.com"
    assert p.source == "user_profile"


def test_country_dropdown_uses_profile_when_label_matches():
    """Even though it's a select, profile lookup happens first when the label
    is a profile key. Custom-select-pick layer downstream matches the option."""
    p = plan_field_value(
        _f(label="Country", field_type="select", options=["United States", "Turkey", "Germany"]),
        _PROFILE, {}, {},
    )
    assert p.value == "Turkey"
    assert p.source == "user_profile"


# ─── File upload paths ───────────────────────────────────────────────────────

def test_file_resume_with_path_in_documents():
    docs = {"CV": {"file_path": "/tmp/koray_cv.pdf", "content": "..."}}
    p = plan_field_value(_f(label="Resume/CV", field_type="file"), _PROFILE, docs, {})
    assert p.status == "filled"
    assert p.source == "user_document"
    assert p.value == "/tmp/koray_cv.pdf"


def test_file_with_no_path_emits_doc_sentinel_for_adapter():
    """When the doc exists as text but has no on-disk path yet, value carries
    a sentinel '<<doc:KEY>>' so the backend adapter can resolve it to a tmpfile
    using ReportLab before invoking the filler."""
    docs = {"CV": {"content": "Long CV text..."}}
    p = plan_field_value(_f(label="Resume", field_type="file"), _PROFILE, docs, {})
    assert p.status == "filled"
    assert p.source == "user_document"
    assert p.value == "<<doc:CV>>"


def test_file_with_no_matching_document_skips():
    p = plan_field_value(_f(label="Resume", field_type="file"), _PROFILE, {}, {})
    assert p.status == "skipped"
    assert p.source == "no_value"


def test_cover_letter_resolves_to_cover_letter_doc():
    docs = {"Cover Letter": {"file_path": "/tmp/cover.pdf"}}
    p = plan_field_value(_f(label="Cover Letter", field_type="file"), _PROFILE, docs, {})
    assert p.value == "/tmp/cover.pdf"
    assert p.source == "user_document"


# ─── Date / select / radio / checkbox ────────────────────────────────────────

def test_date_field_uses_today():
    p = plan_field_value(_f(label="Start Date", field_type="date"), _PROFILE, {}, {})
    assert p.status == "filled"
    assert p.source == "computed"
    assert p.value == date.today().isoformat()


def test_select_picks_first_real_option_when_no_profile_match():
    p = plan_field_value(
        _f(label="Visa Sponsorship", field_type="select",
           options=["Select...", "Yes", "No"]),
        _PROFILE, {}, {},
    )
    assert p.status == "filled"
    assert p.source == "mock"
    assert p.value == "Yes"


def test_select_skips_when_only_placeholder_options():
    p = plan_field_value(
        _f(label="Some Question", field_type="select",
           options=["Choose…", "Select an option"]),
        _PROFILE, {}, {},
    )
    assert p.status == "skipped"
    assert p.reason == "no_real_option"


def test_radio_picks_first_real_option():
    p = plan_field_value(
        _f(label="Pronouns", field_type="radio",
           options=["He/Him", "She/Her", "They/Them"]),
        _PROFILE, {}, {},
    )
    assert p.status == "filled"
    assert p.value == "He/Him"


def test_required_checkbox_filled():
    p = plan_field_value(_f(label="I agree to terms", field_type="checkbox", required=True),
                         _PROFILE, {}, {})
    assert p.status == "filled"
    assert p.value == "true"


def test_optional_checkbox_skipped():
    p = plan_field_value(_f(label="Subscribe to newsletter", field_type="checkbox", required=False),
                         _PROFILE, {}, {})
    assert p.status == "skipped"


# ─── Free-text / mock answer ─────────────────────────────────────────────────

def test_textarea_uses_mock_answer():
    p = plan_field_value(_f(label="Why join us?", field_type="textarea"), _PROFILE, {}, {})
    assert p.status == "filled"
    assert p.source == "mock"
    assert "Why join us" in p.value


def test_user_answer_text_field_uses_mock():
    p = plan_field_value(
        _f(label="Tell us about yourself", field_type="text", expected_source="user_answer"),
        _PROFILE, {}, {},
    )
    assert p.status == "filled"
    assert p.source == "mock"


# ─── Bulk + dict-input compatibility ─────────────────────────────────────────

def test_compute_form_values_accepts_dict_inputs():
    """Real call sites pass DB-deserialized dicts; we accept both."""
    fields = [
        {"label": "First Name", "field_type": "text", "name": "first_name",
         "required": True, "options": [], "accepts_file": [],
         "expected_source": "user_profile"},
        {"label": "Email", "field_type": "email", "name": "email",
         "required": True, "options": [], "accepts_file": [],
         "expected_source": "user_profile"},
    ]
    plans = compute_form_values(fields, _PROFILE, {}, {})
    assert len(plans) == 2
    assert plans[0].value == "Koray"
    assert plans[1].value == "koraysevil@gmail.com"


def test_compute_form_values_handles_empty_input():
    assert compute_form_values([], _PROFILE, {}, {}) == []
    assert compute_form_values(None, _PROFILE, {}, {}) == []  # type: ignore[arg-type]


def test_first_real_option_filters_placeholders():
    assert _first_real_option(["Select...", "Choose…", "—", "Yes", "No"]) == "Yes"
    assert _first_real_option(["", None, "United States"]) == "United States"
    assert _first_real_option([]) is None
    assert _first_real_option(["Select an option"]) is None
