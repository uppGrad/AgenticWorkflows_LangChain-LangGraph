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


# ─── Gate-1 integration: per-field skip via human_review_1 ──────────────────

_DOC_REQUIREMENT_ITEMS = [
    {
        "id": 0, "category": "document", "label": "CV", "required": True,
        "document_type": "CV", "form_field_index": 0,
    },
    {
        "id": 1, "category": "text", "label": "Why us?", "required": True,
        "form_field_index": 1,
    },
]


def test_skip_choice_short_circuits_field():
    """Optional document chosen as `skip` at gate 1 → planner emits
    user_skipped regardless of profile match."""
    fields = [_f(label="Portfolio", field_type="file")]
    items = [{
        "id": 0, "category": "document", "label": "Portfolio", "required": False,
        "document_type": "Portfolio", "form_field_index": 0,
    }]
    review = {"requirements": {"0": {"choice": "skip"}}, "misc_strategy": "auto_fill"}
    plans = compute_form_values(
        fields, _PROFILE, {}, {},
        requirement_items=items, human_review_1=review,
    )
    assert plans[0].status == "skipped"
    assert plans[0].source == "user_skipped"
    assert "skip" in plans[0].reason


def test_ignore_for_now_optional_treated_as_skip():
    """Optional + ignore_for_now is functionally identical to skip at fill time."""
    fields = [_f(label="Portfolio", field_type="file")]
    items = [{
        "id": 0, "category": "document", "label": "Portfolio", "required": False,
        "document_type": "Portfolio", "form_field_index": 0,
    }]
    review = {"requirements": {"0": {"choice": "ignore_for_now"}}, "misc_strategy": "auto_fill"}
    plans = compute_form_values(
        fields, _PROFILE, {}, {},
        requirement_items=items, human_review_1=review,
    )
    assert plans[0].status == "skipped"
    assert plans[0].source == "user_skipped"


def test_required_ignore_for_now_field_skipped_at_per_field_layer():
    """The session-level kill-switch lives in the adapter; if the adapter
    still hands us a payload with required+ignore_for_now (e.g. partial
    rollout, or non-required items already filtered), the per-field rule
    still applies — emit user_skipped."""
    fields = [_f(label="Resume", field_type="file"), _f(label="Why us?", field_type="textarea")]
    review = {
        "requirements": {
            "0": {"choice": "auto_generate"},
            "1": {"choice": "ignore_for_now"},
        },
        "misc_strategy": "auto_fill",
    }
    plans = compute_form_values(
        fields, _PROFILE, {"CV": {"file_path": "/tmp/cv.pdf"}}, {},
        requirement_items=_DOC_REQUIREMENT_ITEMS, human_review_1=review,
    )
    assert plans[0].status == "filled"   # CV auto_generate → resolves
    assert plans[1].status == "skipped"
    assert plans[1].source == "user_skipped"


# ─── Gate-1 integration: tailored_answers replaces mock placeholder ─────────

def test_tailored_answer_used_when_present():
    """Textarea question with a real tailored answer in state → use the
    real answer; do NOT fall back to mock."""
    fields = [_f(label="Why us?", field_type="textarea", required=True)]
    items = [{
        "id": 0, "category": "text", "label": "Why us?", "required": True,
        "form_field_index": 0,
    }]
    review = {"requirements": {"0": {"choice": "auto_generate"}}, "misc_strategy": "auto_fill"}
    tailored_answers = {"0": {"content": "I love your mission and culture."}}
    plans = compute_form_values(
        fields, _PROFILE, {}, {},
        tailored_answers=tailored_answers,
        requirement_items=items, human_review_1=review,
    )
    assert plans[0].status == "filled"
    assert plans[0].source == "user_answer"
    assert plans[0].value == "I love your mission and culture."
    assert "tailored_answers[0]" in plans[0].reason


def test_tailored_answer_string_value_also_supported():
    """Backend may stash the answer as a flat string instead of a dict."""
    fields = [_f(label="Why us?", field_type="textarea")]
    plans = compute_form_values(
        fields, _PROFILE, {}, {},
        tailored_answers={"0": "Direct string answer."},
    )
    assert plans[0].value == "Direct string answer."
    assert plans[0].source == "user_answer"


def test_mock_fallback_when_tailored_answer_missing():
    fields = [_f(label="Why us?", field_type="textarea")]
    plans = compute_form_values(fields, _PROFILE, {}, {}, tailored_answers={})
    assert plans[0].status == "filled"
    assert plans[0].source == "mock"
    assert "Mock answer" in plans[0].value


def test_empty_tailored_answer_falls_through_to_mock():
    """An entry that exists but has empty content shouldn't suppress mock."""
    fields = [_f(label="Why us?", field_type="textarea")]
    plans = compute_form_values(
        fields, _PROFILE, {}, {},
        tailored_answers={"0": {"content": "   "}},
    )
    assert plans[0].source == "mock"


# ─── Gate-1 integration: misc strategy ──────────────────────────────────────

def test_misc_ignore_skips_unowned_fields():
    """Misc fields (form fields with no owning RequirementItem) are skipped
    when misc_strategy=ignore. The user's "package and bounce" path can
    still be partially auto-filled by leaving misc on auto_fill while
    ignore-ing the document/text items individually."""
    # form_fields: [Resume (doc), Why us? (text), Email (misc), Country (misc)]
    fields = [
        _f(label="Resume", field_type="file"),
        _f(label="Why us?", field_type="textarea"),
        _f(label="Email", field_type="email"),
        _f(label="Country", field_type="select", options=["Choose...", "Turkey"]),
    ]
    items = [
        {"id": 0, "category": "document", "label": "Resume", "required": True,
         "document_type": "CV", "form_field_index": 0},
        {"id": 1, "category": "text", "label": "Why us?", "required": True,
         "form_field_index": 1},
        # No RequirementItem points to indices 2, 3 → they're misc.
        {"id": 2, "category": "misc", "label": "Profile / identity (2)",
         "required": False, "form_field_index": None},
    ]
    review = {
        "requirements": {
            "0": {"choice": "auto_generate"},
            "1": {"choice": "auto_generate"},
        },
        "misc_strategy": "ignore",
    }
    plans = compute_form_values(
        fields, _PROFILE, {"CV": {"file_path": "/tmp/cv.pdf"}}, {},
        tailored_answers={"1": "answer"},
        requirement_items=items, human_review_1=review,
    )
    # Resume and Why-us auto-filled.
    assert plans[0].status == "filled"
    assert plans[1].status == "filled"
    # Email and Country (misc) — skipped due to misc_strategy=ignore even
    # though profile lookup would have filled Email.
    assert plans[2].status == "skipped"
    assert plans[2].source == "user_skipped"
    assert "misc_strategy=ignore" in plans[2].reason
    assert plans[3].status == "skipped"


def test_misc_auto_fill_runs_normal_rules():
    fields = [_f(label="Email", field_type="email")]
    items = [{"id": 0, "category": "misc", "label": "Profile (1)",
              "required": False, "form_field_index": None}]
    review = {"requirements": {}, "misc_strategy": "auto_fill"}
    plans = compute_form_values(
        fields, _PROFILE, {}, {},
        requirement_items=items, human_review_1=review,
    )
    # Profile lookup wins for Email even when its owner is the misc bucket.
    assert plans[0].status == "filled"
    assert plans[0].source == "user_profile"
    assert plans[0].value == _PROFILE["email"]


def test_no_requirement_items_disables_misc_rule():
    """Backward compat: when caller doesn't pass requirement_items, the
    is_misc check defaults to False — every field runs the normal rules."""
    fields = [_f(label="Email", field_type="email")]
    plans = compute_form_values(fields, _PROFILE, {}, {})  # no kwargs
    assert plans[0].status == "filled"
    assert plans[0].source == "user_profile"


# ─── Gate-2 clarifying-question per-field answers ───────────────────────────

def test_gate_2_user_answer_wins_over_mock():
    """User typed an answer at gate 2 → use it; do NOT fall through to
    profile lookup, tailored_answers, or mock placeholder."""
    fields = [_f(label="Visa sponsorship?", field_type="text")]
    plans = compute_form_values(
        fields, _PROFILE, {}, {},
        human_review_2={"field_answers": {"0": {"answer": "No"}}},
    )
    assert plans[0].status == "filled"
    assert plans[0].source == "user_answer"
    assert plans[0].value == "No"
    assert plans[0].reason == "gate_2_user_answer"


def test_gate_2_user_answer_wins_over_profile_lookup():
    """If user explicitly answered, it wins even when profile_lookup would
    have matched (e.g. user wants different value than their profile)."""
    fields = [_f(label="Email", field_type="email")]
    plans = compute_form_values(
        fields, _PROFILE, {}, {},
        human_review_2={"field_answers": {"0": {"answer": "alt@example.com"}}},
    )
    assert plans[0].value == "alt@example.com"
    assert plans[0].source == "user_answer"


def test_gate_2_skip_choice_emits_user_skipped():
    fields = [_f(label="Veteran Status", field_type="text")]
    plans = compute_form_values(
        fields, _PROFILE, {}, {},
        human_review_2={"field_answers": {"0": {"choice": "skip"}}},
    )
    assert plans[0].status == "skipped"
    assert plans[0].source == "user_skipped"
    assert "gate_2_choice=skip" in plans[0].reason


def test_gate_2_ignore_for_now_choice_emits_user_skipped():
    """Per-field ignore_for_now at gate 2 still skips the field; the
    session-level kill-switch is the adapter's concern, not the planner's."""
    fields = [_f(label="Visa sponsorship?", field_type="text", required=True)]
    plans = compute_form_values(
        fields, _PROFILE, {}, {},
        human_review_2={"field_answers": {"0": {"choice": "ignore_for_now"}}},
    )
    assert plans[0].status == "skipped"
    assert plans[0].source == "user_skipped"
    assert "ignore_for_now" in plans[0].reason


def test_gate_2_empty_answer_falls_through():
    """A blank answer string shouldn't suppress downstream rules."""
    fields = [_f(label="Email", field_type="email")]
    plans = compute_form_values(
        fields, _PROFILE, {}, {},
        human_review_2={"field_answers": {"0": {"answer": "   "}}},
    )
    # Falls through to profile_lookup
    assert plans[0].source == "user_profile"
    assert plans[0].value == _PROFILE["email"]


def test_gate_2_field_answers_absent_uses_existing_rules():
    """Backward compat: human_review_2 absent → planner unchanged."""
    fields = [_f(label="Email", field_type="email")]
    plans = compute_form_values(fields, _PROFILE, {}, {})
    assert plans[0].source == "user_profile"
