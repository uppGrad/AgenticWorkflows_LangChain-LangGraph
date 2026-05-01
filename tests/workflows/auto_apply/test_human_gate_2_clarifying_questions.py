"""Gate-2 clarifying-questions extension.

Locks in the contract for `needs_user_answer` (which residual misc fields
get surfaced) and the `field_answers` validator (skip vs ignore_for_now,
required vs optional, answer length cap).
"""
from uppgrad_agentic.workflows.auto_apply.nodes.human_gate_2 import (
    _build_needs_user_answer,
    _validate_field_answers,
    _is_eeo_label,
)


def _ff(idx, label, *, field_type="text", required=False, options=None):
    """Form-field dict matching state['form_fields'] shape."""
    return {
        "label": label, "field_type": field_type, "name": "",
        "required": required, "options": options or [],
        "accepts_file": [], "expected_source": "unknown",
        "canonical_document_type": "",
    }


# ─── needs_user_answer construction ─────────────────────────────────────────

def test_needs_user_answer_skips_non_misc_fields():
    """A form field already pointed to by a non-misc RequirementItem (gate
    1 already covered it) is NOT in the residual list."""
    form_fields = [_ff(0, "Resume/CV", field_type="file", required=True)]
    requirement_items = [{
        "id": 0, "category": "document", "label": "Resume/CV",
        "required": True, "form_field_index": 0,
    }]
    needs = _build_needs_user_answer(form_fields, requirement_items, {}, {})
    assert needs == []


def test_needs_user_answer_skips_fields_with_tailored_answer():
    """If application_tailoring already produced a tailored_answer for this
    form field, no need to ask the user — auto-fill will use it."""
    form_fields = [_ff(0, "Why us?", field_type="textarea", required=True)]
    needs = _build_needs_user_answer(
        form_fields, [], {"0": {"content": "drafted answer"}}, {},
    )
    assert needs == []


def test_needs_user_answer_skips_profile_resolvable_fields():
    """If profile_lookup matches the label, auto-fill will resolve it from
    the profile snapshot — no need to ask."""
    form_fields = [
        _ff(0, "Email", field_type="text", required=True),
        _ff(1, "First Name", field_type="text", required=True),
    ]
    profile = {"email": "x@y.com", "first_name": "Stu"}
    needs = _build_needs_user_answer(form_fields, [], {}, profile)
    assert needs == []


def test_needs_user_answer_surfaces_residual_unknowns():
    """A misc field with no profile match and no tailored_answer is
    exactly what we want the user to address at gate 2."""
    form_fields = [
        _ff(0, "Email", field_type="text", required=True),                   # profile-resolvable
        _ff(1, "Do you have expertise in Python?", field_type="text", required=True),  # residual
        _ff(2, "Veteran Status", field_type="text", required=False),        # residual + EEO
    ]
    profile = {"email": "x@y.com"}
    needs = _build_needs_user_answer(form_fields, [], {}, profile)
    assert len(needs) == 2
    assert needs[0]["label"] == "Do you have expertise in Python?"
    assert needs[0]["required"] is True
    assert needs[0]["is_eeo"] is False
    assert needs[1]["label"] == "Veteran Status"
    assert needs[1]["is_eeo"] is True


def test_eeo_label_detection():
    assert _is_eeo_label("Gender")
    assert _is_eeo_label("Are you Hispanic/Latino?")
    assert _is_eeo_label("Veteran Status")
    assert _is_eeo_label("Disability Status")
    assert _is_eeo_label("Race / Ethnicity")
    assert not _is_eeo_label("Email")
    assert not _is_eeo_label("First Name")
    assert not _is_eeo_label("Why us?")


# ─── field_answers validator ────────────────────────────────────────────────

_NEED_REQUIRED = [{"form_field_index": 5, "label": "Visa?", "required": True, "field_type": "text", "options": [], "is_eeo": False}]
_NEED_OPTIONAL = [{"form_field_index": 5, "label": "Veteran", "required": False, "field_type": "text", "options": [], "is_eeo": True}]


def test_field_answers_absent_is_valid():
    out, errors = _validate_field_answers({}, _NEED_REQUIRED)
    assert out == {}
    assert errors == []


def test_field_answers_must_be_dict():
    out, errors = _validate_field_answers({"field_answers": "bogus"}, _NEED_REQUIRED)
    assert out is None
    assert errors and "must be an object" in errors[0]


def test_field_answers_entry_must_have_answer_or_choice():
    out, errors = _validate_field_answers({"field_answers": {"5": {}}}, _NEED_REQUIRED)
    assert errors and "must have either" in errors[0]


def test_field_answers_required_skip_rejected():
    """skip on a required field is invalid — user should pick ignore_for_now
    if they want to handle it manually after handoff (kill-switch trigger)."""
    out, errors = _validate_field_answers(
        {"field_answers": {"5": {"choice": "skip"}}}, _NEED_REQUIRED,
    )
    assert errors and "required field cannot be skipped" in errors[0]


def test_field_answers_required_ignore_for_now_allowed():
    """Required + ignore_for_now is the kill-switch trigger — allowed at
    validator level. Adapter checks it and short-circuits auto-fill."""
    out, errors = _validate_field_answers(
        {"field_answers": {"5": {"choice": "ignore_for_now"}}}, _NEED_REQUIRED,
    )
    assert errors == []
    assert out == {"5": {"choice": "ignore_for_now"}}


def test_field_answers_optional_skip_allowed():
    out, errors = _validate_field_answers(
        {"field_answers": {"5": {"choice": "skip"}}}, _NEED_OPTIONAL,
    )
    assert errors == []
    assert out == {"5": {"choice": "skip"}}


def test_field_answers_string_answer_passes():
    out, errors = _validate_field_answers(
        {"field_answers": {"5": {"answer": "Yes, in 4 production projects."}}},
        _NEED_REQUIRED,
    )
    assert errors == []
    assert out["5"]["answer"] == "Yes, in 4 production projects."


def test_field_answers_answer_length_capped():
    long = "x" * 5001
    out, errors = _validate_field_answers(
        {"field_answers": {"5": {"answer": long}}}, _NEED_REQUIRED,
    )
    assert errors and "exceeds" in errors[0]


def test_field_answers_unknown_index_still_validated_for_shape():
    """Per-id entries for indices outside needs_user_answer aren't blocked
    (frontend may want to override a profile-resolved field) — but their
    shape is still checked."""
    out, errors = _validate_field_answers(
        {"field_answers": {"99": {"answer": "x"}}}, _NEED_REQUIRED,
    )
    assert errors == []
    assert out == {"99": {"answer": "x"}}
