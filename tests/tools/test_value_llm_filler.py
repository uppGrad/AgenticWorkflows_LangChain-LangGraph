"""Tier 4b — value_llm_filler.

Locks the contract that:
  * Only entries with status='skipped' AND source='no_value' are eligible.
  * The deny-list blocks compensation / identifier / sensitive-demographic
    labels BEFORE the LLM is consulted (no waste of budget).
  * When the LLM returns a non-null, in-options, confident value, the
    entry is promoted to filled+llm_inferred.
  * When the LLM returns null / low-confidence / out-of-options, the
    entry stays skipped.
  * The budget cap is honoured — past N calls, remaining skipped entries
    are NOT consulted.
  * `llm=None` → no-op (matches the rest of the agentic stack's
    "degrade gracefully without an LLM" pattern).
"""
from unittest.mock import MagicMock

from uppgrad_agentic.tools.value_llm_filler import (
    FieldGuess,
    _is_eligible_skip,
    _label_is_denied,
    llm_fill_skipped_fields,
)
from uppgrad_agentic.workflows.auto_apply.schemas import (
    FormField,
    FormFieldFillPlan,
)


def _f(**kw) -> FormField:
    base = {
        "label": "X", "field_type": "text", "name": "", "required": False,
        "options": [], "accepts_file": [], "expected_source": "unknown",
    }
    base.update(kw)
    return FormField(**base)


def _skipped(field: FormField) -> FormFieldFillPlan:
    return FormFieldFillPlan(
        field=field, value="", status="skipped", source="no_value",
        reason="unhandled",
    )


def _filled(field: FormField, value: str) -> FormFieldFillPlan:
    return FormFieldFillPlan(
        field=field, value=value, status="filled", source="user_profile",
        reason="profile lookup",
    )


def _fake_llm_returning(*guesses):
    """Build a fake LLM whose `with_structured_output(...).invoke(...)`
    returns each FieldGuess in `guesses` in order."""
    seq = list(guesses)
    structured = MagicMock()
    structured.invoke.side_effect = seq
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


# ─── Eligibility / deny-list ─────────────────────────────────────────────────

def test_eligible_skip_only_promotes_skipped_no_value():
    """A 'mock' filled field shouldn't be re-LLM-ed into a different
    answer; only the deterministic 'I gave up' case is in scope."""
    f = _f(label="Years of Python")
    assert _is_eligible_skip(_skipped(f)) is True
    filled_mock = FormFieldFillPlan(
        field=f, value="placeholder", status="filled", source="mock",
        reason="mock_answer",
    )
    assert _is_eligible_skip(filled_mock) is False


def test_eligible_skip_rejects_short_labels():
    """`label='Y'` is too short to be a meaningful question; skip."""
    assert _is_eligible_skip(_skipped(_f(label="Y"))) is False


def test_deny_list_blocks_salary_questions():
    assert _label_is_denied("Expected salary") is True
    assert _label_is_denied("Hourly rate (USD)") is True
    assert _label_is_denied("base pay range") is True


def test_deny_list_blocks_demographics_and_ids():
    assert _label_is_denied("Date of Birth") is True
    assert _label_is_denied("Disability status") is True
    assert _label_is_denied("Veteran Status") is True
    assert _label_is_denied("SSN") is True


def test_deny_list_does_not_block_normal_questions():
    """Years of experience, degree, location → all fair game."""
    assert _label_is_denied("Years of Python experience") is False
    assert _label_is_denied("Highest degree") is False
    assert _label_is_denied("LinkedIn URL") is False


# ─── llm_fill_skipped_fields ─────────────────────────────────────────────────

def test_no_op_when_llm_none():
    plan = [_skipped(_f(label="Years of Python"))]
    out = llm_fill_skipped_fields(plan, {}, "", {}, llm=None)
    assert out == plan


def test_no_op_when_plan_empty():
    out = llm_fill_skipped_fields([], {}, "", {}, llm=MagicMock())
    assert out == []


def test_promotes_to_llm_inferred_when_guess_confident():
    plan = [_skipped(_f(label="Years of Python experience"))]
    llm = _fake_llm_returning(
        FieldGuess(value="4", confidence=0.85,
                   reason="HAVELSAN AI internship 2022-now in CV"),
    )
    out = llm_fill_skipped_fields(
        plan, profile={"name": "Ali"}, cv_text="x" * 200,
        opportunity_data={"title": "SWE"}, llm=llm,
    )
    assert len(out) == 1
    assert out[0].status == "filled"
    assert out[0].source == "llm_inferred"
    assert out[0].value == "4"
    assert "conf=0.85" in out[0].reason


def test_keeps_skipped_when_guess_is_null():
    """LLM returned `value=None` — stays skipped, no fabrication."""
    plan = [_skipped(_f(label="What is your favourite colour?"))]
    llm = _fake_llm_returning(
        FieldGuess(value=None, confidence=0.0, reason="not in CV/profile"),
    )
    out = llm_fill_skipped_fields(
        plan, profile={"name": "Ali"}, cv_text="some cv",
        opportunity_data={}, llm=llm,
    )
    assert out[0].status == "skipped"
    assert out[0].source == "no_value"


def test_keeps_skipped_when_low_confidence():
    """confidence < 0.5 → treat as null. Never submit a maybe-wrong answer."""
    plan = [_skipped(_f(label="Years of Rust experience"))]
    llm = _fake_llm_returning(
        FieldGuess(value="3", confidence=0.3, reason="guess"),
    )
    out = llm_fill_skipped_fields(
        plan, profile={"name": "Ali"}, cv_text="cv mentions Python",
        opportunity_data={}, llm=llm,
    )
    assert out[0].status == "skipped"


def test_keeps_skipped_when_value_not_in_options():
    """For select/radio with options, the LLM value MUST match an option
    verbatim. Otherwise the form will reject the submission."""
    plan = [_skipped(_f(
        label="Highest education level",
        field_type="select",
        options=["High school", "Bachelor's", "Master's", "PhD"],
    ))]
    llm = _fake_llm_returning(
        FieldGuess(
            value="Bachelor",  # missing the apostrophe-s
            confidence=0.9, reason="from profile.degree_level",
        ),
    )
    out = llm_fill_skipped_fields(
        plan, profile={"degree_level": "Bachelor's"}, cv_text="",
        opportunity_data={}, llm=llm,
    )
    assert out[0].status == "skipped"


def test_promotes_when_value_matches_option():
    plan = [_skipped(_f(
        label="Highest education level",
        field_type="select",
        options=["High school", "Bachelor's", "Master's", "PhD"],
    ))]
    llm = _fake_llm_returning(
        FieldGuess(value="Bachelor's", confidence=0.9, reason="from profile"),
    )
    out = llm_fill_skipped_fields(
        plan, profile={"degree_level": "Bachelor's"}, cv_text="",
        opportunity_data={}, llm=llm,
    )
    assert out[0].status == "filled"
    assert out[0].source == "llm_inferred"
    assert out[0].value == "Bachelor's"


def test_skips_denied_labels_without_llm_call():
    """Salary / SSN / DOB labels never reach the LLM — saves budget AND
    avoids the planner suggesting numbers we shouldn't submit."""
    fields = [
        _f(label="Expected salary"),
        _f(label="Date of Birth"),
        _f(label="SSN"),
    ]
    plan = [_skipped(f) for f in fields]
    structured = MagicMock()
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    out = llm_fill_skipped_fields(
        plan, profile={}, cv_text="", opportunity_data={}, llm=llm,
    )
    structured.invoke.assert_not_called()
    assert all(p.status == "skipped" for p in out)


def test_budget_caps_llm_calls():
    """With 3 eligible-skipped fields and budget=2, only the first 2 are
    consulted. The 3rd stays skipped."""
    plan = [
        _skipped(_f(label="Years of Python experience")),
        _f(label="Years of Java experience"),
        _f(label="Years of Go experience"),
    ]
    plan[1] = _skipped(plan[1])
    plan[2] = _skipped(plan[2])
    llm = _fake_llm_returning(
        FieldGuess(value="4", confidence=0.9, reason="from CV"),
        FieldGuess(value="2", confidence=0.9, reason="from CV"),
    )
    out = llm_fill_skipped_fields(
        plan, profile={}, cv_text="cv body", opportunity_data={},
        llm=llm, budget=2,
    )
    promoted = [p for p in out if p.source == "llm_inferred"]
    skipped = [p for p in out if p.status == "skipped"]
    assert len(promoted) == 2
    assert len(skipped) == 1


def test_already_filled_entries_passed_through_untouched():
    """value_planner-filled entries (user_profile / user_document /
    user_answer / mock / computed) must pass through unchanged."""
    f1 = _f(label="First Name")
    f2 = _f(label="Years of Python experience")
    plan = [
        _filled(f1, "Ali"),
        _skipped(f2),
    ]
    llm = _fake_llm_returning(
        FieldGuess(value="4", confidence=0.9, reason="from CV"),
    )
    out = llm_fill_skipped_fields(
        plan, profile={"first_name": "Ali"}, cv_text="cv",
        opportunity_data={}, llm=llm,
    )
    # First entry untouched
    assert out[0].source == "user_profile"
    assert out[0].value == "Ali"
    # Second promoted
    assert out[1].source == "llm_inferred"
    assert out[1].value == "4"


def test_llm_invoke_failure_keeps_entry_skipped():
    """LLM throws → degrade gracefully. The entry stays skipped, the
    rest of the plan still goes through."""
    plan = [_skipped(_f(label="Years of Python experience"))]
    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("api timeout")
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    out = llm_fill_skipped_fields(
        plan, profile={}, cv_text="cv", opportunity_data={}, llm=llm,
    )
    assert out[0].status == "skipped"
