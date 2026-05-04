"""Unit tests for the Phase-1+2 verification pipeline (Tier-0 combobox
predicate + post-fill DOM probe + LLM batch verifier + drift correction).

Architecture (post-Phase-1+2 redesign):

  - `_is_autocomplete_field(field)`: Tier-0 combobox/autocomplete
    predicate — runs BEFORE the standard tier strategy when ARIA
    signals indicate a typed-search dropdown. Mirrors browser-use.
  - `_probe_field_state(page, plan, locator) -> str`: reads the
    field's observed value from the DOM. NO sane/insane verdict here
    — just data extraction.
  - `_llm_verify_batch(plan, llm) -> Dict[idx, _FieldVerdict]`: ONE
    LLM call decides sane/insane for every filled field. Catches
    semantic mismatches (e.g. "Are you open to relocation?" filled
    with "Ankara, Turkey") + autocomplete-rewrite false-positives
    in the same call.
  - `_correct_field_drift(...)`: per-insane-field corrective action,
    sees only the field's container HTML.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from uppgrad_agentic.tools.playwright_filler import (
    _MAX_CONTAINER_HTML_CHARS,
    _is_autocomplete_field,
    _llm_verify_batch,
    _normalise_for_compare,
    _probe_field_state,
)
from uppgrad_agentic.workflows.auto_apply.schemas import (
    FormField,
    FormFieldFillPlan,
)


def _plan(field_type: str, value: str, *, label="X", name="x", **field_kw):
    field = FormField(
        label=label, field_type=field_type, name=name, required=False,
        options=field_kw.get("options", []), accepts_file=[],
        expected_source="unknown",
        role=field_kw.get("role", ""),
        aria_haspopup=field_kw.get("aria_haspopup", ""),
        aria_controls=field_kw.get("aria_controls", ""),
        aria_owns=field_kw.get("aria_owns", ""),
        aria_autocomplete=field_kw.get("aria_autocomplete", ""),
        list_id=field_kw.get("list_id", ""),
    )
    p = FormFieldFillPlan(field=field, value=value)
    p.status = "filled"
    return p


def _mock_locator(observed_payload):
    locator = MagicMock()
    if isinstance(observed_payload, Exception):
        locator.evaluate = AsyncMock(side_effect=observed_payload)
    else:
        locator.evaluate = AsyncMock(return_value=observed_payload)
    return locator


# ─── _is_autocomplete_field — Tier-0 combobox predicate ─────────────────────

class TestComboboxPredicate:
    """Mirrors browser-use's `_is_autocomplete_field`. The signal that
    distinguishes a normal text input from a typed-search combobox."""

    def test_role_combobox_triggers(self):
        f = FormField(label="C", field_type="text", required=False,
                      options=[], accepts_file=[], expected_source="unknown",
                      role="combobox")
        assert _is_autocomplete_field(f) is True

    def test_aria_autocomplete_list_triggers(self):
        f = FormField(label="C", field_type="text", required=False,
                      options=[], accepts_file=[], expected_source="unknown",
                      aria_autocomplete="list")
        assert _is_autocomplete_field(f) is True

    def test_aria_autocomplete_inline_triggers(self):
        f = FormField(label="C", field_type="text", required=False,
                      options=[], accepts_file=[], expected_source="unknown",
                      aria_autocomplete="inline")
        assert _is_autocomplete_field(f) is True

    def test_list_id_triggers(self):
        f = FormField(label="C", field_type="text", required=False,
                      options=[], accepts_file=[], expected_source="unknown",
                      list_id="countries")
        assert _is_autocomplete_field(f) is True

    def test_haspopup_with_controls_triggers(self):
        f = FormField(label="C", field_type="text", required=False,
                      options=[], accepts_file=[], expected_source="unknown",
                      aria_haspopup="listbox", aria_controls="opts")
        assert _is_autocomplete_field(f) is True

    def test_haspopup_alone_does_not_trigger(self):
        """Without `aria-controls` or `aria-owns`, haspopup isn't enough —
        the popup target is unknown so we can't reliably click+pick."""
        f = FormField(label="C", field_type="text", required=False,
                      options=[], accepts_file=[], expected_source="unknown",
                      aria_haspopup="listbox")
        assert _is_autocomplete_field(f) is False

    def test_aria_autocomplete_none_does_not_trigger(self):
        f = FormField(label="C", field_type="text", required=False,
                      options=[], accepts_file=[], expected_source="unknown",
                      aria_autocomplete="none")
        assert _is_autocomplete_field(f) is False

    def test_plain_text_input_does_not_trigger(self):
        f = FormField(label="First name", field_type="text", required=False,
                      options=[], accepts_file=[], expected_source="unknown")
        assert _is_autocomplete_field(f) is False


# ─── _probe_field_state — DOM readback (no verdict) ─────────────────────────

class TestProbeReadsObserved:
    @pytest.mark.asyncio
    async def test_text_returns_value(self):
        plan = _plan("text", "Koray")
        loc = _mock_locator({"observed": "Koray", "notes": "value"})
        observed = await _probe_field_state(MagicMock(), plan, loc)
        assert observed == "Koray"

    @pytest.mark.asyncio
    async def test_select_native_returns_option_label(self):
        plan = _plan("select", "United States")
        loc = _mock_locator({"observed": "United States", "notes": "native_select"})
        assert await _probe_field_state(MagicMock(), plan, loc) == "United States"

    @pytest.mark.asyncio
    async def test_checkbox_returns_true_or_false_string(self):
        plan = _plan("checkbox", "true")
        loc = _mock_locator({"observed": "true", "notes": "checked"})
        assert await _probe_field_state(MagicMock(), plan, loc) == "true"

    @pytest.mark.asyncio
    async def test_radio_returns_checked_value(self):
        plan = _plan("radio", "yes")
        loc = _mock_locator({"observed": "yes", "notes": "group_checked"})
        assert await _probe_field_state(MagicMock(), plan, loc) == "yes"

    @pytest.mark.asyncio
    async def test_file_returns_count_marker(self):
        plan = _plan("file", "/tmp/cv.pdf")
        loc = _mock_locator({"observed": "1_files", "notes": "file_count"})
        assert await _probe_field_state(MagicMock(), plan, loc) == "1_files"

    @pytest.mark.asyncio
    async def test_evaluate_exception_returns_empty(self):
        plan = _plan("text", "X")
        loc = _mock_locator(RuntimeError("element detached"))
        assert await _probe_field_state(MagicMock(), plan, loc) == ""


# ─── _llm_verify_batch — Phase 2 verification ───────────────────────────────

class TestLLMBatchVerify:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_llm(self):
        """No LLM means no verification; caller treats every field as sane.
        Preserves the no-LLM heuristic path used in tests + offline runs."""
        plan = [_plan("text", "Koray", name="first_name")]
        plan[0].observed_value = "Koray"
        verdicts = await _llm_verify_batch(plan, llm=None)
        assert verdicts == {}

    @pytest.mark.asyncio
    async def test_short_circuits_exact_match_without_llm_call(self):
        """When intended exactly equals observed (modulo case/whitespace),
        the field is trivially sane — no LLM call needed. Bounds cost
        on forms where most fills are direct (name, email, etc.)."""
        plan = [
            _plan("text", "Koray Sevil", name="name"),
            _plan("email", "x@example.com", name="email"),
        ]
        plan[0].observed_value = "koray sevil"  # case-only diff → sane
        plan[1].observed_value = "x@example.com"
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke = MagicMock()
        llm.with_structured_output = MagicMock(return_value=structured)

        verdicts = await _llm_verify_batch(plan, llm)
        assert verdicts[0].sane is True
        assert verdicts[1].sane is True
        # LLM never called because every field was trivially sane.
        structured.invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_circuits_empty_intended(self):
        """An empty intended (optional textarea blank) is always sane —
        nothing to verify."""
        plan = [_plan("textarea", "", name="why")]
        plan[0].observed_value = ""
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke = MagicMock()
        llm.with_structured_output = MagicMock(return_value=structured)
        verdicts = await _llm_verify_batch(plan, llm)
        assert verdicts[0].sane is True
        structured.invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_llm_for_non_trivial_rows(self):
        """When intended != observed (after normalisation), the row is
        sent to the LLM. The LLM's verdict wins."""
        from uppgrad_agentic.tools.playwright_filler import (
            _BatchVerifyResult,
            _FieldVerdict,
        )
        plan = [
            _plan("text", "Yes", name="relocation",
                  label="Are you open to relocation?"),
        ]
        plan[0].observed_value = "Ankara, Turkey"  # semantic mismatch
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke = MagicMock(return_value=_BatchVerifyResult(
            verdicts=[_FieldVerdict(idx=0, sane=False, reason="city_for_yes_no", suggested_value="Yes")],
        ))
        llm.with_structured_output = MagicMock(return_value=structured)

        verdicts = await _llm_verify_batch(plan, llm)
        assert verdicts[0].sane is False
        assert verdicts[0].suggested_value == "Yes"
        structured.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_sane(self):
        """An LLM call exception must not break the fill loop. Treat
        every non-trivial row as sane (no verdict returned) and let
        downstream surface whatever the deterministic fill produced."""
        plan = [_plan("text", "Yes", name="r")]
        plan[0].observed_value = "city"
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke = MagicMock(side_effect=RuntimeError("llm 500"))
        llm.with_structured_output = MagicMock(return_value=structured)
        verdicts = await _llm_verify_batch(plan, llm)
        # No verdict for idx=0 (LLM failed) → caller treats absence as sane.
        assert 0 not in verdicts


# ─── _normalise_for_compare — used by short-circuit ─────────────────────────

class TestNormalise:
    def test_lowercases_and_collapses_whitespace(self):
        assert _normalise_for_compare("  United  States  ") == "united states"

    def test_handles_none_and_non_str(self):
        assert _normalise_for_compare(None) == ""
        assert _normalise_for_compare(42) == "42"


# ─── _correct_field_drift — control flow ────────────────────────────────────

def _selector_plan(**kw):
    from uppgrad_agentic.tools.playwright_filler import _SelectorPlan
    return _SelectorPlan(**{
        "selector": kw.get("selector", "#fixed-element"),
        "action": kw.get("action", "click_then_pick_option"),
        "option_text": kw.get("option_text", ""),
        "linked_input_id": kw.get("linked_input_id", ""),
        "notes": kw.get("notes", ""),
    })


class TestDriftCorrector:
    @pytest.mark.asyncio
    async def test_skips_when_no_container_html(self):
        from uppgrad_agentic.tools.playwright_filler import _correct_field_drift
        plan = _plan("select", "USA")
        plan.observed_value = "Select country"
        loc = MagicMock()
        loc.evaluate = AsyncMock(return_value="")
        outcome, detail = await _correct_field_drift(MagicMock(), plan, loc, llm=MagicMock())
        assert outcome == "llm_skipped"
        assert "no_container" in detail

    @pytest.mark.asyncio
    async def test_skips_when_llm_returns_empty_selector(self):
        from uppgrad_agentic.tools.playwright_filler import _correct_field_drift
        plan = _plan("select", "USA")
        loc = MagicMock()
        loc.evaluate = AsyncMock(return_value="<fieldset>...</fieldset>")
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke = MagicMock(return_value=_selector_plan(selector=""))
        llm.with_structured_output = MagicMock(return_value=structured)
        outcome, detail = await _correct_field_drift(MagicMock(), plan, loc, llm=llm)
        assert outcome == "llm_skipped"
        assert "no_proposal" in detail

    @pytest.mark.asyncio
    async def test_refuses_submit_target(self):
        from uppgrad_agentic.tools.playwright_filler import _correct_field_drift
        plan = _plan("select", "USA")
        loc = MagicMock()
        loc.evaluate = AsyncMock(return_value="<fieldset>...</fieldset>")
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke = MagicMock(return_value=_selector_plan(selector="#submit", action="click"))
        llm.with_structured_output = MagicMock(return_value=structured)
        page = MagicMock()
        loc_target = MagicMock()
        loc_target.count = AsyncMock(return_value=1)
        loc_target.first.text_content = AsyncMock(return_value="Submit Application")
        loc_target.first.get_attribute = AsyncMock(return_value="submit")
        loc_target.first.scroll_into_view_if_needed = AsyncMock()
        page.locator = MagicMock(return_value=loc_target)
        outcome, _ = await _correct_field_drift(page, plan, loc, llm=llm)
        assert outcome == "llm_refused_submit"

    @pytest.mark.asyncio
    async def test_ambiguous_selector_rejected(self):
        from uppgrad_agentic.tools.playwright_filler import _correct_field_drift
        plan = _plan("select", "USA")
        loc = MagicMock()
        loc.evaluate = AsyncMock(return_value="<fieldset>...</fieldset>")
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke = MagicMock(return_value=_selector_plan(selector=".country", action="fill"))
        llm.with_structured_output = MagicMock(return_value=structured)
        page = MagicMock()
        loc_target = MagicMock()
        loc_target.count = AsyncMock(return_value=4)
        page.locator = MagicMock(return_value=loc_target)
        outcome, detail = await _correct_field_drift(page, plan, loc, llm=llm)
        assert outcome == "no_locator"
        assert "ambiguous" in detail


# ─── Container HTML cap ─────────────────────────────────────────────────────

class TestContainerCap:
    @pytest.mark.asyncio
    async def test_container_html_truncated_to_cap(self):
        from uppgrad_agentic.tools.playwright_filler import _container_html_for_field
        loc = MagicMock()
        huge = "<div>" + ("x" * (_MAX_CONTAINER_HTML_CHARS * 3)) + "</div>"
        loc.evaluate = AsyncMock(return_value=huge)
        out = await _container_html_for_field(loc)
        assert len(out) == _MAX_CONTAINER_HTML_CHARS
