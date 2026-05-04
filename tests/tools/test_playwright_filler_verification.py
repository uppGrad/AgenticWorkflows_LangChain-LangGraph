"""Unit tests for the post-fill verification + drift correction tier
(Tier 5 in `playwright_filler.py`).

These don't spin up a real browser — Playwright `Locator`/`Page` calls
are mocked so we can exercise the comparison rules + the corrector's
control flow in isolation. The integration tests in
`test_playwright_filler.py` run a real Chromium against a static form
and exercise the same code path end-to-end (slow, opt-in).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uppgrad_agentic.tools.playwright_filler import (
    _MAX_CONTAINER_HTML_CHARS,
    _correct_field_drift,
    _normalise_for_compare,
    _probe_field_state,
)
from uppgrad_agentic.workflows.auto_apply.schemas import (
    FormField,
    FormFieldFillPlan,
)


def _plan(field_type: str, value: str, *, label="X", name="x", options=None):
    field = FormField(
        label=label, field_type=field_type, name=name, required=False,
        options=options or [], accepts_file=[], expected_source="unknown",
    )
    return FormFieldFillPlan(field=field, value=value)


def _mock_locator_with_evaluate(observed_payload: dict | Exception):
    """Return a mock Locator whose .evaluate(...) yields `observed_payload`."""
    locator = MagicMock()
    if isinstance(observed_payload, Exception):
        locator.evaluate = AsyncMock(side_effect=observed_payload)
    else:
        locator.evaluate = AsyncMock(return_value=observed_payload)
    return locator


# ─── _normalise_for_compare ─────────────────────────────────────────────────

class TestNormalise:
    def test_lowercases_and_collapses_whitespace(self):
        assert _normalise_for_compare("  United  States  ") == "united states"

    def test_handles_none_and_non_str(self):
        assert _normalise_for_compare(None) == ""
        assert _normalise_for_compare(42) == "42"


# ─── _probe_field_state — text-like fields ──────────────────────────────────

class TestProbeText:
    @pytest.mark.asyncio
    async def test_text_value_match_marks_verified(self):
        plan = _plan("text", "Koray Sevil")
        loc = _mock_locator_with_evaluate({"observed": "Koray Sevil", "notes": "value"})
        verified, observed = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is True
        assert observed == "Koray Sevil"

    @pytest.mark.asyncio
    async def test_text_normalisation_handles_case_and_whitespace(self):
        """Tier 5 must accept Greenhouse-style "United  States " ↔ "united states"."""
        plan = _plan("text", "united states")
        loc = _mock_locator_with_evaluate({"observed": "United  States", "notes": "value"})
        verified, _ = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is True

    @pytest.mark.asyncio
    async def test_text_substring_match_passes(self):
        """Some forms append " — primary" / "(US)" decorations to the
        displayed value. As long as intended ⊆ observed (or vice versa),
        treat as verified."""
        plan = _plan("text", "United States")
        loc = _mock_locator_with_evaluate({"observed": "United States — primary", "notes": "value"})
        verified, _ = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is True

    @pytest.mark.asyncio
    async def test_text_unrelated_observed_marks_drift(self):
        """The exact failure mode that motivated this tier: combobox
        treated as text input — we typed 'United States' but no option
        was selected, so observed comes back as the placeholder."""
        plan = _plan("text", "United States")
        loc = _mock_locator_with_evaluate({"observed": "Select your country", "notes": "value"})
        verified, observed = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is False
        assert observed == "Select your country"

    @pytest.mark.asyncio
    async def test_empty_intended_is_always_verified(self):
        """An empty intended value (optional textarea / blank) shouldn't
        be flagged as drift — there's nothing to verify."""
        plan = _plan("textarea", "")
        loc = _mock_locator_with_evaluate({"observed": "", "notes": "value"})
        verified, _ = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is True


# ─── _probe_field_state — checkbox / radio / file / select ──────────────────

class TestProbeStructured:
    @pytest.mark.asyncio
    async def test_checkbox_truthy_intended_matches_checked(self):
        plan = _plan("checkbox", "true")
        loc = _mock_locator_with_evaluate({"observed": "true", "notes": "checked"})
        verified, _ = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is True

    @pytest.mark.asyncio
    async def test_checkbox_falsy_intended_matches_unchecked(self):
        plan = _plan("checkbox", "false")
        loc = _mock_locator_with_evaluate({"observed": "false", "notes": "checked"})
        verified, _ = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is True

    @pytest.mark.asyncio
    async def test_checkbox_intended_truthy_observed_unchecked_drifts(self):
        plan = _plan("checkbox", "true")
        loc = _mock_locator_with_evaluate({"observed": "false", "notes": "checked"})
        verified, _ = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is False

    @pytest.mark.asyncio
    async def test_radio_group_checked_value_matches(self):
        plan = _plan("radio", "yes")
        loc = _mock_locator_with_evaluate({"observed": "yes", "notes": "group_checked"})
        verified, observed = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is True
        assert observed == "yes"

    @pytest.mark.asyncio
    async def test_radio_group_no_check_marks_drift(self):
        """A yes/no question collected as free text and stuffed at a
        radio group leaves no radio checked — observed='', verified=False."""
        plan = _plan("radio", "yes")
        loc = _mock_locator_with_evaluate({"observed": "", "notes": "group_no_check"})
        verified, _ = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is False

    @pytest.mark.asyncio
    async def test_file_observed_count_marks_verified(self):
        plan = _plan("file", "/tmp/cv.pdf")
        loc = _mock_locator_with_evaluate({"observed": "1_files", "notes": "file_count"})
        verified, _ = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is True

    @pytest.mark.asyncio
    async def test_file_no_files_marks_drift(self):
        plan = _plan("file", "/tmp/cv.pdf")
        loc = _mock_locator_with_evaluate({"observed": "", "notes": "file_count"})
        verified, _ = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is False

    @pytest.mark.asyncio
    async def test_select_native_label_match(self):
        plan = _plan("select", "United States", options=["United States", "Turkey"])
        loc = _mock_locator_with_evaluate({"observed": "United States", "notes": "native_select"})
        verified, _ = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is True

    @pytest.mark.asyncio
    async def test_select_custom_widget_aria_selected_match(self):
        plan = _plan("select", "United States")
        loc = _mock_locator_with_evaluate({"observed": "United States", "notes": "aria_selected"})
        verified, _ = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is True

    @pytest.mark.asyncio
    async def test_select_custom_widget_no_selection_marks_drift(self):
        """Combobox-with-search where text was typed but no option chosen.
        The visible widget falls back to value/textContent which usually
        repeats the typed text — so this scenario can pass the substring
        check (typed ⊆ observed). The harder failure mode is when the
        custom widget shows the placeholder; that's covered here."""
        plan = _plan("select", "United States")
        loc = _mock_locator_with_evaluate({"observed": "Select your country", "notes": "fallback_text"})
        verified, _ = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is False


# ─── _probe_field_state — error paths ───────────────────────────────────────

class TestProbeErrors:
    @pytest.mark.asyncio
    async def test_evaluate_exception_returns_unverified(self):
        plan = _plan("text", "X")
        loc = _mock_locator_with_evaluate(RuntimeError("element detached"))
        verified, observed = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is False
        assert observed == ""

    @pytest.mark.asyncio
    async def test_empty_payload_returns_unverified(self):
        plan = _plan("text", "X")
        loc = _mock_locator_with_evaluate({})
        verified, _ = await _probe_field_state(MagicMock(), plan, loc)
        assert verified is False


# ─── _correct_field_drift — control flow ────────────────────────────────────

def _selector_plan(**kw):
    """Build a `_SelectorPlan` instance with sensible defaults — that's
    the structured output the LLM returns."""
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
        plan = _plan("select", "USA")
        plan.observed_value = "Select country"
        # locator.evaluate returning empty string → no container found.
        loc = MagicMock()
        loc.evaluate = AsyncMock(return_value="")
        outcome, detail = await _correct_field_drift(MagicMock(), plan, loc, llm=MagicMock())
        assert outcome == "llm_skipped"
        assert "no_container" in detail

    @pytest.mark.asyncio
    async def test_skips_when_llm_returns_empty_selector(self):
        """LLM signalling 'I can't propose a fix' (empty selector) must
        not silently succeed — caller treats it as drift unresolved."""
        plan = _plan("select", "USA")
        loc = MagicMock()
        loc.evaluate = AsyncMock(return_value="<fieldset><label>Country</label></fieldset>")
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke = MagicMock(return_value=_selector_plan(selector=""))
        llm.with_structured_output = MagicMock(return_value=structured)

        outcome, detail = await _correct_field_drift(MagicMock(), plan, loc, llm=llm)
        assert outcome == "llm_skipped"
        assert "no_proposal" in detail

    @pytest.mark.asyncio
    async def test_refuses_submit_target(self):
        """Even on a correction, we never click submit/apply buttons.
        The Tier-4 denylist applies here too."""
        plan = _plan("select", "USA")
        loc = MagicMock()
        loc.evaluate = AsyncMock(return_value="<fieldset>...</fieldset>")
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke = MagicMock(return_value=_selector_plan(
            selector="#submit", action="click",
        ))
        llm.with_structured_output = MagicMock(return_value=structured)

        page = MagicMock()
        loc_target = MagicMock()
        loc_target.count = AsyncMock(return_value=1)
        loc_target.first.text_content = AsyncMock(return_value="Submit Application")
        loc_target.first.get_attribute = AsyncMock(return_value="submit")
        loc_target.first.scroll_into_view_if_needed = AsyncMock()
        page.locator = MagicMock(return_value=loc_target)

        outcome, detail = await _correct_field_drift(page, plan, loc, llm=llm)
        assert outcome == "llm_refused_submit"
        assert "Submit Application" in detail or "submit" in detail.lower()

    @pytest.mark.asyncio
    async def test_ambiguous_selector_rejected(self):
        plan = _plan("select", "USA")
        loc = MagicMock()
        loc.evaluate = AsyncMock(return_value="<fieldset>...</fieldset>")
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke = MagicMock(return_value=_selector_plan(selector=".country", action="fill"))
        llm.with_structured_output = MagicMock(return_value=structured)

        page = MagicMock()
        loc_target = MagicMock()
        loc_target.count = AsyncMock(return_value=4)  # ambiguous
        page.locator = MagicMock(return_value=loc_target)

        outcome, detail = await _correct_field_drift(page, plan, loc, llm=llm)
        assert outcome == "no_locator"
        assert "ambiguous" in detail


# ─── Container HTML cap ─────────────────────────────────────────────────────

class TestContainerCap:
    @pytest.mark.asyncio
    async def test_container_html_truncated_to_cap(self):
        """The corrector must never receive the whole form/page HTML —
        only the field's container, capped to the configured limit so
        a single correction call has bounded context cost."""
        from uppgrad_agentic.tools.playwright_filler import _container_html_for_field
        loc = MagicMock()
        # Pretend the container is a giant blob (e.g. nested fieldset
        # accidentally containing many siblings).
        huge = "<div>" + ("x" * (_MAX_CONTAINER_HTML_CHARS * 3)) + "</div>"
        loc.evaluate = AsyncMock(return_value=huge)
        out = await _container_html_for_field(loc)
        assert len(out) == _MAX_CONTAINER_HTML_CHARS
