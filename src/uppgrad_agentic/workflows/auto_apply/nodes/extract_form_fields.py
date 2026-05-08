"""Extracts structured form-field information from a rendered application
page so a future auto-submit step can fill the form without re-extracting.

Inputs (from state):
  - opportunity_type ("job"; node skips other types)
  - discovered_form_url (per-ATS resolved URL)
  - scraped_requirements.raw_html (rendered DOM from scrape_application_page)

Outputs:
  - form_fields: List[FormField as dict]
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.tools.form_extractor import extract_ats_iframe_src, extract_form_html
from uppgrad_agentic.tools.web_fetcher import fetch_url_with_fallback, force_browser_fetch
from uppgrad_agentic.workflows.auto_apply.schemas import FormSchema
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)

_MAX_FORM_HTML_FOR_LLM = 160_000  # gpt-4o-mini handles 128k tokens; ~160k chars
# fits within budget AND covers single-page forms with substantial markup
# (Workable's `/apply/` route ships ~95k chars including all 11 inputs;
# the previous 80k cap truncated the file-upload + Q&A sections off and
# the LLM only saw the contact-info portion).

_SYSTEM = """You are extracting the structured fields of a job application form.

You will receive the raw HTML of a single <form> element. Identify EVERY input on the form and return a structured list. For each field:

- label: the human-readable label (from <label>, surrounding text, or `aria-label`/`placeholder` if no label tag). Be concise.
- field_type: one of "file", "text", "textarea", "select", "checkbox", "radio", "number", "email", "url", "date", "tel". Use the input's `type` attribute when present; for `<select>` always return "select"; for `<textarea>` always return "textarea".
- name: the input's `name` attribute exactly as it appears in the markup. Empty string only if absent.
- required: true if the input has the `required` attribute or its label/legend ends with "*" or contains "(required)".
- options: for `select` and grouped `radio` fields, list the visible option labels in order. Empty for other types.
- accepts_file: for `file` fields only, list the values of the `accept` attribute split on commas (e.g. [".pdf", ".docx"]). Empty for non-file fields.
- expected_source: classify where the value should come from when auto-filling:
    * "user_document" — file inputs whose label suggests an uploadable document (resume/CV, cover letter, portfolio, transcript).
    * "user_profile" — fields whose label maps to a profile attribute (name, email, phone, country, LinkedIn URL, GitHub URL, location, work authorization status).
    * "user_answer" — free-form questions (textareas like "Why do you want to join us?", "Describe a project you led", screening multiple-choice questions).
    * "computed" — fields the system can derive without the user (today's date).
    * "unknown" — when none of the above clearly apply.

ARIA / combobox-detection signals — capture these EXACTLY as they appear on the markup, empty string when absent. They are critical for the auto-fill stage to recognise comboboxes that look like text inputs but back a dropdown:
- role: the explicit `role` attribute on the input or its closest `[role]` ancestor (commonly "combobox", "listbox"). Empty if none.
- aria_haspopup: `aria-haspopup` attribute value.
- aria_controls: `aria-controls` attribute value (target listbox id).
- aria_owns: `aria-owns` attribute value.
- aria_autocomplete: `aria-autocomplete` attribute value ("list", "both", "inline", "none").
- list_id: `list` attribute value (datalist id) for `<input list="...">`.

Capture these per-field even when field_type is "text" — a text input that ALSO has role="combobox" or aria-autocomplete="list" is the exact case the auto-filler needs to know about (Lever country picker, Greenhouse location autocomplete, etc.).

Return one entry per visible field, in document order. Do NOT invent fields that are not in the markup.
"""


def extract_form_fields(state: AutoApplyState) -> dict:
    updates = {"current_step": "extract_form_fields", "step_history": ["extract_form_fields"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    if state.get("opportunity_type") != "job":
        return updates

    # When discovery couldn't resolve a form URL (Workday auth wall, failed
    # discovery), the form is unreachable — return an empty list so the
    # downstream package_and_handoff node knows there's nothing to fill.
    form_url = state.get("discovered_form_url")
    if not form_url:
        return {**updates, "form_fields": []}

    # ─── Primary path: live-DOM Playwright walker ────────────────────
    # The walker captures everything the downstream layers need
    # (options for native selects, options for react-select comboboxes
    # via listbox-open probe, ARIA shape, label resolution chain) in
    # ONE Playwright session.
    #
    # When `UPPGRAD_FORM_DISCOVERY_VERIFY=1`, also captures a screenshot
    # of the form area and runs a vision-LLM verifier on the walker's
    # output. The verifier proposes label/type/options corrections,
    # additions for missed fields, and removals for phantom entries
    # (e.g. OpenAI Ashby radio groups the walker silently folds into a
    # single "first option" entry). Bounded single LLM call;
    # never blocks discovery if it fails.
    discovered: list = []
    verify_enabled = (
        (os.environ.get("UPPGRAD_FORM_DISCOVERY_VERIFY") or "").strip().lower()
        in ("1", "true", "yes", "on")
    )
    if verify_enabled:
        try:
            import asyncio
            from uppgrad_agentic.tools.form_discoverer import (
                discover_form_fields_with_screenshot_async,
            )
            from uppgrad_agentic.tools.form_verifier import verify_fields_with_vision
            walker_fields, screenshot = asyncio.run(
                discover_form_fields_with_screenshot_async(form_url)
            )
            if walker_fields:
                discovered = verify_fields_with_vision(walker_fields, screenshot)
                logger.info(
                    "extract_form_fields (verified): %d field(s) for %s "
                    "(walker emitted %d before verifier corrections)",
                    len(discovered), form_url, len(walker_fields),
                )
        except Exception as exc:
            logger.warning(
                "extract_form_fields: verifier path raised (%s) — falling "
                "back to plain walker", exc,
            )
            discovered = []

    if not discovered:
        try:
            from uppgrad_agentic.tools.form_discoverer import discover_form_fields
            discovered = discover_form_fields(form_url)
        except Exception as exc:
            logger.warning("extract_form_fields: form_discoverer raised — %s", exc)
            discovered = []
    if discovered:
        logger.info(
            "extract_form_fields: discovered %d field(s) live for %s "
            "(%d combobox-shape, %d with options)",
            len(discovered), form_url,
            sum(1 for f in discovered if (f.get("role") or "") == "combobox"
                or (f.get("aria_autocomplete") or "") in ("list", "both", "inline")),
            sum(1 for f in discovered if f.get("options")),
        )
        return {**updates, "form_fields": discovered}
    logger.info(
        "extract_form_fields: live walker returned 0 fields for %s — "
        "falling back to LLM-on-static-HTML parser",
        form_url,
    )

    overview_url = state.get("discovered_apply_url") or ""
    # Read from the top-level state field directly — `scraped_requirements`
    # is rewritten by `evaluate_scrape` after `scrape_application_page` and
    # would clobber any raw_html stashed inside it.
    raw_html = state.get("discovered_raw_html") or ""

    # If form URL differs from overview URL (Ashby /application, Lever /apply,
    # SmartRecruiters /apply), fetch the form page itself — overview-page HTML
    # doesn't contain the <form> for these split-URL ATSes.
    if form_url != overview_url:
        logger.info(
            "extract_form_fields: form URL differs from overview — fetching %s",
            form_url,
        )
        fetch = fetch_url_with_fallback(form_url)
        if not fetch.success or not fetch.raw_html:
            logger.warning(
                "extract_form_fields: could not fetch form page %s (status=%s)",
                form_url, fetch.http_status,
            )
            return {**updates, "form_fields": []}
        raw_html = fetch.raw_html

    form_html = extract_form_html(raw_html)
    forced_html = ""

    # Tier 2: retry with browser if the existing HTML had no form/inputs.
    # Many career pages (mongodb.com/careers, Anthropic careers index, generally
    # any company-direct site) return server-rendered HTML that's NOT thin
    # (full JD content visible) but the form area itself is hydrated client-
    # side. The existing fetch_url_with_fallback only escalates on thin
    # verdicts, so we explicitly force browser here when the parse finds nothing.
    if not form_html:
        logger.info(
            "extract_form_fields: no inputs in cached HTML for %s — forcing browser retry",
            form_url,
        )
        forced = force_browser_fetch(form_url)
        if forced and forced.success and forced.raw_html:
            forced_html = forced.raw_html
            form_html = extract_form_html(forced_html)

    # Tier 2b: still no form? Retry with the apply-CTA click pass. Some
    # ATSes (Workable's `/j/<slug>/` listing, SmartRecruiters listings, some
    # company-direct careers pages) show metadata + an "Apply for this job"
    # button on the listing URL, and only render the actual form fields after
    # the button is clicked. The click-through pass dispatches a click on the
    # first visible apply-style CTA after hydration and waits for a form/input
    # to appear before returning. Domain-agnostic — matches any text in the
    # patterns list in `_build_crawler_run_config`.
    if not form_html:
        logger.info(
            "extract_form_fields: no form after first browser pass for %s "
            "— retrying with apply-CTA click",
            form_url,
        )
        forced_click = force_browser_fetch(form_url, click_apply_cta=True)
        if forced_click and forced_click.success and forced_click.raw_html:
            forced_html = forced_click.raw_html
            form_html = extract_form_html(forced_html)

    # Tier 3: follow ATS iframe. Some company-direct careers pages
    # (mongodb.com/careers/<id>, similar) embed the apply form inside a
    # cross-origin iframe served by Greenhouse / Lever / etc. Our normal
    # extraction can't see inside cross-origin iframes; instead we fetch
    # the iframe's src directly.
    if not form_html:
        # Search both the cached HTML and the browser-rendered HTML.
        iframe_src = extract_ats_iframe_src(forced_html or raw_html)
        if iframe_src:
            logger.info(
                "extract_form_fields: following ATS iframe %s (parent=%s)",
                iframe_src, form_url,
            )
            iframe_fetch = fetch_url_with_fallback(iframe_src)
            if iframe_fetch.success and iframe_fetch.raw_html:
                form_html = extract_form_html(iframe_fetch.raw_html)

    if not form_html:
        logger.info("extract_form_fields: no <form>/inputs found for %s after all tiers", form_url)
        return {**updates, "form_fields": []}

    # Diagnostic: count input-like markers in the form HTML before the
    # LLM call. Lets us tell apart "LLM saw the markup but didn't
    # extract everything" (raw_inputs >> len(fields), prompt issue) from
    # "the markup was missing the fields in the first place" (raw_inputs
    # already low, browser/extraction issue) when triaging from prod logs.
    import re as _re
    _input_count = len(_re.findall(r"<(input|textarea|select)\b", form_html, _re.IGNORECASE))
    logger.info(
        "extract_form_fields: form_html=%d chars, raw_inputs=%d for %s",
        len(form_html), _input_count, form_url,
    )

    llm = get_llm()
    if llm is None:
        logger.warning(
            "extract_form_fields: no LLM configured — form extraction has no useful heuristic fallback"
        )
        return {**updates, "form_fields": []}

    structured = llm.with_structured_output(FormSchema)
    snippet = form_html[:_MAX_FORM_HTML_FOR_LLM]
    try:
        schema: FormSchema = structured.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"Form HTML:\n\n{snippet}"),
        ])
    except Exception as exc:
        logger.warning("extract_form_fields: LLM call failed — %s", exc)
        return {**updates, "form_fields": []}

    fields: List[Dict[str, Any]] = [f.model_dump() for f in (schema.fields or [])]
    logger.info(
        "extract_form_fields: extracted %d field(s) for %s", len(fields), form_url,
    )
    return {**updates, "form_fields": fields}
