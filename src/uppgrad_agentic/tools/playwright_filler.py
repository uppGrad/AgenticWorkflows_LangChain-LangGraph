"""Drive Playwright to fill an application form deterministically with an
LLM-pick fallback. Pure agentic-side helper — no LangGraph imports, no
references to AutoApplyState. Consumed by `auto_apply_adapter.attempt_auto_fill`
on the backend.

CONTRACT
--------
- Never clicks submit / apply / send buttons. The `submit_clicked` bool on
  the result is always False; we don't expose a code path that could flip it.
- `dry_run` parameter is currently informational only (everything is
  effectively dry-run). Reserved for a future signed-off submission feature.
- Returns a `FormFillResult` populated entirely from observed Playwright
  outcomes; never optimistic.

TIER STRATEGY
-------------
For each FormFieldFillPlan with status="filled":
  Tier 1 (deterministic, free):
    - Locate via [name="X"] OR [id="X"] (the LLM's "name" field is really
      the input's primary identifier — Greenhouse uses id, Ashby uses name).
    - Type-specific action: fill / select_option / set_input_files / check.
  Tier 2 (deterministic, free):
    - get_by_label(label) when name/id miss.
  Tier 3 (deterministic, free):
    - For select that's actually a custom React dropdown: click trigger,
      then click matching option by visible text. Falls back to type-and-Enter.
  Tier 4 (LLM, ~$0.001-0.005 per call):
    - When tiers 1-3 fail, ask gpt-4o-mini to look at the form HTML and
      return a Playwright selector + action. Validate the selector resolves
      to exactly one element, refuse submit-button targets, then act.
    - Bounded by `llm_picker_budget` (default 10 calls per session).

The LLM in Tier 4 is OPTIONAL — pass llm=None to skip the tier entirely. The
caller owns the LLM client (langchain BaseChatModel) so this module has no
LLM dependency at import time.
"""

from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

from uppgrad_agentic.workflows.auto_apply.schemas import (
    FormFieldFillPlan,
    FormFieldFillReport,
    FormFillResult,
    FillFieldOutcome,
)

logger = logging.getLogger(__name__)


# Phrases on a target's text content that disqualify it as a fill target —
# defense-in-depth against an LLM-picker pointing at submit/apply buttons.
_SUBMIT_TEXT_DENYLIST = (
    "submit", "apply now", "send application", "send", "apply",
    "submit application",
)


_LLM_PICKER_SYSTEM = """You are a Playwright selector finder. Given the rendered HTML of an application form and a target field, return a CSS or text-based selector that uniquely identifies the input element to interact with, plus the appropriate Playwright action.

Rules:
- Prefer #id over [name=...] over text-based locators.
- For React-based custom dropdowns (Greenhouse, Ashby), return the trigger
  button or div selector with action="click_then_pick_option" and put the
  option's visible text in `option_text`.
- For hidden file inputs, return the actual <input type=file> selector with
  action="set_input_files".
- For radio groups, prefer input[name=X][value=Y] — never the group container.
- For checkbox/label-wrapped patterns, prefer clicking the <label for=X> via
  action="click_label_for_input" with `linked_input_id` set to the input's id.
- The selector MUST match exactly one element on the page; if you cannot be
  precise, return an empty selector string.
- NEVER return selectors targeting submit / apply / send buttons.

Return one SelectorPlan."""


class _SelectorPlan(BaseModel):
    """Tier 4 LLM output schema."""
    selector: str = Field(default="", description="Playwright selector. Empty if no unique match.")
    action: Literal[
        "fill", "click", "set_input_files", "select_option",
        "check", "click_then_pick_option", "click_label_for_input",
    ] = Field(default="fill")
    option_text: str = Field(default="", description="For click_then_pick_option")
    linked_input_id: str = Field(default="", description="For click_label_for_input")
    notes: str = Field(default="")


# ─── Locator tiers (1-3, deterministic) ──────────────────────────────────────

async def _dismiss_cookie_banners(page) -> None:
    """Best-effort dismiss of OneTrust / TrustE / generic cookie banners."""
    selectors = (
        "#onetrust-accept-btn-handler",
        "button#truste-consent-button",
        'button:has-text("Accept All")',
        'button:has-text("I agree")',
        'button:has-text("Accept")',
    )
    for sel in selectors:
        try:
            btn = await page.query_selector(sel)
            if btn:
                await btn.click(timeout=1500)
                await page.wait_for_timeout(300)
                return
        except Exception:
            continue


async def _force_hydrate(page) -> None:
    """Trigger lazy-mount of off-screen form sections by scrolling."""
    try:
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(800)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
        await page.wait_for_timeout(400)
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(400)
    except Exception:
        pass


async def _detect_captcha(page) -> bool:
    """Look for reCAPTCHA / hCaptcha widgets that would block a submit. We
    surface this in the result; we never try to solve a captcha."""
    selectors = (
        'iframe[src*="recaptcha/api2"]',
        'iframe[src*="hcaptcha.com"]',
        '.g-recaptcha',
        '[class*="captcha-challenge"]',
    )
    for sel in selectors:
        try:
            if await page.locator(sel).count() > 0:
                return True
        except Exception:
            continue
    return False


async def _locate(page, field) -> tuple[Any, str]:
    """Tier 1 + 2. Returns (locator, method) or (None, "none")."""
    name = (field.name or "").strip()
    label = (field.label or "").strip()
    field_type = field.field_type

    def _selectors_for(tag: Optional[str]) -> List[str]:
        if not name:
            return []
        if tag:
            return [f'{tag}[name="{name}"]', f'{tag}[id="{name}"]']
        return [f'[name="{name}"]', f'[id="{name}"]']

    if name:
        if field_type == "select":
            for sel in _selectors_for("select"):
                loc = page.locator(sel)
                if await loc.count() > 0:
                    return loc.first, "name_or_id_native_select"
        if field_type == "textarea":
            for sel in _selectors_for("textarea"):
                loc = page.locator(sel)
                if await loc.count() > 0:
                    return loc.first, "name_or_id"
        elif field_type in ("radio", "checkbox"):
            for sel in _selectors_for("input"):
                loc = page.locator(sel)
                if await loc.count() > 0:
                    return loc.first, "name_or_id"
        elif field_type == "file":
            for sel in (f'input[type="file"][name="{name}"]', f'input[type="file"][id="{name}"]'):
                loc = page.locator(sel)
                if await loc.count() > 0:
                    return loc.first, "name_or_id_file"
            loc = page.locator('input[type="file"]')
            if await loc.count() > 0:
                return loc.first, "first_file_input"
        else:
            for sel in _selectors_for(None):
                loc = page.locator(sel)
                if await loc.count() > 0:
                    return loc.first, "name_or_id"
    if label:
        try:
            loc = page.get_by_label(label, exact=False)
            if await loc.count() > 0:
                return loc.first, "label"
        except Exception:
            pass
    return None, "none"


async def _custom_select_pick(page, locator, option_text: str) -> tuple[bool, str]:
    """Tier 3 for selects that are React custom components.
    Click the trigger, then click an option matching `option_text`."""
    try:
        await locator.click(timeout=3000)
    except Exception:
        try:
            await locator.locator("xpath=..").click(timeout=2000)
        except Exception:
            return False, "trigger_click_failed"
    await page.wait_for_timeout(400)
    candidate_selectors = (
        f'[role="option"]:has-text("{option_text}")',
        f'li:has-text("{option_text}")',
        f'div[role="option"]:has-text("{option_text}")',
    )
    for sel in candidate_selectors:
        try:
            opt = page.locator(sel).first
            if await opt.count() > 0:
                await opt.click(timeout=2000)
                return True, f"clicked:{sel[:30]}"
        except Exception:
            continue
    # Type-and-Enter fallback for autocomplete-style dropdowns
    try:
        await locator.fill(option_text, timeout=1500)
        await page.keyboard.press("Enter")
        return True, "type_enter"
    except Exception:
        return False, "no_option_match"


async def _fill_deterministic(page, plan: FormFieldFillPlan) -> tuple[FillFieldOutcome, str]:
    """Tiers 1-3."""
    field = plan.field
    value = plan.value
    field_type = field.field_type
    name = (field.name or "").strip()

    locator, method = await _locate(page, field)
    if locator is None:
        return ("no_locator", f"name={name!r} label={field.label[:30]!r}")

    try:
        await locator.scroll_into_view_if_needed(timeout=2000)
    except Exception:
        pass

    if field_type == "file":
        try:
            await locator.set_input_files(value, timeout=3000)
            return ("ok", "set_input_files")
        except Exception as exc:
            return ("file_error", str(exc)[:90])

    if field_type == "select":
        if method == "name_or_id_native_select":
            try:
                await locator.select_option(label=str(value), timeout=2000)
                return ("ok", "select_option:label")
            except Exception:
                try:
                    await locator.select_option(value=str(value), timeout=2000)
                    return ("ok", "select_option:value")
                except Exception:
                    pass
        ok, detail = await _custom_select_pick(page, locator, str(value))
        return ("ok" if ok else "select_error", f"custom_select:{detail}")

    if field_type == "checkbox":
        if value and value.lower() not in ("false", "no", "0"):
            try:
                await locator.check(timeout=2000)
                return ("ok", "checked")
            except Exception as exc:
                return ("checkbox_error", str(exc)[:90])
        return ("plan_skip", "value_falsy")

    if field_type == "radio":
        try:
            if name:
                target = page.locator(f'input[name="{name}"][value="{value}"]')
                if await target.count() > 0:
                    await target.first.check(timeout=2000)
                    return ("ok", "by_value")
            await locator.check(timeout=2000)
            return ("ok", "first_in_group")
        except Exception as exc:
            return ("radio_error", str(exc)[:90])

    # text-like
    try:
        await locator.fill(str(value), timeout=2000)
        return ("ok", str(value)[:60])
    except Exception as exc:
        return ("fill_error", str(exc)[:90])


# ─── Tier 4 — LLM picker ──────────────────────────────────────────────────────

async def _get_form_subtree_html(page) -> str:
    """Pull the form area only, to keep tokens bounded."""
    try:
        full_html = await page.content()
    except Exception:
        return ""
    try:
        from uppgrad_agentic.tools.form_extractor import extract_form_html
        out = extract_form_html(full_html)
        if out:
            return out[:30_000]
    except Exception:
        pass
    return full_html[:30_000]


def _is_submit_target_text(text: Optional[str], type_attr: Optional[str]) -> bool:
    if (type_attr or "").strip().lower() == "submit":
        return True
    text_norm = (text or "").strip().lower()
    return any(text_norm == phrase for phrase in _SUBMIT_TEXT_DENYLIST)


async def _llm_pick_and_act(page, plan: FormFieldFillPlan, llm) -> tuple[FillFieldOutcome, str]:
    """Tier 4. ONE LLM call, validated, executed."""
    from langchain_core.messages import HumanMessage, SystemMessage

    form_html = await _get_form_subtree_html(page)
    if not form_html:
        return ("llm_skipped", "no_form_html")

    field = plan.field
    field_summary = (
        f"label={field.label!r}\n"
        f"field_type={field.field_type!r}\n"
        f"name_or_id_extracted={field.name!r}\n"
        f"options={field.options[:8]}\n"
        f"required={field.required}"
    )
    structured = llm.with_structured_output(_SelectorPlan)
    try:
        sp: _SelectorPlan = structured.invoke([
            SystemMessage(content=_LLM_PICKER_SYSTEM),
            HumanMessage(content=f"Field:\n{field_summary}\n\nValue to set: {plan.value!r}\n\nForm HTML:\n{form_html}"),
        ])
    except Exception as exc:
        return ("llm_exec_error", f"llm_call:{type(exc).__name__}:{str(exc)[:60]}")

    if not (sp.selector or "").strip():
        return ("llm_skipped", "empty_selector")

    try:
        loc = page.locator(sp.selector)
        count = await loc.count()
    except Exception as exc:
        return ("llm_exec_error", f"locator_eval:{str(exc)[:60]}")
    if count == 0:
        return ("no_locator", f"llm_no_match: selector resolves to 0")
    if count > 1:
        return ("no_locator", f"llm_ambiguous: selector resolves to {count}")
    target = loc.first

    # Submit-button refusal
    try:
        text = await target.text_content(timeout=1000) or ""
        type_attr = await target.get_attribute("type") or ""
    except Exception:
        text, type_attr = "", ""
    if _is_submit_target_text(text, type_attr):
        return ("llm_refused_submit", f"text={text[:40]!r} type={type_attr!r}")

    try:
        await target.scroll_into_view_if_needed(timeout=2000)
    except Exception:
        pass

    try:
        if sp.action == "click":
            await target.click(timeout=3000)
        elif sp.action == "fill":
            await target.fill(str(plan.value), timeout=3000)
        elif sp.action == "check":
            await target.check(timeout=3000)
        elif sp.action == "set_input_files":
            await target.set_input_files(plan.value, timeout=3000)
        elif sp.action == "select_option":
            try:
                await target.select_option(label=str(plan.value), timeout=2000)
            except Exception:
                await target.select_option(value=str(plan.value), timeout=2000)
        elif sp.action == "click_then_pick_option":
            ok, detail = await _custom_select_pick(page, target, sp.option_text or str(plan.value))
            if ok:
                return ("ok_llm", f"click_pick:{detail}")
            return ("llm_exec_error", f"click_pick:{detail}")
        elif sp.action == "click_label_for_input":
            await target.click(timeout=3000)
            # Verify the linked input is now checked, when one was specified.
            if sp.linked_input_id:
                try:
                    is_checked = await page.evaluate(
                        f'(id) => document.getElementById(id)?.checked ?? null',
                        sp.linked_input_id,
                    )
                    if is_checked is False:
                        # Try one more click to toggle
                        await target.click(timeout=1500)
                except Exception:
                    pass
            return ("ok_llm", f"click_label:{sp.linked_input_id or sp.selector[:30]}")
        else:
            return ("llm_exec_error", f"unknown_action:{sp.action}")
        return ("ok_llm", f"{sp.action}:{sp.selector[:50]}")
    except Exception as exc:
        return ("llm_exec_error", f"{sp.action}:{str(exc)[:60]}")


# ─── Public entrypoint ───────────────────────────────────────────────────────

async def fill_form_async(
    form_url: str,
    plan: List[FormFieldFillPlan],
    *,
    llm: Any = None,
    headless: bool = True,
    llm_picker_budget: int = 10,
    nav_timeout_ms: int = 30_000,
    dry_run: bool = True,
) -> FormFillResult:
    """Drive a Playwright session to fill the form.

    Args:
        form_url: URL of the application form to fill.
        plan: Fill plan from `compute_form_values`.
        llm: Optional langchain BaseChatModel for Tier 4. None disables Tier 4.
        headless: Run Chromium headless. Set False for visual debugging.
        llm_picker_budget: Max Tier 4 calls per session.
        nav_timeout_ms: Page load timeout.
        dry_run: Currently informational; never clicks submit regardless.

    Never clicks submit/apply buttons. Closing the browser at the end is the
    only "side effect" — the form's filled state is discarded.
    """
    from playwright.async_api import async_playwright

    result = FormFillResult(
        form_url=form_url, success=False, fields_total=len(plan),
        submit_clicked=False,
    )
    llm_calls = 0

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            ctx = await browser.new_context(viewport={"width": 1280, "height": 1800})
            page = await ctx.new_page()

            try:
                await page.goto(form_url, wait_until="domcontentloaded", timeout=nav_timeout_ms)
                try:
                    await page.wait_for_load_state("networkidle", timeout=10_000)
                except Exception:
                    pass
                await _dismiss_cookie_banners(page)
                await _force_hydrate(page)

                result.captcha_detected = await _detect_captcha(page)

                for entry in plan:
                    label = entry.field.label[:50]
                    ftype = entry.field.field_type
                    if entry.status != "filled":
                        result.reports.append(FormFieldFillReport(
                            label=label, field_type=ftype,
                            outcome="plan_skip", detail=entry.reason,
                        ))
                        result.fields_skipped += 1
                        continue

                    outcome, detail = await _fill_deterministic(page, entry)
                    if outcome == "ok":
                        result.fields_filled_native += 1
                        result.reports.append(FormFieldFillReport(
                            label=label, field_type=ftype, outcome="ok", detail=detail,
                        ))
                        continue

                    # Failure on Tier 1-3. Try Tier 4 if LLM available + within budget.
                    if llm is not None and llm_calls < llm_picker_budget:
                        llm_calls += 1
                        llm_outcome, llm_detail = await _llm_pick_and_act(page, entry, llm)
                        if llm_outcome == "ok_llm":
                            result.fields_filled_llm += 1
                            result.reports.append(FormFieldFillReport(
                                label=label, field_type=ftype,
                                outcome="ok_llm", detail=llm_detail,
                            ))
                            continue
                        result.fields_failed += 1
                        result.reports.append(FormFieldFillReport(
                            label=label, field_type=ftype,
                            outcome=llm_outcome,
                            detail=f"{detail} → {llm_detail}",
                        ))
                        continue

                    result.fields_failed += 1
                    result.reports.append(FormFieldFillReport(
                        label=label, field_type=ftype, outcome=outcome, detail=detail,
                    ))

                result.llm_picker_calls = llm_calls
                result.success = (
                    result.fields_filled_native + result.fields_filled_llm > 0
                    and not result.submit_clicked
                )
            finally:
                await ctx.close()
                await browser.close()
    except Exception as exc:
        logger.exception("fill_form_async: top-level error")
        result.error = f"{type(exc).__name__}: {str(exc)[:200]}"
        result.success = False

    return result
