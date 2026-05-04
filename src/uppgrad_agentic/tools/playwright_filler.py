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
import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from uppgrad_agentic.workflows.auto_apply.schemas import (
    FormFieldFillPlan,
    FormFieldFillReport,
    FormFillResult,
    FillFieldOutcome,
)

logger = logging.getLogger(__name__)


# Env var that lets a demo / debug session see the browser fill the form
# instead of running headless. Falsy values: "0", "false", "no", "off"
# (case-insensitive). Anything else, including unset, → headless. Read on
# every call (not at import) so a server can flip the flag mid-process for
# a one-off demo without restarting.
_HEADLESS_ENV_VAR = "UPPGRAD_AUTO_FILL_HEADLESS"


def _default_headless() -> bool:
    """Resolve the default value of `headless` for `fill_form_async` from
    the `UPPGRAD_AUTO_FILL_HEADLESS` env var. Returns True (headless) by
    default — production should never run headed unless someone deliberately
    sets the env var."""
    raw = os.environ.get(_HEADLESS_ENV_VAR, "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True


# Phrases on a target's text content that disqualify it as a fill target —
# defense-in-depth against an LLM-picker pointing at submit/apply buttons.
_SUBMIT_TEXT_DENYLIST = (
    "submit", "apply now", "send application", "send", "apply",
    "submit application",
)


_LLM_PICKER_SYSTEM = """You are a Playwright selector finder. Given the rendered HTML of an application form and a target field, return a CSS or text-based selector that uniquely identifies the input element to interact with, plus the appropriate Playwright action.

Rules:
- Prefer #id over [name=...] over text-based locators.
- For React-based custom dropdowns (Greenhouse, Ashby, Anthropic careers,
  Workable, Lever country/location pickers): the input element has typically
  ALREADY had the value typed into it. The LISTBOX is currently open and
  showing options. Your job is to find the option element (NOT the input)
  whose visible text matches the intended value, return its selector, and
  set action="click". Class names vary across vendors (Greenhouse uses
  `.select__option`, `[role="option"]`; Workable uses `.styles_menu_option_*`;
  Anthropic uses `.select__menu` descendants). Read the ACTUAL HTML you
  receive — don't guess class names.
  - If the listbox is NOT open in the HTML you see (no visible options),
    return action="click_then_pick_option" with the trigger element's
    selector + the option's visible text in `option_text`. The fill
    layer will then re-open and pick.
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


# Caps for the post-fill drift correction loop (Tier 5).
# Per-session bumped from 5 → 15 because Anthropic / Greenhouse forms
# routinely have 5+ combobox fields (visa Q, AI policy, in-person Q,
# etc.) that all fail combobox_pick's deterministic listbox match and
# legitimately need LLM correction.
_MAX_CORRECTIONS_PER_FIELD = 2
_MAX_CORRECTIONS_PER_SESSION = 15
# Container HTML cap when sending to the LLM corrector. Field containers
# are typically 300-1500 bytes; 4 KB covers the longest realistic case
# (radio group with 6+ options + nested labels) without ever passing a
# whole form's worth of HTML.
_MAX_CONTAINER_HTML_CHARS = 4_000


def _normalise_for_compare(s: Any) -> str:
    """Lowercase + collapse whitespace for value comparison. Both intended
    and observed go through this before comparing — the DOM may report
    the canonical form ("United States") while we wrote "united states"."""
    if s is None:
        return ""
    return " ".join(str(s).strip().lower().split())


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


async def _locate_file_input(page, field) -> tuple[Any, str]:
    """File-input-specific lookup that survives Greenhouse/Workable-style
    custom uploaders (visible `<button>Attach</button>` + hidden
    `<input type="file">`).

    Walks four tiers, in order:
      1. `input[type="file"]` matched by `name` / `id` — the obvious case
         when the LLM extracted the actual input's attrs.
      2. Inputs scoped under the section labelled with `field.label`. We
         locate the label (heading / strong / `<label>`) by text, walk up
         to the nearest reasonable container (the closest `<fieldset>`,
         `<section>`, or 5 ancestor levels), and look for a file input
         inside. This is what disambiguates "Resume/CV" from "Cover Letter"
         on Greenhouse (each block has its own hidden file input).
      3. First `input[type="file"]` on the page — fine when there's only
         one.
      4. None — caller falls back to LLM picker (Tier 4).
    """
    name = (field.name or "").strip()
    label = (field.label or "").strip()

    if name:
        for sel in (
            f'input[type="file"][name="{name}"]',
            f'input[type="file"][id="{name}"]',
        ):
            loc = page.locator(sel)
            if await loc.count() > 0:
                return loc.first, "file_name_or_id"

    if label:
        # Use Playwright's text matcher to find the label heading. This is
        # broader than `get_by_label` (which would return the FIRST control
        # the label is associated with — typically the button). We want the
        # text node itself, then walk to its parent section.
        for sel in (
            f'label:has-text("{label}")',
            f'div:has-text("{label}")',
            f'h2:has-text("{label}")', f'h3:has-text("{label}")',
            f'h4:has-text("{label}")', f'strong:has-text("{label}")',
        ):
            try:
                heading = page.locator(sel).first
                if await heading.count() == 0:
                    continue
            except Exception:
                continue
            # Walk up via closest() until we hit a meaningful container,
            # then look inside it.
            for ancestor_sel in (
                "xpath=ancestor::fieldset[1]",
                "xpath=ancestor::section[1]",
                "xpath=ancestor::div[descendant::input[@type='file']][1]",
                "xpath=ancestor::*[5]",
            ):
                try:
                    container = heading.locator(ancestor_sel)
                    file_in = container.locator('input[type="file"]').first
                    if await file_in.count() > 0:
                        return file_in, "file_in_labelled_container"
                except Exception:
                    continue

    loc = page.locator('input[type="file"]')
    if await loc.count() > 0:
        return loc.first, "first_file_input"
    return None, "none"


# ─── Phase 1: combobox detection + native-setter dispatch ───────────────────


def _is_autocomplete_field(field) -> bool:
    """Predicate: should this field be treated as a combobox/autocomplete
    that needs click+pick instead of plain text-fill?

    Mirrors `browser-use`'s `_is_autocomplete_field` rule (the discriminator
    none of our previous tiers had access to). Without this, every
    `<input type="text" role="combobox" aria-autocomplete="list">` looks
    identical to a plain text input and Tier 1's `.fill()` silently
    succeeds while the React state never registers an option pick.

    Triggers when ANY of:
      - role == "combobox"
      - aria_autocomplete in ("list", "both", "inline")
      - list_id (datalist target) is present
      - aria_haspopup is set (and not "false") AND the popup is wired
        to a controlled element (aria_controls or aria_owns)
    """
    role = (getattr(field, "role", "") or "").strip().lower()
    if role == "combobox":
        return True
    aria_ac = (getattr(field, "aria_autocomplete", "") or "").strip().lower()
    if aria_ac in ("list", "both", "inline"):
        return True
    if (getattr(field, "list_id", "") or "").strip():
        return True
    haspopup = (getattr(field, "aria_haspopup", "") or "").strip().lower()
    if haspopup and haspopup != "false":
        if (getattr(field, "aria_controls", "") or "").strip() or (
            getattr(field, "aria_owns", "") or ""
        ).strip():
            return True
    return False


async def _native_setter_dispatch(locator, value: str) -> bool:
    """Set an input's value via the native React/Vue setter, then dispatch
    the input/change events the framework listens for.

    Some controlled inputs (React's onChange-bound `<input>`) silently
    discard `.fill()` writes because Playwright's keyboard simulation
    doesn't propagate to the framework's value setter. browser-use uses
    this exact trick (`Object.getOwnPropertyDescriptor(...).set.call(el)`)
    as its reliability backstop.

    Returns True when the value successfully landed (post-write `el.value`
    matches), False otherwise."""
    try:
        ok = await locator.evaluate(
            r"""(el, v) => {
                const proto = el.tagName === 'TEXTAREA'
                    ? HTMLTextAreaElement.prototype
                    : HTMLInputElement.prototype;
                const desc = Object.getOwnPropertyDescriptor(proto, 'value');
                if (!desc || !desc.set) return false;
                desc.set.call(el, v);
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                return el.value === v;
            }""",
            value,
        )
        return bool(ok)
    except Exception:
        return False


async def _combobox_pick(page, locator, option_text: str) -> tuple[bool, str]:
    """Type into the combobox trigger and click the matching dropdown
    option. Mirrors browser-use's combobox interaction:

      1. Focus + clear the input.
      2. Type the option text character-by-character (some autocomplete
         widgets only react to keypress events, not bulk fill).
      3. Wait briefly for the listbox/options to populate.
      4. Click the visible option whose text matches.

    Falls back to Enter-after-type when no clickable option appears
    (datalist-style autocomplete commits on Enter)."""
    try:
        await locator.click(timeout=2000)
    except Exception:
        pass
    try:
        await locator.fill("", timeout=1500)
    except Exception:
        pass
    try:
        await locator.press_sequentially(str(option_text), delay=20, timeout=4000)
    except Exception as exc:
        return (False, f"type_failed:{str(exc)[:60]}")

    # Wait briefly for the listbox to populate. Different ATSes use
    # different markup — Lever uses [role="listbox"], Greenhouse uses
    # `.select__menu`, Workable uses `.styles_menu__*`. Cast a wide net.
    listbox_selectors = (
        '[role="listbox"] [role="option"]',
        '[role="option"]',
        '.select__option',
        'ul[role="listbox"] li',
        'li[role="option"]',
    )
    for sel in listbox_selectors:
        try:
            await page.wait_for_selector(sel, timeout=1500, state="visible")
            break
        except Exception:
            continue

    # Find the option whose visible text matches the value.
    needle = str(option_text).strip().lower()
    for sel in listbox_selectors:
        try:
            opts = page.locator(sel)
            count = await opts.count()
            for i in range(min(count, 30)):
                opt = opts.nth(i)
                text = (await opt.text_content(timeout=500) or "").strip().lower()
                if not text:
                    continue
                if text == needle or needle in text or text in needle:
                    await opt.click(timeout=2000)
                    return (True, f"picked_option:{sel}")
        except Exception:
            continue

    # Enter-commit fallback for datalist / inline autocomplete.
    try:
        await locator.press("Enter", timeout=1500)
        return (True, "enter_commit")
    except Exception:
        pass
    return (False, "no_option_matched")


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

    if field_type == "file":
        return await _locate_file_input(page, field)

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
    """Tiers 0-3."""
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

    # ─── Tier 0: combobox / autocomplete predicate ────────────────────────
    # Runs BEFORE the type-specific tiers below. When ARIA signals say
    # "this is a combobox," skip the text-fill path entirely and go
    # straight to type-then-pick. This is the discriminator we were
    # missing — fixes the Lever country picker / Greenhouse location
    # autocomplete cases without any LLM call.
    #
    # Crucially: when combobox_pick FAILS (selector miss on the option
    # listbox; vendor uses class names we don't hardcode — Greenhouse
    # `.select__option`, Workable `.styles_menu_option_*`, etc.), we
    # return a non-"ok" outcome to escalate to the Tier-4 LLM picker
    # rather than falling through to plain text-fill. The LLM sees the
    # freshly-rendered DOM (listbox is open after typing) and picks the
    # matching option by visible text — no hardcoded class names
    # needed. Without this escalation, the typed text stays in the
    # input, .fill() returns "ok", and the bad fill is invisible to
    # every downstream tier.
    if _is_autocomplete_field(field) and field_type in (
        "text", "select", "url", "email", "tel"
    ):
        ok, detail = await _combobox_pick(page, locator, str(value))
        if ok:
            return ("ok", f"combobox_pick:{detail}")
        return ("select_error", f"combobox_pick_failed:{detail}")

    if field_type == "file":
        try:
            await locator.set_input_files(value, timeout=3000)
            return ("ok", "set_input_files")
        except Exception as exc:
            err = str(exc)
            # Greenhouse-style "Attach" buttons: the resolved locator is
            # a `<button>`, not the hidden `<input type="file">`. Retry
            # by walking from the locator to its nearest sibling/
            # descendant file input within a small ancestor window.
            if "HTMLInputElement" in err or "set_input_files" in err:
                try:
                    sibling = locator.locator(
                        "xpath=ancestor-or-self::*"
                        "[descendant-or-self::input[@type='file']][1]"
                    ).locator('input[type="file"]').first
                    if await sibling.count() > 0:
                        await sibling.set_input_files(value, timeout=3000)
                        return ("ok", "set_input_files:sibling_recovery")
                except Exception:
                    pass
            return ("file_error", err[:90])

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

    # text-like — Recruitee / SmartRecruiters / similar use React-wrapped
    # inputs where the visible element is `[name="candidate[first_name]"]`
    # but the *actionable* input only becomes interactive after a focus
    # event or hydration animation. Plain `.fill()` with a 2s budget
    # times out before the input accepts input.
    #
    # Three-step recovery:
    #   1. `.fill()` with a slightly longer 5s budget — covers normal
    #      hydration delay (Recruitee usually settles within ~1-3s).
    #   2. On timeout, click to focus + retry `.fill()` (3s). The click
    #      kicks React into "actionable" state for the wrapped input.
    #   3. On timeout still, click + `keyboard.press_sequentially()`
    #      types char-by-char, which most React onChange handlers accept.
    try:
        await locator.fill(str(value), timeout=5000)
        return ("ok", str(value)[:60])
    except Exception as fill_exc:
        # Step 2: click first, then fill again
        try:
            await locator.click(timeout=2000)
            await locator.fill(str(value), timeout=3000)
            return ("ok", f"fill_after_click:{str(value)[:50]}")
        except Exception:
            pass
        # Step 3: type char-by-char
        try:
            await locator.click(timeout=2000)
            await locator.press_sequentially(str(value), delay=20, timeout=5000)
            return ("ok", f"press_sequentially:{str(value)[:40]}")
        except Exception:
            pass
        # Step 4: native-setter fallback — last-resort for React/Vue
        # controlled inputs that swallow Playwright's fill/type entirely.
        # `Object.getOwnPropertyDescriptor(...).set.call(el, v)` writes
        # directly to the DOM property and dispatches input/change so the
        # framework's onChange listener fires.
        if await _native_setter_dispatch(locator, str(value)):
            return ("ok", f"native_setter:{str(value)[:40]}")
        return ("fill_error", str(fill_exc)[:90])


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


# ─── Tier 5 — post-fill state verification + drift correction ─────────────────
#
# The deterministic tiers (1-3) and the LLM picker (Tier 4) only check
# "did Playwright's action throw?" — not "is the form's React state
# coherent with what we intended?". Two real failure modes that mode
# can't catch:
#
#   * Combobox-with-search treated as text input. Tier 1 finds the
#     visible <input> backing a custom dropdown and runs .fill("USA").
#     The input's value is "USA", we declare success, but no option is
#     selected in the React state — submit time treats the field as
#     empty.
#
#   * .fill() that silently no-ops (read-only inputs, validation hooks
#     that revert, disabled inputs that look enabled).
#
# Phase 1 (deterministic state probe) reads the post-fill DOM state
# per field via a single page.evaluate. Phase 2 (LLM corrector) only
# runs on observed mismatches and gets ONLY that field's container
# (not the whole form, not the page) — bounded by `_MAX_CONTAINER_HTML_CHARS`.

_PROBE_JS = r"""
(args) => {
  const { selector, fieldType } = args;
  const el = document.querySelector(selector);
  if (!el) return { found: false, observed: '', notes: 'no_element' };

  const t = (s) => (s == null ? '' : String(s).trim());

  if (fieldType === 'select') {
    if (el.tagName === 'SELECT') {
      const opt = el.options[el.selectedIndex];
      return {
        found: true,
        observed: t(opt && (opt.label || opt.text || opt.value)),
        notes: 'native_select',
      };
    }
    // Custom widget: walk up to the nearest container that has labelled
    // selected state, then read either an aria-selected descendant's text
    // or the trigger's visible value.
    const container = el.closest('[role="combobox"], [role="listbox"], .select__container, .select-shell') || el.parentElement;
    if (container) {
      const sel = container.querySelector('[aria-selected="true"], [data-selected="true"], .select__single-value, .select__multi-value');
      if (sel) return { found: true, observed: t(sel.textContent), notes: 'aria_selected' };
    }
    return { found: true, observed: t(el.value || el.textContent), notes: 'fallback_text' };
  }

  if (fieldType === 'checkbox') {
    return { found: true, observed: el.checked ? 'true' : 'false', notes: 'checked' };
  }

  if (fieldType === 'radio') {
    // Walk siblings in the same name-group and return the checked one's value.
    const name = el.getAttribute('name');
    if (name) {
      const group = document.querySelectorAll(`input[type="radio"][name="${CSS.escape(name)}"]`);
      for (const r of group) {
        if (r.checked) return { found: true, observed: t(r.value || r.id), notes: 'group_checked' };
      }
      return { found: true, observed: '', notes: 'group_no_check' };
    }
    return { found: true, observed: el.checked ? 'true' : 'false', notes: 'lone_radio' };
  }

  if (fieldType === 'file') {
    return {
      found: true,
      observed: el.files && el.files.length > 0 ? `${el.files.length}_files` : '',
      notes: 'file_count',
    };
  }

  // text-like (text/email/tel/url/number/date/textarea)
  return { found: true, observed: t(el.value), notes: 'value' };
}
"""


async def _probe_field_state(
    page,
    plan: FormFieldFillPlan,
    locator,
) -> str:
    """Read the post-fill DOM state of `plan.field` and return the
    observed value. Side-effect: writes any form-validation error
    discovered (aria-invalid / sibling error text) onto
    `plan.validation_error` so the verifier + corrector can use it
    without changing the function signature.

    The sane/insane verdict is delegated to the LLM batch verifier
    (`_llm_verify_batch`). This function ONLY reads state. The
    previous deterministic comparison was prone to false-positives
    (combobox typed without option-pick still has the typed string in
    `el.value`) and false-negatives (autocomplete rewrites).

    Best-effort — when the probe can't read state (element detached,
    JS error), returns "".
    """
    field_type = plan.field.field_type
    try:
        result = await locator.evaluate(
            r"""(el, args) => {
                const { fieldType } = args;
                const t = (s) => (s == null ? '' : String(s).trim());

                // ── Form-validation read: aria-invalid + nearest error sibling. ──
                // ATSes mark validation errors in different ways:
                //   - aria-invalid="true" on the input (Greenhouse, Workable)
                //   - <span class="field-error"> sibling (Lever, SmartRecruiters)
                //   - <div class="error"> in the closest form-row container
                //   - [role="alert"] descendant
                // Walk up to the field's container, then look for ANY of these
                // shapes. Return empty string when no error is visible.
                let validation_error = '';
                const ariaInvalid = (el.getAttribute('aria-invalid') || '').toLowerCase();
                if (ariaInvalid === 'true') validation_error = 'aria-invalid';
                const container = el.closest(
                    '.form-field, .field, .form-row, [data-qa-field], .application-question, fieldset, label'
                ) || el.parentElement;
                if (container) {
                    const errSel = container.querySelector(
                        '.field-error, .form-error, .error-message, [class*="error" i]:not(input):not(select):not(textarea):not(label), [role="alert"]'
                    );
                    if (errSel) {
                        const txt = t(errSel.textContent);
                        // Filter out generic "error" placeholders that aren't real
                        // user-facing messages (some forms always render the slot).
                        if (txt && txt.length > 0 && txt.length < 300) {
                            validation_error = txt;
                        }
                    }
                }

                if (fieldType === 'select') {
                    if (el.tagName === 'SELECT') {
                        const opt = el.options[el.selectedIndex];
                        return { observed: t(opt && (opt.label || opt.text || opt.value)), notes: 'native_select', validation_error };
                    }
                    const c = el.closest('[role="combobox"], [role="listbox"], .select__container, .select-shell') || el.parentElement;
                    if (c) {
                        const sel = c.querySelector('[aria-selected="true"], [data-selected="true"], .select__single-value, .select__multi-value');
                        if (sel) return { observed: t(sel.textContent), notes: 'aria_selected', validation_error };
                    }
                    return { observed: t(el.value || el.textContent), notes: 'fallback_text', validation_error };
                }
                if (fieldType === 'checkbox') return { observed: el.checked ? 'true' : 'false', notes: 'checked', validation_error };
                if (fieldType === 'radio') {
                    const name = el.getAttribute('name');
                    if (name) {
                        const group = document.querySelectorAll(`input[type="radio"][name="${CSS.escape(name)}"]`);
                        for (const r of group) {
                            if (r.checked) return { observed: t(r.value || r.id), notes: 'group_checked', validation_error };
                        }
                        return { observed: '', notes: 'group_no_check', validation_error };
                    }
                    return { observed: el.checked ? 'true' : 'false', notes: 'lone_radio', validation_error };
                }
                if (fieldType === 'file') {
                    return { observed: el.files && el.files.length > 0 ? `${el.files.length}_files` : '', notes: 'file_count', validation_error };
                }
                return { observed: t(el.value), notes: 'value', validation_error };
            }""",
            {"fieldType": field_type},
        )
    except Exception as exc:
        logger.debug("_probe_field_state: probe failed for %s — %s", plan.field.label[:30], exc)
        plan.validation_error = ""
        return ""

    payload = result or {}
    plan.validation_error = (payload.get("validation_error") or "").strip()
    return (payload.get("observed") or "")


# ─── Phase 2: LLM batch sanity check ─────────────────────────────────────────


_BATCH_VERIFY_SYSTEM = """You are verifying that an auto-filler set the right value in each field of a job application form.

For each row you receive, decide whether the `observed` value is a SENSIBLE answer to the question described by `label` (and the original `intended` value we tried to set).

Rules:
- If `validation_error` on a row is non-empty, the form has REJECTED the fill — that's drift regardless of value match. Use the error text to choose `suggested_value` (e.g. error "Phone must be 10 digits" + intended "+90 555 555 5555" → suggested "5555555555").
- Treat semantic mismatch as drift. Example: label="Are you open to relocation for this role?", intended="Yes", observed="Ankara, Turkey" → drift, because a city name isn't a yes/no answer.
- Treat autocomplete decoration as match. Example: label="Country", intended="United States", observed="United States — primary" → sane (autocomplete added a hint). Likewise "san fr" → "San Francisco, CA" is sane on a city autocomplete.
- An empty `observed` while `intended` is non-empty → drift (the form rejected the value or the input was the wrong one).
- A combobox where the user typed text but didn't select an option (observed equals intended exactly but the field needed a pick from a list) → drift. Hint: such fields will have field_type="select" or aria_autocomplete on the metadata.
- File uploads: observed of "1_files" or "2_files" → sane. Empty → drift.
- For yes/no labels, accept "Yes"/"No" or any synonym (e.g. "True"/"False", "Authorized"/"Not authorized") that matches intent.

When drift is detected, suggest the value the form likely expects in `suggested_value` (when obvious from context). Otherwise leave it empty.

Return one verdict per row, in the same order as the input."""


class _FieldVerdict(BaseModel):
    """One row of `_BatchVerifyResult`. Indexed by `idx` so the caller
    can match each verdict back to its FormFieldFillPlan."""
    idx: int = Field(..., description="Index into the input list (matches the row sent)")
    sane: bool = Field(..., description="True when observed is a sensible answer for label")
    reason: str = Field(default="", description="Short explanation, ≤200 chars")
    suggested_value: str = Field(
        default="",
        description="When sane=False AND a better value is obvious from the label, suggest it. Empty otherwise.",
    )


class _BatchVerifyResult(BaseModel):
    verdicts: List[_FieldVerdict] = Field(default_factory=list)


async def _llm_verify_batch(
    plan: List[FormFieldFillPlan],
    llm,
) -> Dict[int, _FieldVerdict]:
    """Single LLM call to sanity-check every filled field's observed
    value against its label + intended value.

    Returns a dict mapping plan-index → verdict. Fields whose status
    isn't 'filled' or whose observed_value matches intended trivially
    are short-circuited and not sent to the LLM (still returned with
    sane=True).

    Bounded cost: 1 LLM call per session regardless of field count.
    Returns `{}` (no verdicts) when llm is None — caller treats every
    field as sane in that case (preserves the no-LLM heuristic path).
    """
    if llm is None:
        return {}

    from langchain_core.messages import HumanMessage, SystemMessage  # lazy

    items: List[Dict[str, Any]] = []
    trivial_sane: Dict[int, _FieldVerdict] = {}
    trivial_drift: Dict[int, _FieldVerdict] = {}
    for i, p in enumerate(plan):
        if p.status != "filled":
            continue
        intended = (p.value or "").strip()
        observed = (p.observed_value or "").strip()

        # Form-validation error trap: ALWAYS drift, regardless of value
        # match. The form told us it rejected the fill (aria-invalid /
        # sibling error message); the corrector needs the error text
        # to know how to fix it.
        validation_error = (getattr(p, "validation_error", "") or "").strip()
        if validation_error:
            trivial_drift[i] = _FieldVerdict(
                idx=i, sane=False,
                reason=f"validation_error:{validation_error[:120]}",
                suggested_value="",
            )
            continue

        # Empty intended is always trivially sane — nothing was supposed
        # to be set, observed is irrelevant.
        if not intended:
            trivial_sane[i] = _FieldVerdict(idx=i, sane=True, reason="empty_intended")
            continue

        # Deterministic drift trap for comboboxes: ARIA says combobox AND
        # observed echoes intended (typed but not picked). This is the
        # canonical combobox-typed-without-pick failure — cheap to flag,
        # expensive to miss. Drift corrector then fires with the intended
        # value; LLM picks the option from the rendered listbox.
        if _is_autocomplete_field(p.field) and (
            _normalise_for_compare(intended) == _normalise_for_compare(observed)
        ):
            trivial_drift[i] = _FieldVerdict(
                idx=i, sane=False,
                reason="combobox_typed_without_option_pick",
                suggested_value=intended,
            )
            continue

        # When intended doesn't match observed at all, send to LLM —
        # could be drift (combobox echo, wrong sibling) or could be
        # benign (autocomplete decoration like "United States — primary",
        # form normalisation like phone "+90 555 555 5555" → "+90 555
        # 555 55 55"). LLM decides.
        if _normalise_for_compare(intended) != _normalise_for_compare(observed):
            items.append({
                "idx": i,
                "label": p.field.label,
                "field_type": p.field.field_type,
                "name": p.field.name or "",
                "options": list(p.field.options or [])[:8],
                "aria_autocomplete": getattr(p.field, "aria_autocomplete", "") or "",
                "role": getattr(p.field, "role", "") or "",
                "intended": intended,
                "observed": observed,
                "validation_error": validation_error,
            })
            continue

        # Exact match. For VANILLA inputs (no ARIA / role decoration,
        # plain field_type=text/email/tel/url/number/textarea), trust
        # the match — short-circuit. For everything else (custom roles,
        # aria-decorated, indexed names like urls[X], file inputs, radio
        # groups), force LLM verification because exact-string-match can
        # still be semantically wrong:
        #   - urls[LinkedIn] vs urls[GitHub]: both accept the same string
        #     so we typed "https://linkedin.com/in/x" into either, then
        #     read back the same string — exact match, but if we hit
        #     urls[GitHub] the LinkedIn URL is now in the wrong field.
        #   - hidden file input behind a custom <button>: set_input_files
        #     reports success on the hidden input but the visible widget
        #     may show a different state.
        #   - any field with `role` set: by definition not a vanilla
        #     control; the framework owns its semantics, deterministic
        #     match is unreliable.
        ftype = p.field.field_type
        is_vanilla_text = ftype in ("text", "email", "tel", "url", "number", "textarea", "date")
        has_aria_decoration = bool(
            (getattr(p.field, "role", "") or "").strip()
            or (getattr(p.field, "aria_autocomplete", "") or "").strip()
            or (getattr(p.field, "aria_haspopup", "") or "").strip()
            or (getattr(p.field, "aria_controls", "") or "").strip()
            or (getattr(p.field, "list_id", "") or "").strip()
        )
        # Indexed-name pattern (urls[LinkedIn], candidate[first_name],
        # etc.) → groups of similar inputs where exact-match alone
        # doesn't prove we hit the right index.
        name = (p.field.name or "")
        is_indexed_group = "[" in name and "]" in name

        if is_vanilla_text and not has_aria_decoration and not is_indexed_group:
            trivial_sane[i] = _FieldVerdict(idx=i, sane=True, reason="vanilla_exact_match")
            continue

        # Non-vanilla exact match → still send to LLM. Cheap (one row per
        # ambiguous field on top of a single batched call) but covers
        # the wrong-input-but-same-value case.
        items.append({
            "idx": i,
            "label": p.field.label,
            "field_type": p.field.field_type,
            "name": p.field.name or "",
            "options": list(p.field.options or [])[:8],
            "aria_autocomplete": getattr(p.field, "aria_autocomplete", "") or "",
            "role": getattr(p.field, "role", "") or "",
            "intended": intended,
            "observed": observed,
        })

    if not items:
        # Even with no LLM-bound items, the deterministic combobox drift
        # traps still apply.
        return {**trivial_sane, **trivial_drift}

    structured = llm.with_structured_output(_BatchVerifyResult)
    try:
        result: _BatchVerifyResult = structured.invoke([
            SystemMessage(content=_BATCH_VERIFY_SYSTEM),
            HumanMessage(content=f"Verify these filled fields. Return one verdict per row.\n\n{items}"),
        ])
    except Exception as exc:
        logger.warning("_llm_verify_batch: LLM call failed — treating all non-trapped as sane (%s)", exc)
        return {**trivial_sane, **trivial_drift}

    out = {**trivial_sane, **trivial_drift}
    for v in result.verdicts:
        out[v.idx] = v
    return out


# ─── Drift corrector — Phase 2 (LLM, on-demand, container-scoped) ─────────────

_DRIFT_CORRECTOR_SYSTEM = """You are a Playwright drift corrector for a form field that was filled but whose DOM state diverged from the intended value, OR whose form-validation rejected the fill.

You will receive:
  - the field's label and type
  - role + aria_autocomplete (combobox indicators)
  - the value we intended to set
  - the value the DOM is currently showing
  - validation_error: form-validation message captured from the DOM
    (aria-invalid="true" OR error-text in a sibling). When non-empty,
    the form has REJECTED the fill — propose a corrective action that
    addresses the error (e.g. "Phone must be 10 digits" → reformat the
    value; "This field is required" → re-attempt the fill; "Invalid
    email" → check format).
  - the field's CONTAINER HTML (the element wrapping label + control —
    typically a fieldset, [role=group], .form-row, or label). This is
    NOT the whole form. Reason about it as a self-contained widget.

Common drift patterns and the fix shape:

  1. Combobox-with-search where we typed text into the input but no
     option is selected. Fix: action="click_then_pick_option" with
     selector targeting the trigger / input, option_text=intended.

  2. Yes/No collected as free text and stuffed into a radio group.
     Fix: action="click" with selector input[name=GROUP][value=VALUE],
     where VALUE is the option matching intended.

  3. Hidden native <select> covered by a custom widget. Fix: target the
     visible custom trigger with action="click_then_pick_option".

  4. Checkbox that needs .check() not .click() (toggle race). Fix:
     action="check" on the input itself.

Selector rules:
- MUST resolve to exactly one element on the page.
- NEVER target submit / apply / send buttons.
- Prefer #id over [name=...] over text-based locators.
- If you can't propose a high-confidence fix, return an empty selector.

Return a single SelectorPlan."""


async def _container_html_for_field(locator) -> str:
    """Return the smallest meaningful container around the field (label +
    control + sibling options). Capped at `_MAX_CONTAINER_HTML_CHARS` so
    a single drift correction call can never receive a whole form's
    worth of HTML."""
    try:
        html = await locator.evaluate(
            r"""(el) => {
                const container = el.closest('[role="group"], fieldset, .form-row, .form-field, label, [data-qa-field], .field-container, .application-question') || el.parentElement;
                return (container || el).outerHTML || '';
            }"""
        )
    except Exception:
        return ""
    return (html or "")[:_MAX_CONTAINER_HTML_CHARS]


async def _correct_field_drift(
    page,
    plan: FormFieldFillPlan,
    locator,
    llm,
) -> tuple[FillFieldOutcome, str]:
    """Single iteration of the drift correction loop. The outer loop
    (`fill_form_async`) handles per-field and per-session caps."""
    from langchain_core.messages import HumanMessage, SystemMessage

    container_html = await _container_html_for_field(locator)
    if not container_html:
        return ("llm_skipped", "no_container_html")

    field = plan.field
    field_summary = (
        f"label={field.label!r}\n"
        f"field_type={field.field_type!r}\n"
        f"name_or_id_extracted={field.name!r}\n"
        f"role={getattr(field, 'role', '')!r}\n"
        f"aria_autocomplete={getattr(field, 'aria_autocomplete', '')!r}\n"
        f"options={field.options[:8]}\n"
        f"intended_value={plan.value!r}\n"
        f"observed_value={plan.observed_value!r}\n"
        f"validation_error={getattr(plan, 'validation_error', '')!r}\n"
    )

    structured = llm.with_structured_output(_SelectorPlan)
    try:
        sp: _SelectorPlan = structured.invoke([
            SystemMessage(content=_DRIFT_CORRECTOR_SYSTEM),
            HumanMessage(content=f"Field state:\n{field_summary}\nContainer HTML:\n{container_html}"),
        ])
    except Exception as exc:
        return ("llm_exec_error", f"corrector_call:{type(exc).__name__}:{str(exc)[:60]}")

    if not (sp.selector or "").strip():
        return ("llm_skipped", "corrector_no_proposal")

    # Reuse the existing Tier-4 validation + execution path. The submit-
    # button denylist + selector uniqueness check live there — we want
    # the same guardrails for corrections.
    try:
        loc = page.locator(sp.selector)
        count = await loc.count()
    except Exception as exc:
        return ("llm_exec_error", f"corrector_locator:{str(exc)[:60]}")
    if count == 0:
        return ("no_locator", "corrector_no_match")
    if count > 1:
        return ("no_locator", f"corrector_ambiguous:{count}")
    target = loc.first

    try:
        text = await target.text_content(timeout=1000) or ""
        type_attr = await target.get_attribute("type") or ""
    except Exception:
        text, type_attr = "", ""
    if _is_submit_target_text(text, type_attr):
        return ("llm_refused_submit", f"corrector_submit:text={text[:40]!r}")

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
            if not ok:
                return ("llm_exec_error", f"corrector_pick:{detail}")
        elif sp.action == "click_label_for_input":
            await target.click(timeout=3000)
        else:
            return ("llm_exec_error", f"corrector_unknown_action:{sp.action}")
    except Exception as exc:
        return ("llm_exec_error", f"corrector_exec:{sp.action}:{str(exc)[:60]}")

    return ("ok_corrected", f"{sp.action}:{sp.selector[:50]}")


# ─── Public entrypoint ───────────────────────────────────────────────────────

async def fill_form_async(
    form_url: str,
    plan: List[FormFieldFillPlan],
    *,
    llm: Any = None,
    headless: Optional[bool] = None,
    llm_picker_budget: int = 10,
    nav_timeout_ms: int = 30_000,
    dry_run: bool = True,
) -> FormFillResult:
    """Drive a Playwright session to fill the form.

    Args:
        form_url: URL of the application form to fill.
        plan: Fill plan from `compute_form_values`.
        llm: Optional langchain BaseChatModel for Tier 4. None disables Tier 4.
        headless: Run Chromium headless. ``None`` (default) reads the
            ``UPPGRAD_AUTO_FILL_HEADLESS`` env var — falsy values
            (``0|false|no|off``, case-insensitive) launch a visible
            browser, suitable for demo recordings or visual debugging.
            Anything else / unset → headless. Tests / scripts that pass
            ``True`` or ``False`` explicitly bypass the env var.
        llm_picker_budget: Max Tier 4 calls per session.
        nav_timeout_ms: Page load timeout.
        dry_run: Currently informational; never clicks submit regardless.

    Never clicks submit/apply buttons. Closing the browser at the end is the
    only "side effect" — the form's filled state is discarded.
    """
    if headless is None:
        headless = _default_headless()
        logger.info("fill_form_async: headless resolved from env → %s", headless)

    # Auto-resolve the LLM from the agentic provider factory when the
    # caller didn't supply one. Without this, Tier 4 LLM picker, Phase 2
    # batch verifier, and drift corrector all stay silent — and the
    # deterministic Tier 0-3 + state probe is *known* to be insufficient
    # for combobox-typed-without-pick / urls[X] disambiguation /
    # form-validation-error correction. The backend's auto_apply_adapter
    # currently passes `llm=None`; this fallback is what gives every
    # call site automatic LLM recovery when OPENAI_API_KEY is set.
    if llm is None:
        try:
            from uppgrad_agentic.common.llm import get_llm
            llm = get_llm()
            if llm is not None:
                logger.info("fill_form_async: resolved LLM via common.llm.get_llm()")
        except Exception as exc:
            logger.debug("fill_form_async: get_llm() failed (%s) — staying llm=None", exc)
            llm = None

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

                drift_calls = 0  # LLM correction calls this session

                # ─── Pass 1: fill every field (Tiers 0-4) + read observed
                # state. The LLM batch verifier (Pass 2) decides sane/insane.
                # `pending_reports[i]` holds the in-progress report for
                # plan[i]; we mutate it post-verification.
                pending_reports: List[Optional[FormFieldFillReport]] = [None] * len(plan)
                fill_sources: List[str] = ["none"] * len(plan)
                outcomes: List[str] = ["pending"] * len(plan)

                for i, entry in enumerate(plan):
                    label = entry.field.label[:50]
                    ftype = entry.field.field_type
                    if entry.status != "filled":
                        pending_reports[i] = FormFieldFillReport(
                            label=label, field_type=ftype,
                            outcome="plan_skip", detail=entry.reason,
                        )
                        outcomes[i] = "plan_skip"
                        continue

                    outcome, detail = await _fill_deterministic(page, entry)
                    fill_source = "native"

                    if outcome != "ok":
                        if llm is not None and llm_calls < llm_picker_budget:
                            llm_calls += 1
                            llm_outcome, llm_detail = await _llm_pick_and_act(page, entry, llm)
                            if llm_outcome == "ok_llm":
                                outcome, detail = "ok_llm", llm_detail
                                fill_source = "llm"
                            else:
                                pending_reports[i] = FormFieldFillReport(
                                    label=label, field_type=ftype,
                                    outcome=llm_outcome,
                                    detail=f"{detail} → {llm_detail}",
                                )
                                outcomes[i] = "fail"
                                continue
                        else:
                            pending_reports[i] = FormFieldFillReport(
                                label=label, field_type=ftype, outcome=outcome, detail=detail,
                            )
                            outcomes[i] = "fail"
                            continue

                    # Read post-fill state. The verdict comes later from
                    # the LLM batch verifier.
                    locator, _method = await _locate(page, entry.field)
                    if locator is not None:
                        try:
                            await page.wait_for_timeout(300)
                        except Exception:
                            pass
                        entry.observed_value = await _probe_field_state(page, entry, locator)

                    fill_sources[i] = fill_source
                    outcomes[i] = "filled_pending_verify"
                    pending_reports[i] = FormFieldFillReport(
                        label=label, field_type=ftype,
                        outcome="ok" if fill_source == "native" else "ok_llm",
                        detail=detail,
                    )

                # ─── Pass 2: ONE batch LLM verification across all filled
                # fields. Catches semantic mismatches the deterministic
                # compare can't (e.g. "Are you open to relocation?" →
                # "Ankara, Turkey") + autocomplete-rewrite false positives
                # (e.g. "san fr" → "San Francisco, CA") in the same call.
                verdicts = await _llm_verify_batch(plan, llm)

                # ─── Pass 3: per-insane-field drift correction. Bounded.
                for i, entry in enumerate(plan):
                    if outcomes[i] != "filled_pending_verify":
                        continue
                    verdict = verdicts.get(i)
                    is_sane = verdict.sane if verdict is not None else True
                    entry.verified = is_sane
                    if fill_sources[i] == "native":
                        result.fields_filled_native += 1
                    else:
                        result.fields_filled_llm += 1
                    if is_sane:
                        result.fields_verified += 1
                        continue

                    # Insane → drift correction, bounded by session budget.
                    locator, _method = await _locate(page, entry.field)
                    if locator is None or llm is None:
                        result.fields_drift_unresolved += 1
                        rep = pending_reports[i]
                        if rep is not None:
                            rep.outcome = "drift_unresolved"
                            rep.detail = (
                                f"{rep.detail} → drift "
                                f"observed={entry.observed_value!r} "
                                f"reason={(verdict.reason if verdict else 'no_locator_or_llm')!r}"
                            )
                        continue

                    correction_succeeded = False
                    correction_detail = (
                        f"observed={entry.observed_value!r} "
                        f"reason={(verdict.reason if verdict else '')!r}"
                    )
                    # Use suggested_value as a hint on retry when the LLM
                    # offered one — overrides the planner's intended value
                    # for the correction attempt only.
                    if verdict and verdict.suggested_value:
                        entry.value = verdict.suggested_value

                    for attempt in range(_MAX_CORRECTIONS_PER_FIELD):
                        if drift_calls >= _MAX_CORRECTIONS_PER_SESSION:
                            correction_detail = f"{correction_detail} → session_budget_exhausted"
                            break
                        drift_calls += 1
                        entry.correction_attempts += 1
                        corr_outcome, corr_detail = await _correct_field_drift(
                            page, entry, locator, llm,
                        )
                        if corr_outcome != "ok_corrected":
                            correction_detail = f"{correction_detail} → corr#{attempt+1}:{corr_outcome}:{corr_detail}"
                            continue
                        try:
                            await page.wait_for_timeout(300)
                        except Exception:
                            pass
                        entry.observed_value = await _probe_field_state(page, entry, locator)
                        # Re-verify ONLY this field with a fresh LLM call —
                        # cheap (single-row batch).
                        single = await _llm_verify_batch([entry], llm)
                        if single.get(0) and single[0].sane:
                            correction_succeeded = True
                            correction_detail = f"{corr_detail} (verified after {attempt+1} correction)"
                            break
                        correction_detail = f"{correction_detail} → corr#{attempt+1} acted but still insane"

                    rep = pending_reports[i]
                    if rep is None:
                        continue
                    if correction_succeeded:
                        result.fields_drift_corrected += 1
                        result.fields_verified += 1
                        rep.outcome = "ok_corrected"
                        rep.detail = correction_detail
                    else:
                        result.fields_drift_unresolved += 1
                        rep.outcome = "drift_unresolved"
                        rep.detail = f"{rep.detail} → drift {correction_detail}"

                # Flush reports + counters.
                for rep in pending_reports:
                    if rep is None:
                        continue
                    result.reports.append(rep)
                    if rep.outcome == "plan_skip":
                        result.fields_skipped += 1
                    elif rep.outcome in (
                        "fill_error", "select_error", "checkbox_error",
                        "radio_error", "file_error", "no_locator",
                        "llm_skipped", "llm_refused_submit", "llm_exec_error",
                    ):
                        result.fields_failed += 1

                result.llm_picker_calls = llm_calls
                result.drift_correction_calls = drift_calls
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
