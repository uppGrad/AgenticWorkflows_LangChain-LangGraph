"""Live-DOM Playwright walker that produces canonical FormField records.

Replaces the LLM-on-static-HTML parser as the primary form-extraction path.
The LLM parser is preserved as a fallback (extract_form_fields keeps the
old tier-2/2b/3 paths for the rare cases this walker finds nothing —
auth-walled forms, exotic SPA frames).

What this captures (in ONE Playwright session):

  - tag, type, name, id, required, accept attributes
  - label resolved via the same chain `_locate_file_input` does at
    fill-time: aria-labelledby → <label for=id> → enclosing <label> →
    closest container heading → aria-label → placeholder
  - ARIA: role, aria-haspopup, aria-controls, aria-owns,
    aria-autocomplete, list (datalist target)
  - native <select> options enumerated from <option> children
  - **combobox options captured by opening the listbox once** — the
    single thing the LLM-on-static-HTML parser CAN'T see, because
    react-select mounts `.select__option` divs only when the listbox is
    open. This is the duplicate-knowledge fix; gate-1 / value_planner /
    asset_mapping / fill-time all share the same `field.options` instead
    of fill-time re-discovering them ad-hoc.
  - radio group options: enumerated from sibling
    input[name=X][value=Y]
  - file accept attribute split into accepts_file
  - canonical_document_type for file inputs via existing classifier
  - expected_source classified by a small heuristic table (mirrors the
    LLM parser's behaviour)

The output dicts match `FormField` model schema exactly (label,
field_type, name, required, options, accepts_file, expected_source,
canonical_document_type, role, aria_*, list_id) so they drop into
state["form_fields"] without any adapter.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from uppgrad_agentic.tools.canonical_doc_types import classify_label

logger = logging.getLogger(__name__)


# ─── Field-shape walker (single Page.evaluate call) ──────────────────────
#
# Returns a list of pre-FormField dicts plus a `walker_id` so we can
# re-target combobox elements for the listbox-open probe afterwards.
_WALKER_JS = r"""
() => {
  const t = (s) => (s == null ? '' : String(s).trim());
  const lower = (s) => (t(s) || '').toLowerCase();

  // ── Label resolution chain (same as fill-time _locate fallback) ──
  function resolveLabel(el) {
    // Generic upload-widget captions to reject (the user wants the
    // surrounding question heading like "Resume/CV*", not the visible
    // button text). Greenhouse / Workable / Lever wrap the hidden
    // <input type="file"> inside a <label> with text "Attach" /
    // "Upload" / etc. — that label wins step 3 and hides the actual
    // question heading at step 4.
    const isGenericButtonCaption = (s) => {
      if (!s) return false;
      const lc = s.toLowerCase().replace(/\*/g, '').trim();
      const denied = [
        'attach', 'attach a file', 'attach file', 'attach files',
        'browse', 'browse files', 'browse file',
        'choose file', 'choose files',
        'upload', 'upload file', 'upload files', 'upload a file',
        'add file', 'add files',
        'drop file', 'drop files', 'drag and drop', 'drag & drop',
        'select file', 'select files', 'pick file', 'pick a file',
      ];
      return denied.includes(lc);
    };

    // 1. aria-labelledby on the input itself OR any ancestor up to
    //    the form root. Greenhouse wraps file inputs in
    //    <div role="group" aria-labelledby="upload-label-resume">
    //    where #upload-label-resume → "Resume/CV". The input itself
    //    doesn't have aria-labelledby, so we walk ancestors.
    const resolveIdsToText = (lb) => {
      if (!lb) return '';
      const ids = lb.split(/\s+/).filter(Boolean);
      return ids.map(id => {
        const n = document.getElementById(id);
        return n ? t(n.textContent) : '';
      }).filter(Boolean).join(' ');
    };
    const inputAriaText = resolveIdsToText(el.getAttribute('aria-labelledby'));
    if (inputAriaText) return inputAriaText;
    {
      let ancestor = el.parentElement;
      for (let i = 0; ancestor && i < 8 && ancestor.tagName !== 'FORM'; i++, ancestor = ancestor.parentElement) {
        const lb = ancestor.getAttribute && ancestor.getAttribute('aria-labelledby');
        const txt = resolveIdsToText(lb);
        if (txt) return txt;
      }
    }
    // 2. <label for="id"> matching input.id — but ONLY when the
    //    referenced label isn't itself a generic upload-widget
    //    caption like "Attach" (Greenhouse renders these as
    //    visually-hidden labels for screen readers; the real heading
    //    is a sibling <div>). The generic-caption check happens in
    //    step 3 after we've tried the ancestor heading first below.
    if (el.id) {
      try {
        const lbl = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
        if (lbl) {
          const text = t(lbl.textContent);
          // Defer the decision to the labelCaption fallback in step 3
          // so generic captions get rejected before being returned.
          if (text && !isGenericButtonCaption(text)) {
            return text;
          }
        }
      } catch (e) { /* invalid id */ }
    }
    // 3. enclosing <label> (within 4 ancestor levels). Skip if the
    //    label is just an upload-widget button caption — fall through
    //    to step 4 which finds the real question heading.
    let labelCaption = null;
    let n = el.parentElement;
    for (let i = 0; n && i < 4; i++, n = n.parentElement) {
      if (n.tagName === 'LABEL') {
        labelCaption = t(n.textContent);
        break;
      }
    }
    if (labelCaption && !isGenericButtonCaption(labelCaption)) {
      return labelCaption;
    }
    // 4. closest container's heading element
    const container = el.closest(
      '.application-question, .form-field, .field, .form-row, ' +
      '.field-wrapper, fieldset, [data-qa-field]'
    );
    if (container) {
      const heading = container.querySelector(
        'label, legend, .question-label, .field-label, [class*="label" i]:not(input):not(select):not(textarea)'
      );
      if (heading && heading !== el && !heading.contains(el)) {
        const headingText = t(heading.textContent);
        // Skip headings that are themselves generic captions (the
        // first <label> inside an upload widget container).
        if (headingText && !isGenericButtonCaption(headingText)) {
          return headingText;
        }
      }
      // Fallback: the container's own first text node, before any
      // form-control text. Greenhouse renders headings as bare
      // strong/text nodes inside .application-question without a
      // class hook on the heading itself.
      const walker = document.createTreeWalker(
        container, NodeFilter.SHOW_TEXT,
        { acceptNode: (txt) => {
            const v = (txt.textContent || '').trim();
            if (!v || v.length < 2) return NodeFilter.FILTER_REJECT;
            // Reject text that lives inside the actual input/widget area.
            if (txt.parentElement && txt.parentElement.closest('input, select, textarea, button')) {
              return NodeFilter.FILTER_REJECT;
            }
            return NodeFilter.FILTER_ACCEPT;
          },
        },
      );
      const candidates = [];
      let node;
      while ((node = walker.nextNode())) {
        const v = t(node.textContent);
        if (v && !isGenericButtonCaption(v) && v.length < 200) {
          candidates.push(v);
        }
        if (candidates.length >= 3) break;
      }
      if (candidates.length > 0) {
        return candidates[0];
      }
    }
    // 5. accept the previously-rejected button caption rather than
    //    leaving label empty — better than dropping the field. Only
    //    happens when steps 1, 2, 4 all failed.
    if (labelCaption) return labelCaption;
    // 6. aria-label
    const al = el.getAttribute('aria-label');
    if (al) return t(al);
    // 7. placeholder
    const ph = el.getAttribute('placeholder');
    if (ph) return t(ph);
    return '';
  }

  function isHidden(el) {
    if (lower(el.type) === 'hidden') return true;
    if (el.hasAttribute('hidden')) return true;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return true;
    // 0×0 size + offsetParent null → also hidden
    if (el.offsetParent === null && cs.position !== 'fixed') return true;
    return false;
  }

  function isUtility(el) {
    const n = lower(el.name);
    if (n.includes('csrf') || n.includes('_token') || n === 'authenticity_token') return true;
    if (n === 'utf8') return true;  // Rails default
    return false;
  }

  function fieldType(el) {
    const tag = el.tagName;
    if (tag === 'TEXTAREA') return 'textarea';
    if (tag === 'SELECT') return 'select';
    return lower(el.type) || 'text';
  }

  function ariaSnap(el) {
    return {
      role: t(el.getAttribute('role')),
      aria_haspopup: t(el.getAttribute('aria-haspopup')),
      aria_controls: t(el.getAttribute('aria-controls')),
      aria_owns: t(el.getAttribute('aria-owns')),
      aria_autocomplete: t(el.getAttribute('aria-autocomplete')),
      list_id: t(el.getAttribute('list')),
    };
  }

  function isRequired(el, label) {
    if (el.required || el.hasAttribute('required')) return true;
    if (lower(el.getAttribute('aria-required')) === 'true') return true;
    if (label && (label.includes('*') || /\(required\)|required\b/i.test(label))) return true;
    return false;
  }

  // Native <select> options. Skip placeholder-style first entries.
  function nativeSelectOptions(el) {
    const opts = [];
    for (const o of el.options) {
      const txt = t(o.label || o.text || o.value);
      if (!txt) continue;
      const lc = txt.toLowerCase();
      if (lc === 'choose...' || lc === 'select...' || lc === '—' ||
          lc === '-' || lc === '--' || lc === '...' ||
          lc.startsWith('please select') || lc.startsWith('choose ') ||
          lc.startsWith('select ')) continue;
      opts.push(txt);
    }
    return opts;
  }

  function isComboboxShape(el, aria) {
    if (lower(aria.role) === 'combobox') return true;
    const aac = lower(aria.aria_autocomplete);
    if (aac === 'list' || aac === 'both' || aac === 'inline') return true;
    if (aria.aria_haspopup && aria.aria_haspopup !== 'false' &&
        (aria.aria_controls || aria.aria_owns)) return true;
    // react-select structural ancestor
    if (el.closest && el.closest('.select__container, .select__control, .select-shell')) return true;
    return false;
  }

  // Find the form area, or fall back to body when no <form> tag exists
  // (some SPAs render form fields without an enclosing form).
  const form = document.querySelector('form') || document.body;

  const inputs = Array.from(form.querySelectorAll('input, textarea, select'));

  const out = [];
  const radioGroups = new Map();
  let stamp = 0;

  for (const el of inputs) {
    if (isHidden(el)) continue;
    if (isUtility(el)) continue;

    const ftype = fieldType(el);
    if (['submit', 'button', 'reset', 'image', 'password'].includes(ftype)) continue;

    const wid = `walker_${stamp++}`;
    el.setAttribute('data-walker-id', wid);

    const label = resolveLabel(el);
    const aria = ariaSnap(el);
    const name = t(el.getAttribute('name'));
    const id = t(el.id);

    // Radio groups: aggregate to one entry per `name`.
    if (ftype === 'radio') {
      const key = name || `radio_anon_${stamp}`;
      if (!radioGroups.has(key)) {
        radioGroups.set(key, {
          walker_id: wid,
          tag: 'input',
          field_type: 'radio',
          name,
          id: '',
          label,
          required: false,
          accept: '',
          options: [],
          option_walker_ids: [],
          is_combobox_shape: false,
          ...aria,
        });
      }
      const grp = radioGroups.get(key);
      const optVal = t(el.value || el.getAttribute('value') || el.id);
      // Try to read the radio's own label text — for visible options.
      const ownLabel = resolveLabel(el);
      const optText = ownLabel && ownLabel !== label ? ownLabel : optVal;
      if (optText && !grp.options.includes(optText)) grp.options.push(optText);
      grp.option_walker_ids.push(wid);
      grp.required = grp.required || isRequired(el, label);
      continue;
    }

    let options = [];
    let is_combobox_shape = false;
    if (el.tagName === 'SELECT') {
      options = nativeSelectOptions(el);
    } else {
      is_combobox_shape = isComboboxShape(el, aria);
    }

    out.push({
      walker_id: wid,
      tag: el.tagName.toLowerCase(),
      field_type: el.tagName === 'SELECT'
        ? 'select'
        : (el.tagName === 'TEXTAREA' ? 'textarea' : ftype),
      name,
      id,
      label,
      required: isRequired(el, label),
      accept: t(el.getAttribute('accept')),
      options,
      is_combobox_shape,
      ...aria,
    });
  }

  for (const grp of radioGroups.values()) {
    out.push(grp);
  }

  return out;
}
"""


# ─── Combobox listbox open + read + close (no LLM, no fill-side helper) ──
#
# This duplicates the trigger-click + listbox-wait + option-text-read
# loop from playwright_filler._combobox_pick. We copy it here rather
# than import to avoid the discoverer depending on fill-time code.
# ─── Apply-CTA click + re-walk fallback ──────────────────────────────────
#
# Coverage harness (50 prod URLs across Greenhouse / Lever / Ashby /
# Workable / SmartRecruiters) showed real success at 51%. A large chunk
# of "ok-but-sparse" Greenhouse runs were listing URLs where the
# walker scanned a page that wasn't the application form yet — it
# picked up 1-3 chrome inputs and reported success while the actual
# form sat behind an "Apply for this job" CTA. This helper detects
# that case, clicks the CTA, waits for the form to render, and lets
# the walker run a second time.
#
# Conservative pattern matching: only click on text that's
# unambiguously an apply-form CTA, never a final submit button.

_APPLY_CTA_TEXT_PATTERNS = (
    "Apply for this job",
    "Apply Now",
    "Apply now",
    "Apply",
    "Start application",
    "Start your application",
    "Continue to application",
    "Begin application",
)

# Tighter "looks-like-an-apply-CTA" check used after a text match to
# rule out submit / withdraw / hide buttons that may share the prefix.
_DENYLIST_CTA_TEXT = ("Submit", "Send", "Withdraw", "Cancel", "Close")


async def _try_click_apply_cta(page) -> bool:
    """Find a visible Apply / Start-application style CTA and click it.
    Returns True when a click landed. Best-effort: never raises."""
    for pattern in _APPLY_CTA_TEXT_PATTERNS:
        # Prefer button > anchor; visible-only.
        for sel in (
            f'button:visible:has-text("{pattern}")',
            f'a:visible:has-text("{pattern}")',
            f'[role="button"]:visible:has-text("{pattern}")',
        ):
            try:
                loc = page.locator(sel).first
                count = await loc.count()
                if count == 0:
                    continue
                # Filter out denylist substrings (e.g. "Submit application"
                # at the bottom of an already-rendered form is NOT a CTA
                # to click — it's the final submit). The text must START
                # with one of our patterns and not contain a denylisted
                # token elsewhere.
                try:
                    txt = (await loc.text_content(timeout=1000) or "").strip()
                except Exception:
                    txt = ""
                if not txt:
                    continue
                lower_txt = txt.lower()
                if any(b.lower() in lower_txt for b in _DENYLIST_CTA_TEXT):
                    continue
                # Lever / Ashby sometimes show an "Apply" link that
                # points at the SAME URL with a hash fragment — clicking
                # it is harmless and may reveal the form.
                try:
                    await loc.scroll_into_view_if_needed(timeout=1500)
                except Exception:
                    pass
                try:
                    await loc.click(timeout=2500)
                except Exception:
                    continue
                # Wait for the new form area to render. Two-stage:
                # network-idle for any XHR-driven render, then wait for
                # at least one form input to appear.
                try:
                    await page.wait_for_load_state("networkidle", timeout=8000)
                except Exception:
                    pass
                try:
                    await page.wait_for_selector(
                        "form input, form textarea, form select",
                        timeout=5000,
                        state="visible",
                    )
                except Exception:
                    pass
                # Extra hydration buffer.
                try:
                    await page.wait_for_timeout(700)
                except Exception:
                    pass
                logger.info(
                    "form_discoverer: clicked apply CTA %r (selector=%s)",
                    txt[:50], sel,
                )
                return True
            except Exception:
                continue
    return False


async def _open_combobox_and_read_options(page, locator) -> List[str]:
    """Click the trigger, wait for listbox to render, read every visible
    option's text, close via Escape. Returns a deduped list capped at 50.
    """
    # 1. Click the .select__control ancestor when present (real
    #    Playwright click → mousedown chain that react-select listens
    #    for). Fall back to clicking the input itself.
    try:
        control = locator.locator(
            'xpath=ancestor::div[contains(@class,"select__control")][1]'
        )
        if await control.count() > 0:
            try:
                await control.first.click(timeout=1500)
            except Exception:
                pass
        else:
            try:
                await locator.click(timeout=1500)
            except Exception:
                return []
    except Exception:
        return []

    # 2. Wait for any listbox-shape selector to become visible.
    listbox_selectors = (
        '.select__option',                          # Greenhouse / Anthropic
        '[role="listbox"] [role="option"]',         # Lever / generic ARIA
        '[role="option"]',                          # bare role
        'li[role="option"]',                        # Ashby
        '.styles_menu__option__*',                  # Workable
        'ul[role="listbox"] li',                    # SmartRecruiters
    )
    matched_sel = None
    for sel in listbox_selectors:
        try:
            await page.wait_for_selector(sel, timeout=1500, state="visible")
            matched_sel = sel
            break
        except Exception:
            continue

    options: List[str] = []
    seen: set = set()
    if matched_sel:
        try:
            opts = page.locator(matched_sel)
            count = await opts.count()
            for i in range(min(count, 50)):
                try:
                    txt = (await opts.nth(i).text_content(timeout=300) or "").strip()
                except Exception:
                    continue
                if txt and txt not in seen:
                    seen.add(txt)
                    options.append(txt)
        except Exception:
            pass

    # 3. Close listbox so the next field's introspection isn't blocked.
    try:
        await page.keyboard.press("Escape")
    except Exception:
        pass

    return options


# ─── expected_source heuristic ───────────────────────────────────────────
#
# Mirrors the existing LLM classifier behaviour. Deterministic; no model
# call needed for the common cases.
_PROFILE_KEYWORDS = (
    "first name", "given name", "last name", "surname", "family name",
    "full name", "email", "phone", "mobile", "telephone",
    "linkedin", "github", "personal website", "portfolio url",
    "personal site", "website",
    "address from which", "current location", "city", "country",
    "location", "home address", "mailing address", "address",
    "work auth", "authoriz", "citizenship", "nationality",
)
_DOCUMENT_KEYWORDS = (
    "resume", "cv", "curriculum vitae", "cover letter", "motivation letter",
    "letter of motivation", "sop", "statement of purpose", "personal statement",
    "research proposal", "writing sample", "transcript", "english proficiency",
    "ielts", "toefl", "portfolio", "certificate", "passport", "birth certificate",
    "references", "reference letter", "recommendation",
)


def _classify_expected_source(label: str, field_type: str) -> str:
    lc = (label or "").lower()
    if field_type == "file":
        return "user_document"
    if field_type == "date":
        return "computed"
    for kw in _PROFILE_KEYWORDS:
        if kw in lc:
            return "user_profile"
    if field_type in ("textarea",):
        return "user_answer"
    if field_type in ("select", "radio", "checkbox"):
        return "user_answer"
    if field_type in ("text", "email", "tel", "url", "number"):
        # Plain text input that isn't a profile attr → likely a free
        # answer or a combobox-typed-as-text. user_answer.
        return "user_answer"
    return "unknown"


# ─── accepts_file split ──────────────────────────────────────────────────
def _split_accept(accept: str) -> List[str]:
    if not accept:
        return []
    out: List[str] = []
    for tok in accept.split(","):
        v = tok.strip()
        if v:
            out.append(v)
    return out


# ─── Public entrypoint (async) ───────────────────────────────────────────
async def discover_form_fields_async(
    form_url: str,
    *,
    headless: bool = True,
    nav_timeout_ms: int = 30_000,
) -> List[Dict[str, Any]]:
    """Open `form_url` in Playwright, walk the form DOM, and return a
    list of FormField-shaped dicts with options populated for every
    native select, radio group, and live combobox.

    Returns [] on Playwright failure (caller falls back to the LLM-parse
    path). Best-effort by design — never raises.
    """
    if not form_url:
        return []

    from playwright.async_api import async_playwright

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

                # Dismiss cookie banners + scroll-trigger off-screen
                # form sections (Greenhouse lazy-mounts EEOC questions).
                # Inline the helpers rather than depend on playwright_filler.
                for sel in (
                    "#onetrust-accept-btn-handler",
                    "button#truste-consent-button",
                    'button:has-text("Accept All")',
                    'button:has-text("I agree")',
                    'button:has-text("Accept")',
                ):
                    try:
                        btn = await page.query_selector(sel)
                        if btn:
                            await btn.click(timeout=1500)
                            await page.wait_for_timeout(300)
                            break
                    except Exception:
                        continue
                try:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(800)
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                    await page.wait_for_timeout(400)
                    await page.evaluate("window.scrollTo(0, 0)")
                    await page.wait_for_timeout(400)
                except Exception:
                    pass

                # Step 1: walk the DOM in one JS evaluation.
                try:
                    raw = await page.evaluate(_WALKER_JS)
                except Exception as exc:
                    logger.warning("form_discoverer: walker JS failed — %s", exc)
                    return []

                # ─── Sparse-result CTA-click fallback ─────────────────
                # If the walker found <5 inputs the page is probably a
                # listing / preview rather than the application form.
                # Try clicking an Apply CTA once, wait for the form to
                # render, re-walk. Coverage harness showed this alone
                # would recover ~20% of the dataset (Greenhouse listing
                # URLs returning 1-3 chrome inputs).
                if not raw or len(raw) < 5:
                    if await _try_click_apply_cta(page):
                        try:
                            raw2 = await page.evaluate(_WALKER_JS)
                        except Exception as exc:
                            logger.warning(
                                "form_discoverer: post-CTA walker failed — %s",
                                exc,
                            )
                            raw2 = None
                        if raw2 and len(raw2) > len(raw or []):
                            logger.info(
                                "form_discoverer: re-walk after CTA click "
                                "improved coverage %d → %d fields",
                                len(raw or []), len(raw2),
                            )
                            raw = raw2

                if not raw:
                    return []

                # Step 2: for combobox-shape candidates without options,
                # re-locate via [data-walker-id] and open the listbox.
                for entry in raw:
                    if not entry.get("is_combobox_shape"):
                        continue
                    if entry.get("options"):
                        continue
                    wid = entry.get("walker_id")
                    if not wid:
                        continue
                    try:
                        loc = page.locator(f'[data-walker-id="{wid}"]')
                        if await loc.count() == 0:
                            continue
                        opts = await _open_combobox_and_read_options(page, loc.first)
                        if opts:
                            entry["options"] = opts[:50]
                    except Exception as exc:
                        logger.debug(
                            "form_discoverer: combobox probe failed for %s — %s",
                            entry.get("label", "")[:40], exc,
                        )
                        continue

                # Step 3: shape into FormField dicts.
                fields: List[Dict[str, Any]] = []
                for entry in raw:
                    raw_label = (entry.get("label") or "").strip()
                    # Strip trailing required-markers (`*`, `(Required)`)
                    # — they convey the required flag, not part of the
                    # question.
                    label = raw_label
                    while label.endswith("*"):
                        label = label[:-1].strip()
                    if label.lower().endswith("(required)"):
                        label = label[: -len("(required)")].strip()
                    if not label:
                        continue
                    ftype = entry.get("field_type") or "text"
                    if ftype == "password":
                        continue
                    options = list(entry.get("options") or [])
                    accept = entry.get("accept") or ""
                    # Fill-time `_locate` matches `[name=X]` OR `[id=X]`
                    # interchangeably (playwright_filler.py:562). Mirror
                    # that here: emit `name` from the input's name attr,
                    # falling back to id when name is absent (Greenhouse
                    # react-select inputs use id="question_X" without a
                    # name attribute).
                    name_or_id = (entry.get("name") or "").strip() or (entry.get("id") or "").strip()
                    fields.append({
                        "label": label,
                        "field_type": ftype,
                        "name": name_or_id,
                        "required": bool(entry.get("required"))
                        or raw_label.endswith("*")
                        or raw_label.lower().endswith("(required)"),
                        "options": options,
                        "accepts_file": _split_accept(accept) if ftype == "file" else [],
                        "expected_source": _classify_expected_source(label, ftype),
                        "canonical_document_type": (
                            classify_label(label) or "" if ftype == "file" else ""
                        ),
                        "role": entry.get("role") or "",
                        "aria_haspopup": entry.get("aria_haspopup") or "",
                        "aria_controls": entry.get("aria_controls") or "",
                        "aria_owns": entry.get("aria_owns") or "",
                        "aria_autocomplete": entry.get("aria_autocomplete") or "",
                        "list_id": entry.get("list_id") or "",
                        # Internal-only — used by `tools/form_verifier.py` to
                        # match LLM corrections back onto specific entries.
                        # Stripped before persistence by extract_form_fields.
                        "_walker_id": entry.get("walker_id") or "",
                    })

                # Step 4: dedupe by label. React-select widgets often
                # mount a sibling proxy input alongside the visible
                # role=combobox input; both share the same label
                # (resolved via the same .application-question heading).
                # Prefer the combobox-shape entry — it's the one the
                # filler / planner / verifier actually want.
                def _info_score(f: Dict[str, Any]) -> int:
                    score = 0
                    if (f.get("role") or "").lower() == "combobox":
                        score += 4
                    if (f.get("aria_autocomplete") or "").lower() in ("list", "both", "inline"):
                        score += 2
                    if f.get("options"):
                        score += 2
                    if f.get("name"):
                        score += 1
                    return score

                # Two-stage dedupe.
                #
                # Stage A: fold by `name` when present — repeated entries
                # with the same name are the same widget.
                # Stage B: drop nameless entries whose (label, type) pair
                # is already covered by a NAMED, MORE INFORMATIVE entry
                # — kills react-select proxy inputs that mirror their
                # parent combobox's label without contributing anything.
                # File fields labelled "Attach" with different names
                # are preserved by stage B because they all have names.
                stage_a: Dict[tuple, List[Dict[str, Any]]] = {}
                no_name: List[Dict[str, Any]] = []
                for f in fields:
                    nm = (f.get("name") or "").strip()
                    if nm:
                        key = (nm, f.get("field_type"))
                        stage_a.setdefault(key, []).append(f)
                    else:
                        no_name.append(f)
                deduped: List[Dict[str, Any]] = []
                for key, group in stage_a.items():
                    deduped.append(max(group, key=_info_score) if len(group) > 1 else group[0])
                # Build a label-set of named, info-bearing entries.
                covered_labels: set = set()
                for d in deduped:
                    if _info_score(d) >= 4:  # has role=combobox
                        covered_labels.add(((d.get("label") or "").strip().lower(),
                                            d.get("field_type")))
                for f in no_name:
                    label_key = ((f.get("label") or "").strip().lower(),
                                 f.get("field_type"))
                    if label_key in covered_labels:
                        continue  # drop the proxy
                    deduped.append(f)
                return deduped
            finally:
                try:
                    await browser.close()
                except Exception:
                    pass
    except Exception as exc:
        logger.warning("form_discoverer: playwright session failed — %s", exc)
        return []


# ─── Sync wrapper (for the LangGraph node) ───────────────────────────────
async def discover_form_fields_with_screenshot_async(
    form_url: str,
    *,
    headless: bool = True,
    nav_timeout_ms: int = 30_000,
) -> tuple[List[Dict[str, Any]], Optional[bytes]]:
    """Same DOM walker as `discover_form_fields_async`, plus captures a
    screenshot of the form area for downstream vision-LLM verification.

    Returns `(fields_with_walker_id, screenshot_bytes_or_None)`. The
    walker_id remains in each field dict (under `_walker_id`) so the
    verifier can target specific entries when proposing corrections.
    Best-effort: returns `([], None)` on Playwright failure.
    """
    if not form_url:
        return [], None

    from playwright.async_api import async_playwright

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
                for sel in (
                    "#onetrust-accept-btn-handler",
                    "button#truste-consent-button",
                    'button:has-text("Accept All")',
                    'button:has-text("I agree")',
                    'button:has-text("Accept")',
                ):
                    try:
                        btn = await page.query_selector(sel)
                        if btn:
                            await btn.click(timeout=1500)
                            await page.wait_for_timeout(300)
                            break
                    except Exception:
                        continue
                try:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(800)
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                    await page.wait_for_timeout(400)
                    await page.evaluate("window.scrollTo(0, 0)")
                    await page.wait_for_timeout(400)
                except Exception:
                    pass

                # Walk the DOM.
                try:
                    raw = await page.evaluate(_WALKER_JS)
                except Exception as exc:
                    logger.warning(
                        "form_discoverer (with_screenshot): walker JS failed — %s", exc,
                    )
                    return [], None

                # Sparse-result CTA-click fallback. See the standard
                # path for rationale.
                if not raw or len(raw) < 5:
                    if await _try_click_apply_cta(page):
                        try:
                            raw2 = await page.evaluate(_WALKER_JS)
                        except Exception as exc:
                            logger.warning(
                                "form_discoverer (with_screenshot): post-CTA "
                                "walker failed — %s", exc,
                            )
                            raw2 = None
                        if raw2 and len(raw2) > len(raw or []):
                            logger.info(
                                "form_discoverer (with_screenshot): re-walk "
                                "after CTA click improved coverage %d → %d",
                                len(raw or []), len(raw2),
                            )
                            raw = raw2

                if not raw:
                    return [], None

                # Combobox listbox-open probe (same as the standard path).
                for entry in raw:
                    if not entry.get("is_combobox_shape"):
                        continue
                    if entry.get("options"):
                        continue
                    wid = entry.get("walker_id")
                    if not wid:
                        continue
                    try:
                        loc = page.locator(f'[data-walker-id="{wid}"]')
                        if await loc.count() == 0:
                            continue
                        opts = await _open_combobox_and_read_options(page, loc.first)
                        if opts:
                            entry["options"] = opts[:50]
                    except Exception:
                        continue

                # Shape into FormField dicts (with walker_id retained).
                fields: List[Dict[str, Any]] = []
                for entry in raw:
                    raw_label = (entry.get("label") or "").strip()
                    label = raw_label
                    while label.endswith("*"):
                        label = label[:-1].strip()
                    if label.lower().endswith("(required)"):
                        label = label[: -len("(required)")].strip()
                    if not label:
                        continue
                    ftype = entry.get("field_type") or "text"
                    if ftype == "password":
                        continue
                    options = list(entry.get("options") or [])
                    accept = entry.get("accept") or ""
                    name_or_id = (entry.get("name") or "").strip() or (entry.get("id") or "").strip()
                    fields.append({
                        "label": label,
                        "field_type": ftype,
                        "name": name_or_id,
                        "required": bool(entry.get("required"))
                        or raw_label.endswith("*")
                        or raw_label.lower().endswith("(required)"),
                        "options": options,
                        "accepts_file": _split_accept(accept) if ftype == "file" else [],
                        "expected_source": _classify_expected_source(label, ftype),
                        "canonical_document_type": (
                            classify_label(label) or "" if ftype == "file" else ""
                        ),
                        "role": entry.get("role") or "",
                        "aria_haspopup": entry.get("aria_haspopup") or "",
                        "aria_controls": entry.get("aria_controls") or "",
                        "aria_owns": entry.get("aria_owns") or "",
                        "aria_autocomplete": entry.get("aria_autocomplete") or "",
                        "list_id": entry.get("list_id") or "",
                        "_walker_id": entry.get("walker_id") or "",
                    })

                # Same dedupe as the standard path.
                def _info_score(f: Dict[str, Any]) -> int:
                    score = 0
                    if (f.get("role") or "").lower() == "combobox":
                        score += 4
                    if (f.get("aria_autocomplete") or "").lower() in ("list", "both", "inline"):
                        score += 2
                    if f.get("options"):
                        score += 2
                    if f.get("name"):
                        score += 1
                    return score

                stage_a: Dict[tuple, List[Dict[str, Any]]] = {}
                no_name: List[Dict[str, Any]] = []
                for f in fields:
                    nm = (f.get("name") or "").strip()
                    if nm:
                        key = (nm, f.get("field_type"))
                        stage_a.setdefault(key, []).append(f)
                    else:
                        no_name.append(f)
                deduped: List[Dict[str, Any]] = []
                for key, group in stage_a.items():
                    deduped.append(max(group, key=_info_score) if len(group) > 1 else group[0])
                covered_labels: set = set()
                for d in deduped:
                    if _info_score(d) >= 4:
                        covered_labels.add(((d.get("label") or "").strip().lower(),
                                            d.get("field_type")))
                for f in no_name:
                    label_key = ((f.get("label") or "").strip().lower(),
                                 f.get("field_type"))
                    if label_key in covered_labels:
                        continue
                    deduped.append(f)

                # Capture form-area screenshot. Prefer the <form> locator;
                # fall back to a viewport-clamped full-page shot when no
                # form tag exists (rare on SPAs that mount inputs into
                # body without a wrapping form).
                screenshot: Optional[bytes] = None
                try:
                    form_loc = page.locator('form').first
                    if await form_loc.count() > 0:
                        screenshot = await form_loc.screenshot(timeout=8000)
                    else:
                        screenshot = await page.screenshot(
                            full_page=True, timeout=10000,
                        )
                except Exception as exc:
                    logger.debug(
                        "form_discoverer: screenshot capture failed — %s", exc,
                    )
                    screenshot = None

                return deduped, screenshot
            finally:
                try:
                    await browser.close()
                except Exception:
                    pass
    except Exception as exc:
        logger.warning(
            "form_discoverer (with_screenshot): playwright session failed — %s",
            exc,
        )
        return [], None


def discover_form_fields(form_url: str, *, headless: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Sync wrapper. Returns [] on any failure (caller falls back to LLM
    parser).

    `headless` defaults to True; pass False to launch a visible browser
    when debugging the discoverer locally.
    """
    if not form_url:
        return []
    if headless is None:
        # Reuse the auto-fill env knob so the demo path can watch
        # discovery happen too.
        env = (os.environ.get("UPPGRAD_FORM_DISCOVERER_HEADLESS") or "").strip().lower()
        if env in ("0", "false", "no", "off"):
            headless = False
        else:
            headless = True
    try:
        return asyncio.run(
            discover_form_fields_async(form_url, headless=headless)
        )
    except RuntimeError as exc:
        # If we're already inside an event loop (rare in sync nodes but
        # possible under some test harnesses), bail to caller's fallback.
        logger.warning("form_discoverer: cannot run in nested loop — %s", exc)
        return []
    except Exception as exc:
        logger.warning("form_discoverer: %s", exc)
        return []
