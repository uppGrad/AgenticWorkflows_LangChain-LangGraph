"""Live smoke test: pick "Yes" on an Anthropic Greenhouse combobox and
read back the observed value via the EXACT JS the probe uses.

If observed comes back "Yes", the probe walk is correct and the cascade
the user described should be gone. If observed is "" the probe is
still broken — print the wrapper class names we found so we know what
to walk to.

Run:
  PYTHONPATH=src uv run python scripts/probe_anthropic_combobox.py
"""
from __future__ import annotations

import asyncio
import sys


# Pick any Anthropic role with the standard combobox question set.
# Confirmed live via Greenhouse public board API.
URL = "https://job-boards.greenhouse.io/anthropic/jobs/5161980008"


PROBE_JS = r"""
(el) => {
  const t = (s) => (s == null ? '' : String(s).trim());

  const isComboboxShape = (
    el.getAttribute('role') === 'combobox' ||
    ((el.getAttribute('aria-autocomplete') || '').toLowerCase() !== '' &&
     (el.getAttribute('aria-autocomplete') || '').toLowerCase() !== 'none')
  );
  if (!isComboboxShape) return { observed: '<not_combobox_shape>', notes: 'na' };

  // Probe walk — strictly to wrapper classes, NOT [role=combobox]
  // (which would match the input itself).
  const c = (
    el.closest('.select__container') ||
    el.closest('.select-shell') ||
    el.closest('.select__control') ||
    el.closest('.field-wrapper, .field, .form-field, fieldset, .application-question') ||
    el.parentElement && el.parentElement.parentElement
  );

  if (!c) return { observed: '', notes: 'no_wrapper_found' };

  const wrapperClass = c.className || '<no class>';

  const single = c.querySelector('.select__single-value');
  if (single) {
    return {
      observed: t(single.textContent),
      notes: 'combobox_single_value',
      wrapper: wrapperClass,
    };
  }

  const multi = c.querySelectorAll('.select__multi-value, .select__multi-value__label');
  if (multi.length > 0) {
    const labels = Array.from(multi).map(n => t(n.textContent)).filter(Boolean);
    return {
      observed: labels.join(', '),
      notes: 'combobox_multi_value',
      wrapper: wrapperClass,
    };
  }

  const ariaSel = c.querySelector('[aria-selected="true"], [data-selected="true"]');
  if (ariaSel) {
    return {
      observed: t(ariaSel.textContent),
      notes: 'aria_selected_combobox',
      wrapper: wrapperClass,
    };
  }

  // Dump all children's classes so we see what's actually in there.
  const childClasses = [];
  c.querySelectorAll('div, span').forEach(n => {
    const cls = n.className || '';
    if (cls && childClasses.length < 30) childClasses.push(cls);
  });
  return {
    observed: '',
    notes: 'combobox_empty',
    wrapper: wrapperClass,
    sample_child_classes: childClasses.slice(0, 20),
  };
}
"""


async def main() -> None:
    from playwright.async_api import async_playwright  # type: ignore

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # visible
        ctx = await browser.new_context(viewport={"width": 1280, "height": 1800})
        page = await ctx.new_page()

        print(f"navigating to {URL}")
        await page.goto(URL, wait_until="domcontentloaded", timeout=30_000)
        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass

        # Find the first combobox-shaped input on the page.
        combobox = page.locator(
            'input[role="combobox"][aria-autocomplete="list"]'
        ).first
        count = await combobox.count()
        if count == 0:
            print("NO combobox found — page might be the JD page, not the apply form.")
            await browser.close()
            sys.exit(1)

        label_el = await combobox.evaluate(
            r"""(el) => {
                const id = el.getAttribute('aria-labelledby');
                if (id) {
                    const lbl = document.getElementById(id);
                    if (lbl) return lbl.textContent.trim();
                }
                return el.getAttribute('aria-label') || '<unknown>';
            }"""
        )
        print(f"first combobox label: {label_el!r}")

        # Phase 1: probe BEFORE pick — should be empty.
        observed_before = await combobox.evaluate(PROBE_JS)
        print(f"\nBEFORE pick:\n  {observed_before}")

        # Phase 2: open via Playwright locator.click() on .select__control
        # (real event chain, not synthesized JS click).
        await combobox.scroll_into_view_if_needed(timeout=3000)
        try:
            control = combobox.locator(
                'xpath=ancestor::div[contains(@class,"select__control")][1]'
            )
            ccount = await control.count()
            print(f".select__control ancestor count: {ccount}")
            if ccount > 0:
                try:
                    await control.first.click(timeout=2500)
                    print("clicked .select__control via Playwright locator.click()")
                except Exception as exc:
                    print(f"locator click failed ({exc}); dispatching mousedown event chain")
                    await control.first.evaluate(
                        r"""(el) => {
                            el.dispatchEvent(new MouseEvent('mousedown', {bubbles: true, cancelable: true}));
                            el.dispatchEvent(new MouseEvent('mouseup', {bubbles: true, cancelable: true}));
                            el.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
                        }"""
                    )
                    print("dispatched mousedown→mouseup→click")
            else:
                await combobox.click(timeout=3000)
                print("fallback: clicked the input itself")
        except Exception as exc:
            print(f"click failed: {exc}")

        # Wait for the listbox.
        try:
            await page.wait_for_selector(".select__option", timeout=3000, state="visible")
            print("listbox opened ✓")
        except Exception:
            print("listbox didn't open within 3s")

        # Click the first option (typically "Yes").
        opt = page.locator(".select__option").first
        opt_text = (await opt.text_content() or "").strip()
        print(f"about to click option: {opt_text!r}")
        await opt.click(timeout=3000)
        print("clicked option")

        # Brief wait for react-select to settle.
        await page.wait_for_timeout(500)

        # Phase 3: probe AFTER pick — should now read the picked text.
        observed_after = await combobox.evaluate(PROBE_JS)
        print(f"\nAFTER pick:\n  {observed_after}")

        # Verdict.
        observed = (observed_after or {}).get("observed", "")
        if observed and observed.lower() == opt_text.lower():
            print(f"\n✅ PROBE WORKS — read back {observed!r} matching pick {opt_text!r}")
        else:
            print(f"\n❌ PROBE BROKEN — pick was {opt_text!r}, observed {observed!r}")

        await page.wait_for_timeout(2000)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
