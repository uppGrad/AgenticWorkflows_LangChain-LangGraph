# Auto-fill redesign — phased plan (2026-05-04)

## Why we're rewriting

Today's `playwright_filler` pipeline:

```
extract_form_fields (BS4 over rendered HTML, LLM extracts FormField records)
   → value_planner (FormFieldFillPlan per field)
   → fill_form_async:
       Tier 1: [name=X] / [id=X] + .fill / .select_option / .set_input_files / .check
       Tier 2: get_by_label
       Tier 3: React custom-dropdown click+pick
       Tier 4: LLM picker on Tier-1-3 failure
       Tier 5: deterministic post-fill state probe + LLM drift corrector
```

This fails on:

1. **Combobox-with-search treated as text input.** Tier 1 finds `[name="country"]` (the visible `<input>` backing a custom dropdown), runs `.fill("United States")`, the input value is "United States", we declare success — but no option is selected in React state. Submit treats the field as empty.
2. **Yes/No collected as free text and stuffed at a radio group.** Tier 1 fails silently on the radio, Tier 2 fills a wrong sibling, fill_form_async returns "ok".
3. **Hidden native `<select>` covered by a custom widget.** Hidden select accepts the option, visible widget shows nothing, React state wins on submit.
4. **Wrong source value chosen at planning time.** "Are you open to relocation?" → "Ankara, Turkey" because value_planner mapped it to `user.location`. Fill succeeds (string match), DOM verifier passes, semantic mismatch invisible to every tier.
5. **Stale extraction.** Form HTML is captured at session start, hours before fill. By fill time JS has rewritten the DOM.

Six concrete gaps vs `browser-use` / `Stagehand` / `Skyvern`:

- **Schema loses the discriminator.** BS4 sees `role`, `aria-haspopup`, `aria-controls`, `aria-autocomplete` but our `FormField` schema only stores `field_type`. Aria signals dropped at extraction, never reach the filler.
- **No combobox predicate.** Even if signals were preserved, no tier branches on them.
- **No native-setter dispatch.** React/Vue ignore raw `.value` writes. Our `.fill()` silently no-ops on some controlled inputs. browser-use uses `Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, x)` + `dispatchEvent(new Event('input', { bubbles: true }))`.
- **Verification is shallow.** `el.value === intended` can't catch autocomplete-rewrites, combobox-without-selection, or semantic mismatches.
- **No per-step re-observation.** Plan-once-execute-many; can't adapt to JS rewrites mid-form.
- **Tier 4 (LLM) only fires on Tier-1-3 *failure*.** The combobox-typed-as-text case doesn't throw, so the LLM never inspects it.

## Phase 1 — quick wins, no architecture change

**Scope:** ~300 LOC total. Fixes the combobox-as-text + native-setter-no-op classes specifically. Zero LLM cost increase.

1. **Capture aria attributes during extraction.** Add to `FormField`:
   - `role: str = ""` (computed CSS role or aria-role)
   - `aria_haspopup: str = ""`
   - `aria_controls: str = ""`
   - `aria_owns: str = ""`
   - `aria_autocomplete: str = ""`
   - `list_id: str = ""` (from `<input list="...">`)

   Update `extract_form_fields` LLM prompt to extract these. BS4 already sees them.

2. **Add Tier 0 combobox predicate to `playwright_filler`.** Before Tier 1 runs, evaluate `_is_autocomplete_field()`:

   ```python
   def is_combobox(field: FormField) -> bool:
       if field.role == "combobox": return True
       if field.aria_autocomplete and field.aria_autocomplete != "none": return True
       if field.list_id: return True
       if field.aria_haspopup and field.aria_haspopup != "false" and (
           field.aria_controls or field.aria_owns
       ): return True
       return False
   ```

   When it matches, the action policy changes: `await element.fill(""); await element.type(value, delay=20); await page.wait_for_selector('[role="listbox"] [role="option"]', timeout=2000); click matching option by visible text`.

3. **Per-fill DOM readback after every action.** Move the existing Tier 5 probe to run inline after each Tier 1-4 success, not at the end of the loop. Compare to typed value; if the input rewrote it (autocomplete) accept when there's an `aria-expanded=true` or selected option nearby.

4. **Native-setter fallback for `.fill()` no-ops.** When `.fill()` returns but the value didn't stick:
   ```python
   await element.evaluate(
       """(el, v) => {
           const setter = Object.getOwnPropertyDescriptor(
               el.tagName === 'TEXTAREA'
                   ? HTMLTextAreaElement.prototype
                   : HTMLInputElement.prototype,
               'value'
           ).set;
           setter.call(el, v);
           el.dispatchEvent(new Event('input', { bubbles: true }));
           el.dispatchEvent(new Event('change', { bubbles: true }));
       }""",
       value,
   )
   ```

**Estimated impact:** fixes ~80% of the cases that motivated this redesign. Combobox detection alone resolves the common Greenhouse / Anthropic / Lever cases.

## Phase 2 — replace deterministic verification with LLM batch verifier

**Scope:** ~200 LOC. Catches semantic mismatches the string compare can't.

After all fills + per-field readback, batch-send `(label, intended_value, observed_value, surrounding_HTML_snippet)` tuples to ONE LLM call. Structured output:

```python
class _FieldVerdict(BaseModel):
    idx: int
    sane: bool
    reason: str  # short human-readable explanation
    suggested_value: str = ""  # only when sane=False AND a better answer is obvious

class _BatchVerifyResult(BaseModel):
    verdicts: List[_FieldVerdict]
```

The LLM sees question semantics, surrounding labels, error text. Catches:
- "Are you open to relocation?" → "Ankara, Turkey" (insane, semantic mismatch)
- Greenhouse autocomplete rewriting "San Fr" → "San Francisco, CA" (sane despite string drift)
- Empty-but-required fields where the readback says "" but the form expects a value (insane)

For insane rows, run the existing drift corrector with the `suggested_value` as a hint. Bound by the same per-session correction budget.

Cost: ~1 LLM call/session for the batch + 0-5 corrections × ~$0.005. Total ~$0.05/session.

## Phase 3 — accessibility-tree grounding

**Scope:** larger refactor. Adopts the `browser-use` / `Stagehand` grounding model.

Replace BS4 form extraction with Playwright's accessibility tree:

```python
ax_tree = await page.accessibility.snapshot(interesting_only=True)
# Walk the tree; for each node with role in INTERACTIVE_ROLES
# (textbox, combobox, listbox, checkbox, radio, button, link),
# emit a FormField with the COMPUTED role (which reflects React
# custom widgets that render as <div> but expose role=combobox).
```

The AX tree gives the *computed* role — what assistive tech sees, what `browser-use` and `Stagehand` use as ground truth. Resolves React custom widgets that BS4 can't see through.

**Schema migration:** existing `FormField` records keep working; the AX-derived extractor just produces richer ones.

**Compatibility:** existing tests should keep passing because the FormField schema is additive.

## Phase 4 — per-action observe→act loop

**Scope:** rewrite of `fill_form_async`. Largest lift; the right end-state.

Each field becomes a micro-agent: snapshot → plan → act → re-snapshot → verify → loop (≤3 iterations / field). Stagehand's pattern. Higher cost (each field = 2-4 LLM calls), but the only way to handle:

- Multi-step forms with conditional fields.
- Forms that show validation errors mid-fill ("invalid postal code") that we need to read and react to.
- Forms with side-effects (filling Country reveals State).

## Implementation order

| PR | Phase | Estimate | Ships | Risk |
|---|---|---|---|---|
| 1 | Phase 1 | 1-2 days | combobox predicate + native-setter + per-fill readback | low |
| 2 | Phase 2 | 1 day | LLM batch verifier replaces deterministic | low |
| 3 | Phase 3 | 3-5 days | AX-tree grounding | medium (LLM extractor prompt rewrite) |
| 4 | Phase 4 | 1-2 weeks | per-action loop | high (changes the core control flow) |

Phase 1 + 2 in one PR is the right "ship first" target. Phase 3 + 4 are bigger; consider whether to take the `browser-use` library directly instead of rebuilding their work — see the POC plan below.

## Alternative: adopt `browser-use` as a library

`browser-use` (https://github.com/browser-use/browser-use, ~91k stars) implements every gap we identified — AX-tree grounding, combobox predicate, native-setter dispatch, per-action loop, file upload — and has a documented job-application example we can mimic.

Adopting it as a library means:
- Drop our 5-tier filler.
- Build an adapter that constructs a browser-use `Agent` with the task ("apply to opportunity X with profile Y and tailored docs Z"), runs it, and maps the agent's history back to our `FormFillResult` schema.
- Behind a feature flag (`UPPGRAD_AUTO_FILL_BACKEND=browser_use`) so we can A/B against the current filler.

Trade-off: we inherit their architecture decisions (cost model, model family, action grammar) and their bugs. But also their fixes — they've actually solved this problem at scale. The phased plan above is what we'd be reimplementing; running the POC tells us whether to keep building or adopt the library.

The POC lives on `poc/browser-use-integration` (branched off `rollup/agentic-2026-05-04`).
