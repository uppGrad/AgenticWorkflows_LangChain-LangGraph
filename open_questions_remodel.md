# Open questions — gate-0 removal + gate-1 remodel

Tracking gaps and decisions surfaced during the post-Step-7 review.
Resolved items kept here for the audit trail.

---

## 1. The whole `evaluate_scrape` → `normalized_requirements` path is misframed  ❌ ARCHITECTURAL BUG

### 1a. JS-rendering support — still present, just env-gated

**Concern raised:** "Don't we have support for JS-rendered websites like
Workday?"

**Status — confirmed in code:** Yes, intact, not removed. Architecture:
- `web_fetcher.fetch_url_with_fallback` — httpx-first, escalates to
  Crawl4AI/Playwright when (a) httpx returns thin AND (b)
  `UPPGRAD_BROWSER_SCRAPE_ENABLED=true`.
- `web_fetcher.force_browser_fetch` — bypasses the thin gate and renders
  via Crawl4AI directly (used by `extract_form_fields` for non-thin
  server-rendered pages with client-side hydrated forms).
- `_build_crawler_run_config` waits for
  `js:() => document.body.innerText.length > 1000` so React/SPA hydration
  completes before extraction.

**The Workday-specific issue is a different one:** `myworkdayjobs.com`
URLs sit behind an **auth/SSO wall**, not just JS rendering.
`ats_form_urls.resolve_application_form_url` returns `None` for any
`*.myworkdayjobs.com` host — extraction can't proceed even with browser
because there's nothing public to render. So Workday hits Case A below
(empty form_fields) for an *auth* reason, not a *JS rendering* reason.

**Production gap (already in CLAUDE.md "Still open"):**
`UPPGRAD_BROWSER_SCRAPE_ENABLED=false` in the deployed env. Until the
Railway Chromium piece lands, the browser path no-ops in prod. That's a
deploy task, not missing code.

### 1b. Brave discovery + verification pipeline — fully intact

**Concern raised:** "Please don't tell me Brave search discovery is
removed as well!!!"

**Status — confirmed in code: still in place, end-to-end:**
- `tools/search.py` — `BraveSearchProvider`, opt-in via
  `UPPGRAD_SEARCH_PROVIDER=brave` + `BRAVE_SEARCH_API_KEY`.
- `tools/url_discovery.py` — 3-tier orchestration (ATS → careers →
  generic), title-fuzz prerequisite (`_TITLE_FUZZY_MIN=85`),
  multi-factor corroborator scoring, slug fuzzy match
  (`_SLUG_FUZZY_MIN=70`), closed-posting phrase detection,
  `_detect_thin` rejection before scoring.
- `nodes/discover_apply_url.py` — wired between `load_opportunity`
  and `scrape_application_page` for jobs.
- Per-ATS form URL resolution, raw_html propagation, posting_closed
  flag — all carried on `DiscoveryResult` and propagated through
  state.
- Discovery cache (`JobApplyUrlDiscovery` model on backend) with
  14-day staleness gate — also intact.

Verification gates (`score_candidate` in `url_discovery.py`):
- Title fuzzy match ≥85 (hard prerequisite).
- ATS / generic tiers need ≥2 corroborators from
  {company-in-text, location-match, posted-time-match,
  description-keyword-overlap}; careers tier needs 1.
- Thin pages rejected before scoring.

So for the **discovery + verification** purpose, nothing is removed.

### 1c. The actual bug — `evaluate_scrape` drifted from its original purpose

**Correction to earlier diagnosis:** `evaluate_scrape` does NOT write
directly to `asset_mapping`'s input. There is one intermediate node
(`determine_requirements`). Full chain below in §1d.

**Original purpose (per your description):** post-discovery sanity
check. Confirm the URL Brave returned actually serves the right job —
not a 404, not a different role, not a wrong location, not an
expired posting. A second-pass verification on the page contents.

**What `evaluate_scrape.py` actually does today:**
1. Reads `scraped_requirements.raw_content` (the page text).
2. Runs an LLM with this prompt:
   *"You are assessing the quality of a scraped job application page …
   return … requirements: list of objects, each with requirement_type,
   document_type …"*
3. Writes BOTH a `status` (full/partial/failed) AND a `requirements`
   list back into `state['scraped_requirements']`.

The LLM is asked to return a `status` (full/partial/failed) — that part
matches the original verification purpose. But the same LLM call **also**
fishes "CV / cover letter / portfolio" mentions out of the JD prose
and returns them as requirements. Those JD-derived "requirements" then
flow through asset mapping and gate 1 as if they were authoritative
form requirements.

The status verification is correct. The requirement-extraction-from-prose
side is the architectural drift you're flagging. JD copy might say "we'd
love to see your CV" rhetorically while the actual form asks for a
portfolio or transcript instead.

**What we have for the form, properly:** `extract_form_fields` reads the
real `<input>/<select>/<textarea>` tags from the rendered DOM and emits
`FormField` records. THAT is the authoritative requirement source for jobs.

### 1d. Actual data flow (file:line refs)

```
discover_apply_url        Brave search + 3-tier verification + fetch.
  └─→ state['discovered_apply_url']
      state['discovered_page_content']   (markdown / HTML)
      state['discovered_raw_html']       (real DOM)

scrape_application_page   No re-fetch — uses pre-fetched content from
                          discovery. Writes:
                          state['scraped_requirements'] = {
                            status='partial',
                            raw_content=<page text>,
                            raw_html=<DOM>,
                            requirements=[]   # empty here
                          }
                          (scrape_application_page.py:47–58)

evaluate_scrape           LLM-on-prose. Reads raw_content.
                          Overwrites state['scraped_requirements']:
                          {
                            status=full|partial|failed,
                            requirements=[...JD-derived list...],
                            confidence=0.x,
                            source=<url>
                          }
                          (evaluate_scrape.py:191–198)
                          ← Discards raw_content/raw_html on overwrite.

extract_form_fields       LLM-on-DOM. Reads discovered_raw_html.
  └─→ state['form_fields'] = [FormField, ...]
      (or empty list when form unreachable)

determine_requirements    Routes by opportunity_type:
  - jobs:
      Reads scraped_requirements.{status, requirements}.
      If status=='full': use scraped reqs as-is.
      If status=='partial': MERGE scraped + _DEFAULTS["job"] (dedupe).
      If failed/empty: use _DEFAULTS["job"].
      Also tags form_fields with canonical_document_type.
      (determine_requirements.py:289–312)
  - masters/phd: parse opportunity_data['data'] JSON.
      (determine_requirements.py:317–326)
  - scholarship: parse opportunity_data['data'] JSON.
      (determine_requirements.py:331–340)
  └─→ state['normalized_requirements'] = [NormalizedRequirement, ...]

asset_mapping             Reads, in priority order:
  1. state['form_fields']               → _build_from_form_fields
  2. state['normalized_requirements']   → _build_from_normalized_…
  3. _DEFAULTS[opportunity_type]        → _build_defaults
  (asset_mapping.py:243–287)
  └─→ state['requirement_items'] = [RequirementItem, ...]
      state['asset_mapping']    = same list (JSONB column reused)
```

**So the chain that lets JD prose leak into requirements for jobs is:**
`evaluate_scrape` (LLM-on-JD-prose)
  → writes `scraped_requirements.requirements`
  → read by `determine_requirements`
  → written to `normalized_requirements`
  → read by `asset_mapping` as 2nd-priority fallback when `form_fields`
    is empty.

`asset_mapping` itself never imports or references `evaluate_scrape`. The
bug surfaces because `determine_requirements` faithfully passes through
whatever `evaluate_scrape` produced — including the prose-derived
phantom requirements.

### 1e. Why three nodes feel redundant — they're not the same job, but two of them try to answer "what documents does this job require?"

| Node | Source | Output | Legitimate purpose |
|---|---|---|---|
| `evaluate_scrape` | LLM on **JD prose** | `scraped_requirements.{status, requirements}` | **Status verification** (your original intent — is the discovered page the right job, not a 404). The `requirements` field is the drift. |
| `extract_form_fields` | LLM on **rendered form DOM** | `form_fields` (real `<input>/<select>/<textarea>`) | **Authoritative form schema** — what the form actually collects. |
| `determine_requirements` | Reads `scraped_requirements`, `form_fields`, `opportunity_data['data']` | `normalized_requirements` (jobs: merged scrape+defaults; non-jobs: parsed JSON) | **Routing/fallback orchestrator + canonical-type tagging.** Works correctly for non-jobs. For jobs, faithfully forwards the prose-derived list. |

So they're not three nodes doing the same thing. They're:
- **`extract_form_fields`** = ground truth (form-side).
- **`evaluate_scrape`** = doing TWO jobs — verification (legitimate) +
  prose-as-requirements (drift).
- **`determine_requirements`** = an orchestrator that doesn't know
  which input is more trustworthy and forwards both.

### 1f. Where do we gather requirements from outside the form?

This is the right question. The honest answer:

| Opportunity type | Form available? | Legitimate signal |
|---|---|---|
| **Jobs (form reached)** | Yes | `extract_form_fields` — the real DOM. |
| **Jobs (form unreached)** | No (Workday auth wall, hydration failed, browser disabled in prod) | **No good outside-the-form signal.** JD prose is recruiter copy, not authoritative. Best fallback is `_DEFAULTS["job"]` = `[CV, Cover Letter]`. |
| **Masters / PhD** | N/A — no form pipeline | `programs.data` JSON `requirements.{academic, english, other}` field — structured, parsed by `_parse_program_requirements`. |
| **Scholarship** | N/A | `scholarships.data` JSON `required_documents` field — parsed by `_parse_scholarship_requirements`. |

**Punchline:** there is no legitimate prose-based requirement signal for
jobs. The JD on a Greenhouse/Lever overview page is marketing copy. Either
the form gives us the truth, or we use defaults.

### Concrete failure cases under the current code

- **Case A — empty form_fields, opportunity is a job.** Workday auth wall,
  forced browser disabled in prod, or hydration timed out.
  `evaluate_scrape` ran the LLM on the JD prose and produced a
  requirements list from whatever document words appeared. Asset mapping
  uses that JD-derived list as fallback. Wrong source of truth — and
  worse, often produces phantom requirements the form never asked for.
- **Case B — non-empty form_fields, opportunity is a job.** Asset mapping
  uses form_fields and the JD-derived `normalized_requirements` is
  ignored. Correct outcome — but the LLM call still ran. Cost waste only.
- **Case C — non-job opportunity (masters/phd/scholarship).**
  `determine_requirements` parses `opportunity_data['data']` JSON
  directly. Different code path, legitimate.

### Resolution direction (proposed, needs sign-off)

The fix is narrow — keep `evaluate_scrape`'s **verification** role,
strip its **requirement extraction** role:

1. **Reduce `evaluate_scrape` to status-only.** Update the LLM prompt to
   return only `status` (full/partial/failed) + `confidence` — no
   `requirements` field. The status is genuinely useful: telemetry,
   "did we reach the form vs. an error page" gates, and feeding into
   the discovery cache's `last_verified_at` decision.

2. **Stop populating `normalized_requirements` from prose for jobs.**
   When form_fields is empty AND opportunity_type=='job', fall back
   directly to `_DEFAULTS["job"]` in asset_mapping (already implemented
   as the floor — just remove the prose-derived list from competing
   for that slot first).

3. **For non-jobs**, keep parsing `opportunity_data['data']` JSON in
   `determine_requirements`. That's structured data, not LLM-on-prose.

4. **`_DEFAULTS` floor stays** unchanged.

Note on overlap with discovery verification: the 3-tier pipeline already
does its own verification (title fuzz + corroborators + closed-posting
detection). `evaluate_scrape` is a *second* verification pass on the same
page text. There's a separate question — possibly redundant — of whether
the second pass adds anything beyond what the discovery verifier already
caught. Worth a sanity audit but lower priority than fixing the
prose-as-requirements bug.

---

## 2. `misc` collapse threshold  ✅ RESOLVED

**Decision.** No threshold. Misc collapse always fires when ANY non-file,
non-textarea field exists. The whole point of the misc bucket is to spare
the user from being asked about fields the system can fill itself (email,
veteran-status radios with a default-from-context, etc.). Even one such
field belongs in misc.

**Implication.** The current code (`asset_mapping._build_from_form_fields`
emitting misc when `misc_field_indices` is non-empty) is correct as-is.

---

## 3. `expected_source='user_answer'` on non-textarea fields  ❓ OPEN

**Today.** `_build_from_form_fields` puts an item in the **text** bucket
only when `field_type=='textarea'` OR `(field_type=='text' AND
expected_source=='user_answer')`. A `<select>` whose label is "How did you
hear about us?" gets `expected_source='user_answer'` from the LLM
classifier but `field_type='select'`, so it lands in **misc** instead of
**text**.

**The question.** Should the text bucket include ANY non-file field with
`expected_source='user_answer'` regardless of `field_type`? Or is misc
the right home for select/radio "user_answer" fields because their
options are bounded?

---

## 4. USER_SUPPLIED canonical types — `auto_generate` rejection scope  ❓ OPEN (likely OK)

**Today.** `human_gate_1._validate_resume` rejects `auto_generate` for any
`USER_SUPPLIED` document type regardless of whether the item is required.
Plan §H wording was ambiguous — one line said "USER_SUPPLIED rejects
auto_generate" (unconditional), the next said "(only upload / skip when
not required)" (conditional).

**The question.** Confirm the frontend offers Upload / Skip / Ignore for
these — never an Auto-Generate button — for both required and optional
USER_SUPPLIED items.

---

## 5. `auto_submit_feasible_at_gate_1` skips misc unconditionally  ❓ OPEN

**Today.** `human_gate_1._compute_auto_submit_feasible` returns `True` even
when every required item is misc and the user picks `misc_strategy='ignore'`.
Concrete example: a Greenhouse form with only Name + Email (no file, no
textarea) → 1 misc line, user picks `misc_strategy='ignore'` →
`auto_submit_feasible=True` despite no required field having an explicit
choice.

**The question.** Should feasibility require `misc_strategy='auto_fill'`
when misc is the ONLY required category present? Or is that contradiction
better surfaced as a gate-2 warning?

---

## 6. In-flight session cancellation policy on Step-8 deploy  ✅ RESOLVED

**Decision.** No real users yet — cancel silently as part of the pre-deploy
ops sweep. No notification or refund logic needed. Revisit when there's a
production user base.

---

## 7. Gate-1 file upload size + extension whitelist  ❓ OPEN

**Today.** Step 9 plumbs `additional_uploads: List[FileField]` through the
backend serializer. Neither the plan nor `CLAUDE.md` sets caps.

**The question.** What size/extension caps should the gate-1 serializer
enforce? Suggested defaults: 10MB max per file, `.pdf` and `.docx` only.
A 50MB scanned PDF through `_extract_text_with_ocr_fallback` is minutes of
OCR cost.

---

## 8. `attempt_auto_submit=True` UX while auto-submit module not yet wired  ❓ OPEN

**Today.** Gate 2 records `attempt_auto_submit` regardless of feasibility,
but the actual auto-submit module (`feature/playwright-form-filler`)
isn't merged or wired. Today's routing for external opportunities is
unconditionally `package_and_handoff`.

**The question.** When the user clicks "submit for me" at gate 2, what
should the frontend show?
- (a) "Auto-submit isn't available yet — here's your package."
- (b) Treat the click as a normal handoff request, no message.
- (c) Hide the button until auto-submit is wired.

---

## 9. `accepts_file` vs. renderer output format  ❓ OPEN

**Today.** `FormField.accepts_file` carries `[".pdf", ".docx"]` etc. from
the form's `accept` attribute. `document_renderer.render_text_to_pdf`
always emits PDF. No cross-check today.

**The question.** Should gate 2 compare each tailored document's renderer
output against the corresponding FormField's `accepts_file` list and
surface a warning when they disagree (e.g. form accepts `.docx` only)?
Auto-fill at submit-time will silently fail in that case.

---

## 10. Internal short-circuit defaults vs. `jobs_application` schema drift  ❓ OPEN

**Today.** Internal jobs (`employer_id == 1`) skip discovery / scrape /
extract_form_fields and `determine_requirements` emits `[CV, Cover Letter]`
directly. This matches the current `jobs_application` model's non-system
fields (`resume_file` + `cover_letter`).

**The question.** If a teammate adds a third internal field (e.g. salary
expectation), the agentic short-circuit silently misses it. Worth a
schema-vs-defaults assertion test in the backend that fails CI when
`jobs_application`'s field set drifts from the agentic defaults?

---

## 11. PreA `ready_for_polish` short-circuit  ❓ OPEN (tunable)

**Today.** Plan §F: "Always 2-pass; do NOT short-circuit on
`ready_for_polish`." Code matches.

**The question.** Is there a token-cost reason to make
`ready_for_polish` skip T1 (T2-only path)? Real-data-driven decision —
flag as a tunable for after a few live runs.

---

## 12. `tailored_answers` 1500-char cap vs. form's `maxlength`  ❓ OPEN

**Today.** `application_tailoring` caps text answers at 1500 chars.
`FormField` has no `max_length` field — extraction doesn't capture the
form's `maxlength` attribute.

**The question.** Should we plumb `maxlength` from the rendered form into
`FormField` at extraction time and use that as the cap (falling back to
1500 when absent)?

---

## 13. Canonical-type LLM classification cache  ❓ OPEN

**Today.** Discovery cache (`JobApplyUrlDiscovery`) stores the URL +
raw_html but not the LLM-classified `canonical_document_type` per
FormField. Re-running discovery for the same `linkedin_jobs.id` re-runs
the canonical-type LLM batch.

**The question.** Worth caching at the discovery layer if cost becomes
material. Cheap to add when needed.

---

## 14. `AssetMap`/`AssetMappingOutput` deprecation cliff  ❓ OPEN

**Today.** Marked deprecated in `schemas.py` "for one cycle."

**The question.** When is the cycle? After Step 8 ships? After two
releases? Set a removal trigger.

---

## 15. Crawl4AI consolidation vs. auto-fill landing  ❓ OPEN

**Today.** Crawl4AI consolidation is flagged orthogonal in `CLAUDE.md`.
The auto-fill module (`feature/playwright-form-filler`) makes Playwright
a committed dependency.

**The question.** Land auto-fill under the new gate-2 contract first, or
swap fetchers to direct Playwright in the same drop?

---

## 16. PR A (`feature/playwright-form-filler`) rebase  ❓ OPEN

**Today.** PR A's `compute_form_values` + `fill_form_async` were written
against the *old* state shape (`asset_mapping` AssetMap dicts). The new
`requirement_items` payload + `tailored_answers` keyed by
`form_field_index` changes the inputs.

**The question.** Rebase the branch onto the new shape before merging.
Concrete diff:
- `asset_mapping: List[AssetMap]` → `requirement_items: List[RequirementItem]` + `tailored_answers: Dict[str, dict]`
- Document-type lookup via `requirement_items[*].document_type` (not `AssetMap.requirement_type`).
- Text-answer lookup via `tailored_answers[form_field_index]` (not `tailored_documents`).
- Misc handling: `human_review_1.misc_strategy` drives whether misc fields are touched.

---

## 17. Custom-React-control extraction gap  ❓ OPEN

**Today.** The `extract_form_fields` pipeline is biased toward native
`<input>`/`<select>`/`<textarea>` tags end-to-end:

- `form_extractor._score` (`form_extractor.py:80–81`) picks the form by
  counting descendants of `("input", "select", "textarea")` only. A
  real apply UI built entirely as `<div role="combobox">` + `<button>`
  controls scores 0. If the page also contains a stray native `<form>`
  (search bar, newsletter signup) with score ≥ 1, Strategy 1 returns
  that instead of the apply UI.
- Body fallback (Strategy 2) re-uses the same scorer. A 100% custom-
  rendered SPA with no native controls returns `""` and the node
  short-circuits to `form_fields=[]`.
- `extract_form_fields._SYSTEM` prompt enumerates native types only and
  steers the LLM toward `type` attributes — no mention of
  `role="combobox"`, `contenteditable`, `aria-checked`, etc.
- `FormFieldType` enum has no value for custom controls so even if the
  LLM wanted to surface one, there's no slot for it.

**Concrete cases:**

| Control | What's in the DOM | Captured today |
|---|---|---|
| Greenhouse country/select | `<input role="combobox">` + hidden options | ✅ (native input survives) |
| Ashby combobox | `<div role="combobox">` + virtualized list | ❌ no native input in trigger |
| Custom file dropzone | Visible `<div>` + `<input type="file" hidden>` | ✅ (only `type="hidden"` is stripped, `type="file"` survives) |
| Rich-text textarea | `<div contenteditable="true">` | ❌ |
| Yes/no button pair | `<button aria-pressed>` × 2 | ❌ |
| Multi-step Workday form | Only step 1 in the DOM | ❌ steps 2+ never seen |
| GDPR/EEO div checkbox | `<div role="checkbox" aria-checked>` | ❌ |

**Downstream impact.**
- Asset mapping doesn't see missed fields → user isn't asked about them.
- `_DEFAULTS["job"]` floor only kicks in when `form_fields=[]` entirely;
  partial extraction (3 native captured, 2 div-based missed) silently
  skips the floor.
- PR A's auto-fill module (Tier 4 LLM picker on live DOM) compensates
  at fill time but a missed *required* field silently blocks the submit.

**Possible fix paths:**

1. **Cheap (extraction-side):** widen `_score` to also count
   `[role="combobox"]`, `[contenteditable]`, `[role="checkbox"]`, and
   update the LLM prompt to read `role`/`aria-*` attributes. Add a
   `custom_control` value to `FormFieldType` and `unknown_widget` to
   `FormFieldValueSource` so the LLM has a destination for things it
   sees but can't classify.
2. **Right (submit-side):** treat extracted `form_fields` as a *hint*
   for gate 1, and rely on PR A's runtime-DOM Tier 4 picker as the
   source of truth at fill time. Live DOM is the only honest source
   for SPA forms.

**Recommended.** Ship #1 cheap as a follow-up to the prose-bug fix
(Step 1.6 in the plan), and accept #2's hint-only contract for gate 1.

---

## Status legend
- ✅ RESOLVED — decision recorded, no further action needed.
- ❓ OPEN — needs decision before relevant step ships.
- ❓ NEEDS CLARIFICATION — question itself unclear; expanded inline.
