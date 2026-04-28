# Auto-Apply Domain — Status

_Last updated: 2026-04-28. Source of truth: this branch (`feature/form-field-extraction`) merged into main._

This document captures what the auto-apply pipeline does today, what's wired through the Django backend, what runs end-to-end against prod data, and what's still missing for a fully autonomous "apply on the student's behalf" engine. The aspiration is: **a student picks an opportunity, the engine generates tailored materials and either submits the application directly OR hands the student a ready-to-submit package with all form fields pre-answered.**

The current shipped product is the **handoff** half. The auto-submit half — Playwright-driven form filling — is the next major work item.

---

## 1. End-to-end happy paths today

### 1.1 External job (e.g. Anthropic, Notion, MongoDB)

```
Student picks job → backend builds opportunity_snapshot + profile_snapshot
                  ↓
load_opportunity (uses pre-loaded snapshot — no DB call inside the graph)
                  ↓
discover_apply_url   ← Brave Search → Greenhouse/Lever/Ashby/etc., httpx + browser fallback
                  ↓
scrape_application_page   (uses verified content from discovery — single fetch architecture)
                  ↓
evaluate_scrape   ← gpt-4o-mini extracts requirements list (or falls back to assumed [CV, Cover Letter])
                  ↓
extract_form_fields   ← gpt-4o-mini extracts FormField list from the form HTML
                  ↓
determine_requirements
                  ↓
eligibility_and_readiness   (deadline + missing user-supplied docs are hard blocks; everything else → warning)
                  ↓
[gate 0 if missing user-supplied docs → loop back to eligibility, max 2 retries]
                  ↓
asset_mapping   ← LLM picks user CV / generates Cover Letter / SOP / etc. per requirement
                  ↓
[gate 1 — student reviews mapping, can override and upload extra docs]
                  ↓
application_tailoring   ← LLM rewrites/generates per doc with per-type caps
                  ↓
application_evaluation   (loop to tailoring on hallucination or coverage failure, max 2 retries)
                  ↓
[gate 2 — student approves final package]
                  ↓
package_and_handoff   → application_package = {documents, opportunity, warnings, posting_closed, form_fields}
                  ↓
record_application
                  ↓
COMPLETED_HANDOFF — student gets a download link + the URL to submit
```

### 1.2 Internal job (`employer_id == 1`)

Same up to gate 2. Instead of `package_and_handoff` the graph routes to `submit_internal` (records intent only), the backend's `finalize_internal_submission` writes a real `Application` row via Django ORM with the tailored materials, and the session moves to `COMPLETED_SUBMITTED`.

### 1.3 Masters / PhD / Scholarship

`load_opportunity` skips scraping entirely; `determine_requirements` parses the `data` JSON field of the opportunity row. Discovery + form extraction are job-only. Otherwise the pipeline is identical.

---

## 2. What works today (live-verified)

Live-tested against prod `linkedin_jobs` rows on Neon (project `summer-math-90128942`, default branch `production`).

| Capability | Verified case | Notes |
|---|---|---|
| URL discovery (Brave-driven, 3-tier ATS/careers/generic, multi-factor verification) | ✅ MongoDB 227899 → `mongodb.com/careers/jobs/7484657` (generic, conf 0.85) | All four corroborators present |
| Discovery for ATS-tier (Greenhouse) | ✅ Anthropic 228527 → `job-boards.greenhouse.io/anthropic/jobs/5121912008` | Cross-checked: title, company, Dublin all in body |
| Discovery for Ashby-hosted (browser fallback) | ✅ Notion 227322 → `jobs.ashbyhq.com/notion/6ad34426-...` | Crawl4AI + JS hydration `wait_for` |
| ATS slug-mismatch rejection | ✅ GitHub query no longer false-matches Forma.ai's Greenhouse posting | Live regression that motivated `_extract_ats_company_slug` |
| Closed-listing detection | ✅ Notion-via-Base10 (when browser is off) → `method='closed'`, surfaced in handoff package | "no longer accepting applications" phrase set |
| Per-ATS form URL resolver | ✅ Ashby `/application`, Lever `/apply`, SmartRecruiters `/apply`, Greenhouse same URL, Workday returns `None` for auth wall | `tools/ats_form_urls.py` |
| Form-field extraction (Greenhouse server-rendered) | ✅ Anthropic 228527 → 22 fields (First/Last Name, Email, Country dropdown, Phone, Resume, "Why Anthropic?" textarea, visa, relocation, etc.) | LLM with `FormSchema` structured output |
| Form-field extraction (Ashby React) | ✅ Notion 227322 → 10 fields (incl. Pronouns radio with all 5 options, Anchor-Days checkbox, "How did you hear" multi-checkbox, all `expected_source` classified) | Body-fallback path in `form_extractor.py` |
| Browser fallback (Crawl4AI + Chromium) | ✅ Renders Ashby SPA, hits `wait_for` on JS-hydrated body text | Lazy-imported, gated on `UPPGRAD_BROWSER_SCRAPE_ENABLED` |
| Posting-closed warning in handoff | ✅ `application_package.warnings` includes "This posting appears to no longer be accepting applications…" | |
| Compatibility warnings (location/age/discipline/nationality) | ✅ Surfaced in `application_package.warnings`, NOT a hard block | |
| Cancel mid-flight | ✅ `cancel_session(thread_id, checkpointer)` injects error result via `graph.update_state()`; observed at next node boundary | |
| Gate 0 loop (missing user-supplied docs) | ✅ Up to 2 retries; cap-out → `INELIGIBLE` with `error_code='PROFILE_INCOMPLETE_AFTER_RETRIES'` | |
| Internal-job ORM write | ✅ Anthropic-style internal job → real `Application` row with PDF resume + tailored cover letter, FK populated | `auto_apply_adapter.finalize_internal_submission` |
| External-job handoff with real Koray CV + real OpenAI tailoring | ✅ 16-graph-step run, properly formatted output | Verified end-to-end during backend integration phase |

---

## 3. Backend integration status

### 3.1 Models (`backend/ai_services/models.py`)

| Model | Purpose | Status |
|---|---|---|
| `ApplicationSession` | One row per auto-apply attempt. Carries thread_id, status, snapshots, gate responses, tailored documents, application_package, optional FK to `Application`. Partial unique constraint: one active session per `(student, opportunity_type, opportunity_id)`. | ✅ Functional, exercised by tests + live runs |
| `JobApplyUrlDiscovery` | Cross-user cache of resolved apply URLs per `linkedin_jobs.id`. PK on `job_id`. 14-day staleness gate. Only successful methods cached (`url_direct`/`ats`/`careers`/`generic`); `failed`/`closed` skipped. | ✅ Functional |
| `Application` (jobs app, FK from `ApplicationSession`) | Real submission record for internal jobs. Already existed pre-auto-apply. | ✅ Functional |
| Status enum | PROCESSING / AWAITING_PROFILE_COMPLETION / AWAITING_DOCUMENT_MAPPING / AWAITING_FINAL_APPROVAL / FINALIZING / COMPLETED_SUBMITTED / COMPLETED_HANDOFF / INELIGIBLE / ERROR / CANCELLED | ✅ Wired end-to-end |

### 3.2 Adapter (`backend/ai_services/auto_apply_adapter.py`)

| Function | Purpose | Status |
|---|---|---|
| `build_auto_apply_opportunity_snapshot` | Pre-loads opportunity row from the right table (`linkedin_jobs` / `programs` / `scholarships`); skips closed jobs. | ✅ |
| `build_auto_apply_profile_snapshot` | Builds profile dict from `Student` + `StudentCV`. Splits CSV fields, attaches CV file ref + parsed text. | ✅ |
| `_build_checkpointed_auto_apply_graph` | Compiles graph with `PostgresSaver` checkpointer. | ✅ |
| `start_session` / `_run_graph_initial_phase` | Spins up a graph thread, runs to first interrupt (gate 0/1/2) or terminal. | ✅ |
| `_pending_node_after_invoke` | Reads `graph.get_state(config).next` to identify which gate is held. (Bug fix for `interrupt()` lag described in CLAUDE.md.) | ✅ |
| `_persist_state_after_phase` | Maps graph state → DB row. Handles the `INELIGIBLE` / `CANCELLED` / `PROFILE_INCOMPLETE_AFTER_RETRIES` error_codes. | ✅ |
| `resume_session_gate_0/1/2` | Per-gate handlers with `Command(resume=...)` semantics. Gate 0 injects fresh `profile_snapshot` via `graph.update_state` to fix the stale-profile bug. | ✅ |
| `finalize_internal_submission` | After graph terminates with `submission_type='internal'`, writes the real `Application` row and links it back via FK. | ✅ |
| `cancel_session` | Wraps the agentic `control.cancel_session` for backend callers. | ✅ |
| `_lookup_discovery_cache` / `_persist_discovery_to_cache` | Cross-user cache layer with 14-day TTL; skips `failed`, `skipped_internal`, `closed`. | ✅ |

### 3.3 HTTP API (`backend/ai_services/views.py` + `urls.py`)

| Endpoint | Method | Purpose | Status |
|---|---|---|---|
| `/api/ai/application-sessions/` | GET, POST | List sessions / start new session | ✅ |
| `/api/ai/application-sessions/<id>/` | GET | Poll session status | ✅ |
| `/api/ai/application-sessions/<id>/cancel/` | POST | Cancel mid-flight | ✅ |
| `/api/ai/application-sessions/<id>/resume-gate-0/` | POST | User completes missing profile fields | ✅ |
| `/api/ai/application-sessions/<id>/resume-gate-1/` | POST | User confirms / overrides document mapping; uploads extras | ⚠️ Partial — `additional_uploads` accepted by serializer but not threaded into tailoring `content` (planned 1.5 work) |
| `/api/ai/application-sessions/<id>/resume-gate-2/` | POST | User approves final package | ✅ |

### 3.4 Operational glue

- **Janitor (`janitor.py`)**: per-status TTL cleanup. PROCESSING/15min, FINALIZING/30min, AWAITING_*/7 days. ✅
- **Document renderer (`document_renderer.py`)**: ReportLab text→PDF for internal-submit `Application.resume_file`. ✅
- **Test count**: 27 ai_services unit tests passing on the backend repo (1 pre-existing unrelated failure on `test_post_documental_feedback`). ✅
- **Production config**: backend integration runs on Railway with `OPENAI_API_KEY` + (newly) `BRAVE_SEARCH_API_KEY`/`UPPGRAD_SEARCH_PROVIDER`. Browser fallback in prod requires Chromium install via Playwright — separate PR (#5 in backend repo) ships the Dockerfile change.

---

## 4. Workflow node-by-node coverage

All nodes live in `src/uppgrad_agentic/workflows/auto_apply/nodes/`. Implementation status reflects what they currently do, not aspirational behavior.

| Node | Phase | Implementation | Live-verified? |
|---|---|---|---|
| `load_opportunity` | Opportunity Intelligence | Short-circuits on backend-pre-loaded `opportunity_data`; falls back to in-repo stubs for CLI mode (with WARNING log). | ✅ |
| `discover_apply_url` | Opportunity Intelligence | 3-tier search (ATS → careers → generic) + multi-factor verification + closed-posting detection. Skipped for non-job types. | ✅ |
| `scrape_application_page` | Opportunity Intelligence | Uses `state.discovered_page_content` if discovery fetched; else fetches fresh via `web_fetcher`. Surfaces `raw_html` from state into `scraped_requirements`. | ✅ |
| `evaluate_scrape` | Opportunity Intelligence | LLM with `ScrapeResult` structured output; heuristic keyword fallback when LLM unavailable; falls back to assumed `[CV, Cover Letter]` when extraction yields nothing. | ✅ |
| `extract_form_fields` | Opportunity Intelligence | NEW. LLM with `FormSchema` structured output. Three tiers: in-state HTML → forced browser fetch → ATS iframe-follow. | ✅ for Greenhouse + Ashby; ❌ for iframe-embedded (MongoDB) — known gap, see §5. |
| `determine_requirements` | Opportunity Intelligence | Picks scraped requirements when present; else type-defaults (job → `[CV, Cover Letter]`, masters/phd → `[CV, SOP]`, scholarship → `[CV, Cover Letter]`). | ✅ |
| `eligibility_and_readiness` | Eligibility | Hard-blocks ONLY on deadline-passed and missing user-supplied (non-generatable) docs. Compatibility issues → `state.compatibility_warnings`. | ✅ |
| `end_with_explanation` | Eligibility terminal | Sets `result.error_code='INELIGIBLE'` when the verdict is hard-block. | ✅ |
| `human_gate_0` | Eligibility | `interrupt()` cycle, max 2 iterations. Cap-out → `error_code='PROFILE_INCOMPLETE_AFTER_RETRIES'`. Loops back to eligibility on resume. | ✅ |
| `asset_mapping` | Asset Mapping | LLM picks user CV / sets generation depth per requirement. `_GENERATABLE` list controls which doc types the system can write itself. | ✅ |
| `human_gate_1` | Asset Mapping | `interrupt()` for student to confirm/override mapping. Accepts `additional_uploads` in serializer; **adapter doesn't yet thread their content into tailoring** (gap). | ⚠️ Partial |
| `application_tailoring` | Tailoring | LLM apply-mode rewrite per doc with per-type byte caps (CV ≤ 8000, Cover Letter ≤ 3000, SOP/Personal Statement ≤ 6000). | ✅ |
| `application_evaluation` | Tailoring | Length / placeholder / keyword-coverage checks; loop back to tailoring on failure (max 2 retries). | ✅ |
| `human_gate_2` | Tailoring | `interrupt()` for final approval. Reject → cancel (no edit-and-re-tailor loop yet — see §5). | ✅ |
| `submit_internal` | Submission | Records intent only (no fake `platform_application_id`). Backend's `finalize_internal_submission` does the real ORM write. | ✅ |
| `package_and_handoff` | Submission | Assembles `application_package = {documents, opportunity, warnings, posting_closed, form_fields, scrape_status, scrape_confidence, scrape_source}`. Includes `discovered_form_url` in `opportunity.form_url`. | ✅ |
| `record_application` | Submission | Logs outcome (timestamp, doc types, scrape metadata). | ✅ |

---

## 5. Schemas (`schemas.py`)

| Model | Purpose | Status |
|---|---|---|
| `NormalizedRequirement` | One requirement (document type, eligibility, language, other). Used by asset mapping + tailoring. | ✅ Document-centric scope is intentional for v1 |
| `ScrapeResult` | LLM output for `evaluate_scrape`. | ✅ |
| `EligibilityResult` | Decision + reasons + missing fields. | ✅ |
| `AssetMap` / `AssetMappingOutput` | One mapping per requirement (source document, tailoring depth, available, notes). | ✅ |
| `FormField` / `FormSchema` | NEW. Per-input metadata: label, field_type, name, required, options, accepts_file, expected_source ∈ {user_profile, user_document, user_answer, computed, unknown}. Container `FormSchema` carries fields + form_action + form_method. | ✅ Captured but **not yet consumed** by any downstream node — surfaced only in the handoff package, awaiting auto-submit |
| `WorkflowResult` | status / error_code / user_message / details | ✅ |

---

## 6. Test coverage

```
tests/
  common/                         (test counts approx — see pytest output)
  tools/
    test_ats_form_urls.py            11 tests — per-ATS form URL resolver
    test_form_extractor.py           17 tests — <form>/body-fallback/iframe extraction
    test_search.py                    N tests — Brave provider
    test_url_discovery_orchestration.py  16 tests — 3-tier orchestrator + closed-posting handling
    test_url_discovery_verify.py     17 tests — multi-factor verification + slug guard
    test_web_fetcher_browser_fallback.py 10 tests — Crawl4AI fallback policy
    test_web_fetcher_httpx.py        13 tests — thin detector + final_url + word-boundary regression guards
  workflows/auto_apply/
    test_application_tailoring_caps.py    Per-doc-type cap behavior
    test_control_cancel.py                cancel_session via graph.update_state
    test_discover_apply_url.py            Node-level discovery wiring (cache, internal-skip)
    test_eligibility_compatibility_warnings.py / _generatable_docs    Eligibility split
    test_extract_form_fields.py           7 tests — form-field extraction node
    test_graph_discovery_routing.py       Routing through opportunity intelligence phase
    test_graph_gate_0_loop.py             Gate 0 retry loop
    test_human_gate_0.py                  Real interrupt cycle
    test_load_opportunity_preloaded.py    Backend-pre-load short-circuit
    test_package_includes_warnings.py     Compatibility warnings + posting_closed in handoff
    test_resolve_profile.py / _integration   profile_snapshot resolution
    test_scrape_application_page_fetcher.py Pre-fetched-content fast path
    test_state_compatibility_warnings.py / _fields   AutoApplyState shape
    test_stub_fallback_warnings.py        Stub-fallback WARNING logs
    test_submit_internal_intent.py        No fake platform_application_id

Total: 172 tests, all green.
```

Backend `ai_services` adds 27 more (1 pre-existing unrelated failure on `test_post_documental_feedback`).

---

## 7. What's missing for full auto-submit

The aspiration described in the brief is "auto-applies on behalf of students interactively through package handoffs." Today the **handoff** half is shipped. The **submit-on-behalf-of** half is not. Here's the gap, in priority order.

### 7.1 Critical / blocking for autonomy

1. **External form auto-submission via Playwright.** Today external jobs always end in `COMPLETED_HANDOFF` — the student manually submits via the URL we provide. To submit on their behalf, we need a new `submit_external` node that:
   - Drives the form URL via Playwright (renders, waits, fills inputs, uploads files, submits)
   - Maps `form_fields` (which extract_form_fields already captures) to values via `expected_source`:
     - `user_profile` → look up in `Student` row
     - `user_document` → use the tailored doc from `tailored_documents`
     - `user_answer` → LLM-draft using profile + opportunity + question prompt
     - `computed` → derive (today's date, etc.)
   - Handles multi-step forms, file uploads, captcha presence detection (gracefully fall back to handoff when captcha)
   - Gate 2 splits into "approve and submit" vs "approve and handoff"
   - Records the real submission outcome in `Application` row (or a new `ExternalApplication` model)

2. **Crawl4AI → direct Playwright consolidation.** Already documented as future cleanup (see [`/memory/project_crawl4ai_consolidation.md`](../../../.claude/projects/-Users-koraysevil-Desktop-Senior-cs491-2/memory/project_crawl4ai_consolidation.md)). The auto-submit work IS the natural moment to do this — Playwright will be a committed dep, Crawl4AI's only remaining function is HTML→markdown which is ~30 lines of `html2text`. Specifically unblocks:
   - **Iframe-embedded ATS forms** (MongoDB→Greenhouse, others) — `page.frame_locator("#grnhse_iframe")` traverses cross-origin iframes natively. Crawl4AI cannot capture them in the rendering window (verified live with 30s `wait_for` timeout on the grnhse_iframe selector).
   - **Click-to-reveal modal forms** — `page.click('button:has-text("Apply")')` then re-extract.
   - **Multi-step forms** — `await page.click("Next")`, `await page.wait_for_selector(...)`.

3. **`expected_source` classification quality.** Live tests showed noise: Anthropic's `Country` got `unknown`, Notion's `Location` got `user_profile` despite being free-text. The LLM prompt needs few-shot examples to tighten before auto-fill is reliable.

4. **Hard-block ineligibility design.** Today the system blocks on `_check_job_eligibility` / `_check_program_eligibility` / `_check_scholarship_eligibility` for things that should be soft warnings the UI surfaces (location mismatch, age caps). v2 of the eligibility cleanup made compatibility issues warnings-not-blocks; the per-type checks haven't been retired yet.

### 7.2 Important but not blocking

5. **Polymorphic `StudentDocument` store.** v1 only supports CV from `StudentCV`. Cover Letter / SOP / Personal Statement are always *generated*, never read from a stored user file. Adding a generic `StudentDocument` table would let users save and reuse these.

6. **Gate 1 `additional_uploads` plumbing.** Serializer accepts `List[FileField]` but the adapter doesn't yet save those files or thread their extracted text into the tailoring `content` fields. Today gate 1 is "confirm defaults or set `tailoring_depth='none'`."

7. **Gate 2 "edit and re-tailor" loop.** Today rejection at gate 2 hard-cancels. A loop back to `application_tailoring` with per-document feedback as edit instructions would let the user iterate without starting fresh.

8. **Email handoff delivery.** Currently handoff is in-app only — the package sits in `application_package` and the user has to download it. A `send_handoff_email` would deliver to the student's inbox.

9. **LinkedIn ghost-posting upstream detection.** A meaningful share of `linkedin_jobs` rows are tracker-only postings (LinkedIn's apply button doesn't go anywhere external). Discovery correctly returns `failed` for these but burns Brave calls finding nothing real. Real fix is in `bitirme/linkedin_jobspy/scraper.py` — detect apply-flow type at ingestion and store as `apply_type` field.

10. **Negative caching for `failed` discoveries.** Short TTL (e.g. 7 days) would save Brave budget on repeat ghost-posting attempts. Risk: misses cases where the company posts to Greenhouse a few days later.

### 7.3 Operational gaps

11. **Per-session LLM cost budgeting.** Worst-case ~15-20 LLM calls per session (discovery + scrape eval + form fields + asset mapping + tailoring + evaluation + retries). No token cap today. Worth wiring before this scales.

12. **Cooperative cancel inside nodes.** Cancel marker is observed only at the next node boundary — the currently-executing node finishes (worst case ~30-60s of LLM cost wasted).

13. **GDPR / retention policy.** Tailored documents and scraped page content live forever in `ApplicationSession` rows + `PostgresSaver` checkpoints. Need a retention rule (e.g. 90 days post-completion).

14. **Browser fallback in production deployment.** With `UPPGRAD_BROWSER_SCRAPE_ENABLED=false`, success rate against React-driven ATSes (Workday, Ashby, modern Greenhouse SPAs) is ~zero. Backend repo PR #5 ships the Dockerfile change to install Chromium; needs merge + Railway env var.

15. **Captcha handling.** Real apply pages sometimes serve reCAPTCHA / hCaptcha challenges. Today we'd fail. For the auto-submit path, plan for graceful fallback: detect captcha presence in the Playwright session, abort auto-submit, fall back to handoff with a "captcha encountered" warning.

### 7.4 UI / UX gaps the engine assumes the frontend will provide

16. **Closed-posting messaging.** Engine sets `posting_closed=True` and a clear warning in `application_package.warnings`. Frontend should surface this prominently before the user opens the package.

17. **Form-field preview pre-submit.** Once auto-submit ships, the frontend should let the student preview every form field with the auto-populated value before clicking submit. Same gate 2 review surface, just expanded.

18. **Generated document review at gate 1.** Today gate 1 only reviews the *mapping* (which doc goes where). The generated content is reviewed at gate 2. Some users will want to see drafts earlier.

---

## 8. Open architecture decisions

- **Crawl4AI vs direct Playwright** — decided to consolidate to Playwright when auto-submit lands. Saved as memory [`project_crawl4ai_consolidation.md`](../../../.claude/projects/-Users-koraysevil-Desktop-Senior-cs491-2/memory/project_crawl4ai_consolidation.md).
- **`FormField` schema scope** — included `expected_source` classification today even though no node consumes it yet; auto-submit will. Avoids a schema migration later.
- **Iframe-embedded ATS detection** — code shipped (`extract_ats_iframe_src`, tier-3 follow), but rendering-window limitations mean it doesn't help against MongoDB-style late-injected iframes today. Becomes useful immediately when we move to Playwright.

---

## 9. Quick reference

- **Spec**: `docs/superpowers/specs/2026-04-26-auto-apply-backend-integration.md`
- **Plans**: `docs/superpowers/plans/2026-04-26-auto-apply-backend-integration.md`, `docs/superpowers/plans/2026-04-26-discovery-v2.md`
- **Live test scripts**: `scripts/e2e_discovery_test.py`, `scripts/e2e_scrape_test.py`
- **Required env vars** (backend): `OPENAI_API_KEY`, `BRAVE_SEARCH_API_KEY`, `UPPGRAD_SEARCH_PROVIDER=brave`, optional `UPPGRAD_BROWSER_SCRAPE_ENABLED=true`
- **Test counts**: 172 agentic + 27 backend ai_services = 199 unit tests (1 pre-existing unrelated backend failure on `test_post_documental_feedback`)
