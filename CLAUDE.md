# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project context

UppGrad is an AI-powered platform helping students find and apply to jobs,
graduate programs, and scholarships. Two agentic workflows: **document
feedback** (review uploaded CVs/SOPs/cover letters and propose structured
edits) and **auto-apply** (assess eligibility, generate tailored materials,
optionally drive a Playwright form-filler against the real ATS). Both use
human-in-the-loop approval before any consequential action.

This repo holds the LangGraph workflows. The DRF-backed Django service in
`UppGrad/backend/` is the API surface for the frontend; it imports and
drives these graphs.

---

## Commands

`uv` for env and dependency management. Browser fallback (Crawl4AI +
Playwright) ships as the optional `browser` extra; install with
`uv sync --extra browser` then `uv run playwright install chromium`.

```bash
uv sync                                       # install deps
uv sync --extra browser                       # + crawl4ai/playwright
uv run pytest tests/                          # run tests
uv run python -m uppgrad_agentic.workflows.document_feedback.run --file path/to/cv.pdf
uv run python -m uppgrad_agentic.workflows.auto_apply.run \
  --opportunity-type job --opportunity-id 127928
```

For end-to-end auto-apply against real Neon data, use
`scripts/e2e_auto_apply_harness.py` (SqliteSaver checkpointer, three
phases: `start` / `resume-gate-1` / `resume-gate-2`).

## LLM / search / browser configuration

All opt-in via env vars. Without them, nodes fall back to heuristics or
emit empty results — never raise.

| Variable | Description | Default |
|---|---|---|
| `UPPGRAD_LLM_PROVIDER` | `openai` (only one wired) | _heuristic_ |
| `UPPGRAD_OPENAI_MODEL` | OpenAI model | `gpt-5.4-mini` |
| `OPENAI_API_KEY` | required when provider is openai | _(none)_ |
| `UPPGRAD_SEARCH_PROVIDER` | `brave` (only one wired) | _url_direct only_ |
| `BRAVE_SEARCH_API_KEY` | required when provider is brave | _(none)_ |
| `UPPGRAD_BROWSER_SCRAPE_ENABLED` | `true` enables Crawl4AI/Playwright fallback | `false` |

---

## Architecture

### Package layout

```
src/uppgrad_agentic/
  common/          # llm.py (provider factory), logging, guardrails, errors
  config/          # stub
  tools/
    documents.py        # PDF/DOCX/TXT extraction (no OCR — backend has fallback)
    search.py           # Brave search provider, opt-in factory
    web_fetcher.py      # httpx-first + Crawl4AI/Playwright fallback
    url_discovery.py    # 3-tier apply-URL discovery + verification
    ats_form_urls.py    # per-ATS rules: overview URL → application URL
    form_extractor.py   # form area extraction + ATS-iframe-follow
    playwright_filler.py  # Tier 1-4 form-fill driver (no submit)
    value_planner.py    # plan FormField → fill value from profile / tailored / answers
    profile_lookup.py   # canonical profile-attr lookup table
    canonical_doc_types.py  # label phrase → canonical document type
    opportunities.py
  workflows/
    document_feedback/
      state.py     schemas.py     graph.py     run.py     nodes/     prompts.py (unused)
    auto_apply/
      state.py     schemas.py     graph.py     run.py
      _profile.py  # resolve_profile() — prefers injected profile_snapshot
      control.py   # cancel_session() — out-of-band cancel via update_state
      nodes/
```

### General patterns

- **State pattern**: each node receives the full State TypedDict and
  returns a partial dict to merge. Errors set `result.status="error"`;
  downstream nodes short-circuit with `return {}`.
- **LLM pattern**: `common.llm.get_llm()` returns `None` when no provider.
  Every node calling an LLM must handle `None` with a heuristic fallback.
  Reference: `nodes/detect_doc_type.py`.
- **Search/fetcher pattern**: same opt-in factory shape. Callers degrade
  to a safe result on no-provider; never raise.
- **Prompt pattern**: prompts live inline in each node file, not in
  `prompts.py` (which is unused — ignore).
- **Human-in-the-loop**: any consequential action passes through a
  `human_gate_*` node that uses LangGraph `interrupt()`.
- **Backend-first state injection**: nodes that previously owned stub
  data (`fetch_profile_snapshot`, `get_opportunity_context`,
  `load_opportunity`, `eligibility_and_readiness`, `asset_mapping`,
  `application_tailoring`) check for pre-injected state and use it when
  present. CLI/local-dev paths log a WARNING when falling back to stubs.
- **Checkpointer**: `build_graph(checkpointer=...)` accepts an injected
  saver. Production callers (Django adapter) pass `PostgresSaver`. CLI
  defaults to `MemorySaver`. The harness uses `SqliteSaver` so resumes
  span process restarts.

---

## Auto-Apply Workflow

### Opportunity DB tables

Three tables; pick by `opportunity_type` from frontend.

**linkedin_jobs** (jobs) — id, title, company, location, description,
job_type, job_level, job_function, company_industry, is_remote,
is_closed, url (LinkedIn), url_direct (real apply URL when present),
site (`"linkedin"` external / `"manuel"` internal), `employer_id`
(`NULL` external, `1` internal — *the* internal/external signal),
posted_time, salary.

**programs** (masters / phd) — id, url, title, university, location,
duration, degree_type, study_mode, program_type, tuition_fee, venue,
**`data` (jsonb)** primary source: description, requirements
(academic/english/other), curriculum, funding, living_costs, start_dates.

**scholarships** — id, url, title, provider_name, disciplines,
grant_display, location, deadline, scholarship_type, coverage,
description, benefits, eligibility_text, req_disciplines, req_locations,
req_nationality, req_age, req_study_experience, application_info,
**`data` (jsonb)** structured form of all the above.

### Pipeline

```
START → load_opportunity
  if employer_id == 1 (internal job)         → determine_requirements (CV + Cover Letter only)
  elif opportunity_type == job               → discover_apply_url → scrape_application_page
                                              → evaluate_scrape → extract_form_fields
                                              → determine_requirements
  else (masters / phd / scholarship)         → determine_requirements (parse from data jsonb)
→ eligibility_and_readiness
   ineligible / past-deadline                → end_with_explanation → END
   ready                                     → asset_mapping
→ asset_mapping (build categorised RequirementItem list from form_fields or normalized_requirements)
→ human_gate_1 (interrupt; user picks per-item choice + misc_strategy)
→ application_tailoring (per-item: upload→PreA→T1→LA→T2; auto_generate→single-pass; text→single-pass)
→ application_evaluation (informational warnings only — no retry loop)
→ human_gate_2 (interrupt; user approves and may set attempt_auto_submit=True)
   approved=False                            → end_with_error → END
   employer_id == 1                          → submit_internal (intent only, backend writes Application row)
   else                                      → package_and_handoff
→ record_application → END
```

Discovery is job-only and short-circuits internally for internal jobs and
non-job types. Cache hits (cross-user, 14-day staleness) skip the search.
Verified content from discovery propagates via state and is NOT re-fetched
during scrape. Form-field extraction short-circuits internally on no
form URL or LLM unavailable. Never block the workflow on a failed scrape;
fall back to defaults.

### Default requirements (when scrape fails / non-jobs)

| Type | Defaults |
|---|---|
| job | CV, Cover Letter |
| masters / phd | CV, SOP |
| scholarship | CV, Cover Letter |

### Discovery v2 (jobs only)

`tools/url_discovery.py` orchestrates. `tools/search.py` (Brave) +
`tools/web_fetcher.py` (httpx-first, Crawl4AI/Playwright fallback when
httpx is thin AND `UPPGRAD_BROWSER_SCRAPE_ENABLED=true`).

**3 tiers** (in order):
- **Tier 1 ATS**: `"<title>" "<company>" (site:greenhouse.io OR site:lever.co OR site:ashbyhq.com OR site:workable.com OR site:smartrecruiters.com)`
- **Tier 2 careers**: `"<title>" site:<company-domain>` — skipped when
  `company_url` resolves to a blocklisted domain (linkedin.com,
  indeed.com, glassdoor.com, social/aggregator sites).
- **Tier 3 generic**: `"<title>" "<company>" apply` (no site filter).

**Verification gates**: title fuzzy match ≥85 (hard prereq).
Corroborators drawn from {company-in-text, location-match,
posted-time-match, description-keyword-overlap}. ATS / generic tiers need
≥2 corroborators; careers tier needs 1 (domain itself proves company).
Confidence scales with extra corroborators. Thin pages (4xx, captcha,
JS shell) are rejected at the fetcher gate before scoring.

**Per-ATS form-URL rules** (`tools/ats_form_urls.py`): Ashby appends
`/application`, Lever / SmartRecruiters append `/apply`, Greenhouse /
Workable / company-direct keep the same URL, Workday returns `None`
(auth wall — form unreachable via simple navigation).

**Discovery method flags** (`DiscoveryResult.method` / `discovery_method`):
- `url_direct` — `linkedin_jobs.url_direct` was populated; no search.
- `ats` — Tier 1 hit, verified.
- `careers` — Tier 2 hit, verified.
- `generic` — Tier 3 hit, verified.
- `closed` — verified hit but says "no longer accepting applications";
  surfaced as a warning in handoff (`posting_closed=true`).
- `failed` — no tier produced a verified hit.
- `skipped_internal` — `employer_id == 1`; no external apply URL needed.

### Form-field extraction (jobs only)

Runs after `evaluate_scrape`, before `determine_requirements`. Captures
every input on the rendered form so a future auto-submit step has a
complete map. Output surfaced today in `application_package.form_fields`.

**3-tier strategy** in `extract_form_fields`:
1. In-state HTML from discovery (httpx-cached).
2. Forced browser fetch via `web_fetcher.force_browser_fetch` (bypasses
   thin gate; used when in-state HTML has 0 inputs because form area is
   client-side-hydrated — Greenhouse, Anthropic careers, Ashby SPAs).
   Crawl4AI passes `wait_for: 'js:() => document.body.innerText.length
   > 1000'` so extraction defers until React hydration completes.
3. ATS-iframe-follow via `form_extractor.extract_ats_iframe_src` — for
   pages like `mongodb.com/careers/<id>` that embed a third-party
   Greenhouse/Lever form via cross-origin iframe.

**FormSchema → FormField**: structured LLM output. Fields per record:
`label`, `field_type`, `name` (DOM name attr), `required`, `options`,
`accepts_file`, `expected_source`, `canonical_document_type` (file fields).

**FormFieldType flags** (`schemas.FormFieldType`): `file`, `text`,
`textarea`, `select`, `checkbox`, `radio`, `number`, `email`, `url`,
`date`, `tel`.

**FormFieldValueSource flags** (`schemas.FormFieldValueSource`):
- `user_profile` — name, email, phone, country, LinkedIn URL, GitHub URL,
  location, work auth.
- `user_document` — file inputs whose label matches a document type.
  Drawn from `tailored_documents` or stored user files at fill time.
- `user_answer` — free-form question textareas / screening MCQ. LLM-drafted.
- `computed` — system-derivable (today's date, etc.).
- `unknown` — surfaced for manual entry.

**ScrapeStatus flags** (`scraped_requirements.status`): `full`,
`partial`, `failed`.

### Eligibility

Hard-blocks ONLY for: deadline passed, missing user-supplied
(non-generatable) documents (Transcript, English Proficiency Test,
Portfolio, Certificate, Passport, Birth Certificate). Compatibility
issues (location, age cap, degree-level, discipline, nationality) are
non-blocking warnings stored in `state['compatibility_warnings']` and
surfaced via `application_package.warnings`. `_GENERATABLE`
(CV, Cover Letter, Motivation Letter, SOP, Personal Statement, Research
Proposal, Writing Sample, References) and `_USER_SUPPLIED` sets live in
`nodes/asset_mapping.py` and gate canonical-type classification +
auto_generate visibility.

Decision: `ready | ineligible`. The earlier gate-0 / pending decision was
removed; pre-flight profile completeness is the backend's responsibility.

### Asset mapping → RequirementItem

`asset_mapping` produces `state['requirement_items']` (and reuses the
`asset_mapping` JSONB column for stability — only the dict shape changed).

`RequirementItem` (in `schemas.py`):
- `id` — stable index used by gate-1 resume payload.
- `category` — `document | text | misc`.
- `label`, `description`.
- `field_type` — FormFieldType when derived from form_fields.
- `required` — true when form field is required or requirement hard-blocks.
- `document_type` — canonical doc type for `category=document`.
- `question` — for `category=text`, the FormField label (drives generation).
- `form_field_index` — back-pointer into `state['form_fields']`.

Build path:
- Jobs with non-empty form_fields → group `field_type='file'` into
  documents (deduped on canonical_document_type, keep required), `textarea`
  + `text/user_answer` into texts, everything else into one collapsed
  misc line.
- Everything else (non-jobs, form-failed jobs) → document-only items
  from `normalized_requirements`.
- Floor: empty inputs → per-type defaults from `_DEFAULTS`.

### Gate 1 — interrupt + resume contract

Interrupt payload:

```python
{"requirement_items": [...], "opportunity_type": "...", "opportunity_title": "..."}
```

Resume payload:

```python
{
  "requirements": {
    "<id>": {
      "choice": "upload" | "auto_generate" | "ignore_for_now" | "skip",
      "uploaded_text": "<extracted text>" | null,
      "user_prompt": "<≤200 chars>" | null  # documents only
    },
  },
  "misc_strategy": "auto_fill" | "ignore",
}
```

Validation rules (table-driven in `human_gate_1._validate_resume`):
- `skip` is rejected for any required item (document or text).
- `auto_generate` is rejected for `_USER_SUPPLIED` document types.
- `upload` requires non-empty `uploaded_text`.
- `user_prompt` length ≤200.
- `ignore_for_now` is permitted on required items (lets users defer
  optional uploads without blocking the graph).

On invalid resume the node returns no state changes and re-interrupts;
the backend serializer turns this into a 400. Computes
`auto_submit_feasible_at_gate_1` from the choices and stashes on state.

### Tailoring branches

`application_tailoring` consumes `human_review_1.requirements`:

| category | choice | path |
|---|---|---|
| document | `upload` | `PreA → T1 → LA → T2` (always 2-pass; no short-circuit on `ready_for_polish`) |
| document | `auto_generate` | single LLM call; output capped per-doc-type (CV 8000, CL 3000, SOP/PS 6000, default 5000) |
| text | `auto_generate` | single LLM call; capped at 1500 chars; written to `tailored_answers[str(form_field_index)]` |
| any | `ignore_for_now` / `skip` | nothing produced |

Per-doc caps via `_truncate_to_cap` prefer the last `\n\n` boundary in
the upper half of the cap; otherwise hard-cut.

`application_evaluation` runs informational checks (length, placeholder
text, keyword coverage) across `tailored_documents` and `tailored_answers`.
Output is `evaluation_result.warnings: List[str]` — surfaced at gate 2.
No retry loop. The generation prompts forbid `[Date]`-style placeholders
and refuse to fabricate compensation figures.

### Gate 2 — interrupt + resume contract

Interrupt payload includes:
- `tailored_documents` previews (first 400 chars per doc + metadata).
- `tailored_answers` previews.
- `evaluation_warnings`.
- `posting_closed`.
- `auto_submit_feasible` (recomputed: every required item has either
  non-empty tailored content for the chosen path, or skipped misc).
- `opportunity_title`, `opportunity_type`.

Resume payload: `{approved, attempt_auto_submit, feedback}`.
`attempt_auto_submit=true` is recorded on state regardless of feasibility
(intent only — auto-submit itself is driven from the backend after the
graph terminates; see below).

Routing: rejection → `end_with_error`. Approval: `employer_id == 1` →
`submit_internal` (intent only — backend writes the Application row);
else → `package_and_handoff`. Both → `record_application` → END.

### Auto-submit (Playwright form-filler)

Lives in `tools/playwright_filler.py` (524 lines). **Pure helper — no
LangGraph imports.** Driven from the backend's `auto_apply_adapter
.attempt_auto_fill` after the graph terminates with
`attempt_auto_submit=True` and `submission_type='handoff'`.

**Contract**:
- Never clicks submit/apply/send buttons. `submit_clicked` is always
  false; `_SUBMIT_TEXT_DENYLIST` is defense-in-depth against an
  LLM-picker that points at one anyway.
- `dry_run` is informational today; reserved for a future signed-off
  submission feature.
- Returns a `FormFillResult` populated entirely from observed Playwright
  outcomes (never optimistic).

**Tier strategy** per `FormFieldFillPlan`:
- **Tier 1** (deterministic, free): `[name="X"]` OR `[id="X"]` —
  Greenhouse uses id, Ashby uses name. Type-specific action: fill /
  select_option / set_input_files / check.
- **Tier 2** (deterministic, free): `get_by_label(label)`.
- **Tier 3** (deterministic, free): React custom dropdowns — click
  trigger then click matching option by visible text; falls back to
  type-and-Enter.
- **Tier 4** (LLM, ~$0.001-0.005 per call): gpt-4o-mini returns a
  selector + action; validated unique + not on the submit denylist.
  Bounded by `llm_picker_budget` (default 10 calls).

`tools/value_planner.py` plans the `FormFieldFillPlan` for each
`FormField` from profile / tailored documents / tailored answers.
`tools/profile_lookup.py` is the canonical attr lookup table.

**Backend wiring** (`auto_apply_adapter.attempt_auto_fill`):
- Triggered post-gate-2 on the backend, not from the graph.
- Persists per-field outcomes to
  `ApplicationSession.auto_fill_result` (JSONB column added in
  `0011_application_session_auto_fill_result.py`).
- The local E2E harness lives at `backend/scripts/e2e_auto_apply_local.py`.

### State schema (key fields)

Full TypedDict in `workflows/auto_apply/state.py`. Key entries:

- `opportunity_type`, `opportunity_id`, `opportunity_data`
- `profile_snapshot` (injected by backend adapter)
- `discovered_apply_url`, `discovery_method`, `discovery_confidence`
- `discovered_page_content` (markdown for browser path / HTML for
  httpx — feeds prose extraction)
- `discovered_raw_html` (always actual HTML — feeds form extraction;
  separate because the browser path's text is markdown but form
  extraction needs real DOM tags)
- `discovered_form_url` (per-ATS-resolved; equal to apply URL on
  same-URL ATSes; differs for split-URL Ashby/Lever/SmartRecruiters;
  `None` for Workday)
- `posting_closed: bool`
- `scraped_requirements` (rewritten by `evaluate_scrape`; raw_html lives
  at the top-level `discovered_raw_html` instead, since evaluate rewrites
  this dict)
- `form_fields: List[FormField dicts]`
- `compatibility_warnings: List[str]`
- `eligibility_result`, `asset_mapping`, `requirement_items`
- `human_review_1`, `auto_submit_feasible_at_gate_1`
- `tailored_documents` (doc_type → content + metadata),
  `tailored_answers` (str(form_field_index) → content + metadata)
- `evaluation_result.warnings: List[str]`
- `human_review_2`
- `application_package`, `application_record`
- `current_step`, `step_history` (`Annotated[List[str], operator.add]`
  — concurrent writes from parallel-fanout nodes are safe)
- `result: WorkflowResult`

---

## Document Feedback Workflow

Fully implemented across phases 0-6:
- **Phase 0**: load_document, detect_doc_type, end_with_error.
- **Phase 1**: context assembly (fetch_profile_snapshot,
  extract_doc_sections, parse_user_instructions, get_opportunity_context,
  build_context_pack).
- **Phase 2**: parallel analysis fan-out — analyze_structure,
  analyze_style, analyze_content_gaps, analyze_ats,
  analyze_opportunity_alignment, analyze_rhetoric, analyze_narrative
  (LangGraph `Send` from build_context_pack). The 7 parallel nodes write
  only `step_history` (concurrent `current_step` writes would conflict);
  `build_context_pack` sets `current_step="parallel_analysis"`.
  analyze_ats is CV-only; analyze_rhetoric and analyze_narrative are
  SOP/CL-only — all still write step_history so the frontend shape is
  doc-type-agnostic. analyze_rhetoric is per-paragraph; analyze_narrative
  is whole-document (anchor reuse, paragraph progression, closing audit).
- **Phase 3**: synthesize_feedback with grounding validation (drops
  proposals whose `before_text` cannot be fuzzy-matched to the document).
  Doc-type branch: CV → `_SYSTEM_CV` (sentence-level polish); SOP/CL →
  `_SYSTEM_SUBSTANCE` (paragraph rewrites driven by rhetoric findings,
  narrative-driven delete/merge proposals + closing rewrite, polish
  capped at ~30%).
- **Phase 4**: evaluate_output retry loop, capped at
  `MAX_EVAL_ITERATIONS=2`. SOP/CL adds three blocking deterministic
  audits: substance (coverage / preservation / polish-mix), narrative
  (uncovered deletions / repeated-anchor diversity / closing
  commitment), and AI-tell density (0 em-dashes; banned-phrase budget
  ≤1 across all after_text values combined).
- **Phase 5**: human_gate using `interrupt()` with a frontend-friendly
  resume payload.
- **Phase 6**: finalize generates LaTeX via LLM and compiles via tectonic.
  Doc-type branch: CV → resume template (`\resumeItem*` helpers); SOP/CL →
  prose template (article + parskip, no list helpers).
  `_strip_resume_commands_for_prose` defensively unwraps stray list
  commands on the prose path; `_normalize_ai_tells` (SOP/CL only)
  deterministically strips em-dashes (→ comma) and a curated banned-
  phrase list as belt-and-suspenders against synth/finalize regressions.

Schemas in `workflows/document_feedback/schemas.py`:
`DocTypeClassification`, `ChangeProposal` (with `action: rewrite | delete
| merge`), `EvaluationResult`, `NarrativeAnalysis`.

### Doc-type contracts (don't break these without touching both producer + consumer)

- SOP/CL `preserve_sentences` (analyze_rhetoric per-paragraph finding)
  MUST appear verbatim in any proposal's after_text that targets the same
  paragraph, or the evaluator drops the proposal. Paraphrasing is a
  violation. `rewrite_strategy` (augment / restructure / replace) signals
  how aggressive a rewrite is allowed.
- SOP/CL `opportunity_context` menu fields (`mission`, `products`,
  `values`, `distinctive_responsibilities`, `recent_signals`) are a
  *menu, not a checklist* — synth uses ≤1 signal per rewritten paragraph,
  never reuses one across paragraphs, and never invents signals not
  present in the menu.
- SOP/CL anchor diversity: each named anchor (project / internship)
  may serve as the focus of at most ONE paragraph. Every entry in
  `narrative.repeated_anchors` MUST be resolved by refocusing or
  deleting all-but-one of the listed paragraphs. Every entry in
  `narrative.paragraphs_to_delete` MUST get an `action="delete"`
  proposal; closings flagged `conclusion_commits_forward=false` MUST
  get a rewrite that names the org + a concrete contribution. The
  evaluator blocks on all three.
- SOP/CL AI-tell rules: no em-dashes (`—`) anywhere in any after_text;
  banned-phrase budget ≤1 total across all after_text values combined.
  Phrases live in `_BANNED_PHRASES` (evaluator) and `_BANNED_PHRASE_REWRITES`
  (finalize normalize pass) — keep them in sync. CV path is exempt
  (em-dashes legitimate in date ranges).
- SOP/CL `differentiators` (rhetoric per-paragraph finding) MUST appear
  VERBATIM in any rewrite of that paragraph. Stricter than
  `preserve_sentences` (which allows paraphrase). For `action="delete"`
  proposals, every differentiator MUST survive in the projected post-
  application document — either via an unchanged paragraph or via a
  sibling rewrite that injects it. The evaluator's `_check_distinctiveness`
  blocks both cases.
- SOP/CL `candidate_voice_signals` (narrative whole-doc finding) — ≥60%
  of these short verbatim phrases must survive (substring match) in the
  projected post-application document. Protects against the "smoothed
  into generic prose" failure mode where rewrites strip the candidate's
  memorable specifics.
- SOP/CL `posting_phrases` (opportunity_alignment finding) — no
  `after_text` may contain any of these verbatim. Prevents the document
  reading like the candidate is parroting the JD back. Paraphrase
  required.
- `ChangeProposal.action="delete"` carries empty `after_text`; consumers
  (frontend `ProposalReviewPane`, backend serializer, LaTeX prose
  prompt) all handle this. CV synth never emits `delete`/`merge` so
  backwards compat is preserved.
- CV: Summary and Skills-categorisation are CONTEXTUAL, not default —
  Summary only for 5+ years / career-changer / explicit user request;
  categorise Skills only when the list has 12+ entries. Default to NOT
  recommending either.
- CV `well_constructed_bullets` (analyze_content_gaps) are past-tense
  action + numeric outcome OR named tech; the synthesizer must NOT
  propose rewrites of these — only ATS-keyword-synonym injection is
  permitted.
- CV `cv_antipatterns` (References-on-request, generic Hobbies, "CV"
  title, first-person Experience bullets, photo, DOB/marital status) are
  emitted as one removal proposal each; PII removals get
  `requires_confirmation=true` because visa context can justify keeping
  them.

---

## Backend integration (driven from `UppGrad/backend/ai_services/`)

The backend is the API surface for the frontend. Key modules:

- `graph_adapter.py` — document-feedback adapter. ORM→state converters,
  background-thread runners, stale-state cleanup, `PostgresSaver` setup
  (one-time `setup()` over a direct/non-pooler URL with autocommit
  because `CREATE INDEX CONCURRENTLY` cannot run in a transaction).
  OCR fallback layers pdf2image + pytesseract on top of normal
  extraction.
- `auto_apply_adapter.py` — auto-apply adapter. Snapshot builders for
  all four opportunity types, `start_session`, two per-gate resume
  handlers (`resume_session_gate_1`, `resume_session_gate_2`),
  `finalize_internal_submission` (writes the `Application` row via ORM
  after `submission_type='internal'`), `attempt_auto_fill` (post-gate-2
  Playwright driver), `cancel_session` (calls into agentic
  `control.cancel_session`). Discovery cache: pre-invoke
  `_lookup_discovery_cache` (14-day staleness) + post-invoke
  `_persist_discovery_to_cache`. Per-status TTL janitor:
  PROCESSING/15m, FINALIZING/30m, AWAITING_*/7 days. Result-code
  mapping: `INELIGIBLE` → INELIGIBLE; `CANCELLED` → CANCELLED;
  `result.status='error'` → ERROR.
- `models.py` — `FeedbackSession`, `ApplicationSession` (one active per
  `(student, opportunity_type, opportunity_id)` via partial unique
  constraint; carries `requirement_items`, `tailored_answers`,
  `auto_fill_result`), `JobApplyUrlDiscovery` (cross-user apply-URL cache).
- `views.py` + `urls.py` — DRF mounted at
  `/api/ai/feedback-sessions/...` and
  `/api/ai/application-sessions/...`. Per-gate resume views look up
  handlers via `globals()[handler_name]` so test patches take effect.
- `document_renderer.py` — ReportLab CV → PDF for internal-submit
  `Application.resume_file`.

Internal-job auto-submit writes a real `Application` row from the
backend (no HTTP loopback to the agentic repo).

---

## Open / out-of-scope

- **Polymorphic `StudentDocument` store** — v1 only stores CV
  (`StudentCV`). Cover Letter / SOP / Personal Statement are always
  generated, never read from a stored user file.
- **Gate 2 "edit and re-tailor" loop** — v1 hard-cancels on rejection.
- **Submit-stage clicker for auto-submit** — `playwright_filler` fills
  but never clicks submit; needs a separate signed-off path.
- **Crawl4AI → direct Playwright consolidation** — Crawl4AI's value
  today is HTML→markdown + JS hydration wait; both are ~30 lines of
  native Playwright. Consolidating unblocks late-injected iframe
  capture (mongodb.com → Greenhouse) where Crawl4AI's rendering window
  cannot follow.
- **`expected_source` classification quality** — live tests showed
  noise (Anthropic Country=`unknown`, Notion Location=`user_profile`
  for free-text). The `extract_form_fields` system prompt needs
  few-shot examples before auto-fill is reliable across ATSes.
- **LinkedIn ghost-posting detection upstream** — many `linkedin_jobs`
  rows are tracker-only (apply button bookmarks instead of going
  external). Discovery correctly returns `failed` but burns Brave calls.
  Real fix is in the LinkedIn scraper at ingestion time.
- **Negative caching for `failed` discoveries** — would save Brave
  budget on repeat ghost-posting attempts. Short TTL (~7 days).
- **Per-session LLM cost budgeting** — worst case ~15-20 LLM calls per
  session; no token cap today.
- **Cooperative cancel inside nodes** — cancel marker is observed at
  the next node boundary; the currently-executing node finishes
  (worst case ~30-60s of wasted LLM cost).
- **Email handoff delivery** — handoff is in-app only.
- **GDPR / retention policy** — tailored documents and scraped content
  live indefinitely in `ApplicationSession` rows + PostgresSaver
  checkpoints.

### Resume-value contract quirk (not a bug)

LangGraph re-interrupts when `Command(resume=<falsy>)` is passed (empty
dict, None). Backend always sends a non-empty dict for both gates.
