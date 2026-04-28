# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

UppGrad is an AI-powered platform helping students find and apply to jobs, 
graduate programs, and scholarships. It has two core agentic workflows: 
document feedback and auto-apply. Document feedback analyzes uploaded CVs, 
SOPs, and cover letters and proposes structured reviewable edits. Auto-apply 
assesses eligibility, generates tailored application materials, and attempts 
submission. Both workflows use human-in-the-loop approval before any 
consequential action.

## Commands

This project uses [uv](https://github.com/astral-sh/uv) for environment and dependency management.

```bash
# Install dependencies
uv sync

# Run the document feedback workflow against a file
uv run python -m uppgrad_agentic.workflows.document_feedback.run --file path/to/cv.pdf --instructions "Focus on clarity"

# Run tests (currently stubs; add pytest once tests are written)
uv run pytest src/

# Install package in editable mode (if needed outside uv)
pip install -e .
```

## LLM Configuration

The LLM is opt-in via environment variables. Without them, nodes fall back to heuristics.

| Variable | Description | Default |
|---|---|---|
| `UPPGRAD_LLM_PROVIDER` | `openai` (only supported provider currently) | _(none — heuristic mode)_ |
| `UPPGRAD_OPENAI_MODEL` | OpenAI model name | `gpt-5.4-mini` |
| `OPENAI_API_KEY` | Required when provider is `openai` | _(none)_ |
| `UPPGRAD_SEARCH_PROVIDER` | `brave` (only Brave Search supported currently) | _(none — discovery uses url_direct only)_ |
| `BRAVE_SEARCH_API_KEY` | Required when search provider is `brave` | _(none)_ |
| `UPPGRAD_BROWSER_SCRAPE_ENABLED` | `true` to enable Playwright/Crawl4AI fallback for thin httpx fetches | `false` |

## Architecture

### Package layout

```
src/uppgrad_agentic/
  common/          # Shared utilities: LLM factory (llm.py), logging, guardrails, error types
  config/          # Settings (currently stub)
  tools/           # File-level tools: documents.py (PDF/DOCX/TXT extraction),
                   #                  search.py (Brave search), web_fetcher.py
                   #                  (httpx-first + Playwright/Crawl4AI fallback),
                   #                  url_discovery.py (3-tier apply-URL discovery),
                   #                  opportunities.py
  workflows/
    document_feedback/   # Fully implemented end-to-end
      state.py     # DocFeedbackState TypedDict — the single source of truth for graph state
      schemas.py   # Pydantic models used for LLM structured output (DocTypeClassification)
      graph.py     # build_graph() — assembles and compiles the LangGraph StateGraph
      run.py       # CLI entry point: python -m uppgrad_agentic.workflows.document_feedback.run
      nodes/       # One file per node function
      prompts.py   # System/human prompt strings (unused — see Prompt pattern below)
      tests/       # Smoke test + unit test stubs (currently empty)
    auto_apply/          # Fully implemented end-to-end
      state.py     # AutoApplyState TypedDict
      schemas.py   # Pydantic models (NormalizedRequirement, AssetMap, EligibilityResult, etc.)
      graph.py     # build_graph() — assembles and compiles the LangGraph StateGraph
      run.py       # CLI entry point: python -m uppgrad_agentic.workflows.auto_apply.run
      nodes/       # One file per node function
      _profile.py  # resolve_profile() — prefers injected profile_snapshot, falls back to stub
      control.py   # cancel_session() — out-of-band cancel via graph.update_state()
```

### General patterns

- **State pattern**: each node receives the full State TypedDict and returns a partial 
  dict of keys to merge. Errors are signalled by setting `result.status = "error"` 
  in state. Downstream nodes check this and short-circuit with `return {}`.
- **LLM pattern**: `get_llm()` in `common/llm.py` returns `None` when no provider is 
  configured. Every node that calls an LLM must handle the `None` case with a 
  heuristic fallback. See `detect_doc_type.py` for the reference implementation.
- **Search/fetcher pattern**: `tools/search.get_search_provider()` and 
  `tools/web_fetcher` follow the same opt-in factory pattern as `get_llm()`. 
  Callers must handle the no-provider case by returning a degraded result, never 
  by raising.
- **Prompt pattern**: prompts live inline inside each node file, not in prompts.py. 
  This keeps each prompt next to its logic. prompts.py is unused and can be ignored.
- **Human-in-the-loop**: any workflow that triggers external actions must include a 
  human_gate node using LangGraph `interrupt()` before the action.
- **Backend-first state injection**: nodes that historically owned stub data 
  (`fetch_profile_snapshot`, `get_opportunity_context`, `load_opportunity`, 
  `eligibility_and_readiness`, `asset_mapping`, `application_tailoring`) now check 
  for pre-injected state from the backend adapter and use it when present, falling 
  back to in-repo stubs for CLI / local-dev mode (with a WARNING log on the fallback 
  path).
- **Checkpointer**: `build_graph(checkpointer=...)` accepts an injected checkpointer. 
  Production callers (the Django adapter) pass `PostgresSaver`. CLI defaults to 
  `MemorySaver`.

### common/state.py
Intentionally reserved as a shared base state for future workflows. Currently empty. 
Do not delete or use for workflow-specific state.


### Adding a new workflow
1. Create `src/uppgrad_agentic/workflows/<name>/` mirroring the `document_feedback` layout.
2. Define a `State` TypedDict in `state.py`.
3. Put each node in `nodes/<node_name>.py` with signature `(state: YourState) -> dict`.
4. Wire the graph in `graph.py` with `build_graph() -> CompiledGraph`.
5. Add a `run.py` CLI entry point.

---


## Document Feedback Workflow

### Agent responsibilities 

**Intake & Classification Agent** (load_document.py, detect_doc_type.py)
Loads and extracts text from uploaded file, validates minimum length, 
classifies document as CV/SOP/COVER_LETTER, routes accordingly.

**Context Assembly Agent** (fetch_profile_snapshot.py, extract_doc_sections.py, 
parse_user_instructions.py, get_opportunity_context.py, build_context_pack.py)
Fetches user profile snapshot, extracts document sections, parses user 
instructions, optionally retrieves opportunity context, builds unified 
context pack for all downstream agents.

**Document Analysis Agent** (analyze_structure.py, analyze_style.py, 
analyze_content_gaps.py, analyze_ats.py, analyze_opportunity_alignment.py)
Runs parallel analyses: structure, style/grammar, content gaps, ATS 
compatibility (CV only), opportunity alignment (if context provided). 
All analyses are parameterized by doc_type.

**Synthesis & Planning Agent** (synthesize_feedback.py)
Merges parallel analysis outputs, prioritizes issues, generates structured 
ChangeProposal list with section, rationale, before/after text, confidence, 
and confirmation flag per proposal.

**Evaluation Agent** (evaluate_output.py)
Checks proposals for groundedness, hallucinations, and format compliance. 
Triggers refinement loop back to synthesis, capped at 2 iterations.

**Human Review Coordinator** (human_gate.py)
LangGraph interrupt point. Presents proposals to user as reviewable 
checklist, collects accept/reject decisions and comments, holds workflow 
until explicit approval.

**Rewrite Agent** (finalize.py)
Applies only approved edits, resolves conflicts between overlapping changes, 
preserves rejected segments, runs coherence smoothing pass, produces final 
rewritten document and diff.

### Orchestration

The full graph flow is:

START
→ load_document
→ detect_doc_type
→ [route by doc_type: cv / sop / cover_letter / error]
→ fetch_profile_snapshot
→ extract_doc_sections
→ parse_user_instructions
→ [conditional] get_opportunity_context (only if user provided opportunity)
→ build_context_pack
→ [parallel] analyze_structure, analyze_style, analyze_content_gaps
           + analyze_ats (CV only)
           + analyze_opportunity_alignment (only if opportunity context exists)
→ synthesize_feedback
→ evaluate_output
→ [loop back to synthesize_feedback if quality check fails, max 2 iterations]
→ human_gate (interrupt — wait for user approval)
→ finalize
→ END

Key orchestration rules:
- Every node checks for result.status == "error" at the top and returns {} to short-circuit
- Parallel analysis nodes fan out from build_context_pack and merge into synthesize_feedback
- The evaluation loop is capped at 2 retries via an iteration counter in state
- human_gate uses LangGraph interrupt() and resumes only after user submits approved changes
- All three doc types (CV/SOP/COVER_LETTER) share the same nodes after routing; 
  doc_type in state parameterizes behavior inside each node


---

## Auto-Apply Workflow

### Opportunity database tables

Three tables hold opportunities. The workflow determines which table to query
based on opportunity_type passed in by the frontend.

**linkedin_jobs** (jobs)
Key columns for the workflow:
- id, title, company, location, description, job_type, job_level, job_function
- company_industry, is_remote, is_closed
- url (linkedin application url), url_direct (company site url, not always present)
- site: "linkedin" for external, "manuel" for internal UppGrad jobs
- employer_id: NULL for external, 1 for internal (use this to determine internal vs external)
- posted_time, salary

**programs** (masters / phd)
Key columns for the workflow:
- id, url, title, university, location, duration, degree_type, study_mode
- program_type, tuition_fee, venue
- data (json): contains description, requirements (academic, english, other),
  curriculum, funding, living_costs, start_dates — primary source of eligibility info

**scholarships**
Key columns for the workflow:
- id, url, title, provider_name, disciplines, grant_display, location, deadline
- scholarship_type, coverage, description, benefits
- eligibility_text, req_disciplines, req_locations, req_nationality,
  req_age, req_study_experience
- application_info (contains how to apply instructions)
- data (json): contains all of the above in structured form — use as primary source


### Agent responsibilities

**Opportunity Intelligence Agent** (load_opportunity.py, discover_apply_url.py, 
scrape_application_page.py, evaluate_scrape.py, determine_requirements.py)
Loads the opportunity record from the correct DB table based on opportunity_type 
(short-circuits when the backend adapter has pre-loaded `opportunity_data`).
For jobs only: discovers the real apply URL via `discover_apply_url` (Brave 
search through ATS / company-careers / generic tiers, with multi-factor 
verification — see Discovery v2 below), then scrapes the application page 
using the verified content from discovery (no double-fetch). Evaluates scrape 
quality as full, partial, or failed and normalizes scraped content into a 
structured requirements list.
For programs and scholarships: skips scraping entirely and parses requirements 
directly from the data json field in the DB record.
If scraping fails or is partial, falls back to assumed default requirements based 
on opportunity type. Never blocks on a failed scrape.
Stores scrape_status and scrape_confidence in state so downstream agents and the 
user are aware of whether requirements are real or assumed.

**Applicant Eligibility and Readiness Agent** (eligibility_and_readiness.py)
Hard-blocks ONLY for: deadline passed, missing user-supplied (non-generatable) 
documents (Transcript, English Proficiency Test). Compatibility issues 
(location, age cap, degree-level, discipline, nationality) are non-blocking 
warnings stored in `state['compatibility_warnings']` and surfaced to the user 
through the application package. Documents the system can generate from CV + 
profile (Cover Letter, SOP, Personal Statement, Research Proposal, etc.) do 
not block — they flow through to asset_mapping with `tailoring_depth='generate'`.
Decision: ready | pending | ineligible | manual_review. If pending, triggers 
`human_gate_0`.

**Asset Mapping Agent** (asset_mapping.py)
Maps each normalized requirement to the best available user document.
Determines tailoring depth for each: none | light | deep | generate.
Flags requirements with no suitable source document.

**Application Tailoring Agent** (application_tailoring.py)
Reuses document feedback pipeline in apply-mode meaning changes are applied 
directly without a proposal review step at this stage.
Generates tailored version of each required document grounded in opportunity 
context and the normalized requirements from the Opportunity Intelligence Agent.
Parameterized by opportunity_type and tailoring depth from asset mapping.
Per-doc-type output caps via `_truncate_to_cap` (CV ≤ 8000, Cover Letter ≤ 3000, 
SOP / Personal Statement ≤ 6000, default 5000) prevent runaway LLM output from 
breaking PDF rendering and frontend display. Boundary-aware truncation prefers 
the last paragraph break that fits in the upper half of the cap; otherwise 
hard-cut.

**Application Evaluation Agent** (application_evaluation.py)
Verifies full package for groundedness, requirement coverage, and hallucinations 
(length, placeholder text, keyword coverage). Triggers refinement loop back to 
application_tailoring, capped at 2 iterations.

**Human Review Coordinator** (human_gate_0.py, human_gate_1.py, human_gate_2.py)
Gate 0: triggered when eligibility finds missing user-supplied profile fields 
or non-generatable documents. Real `interrupt()` cycle with iteration cap of 2 
(`PROFILE_INCOMPLETE_AFTER_RETRIES`); on resume, graph routes back to 
`eligibility_and_readiness` for a re-check.
Gate 1: after asset mapping, presents document mapping to user, collects 
document selections and any additional uploads from local device.
Gate 2: after evaluation, presents final tailored package for user approval 
before submission or handoff. No materials are submitted without passing this 
gate.
All three use LangGraph interrupt() and require a checkpointer 
(MemorySaver for CLI, PostgresSaver in production).

**Submission Agent** (submit_internal.py, package_and_handoff.py, 
record_application.py)
Routing inside `_route_after_gate2`: internal jobs (employer_id == 1) → 
submit_internal; everything else → package_and_handoff.
For internal jobs: `submit_internal` records submission *intent* only; the 
backend adapter writes the real `Application` row via Django ORM after the 
graph terminates with `submission_type='internal'` (no HTTP loopback from 
agentic → backend).
For all external opportunities (external jobs, masters, phd, scholarships): 
assembles the final tailored package and hands it off to the user.
Records the application outcome in both cases. For jobs, also stores 
scrape_status, scrape_confidence, and compatibility warnings in the 
application record.

### Orchestration

START
→ load_opportunity (receive opportunity_type + id from frontend, query correct table;
                    short-circuits when backend pre-loaded opportunity_data)
if opportunity_type == job:
→ discover_apply_url (3-tier Brave search → verified URL + content)
→ scrape_application_page (uses pre-fetched discovery content; falls back to fresh httpx)
→ evaluate_scrape (assess quality: full | partial | failed)
→ determine_requirements (full → use scraped, partial → merge with defaults, failed → defaults)
if opportunity_type == masters or phd:
→ skip scraping
→ determine_requirements (parse from data json; assume [CV, SOP] as baseline)
if opportunity_type == scholarship:
→ skip scraping
→ determine_requirements (parse from data json; assume [CV, Cover Letter] as baseline)
→ eligibility_and_readiness
→ ineligible: end_with_explanation → END
→ pending missing profile info: human_gate_0 → eligibility_and_readiness (re-check, capped at 2)
→ ready: continue
→ asset_mapping
→ human_gate_1 (user reviews document mapping, selects or uploads documents)
→ application_tailoring
→ application_evaluation
→ if failed and iteration_count < 2: loop back to application_tailoring
→ if passed or iteration_count == 2: continue
→ human_gate_2 (user reviews and approves final package)
→ route_by_source
if opportunity_type == job and employer_id == 1:
→ submit_internal → record_application → END
else:
→ package_and_handoff → record_application → END

Key orchestration rules:
- Discovery is only attempted for job opportunities (skipped for internal jobs and 
  non-job types). Cache hits (cross-user, 14-day staleness) skip the search.
- Verified content from discovery is propagated through state to scrape_application_page 
  to avoid double-fetching the same URL.
- Never block the workflow on a failed scrape, always degrade to assumed requirements
- Store scrape_status and scrape_confidence in state throughout so the user is 
  always informed whether requirements were scraped or assumed
- Internal vs external is determined by employer_id in linkedin_jobs, not site column
- human_gate_0 is conditional and only triggered when eligibility check finds 
  missing user-supplied documents or required profile fields
- For now internal submission only requires CV and Cover Letter fields
- All three opportunity types share the same pipeline after determine_requirements; 
  only the scraping step and the final routing differ

### Assumed default requirements by opportunity type

| Opportunity type | Default required documents |
|---|---|
| Job (external or internal) | CV, Cover Letter |
| Masters / PhD | CV, SOP |
| Scholarship | CV, Cover Letter |

### State schema (AutoApplyState)

Define in workflows/auto_apply/state.py:
- opportunity_type: job | masters | phd | scholarship
- opportunity_id: str
- opportunity_data: dict (raw DB record; pre-loaded by backend adapter in production)
- profile_snapshot: dict (injected by backend adapter; replaces _get_stub_profile lookups)
- discovered_apply_url, discovery_method, discovery_confidence: discovery output
- discovered_page_content, discovered_http_status: verified content propagated to scrape
- posting_closed: bool (true when discovery found a real listing that says the
  posting is closed; surfaced in handoff package)
- scraped_requirements: dict (status, requirements list, confidence, source)
- normalized_requirements: list (final requirements list after scrape or assumption)
- gate_0_iteration_count: int (caps the eligibility re-check loop after gate 0)
- compatibility_warnings: List[str] (non-blocking warnings)
- eligibility_result: dict (decision, reasons, missing_fields)
- asset_mapping: dict (requirement to document mapping with tailoring depth)
- human_review_0: dict (user response to missing profile info gate, if triggered)
- human_review_1: dict (user selections from gate 1)
- tailored_documents: dict (document type to tailored content)
- evaluation_result: dict
- iteration_count: int
- human_review_2: dict (user approval from gate 2)
- application_package: dict (final documents ready for handoff or submission)
- application_record: dict (logged outcome, includes scrape_status and scrape_confidence for jobs)
- current_step, step_history: frontend progress tracking
- result: dict (status, error_code, user_message)

### Future work — agentic-side roadmap

These are intentionally out of scope for the current implementation:

**Requirements review human gate**
After determine_requirements, surface the normalized requirements list to the 
user as a reviewable checklist before asset mapping begins. Similar to the 
ChangeProposal review in document feedback. User can confirm what they can 
provide, flag items they cannot provide right now (e.g. financial documents, 
transcripts), and upload additional documents on the spot. Anything flagged 
as unavailable is noted in the final package rather than blocking the workflow. 
This makes asset mapping cleaner since it works with a confirmed set of assets.

**External application form submission**
For external job opportunities where discovery succeeded with high confidence, 
attempt to automatically fill and submit the application form using browser 
automation (Playwright or equivalent). This requires handling multi-step forms, 
file upload fields, and graceful fallback when anti-bot mechanisms are 
encountered. Scrape confidence and scraped field structure are already stored 
in the application record to make this step easier to add later.

---

## Implementation Status

### Workflows — both fully implemented end-to-end

**Document Feedback Workflow** — all 17 nodes wired across phases 0–6, smoke 
tested across CV/SOP/Cover Letter, bug audit completed.
- Phase 0: load_document, detect_doc_type, end_with_error, doc-type routing.
- Phase 1: Context assembly (fetch_profile_snapshot, extract_doc_sections, 
  parse_user_instructions, get_opportunity_context, build_context_pack); 
  state.py and schemas.py extended with all Phase 1+ fields including 
  ChangeProposal/EvaluationResult schemas.
- Phase 2: Parallel analysis (analyze_structure, analyze_style, 
  analyze_content_gaps, analyze_ats, analyze_opportunity_alignment); LangGraph 
  Send fan-out from build_context_pack. analyze_ats consumes 
  opportunity_context keywords in both LLM and heuristic paths.
- Phase 3: synthesize_feedback with grounding validation (drops proposals 
  whose `before_text` cannot be fuzzy-matched to the document).
- Phase 4: evaluate_output evaluation loop, capped at MAX_EVAL_ITERATIONS=2.
- Phase 5: human_gate using `interrupt()` with frontend-friendly resume payload.
- Phase 6: finalize applies accepted proposals right-to-left, resolves overlapping 
  spans by confidence, runs an LLM coherence smoothing pass, produces a diff 
  summary and final document.
- Frontend progress tracking: `current_step` (Optional[str]) and `step_history` 
  (Annotated[List[str], operator.add]) on DocFeedbackState. Sequential nodes set 
  both fields; the 5 parallel analysis nodes set only `step_history` (concurrent 
  `current_step` writes would conflict). `build_context_pack` sets 
  `current_step="parallel_analysis"` on its successful return so the frontend 
  has a meaningful indicator during the fan-out.

**Auto-Apply Workflow** — all 15 nodes wired end-to-end, smoke tested across 
job/masters/phd/scholarship, routing verified at every conditional edge.
- Opportunity Intelligence: load_opportunity (short-circuits on pre-loaded 
  opportunity_data, falls back to in-repo stubs in CLI), discover_apply_url 
  (Brave + 3-tier search + multi-factor verification), 
  scrape_application_page (consumes pre-fetched discovery content; falls back 
  to fresh httpx fetch via the tiered fetcher), evaluate_scrape (LLM structured 
  output + heuristic fallback), determine_requirements (scrape → parse data 
  json → assumed defaults).
- Eligibility: eligibility_and_readiness (deadline + missing user-supplied 
  docs hard-block; compatibility issues are non-blocking warnings); 
  end_with_explanation; human_gate_0 (real `interrupt()` cycle with iteration 
  cap, routes back to eligibility_and_readiness for a re-check).
- Asset Mapping: asset_mapping (LLM structured output + heuristic depth 
  classification using `_GENERATABLE` / `_USER_SUPPLIED` sets); 
  AssetMap / AssetMappingOutput schemas; human_gate_1 (full interrupt/resume 
  with per-document override and upload support).
- Tailoring & Evaluation: application_tailoring (LLM apply-mode rewrite + 
  heuristic fallback per depth, with per-doc-type output caps); 
  application_evaluation (length, placeholder, keyword-coverage checks with 
  2-iteration retry loop).
- Human Review: human_gate_2 (full interrupt/resume, approve/reject with 
  per-document feedback).
- Submission: submit_internal records intent only (backend ORM writes the 
  Application row); package_and_handoff (assembles package with scrape 
  provenance, compatibility warnings, posting_closed flag for jobs); 
  record_application (logs outcome, timestamp, doc types, scrape metadata).
- graph.py fully wired end-to-end; run.py CLI entry point with `--thread-id`.
- Frontend progress tracking: `current_step` and `step_history` on 
  AutoApplyState; all 15 nodes set both fields. Auto-apply has no parallel 
  fan-out so no `parallel_analysis` sentinel is needed.
- Cancel support: `workflows/auto_apply/control.py:cancel_session(thread_id, 
  checkpointer)` injects a CANCELLED marker into the live checkpoint via 
  `graph.update_state()`. The next-node short-circuit pattern drains the graph 
  to END.

### Backend integration — driven from `backend/ai_services/`

Both agentic workflows are wired into the Django backend via dedicated adapter 
modules. The backend, not the agentic repo, is the API surface for the 
frontend. The agentic repo no longer needs a FastAPI layer of its own.

**Document feedback adapter** (`graph_adapter.py`):
- ORM-to-state converters: `build_profile_snapshot`, `build_opportunity_context` 
  (handles job / program / scholarship / event types).
- `start_session` / `resume_session` background-thread runners with stale-state 
  cleanup via `cleanup_stale_sessions` (flat 15-min TTL).
- `_build_checkpointed_graph` builds a `PostgresSaver`-backed graph; one-time 
  `setup()` uses a direct (non-pooler) Neon URL with autocommit because 
  `CREATE INDEX CONCURRENTLY` cannot run inside a transaction. Normal runtime 
  uses the pooler URL with autocommit (without autocommit, checkpoint writes 
  end up in an uncommitted transaction that rolls back on connection close, 
  and the resumed graph cannot find its state).
- OCR fallback (`_extract_text_with_ocr_fallback`) layers pdf2image + 
  pytesseract on top of normal text extraction; lazy-imports gracefully when 
  OCR libs are not installed.

**Auto-apply adapter** (`auto_apply_adapter.py`):
- Opportunity snapshot builder (`build_auto_apply_opportunity_snapshot`) 
  handles all four types via `JobOpportunity` / `ProgramOpportunity` / 
  `ScholarshipOpportunity` ORM lookups; returns None when the opportunity is 
  missing or closed (gates session creation upstream).
- Profile snapshot builder (`build_auto_apply_profile_snapshot`) reads the 
  `Student` / `StudentCV` ORM, extracts CV text via the document-feedback 
  adapter's OCR fallback, and produces the dict the agentic graph expects. 
  v1 cap: only CV is sourced from a stored user file. Cover Letter / SOP / 
  Personal Statement are always *generated* by application_tailoring 
  (`tailoring_depth='generate'`).
- `start_session` and three per-gate resume handlers 
  (`resume_session_gate_{0,1,2}`).
- Gate 0 resume rebuilds the profile snapshot post-update and injects it via 
  `graph.update_state(config, {"profile_snapshot": fresh})` before resuming, 
  so the eligibility re-check sees the new profile rather than the snapshot 
  frozen at start.
- Gate 2 resume freshness re-check: rebuilds the opportunity snapshot before 
  resuming and short-circuits to INELIGIBLE if the opportunity has just 
  closed during the wait.
- `_pending_node_after_invoke` reads `graph.get_state(config).next` to 
  determine which gate is held. `result_state['current_step']` cannot be 
  used because `interrupt()` suspends BEFORE the node returns its updates dict 
  — `current_step` reflects the last fully-completed node, not the gate.
- `finalize_internal_submission` writes a real `Application` row via Django 
  ORM (Spec A3 — no HTTP loopback to the agentic repo); the internal-submit 
  path renders the tailored CV to PDF using ReportLab in `document_renderer.py` 
  and saves it to `Application.resume_file`.
- `cancel_session` calls into 
  `uppgrad_agentic.workflows.auto_apply.control.cancel_session` to inject a 
  CANCELLED marker into the live PostgresSaver checkpoint.
- Per-status TTL janitor (`janitor.py`): PROCESSING/15m, FINALIZING/30m, 
  AWAITING_*/7 days. The flat 15-min cap from FeedbackSession is wrong for 
  ApplicationSession because human-gate states can legitimately last days.
- Discovery cache: pre-invoke `_lookup_discovery_cache` (gates on 14-day 
  staleness via `last_verified_at`) + post-invoke `_persist_discovery_to_cache` 
  (upserts via `update_or_create`, skips for `failed` and `skipped_internal`).
- Result-code mapping: `INELIGIBLE` and `PROFILE_INCOMPLETE_AFTER_RETRIES` 
  flip session to `INELIGIBLE`; `CANCELLED` flips to `CANCELLED`; everything 
  else with `result.status='error'` flips to `ERROR`.

**Database models** (`models.py`):
- `FeedbackSession` — document feedback state (status enum, proposals, 
  decisions, final document/PDF, error_message, thread_id).
- `ApplicationSession` — auto-apply state (status enum, per-gate response, 
  eligibility result, asset mapping, tailored documents, application package, 
  discovery fields, compatibility_warnings, application FK to 
  `jobs.Application`); partial unique constraint enforces one active session 
  per `(student, opportunity_type, opportunity_id)`.
- `JobApplyUrlDiscovery` — cross-user apply-URL cache (PK=job_id, fields: 
  discovered_url, discovery_method, discovery_confidence, discovered_at, 
  last_verified_at). Stale rows are not deleted; the next successful discovery 
  overwrites them.

**DRF views and URL routes** (`views.py`, `urls.py`):
- Document feedback: `FeedbackSessionListCreateView`, 
  `FeedbackSessionDetailView`, `FeedbackSessionReviewView`, 
  `FeedbackSessionCancelView` mounted at `/api/ai/feedback-sessions/...`.
- Auto-apply: `ApplicationSessionListCreateView`, 
  `ApplicationSessionDetailView`, `ApplicationSessionCancelView`, three 
  per-gate resume views mounted at `/api/ai/application-sessions/...`. The 
  `_GateResumeBaseView` looks up handlers via `globals()[handler_name]` so 
  test patches of the module-level handlers take effect.

### Discovery v2 — apply-URL discovery for jobs

Replaces the old "fetch the LinkedIn page and fail" path with a search-based 
discovery pipeline + a tiered fetcher. Lives entirely in `tools/`.

**Components**:
- `search.py` — `SearchProvider` ABC + `BraveSearchProvider`. Opt-in via 
  `UPPGRAD_SEARCH_PROVIDER=brave` + `BRAVE_SEARCH_API_KEY`. Mirrors the 
  `get_llm()` factory pattern.
- `web_fetcher.py` — Tiered fetcher: httpx-first (fast, cheap, no browser); 
  Playwright/Crawl4AI fallback when (a) httpx is thin AND (b) 
  `UPPGRAD_BROWSER_SCRAPE_ENABLED=true`. Crawl4AI is lazy-imported and no-ops 
  gracefully when not installed. `_detect_thin` flags 4xx, short bodies, and 
  ≥2 anti-bot keywords.
- `url_discovery.py` — Multi-factor verification + 3-tier orchestration.

**Verification gates** (informed by live testing — see Live verification below):
- Title fuzzy match ≥85 (hard prerequisite).
- Corroborator count drawn from {company-in-text, location-match, 
  posted-time-match, description-keyword-overlap}. ATS / generic tiers need 
  ≥2 corroborators; the careers tier needs 1 (the domain itself proves the 
  company).
- Confidence is scaled by extra corroborators beyond the minimum.
- Thin pages (captcha walls, 4xx, JS shells) are rejected at the fetcher gate 
  before scoring; never given a second-look at scrape time.

**3-tier search**:
- Tier 1 ATS: `"<title>" "<company>" (site:greenhouse.io OR site:lever.co OR ...)`.
- Tier 2 careers: `"<title>" site:<company-domain>` — skipped when 
  `company_url` resolves to a blocklisted domain (linkedin.com, indeed.com, 
  glassdoor.com, social/aggregator sites). LinkedIn-jobs `company_url` is 
  usually the LinkedIn company page URL, not a real careers site, so the 
  blocklist saves wasted Brave calls.
- Tier 3 generic: `"<title>" "<company>" apply` (no `site:` constraint).

**Single-fetch architecture**: `DiscoveryResult.text` carries verified content 
forward via `state['discovered_page_content']`; `scrape_application_page` 
consumes it instead of re-fetching the same URL. Halves the request count and 
reduces ban risk.

**Graph wiring**: `discover_apply_url` node sits between `load_opportunity` 
and `scrape_application_page` for jobs (skipped for internal jobs and non-job 
opportunity types). Cache-hit short-circuit honored.

**Live verification** (Neon dev, real Brave + OpenAI, cross-checked against source data):

| Job | Location | Discovery | Cross-check verdict |
|---|---|---|---|
| Celonis CVP (id 202599) | Schwyz, Switzerland | `failed` | ✅ Greenhouse only had Cleveland role; refusing the false positive is correct |
| GitHub Senior Solutions Engineer (id 199838) | Germany | `failed` | ✅ GitHub uses Workday (JS-rendered); needs browser fallback |
| Anthropic SA Munich (id 200082) | Munich, Germany | `failed` | ✅ Greenhouse had Paris and NYC variants only; user confirmed listing is a "tracker-only" ghost posting with no real apply destination |

Cache rows after 3 failed runs: 0 (correct: cache only stores successful matches).

---

## Integration TODO

### Resolved during integration

The following items from earlier versions of this file have been resolved by 
the backend integration and Discovery v2 work. Kept here only as a pointer.

**Backend integration**
- **fetch_profile_snapshot, get_opportunity_context** — Both nodes now check 
  for pre-injected state and use it when present; fallbacks remain for CLI mode. 
  Backend `build_profile_snapshot` / `build_opportunity_context` provide real 
  data via the document-feedback adapter.
- **Graph state persistence (both workflows)** — `build_graph(checkpointer=...)` 
  accepts an injected checkpointer; backend adapters pass `PostgresSaver`. 
  `MemorySaver` remains as the CLI default.
- **`run.py` thread_id** — Both CLIs accept `--thread-id` and auto-generate 
  UUIDs.
- **load_document file ingestion** — Backend stores uploaded files via Django's 
  `FileField` and passes the resolved storage path to the graph.
- **OCR fallback for scanned PDFs** — Backend's 
  `graph_adapter._extract_text_with_ocr_fallback` uses pdf2image + pytesseract 
  when normal extraction yields too little text. Lazy-imports gracefully when 
  OCR libs are not installed. The agentic `tools/documents.py` itself still has 
  no OCR path.
- **analyze_ats.py keyword list** — Now consumes `opportunity_context` keywords 
  in both LLM and heuristic paths.
- **HTTP API layer (both workflows)** — Backend has DRF views and URL routes 
  for list/create/detail/cancel + per-gate resume endpoints. (Stack is Django 
  + DRF rather than FastAPI; the API requirement is met.)
- **Auto-apply: `_get_stub_profile()` shared by three nodes** — 
  `eligibility_and_readiness`, `asset_mapping`, and `application_tailoring` 
  all use `resolve_profile(state)` from `_profile.py`, which prefers 
  `state['profile_snapshot']` and falls back to the in-repo stub.
- **Auto-apply: `_fetch_opportunity()` stub** — `load_opportunity` 
  short-circuits when `state['opportunity_data']` is pre-loaded by the backend 
  adapter; the CLI fallback path emits a WARNING log to surface unexpected use 
  in production.
- **Auto-apply: `submit_internal._post_to_backend` stub** — Removed; 
  `submit_internal` now records intent only, and 
  `auto_apply_adapter.finalize_internal_submission` writes the `Application` 
  row via Django ORM after the graph terminates with 
  `submission_type='internal'`.
- **Auto-apply: `human_gate_0` dead stub** — Now uses real `interrupt()` with 
  iteration cap of 2 (`PROFILE_INCOMPLETE_AFTER_RETRIES`); graph routes back 
  to `eligibility_and_readiness` for a re-check, and the backend rebuilds the 
  profile snapshot before resuming so eligibility sees fresh data.
- **Auto-apply: stub profile blocks all docs at gate 0** — Fixed via 
  `_GENERATABLE` set consulted by `_check_profile_completeness`. Generatable 
  documents (Cover Letter, SOP, Personal Statement, Research Proposal, Writing 
  Sample, Motivation Letter, References) no longer trigger gate 0; user-supplied 
  docs (Transcript, English Proficiency Test) still do.
- **Auto-apply: hard-block ineligibility design** — Compatibility checks 
  (location, age, degree-level, discipline, nationality) became non-blocking 
  warnings carried via `state['compatibility_warnings']` and surfaced through 
  `application_package.warnings`. Users can apply anyway.
- **Auto-apply: `eligibility_and_readiness` document_texts['CV'] hardcoded 
  stub** — Backend's profile snapshot fetches real CV text via the OCR fallback.

**Discovery v2**
- **scrape_application_page User-Agent / JS rendering** — Discovery v2 added 
  httpx-first fetching with optional Playwright/Crawl4AI fallback, env-gated 
  by `UPPGRAD_BROWSER_SCRAPE_ENABLED`. The dependency is lazy-imported. 
  Production deployment of Chromium on Railway is the remaining piece (see 
  Still open below).
- **Stub DB calls in 4 nodes** — All four (`load_opportunity` + the three 
  profile-using nodes) are now backend-injection aware via `resolve_profile()` 
  / `opportunity_data` pre-load.

### Still open

**Agentic — analysis quality**
- **All five `analyze_*` nodes** — `parsed_instructions` lives in `context_pack` 
  but only `synthesize_feedback` consumes it. Analysis nodes should narrow or 
  prioritize findings by the user's stated focus.
- **`synthesize_feedback.py` heuristic path** — `before_text` and `after_text` 
  are placeholder strings rather than actual spans from `doc_sections`. The 
  LLM path uses real spans (validated against the document), so this only 
  matters when the LLM is unavailable.
- **`application_tailoring.py:_heuristic_tailor`** — When LLM is unavailable, 
  the SOP/Personal Statement generator embeds 
  `[Source material summary — real generation requires LLM]`. 
  `application_evaluation` flags it as a placeholder and the loop retries, 
  but the final document still contains the string after MAX_EVAL_ITERATIONS=2. 
  Either drop the placeholder and emit minimal coherent prose, or hard-gate 
  SOP generation on LLM availability and surface a clear message to the user.
- **`AssetMap.requirement_type` is misnamed** — The field stores the document 
  type ("CV", "Cover Letter") not the requirement category. Consider renaming 
  to `document_type`. Downstream consumers (`human_gate_1.py`, 
  `application_tailoring.py`) key `confirmed_mappings` by this field's value, 
  so the rename must be coordinated.

**Agentic — document state**
- **`DocFeedbackState` has no `user_id` field** — Not blocking (the backend 
  injects `profile_snapshot` which already contains `user_id`), but worth 
  adding for explicit auth context if more user-keyed lookups are added.

**Agentic — additional providers**
- **`common/llm.py` only wires OpenAI** — Add Anthropic / Azure as needs arise.
- **`tools/search.py` only wires Brave** — Same factory pattern would apply.

**Agentic — file extraction edge cases**
- **`tools/documents.py`** — DOCX tables and images are silently ignored. 
  Plain PDF text-extraction yields empty for scanned docs; the backend's 
  OCR fallback (`graph_adapter._extract_text_with_ocr_fallback`) covers this 
  in production but the agentic tool itself has no OCR path.

**Resume-value contract** (documented quirk, not a fix-it)
- **`human_gate_1.py`, `human_gate_2.py`** — LangGraph re-interrupts when 
  `Command(resume=<falsy value>)` is passed (empty dict, None). The backend 
  always sends a non-empty dict: `{"confirm": True, ...}` for gate 1 and 
  `{"approved": True, ...}` for gate 2.

**Backend / production**
- **Browser fallback in production** — With `UPPGRAD_BROWSER_SCRAPE_ENABLED=false`, 
  discovery success rate against Workday-hosted careers (a meaningful share of 
  large-company postings) is near zero. Turning on Crawl4AI/Playwright in prod 
  requires Railway Chromium setup (1GB+ per instance, lazy-imported, env-gated). 
  Plan covers the integration; the deployment piece is the open task.
- **LinkedIn ghost-posting detection upstream** — A meaningful share of 
  `linkedin_jobs` rows are tracker-only postings (LinkedIn's apply button 
  doesn't go anywhere external; it just bookmarks the listing in the user's 
  tracker). Discovery correctly returns `failed` but burns Brave calls finding 
  nothing real. The real fix is in the scraper 
  (`bitirme/linkedin_jobspy/scraper.py`) — detect apply-flow type at ingestion 
  and store as an `apply_type` field; backend filters tracker-only jobs out 
  of the auto-apply candidate set entirely. Until then, see "negative caching" 
  below.
- **Negative caching for `failed` discoveries** — Short TTL (e.g. 7 days) 
  would save Brave budget on repeat ghost-posting attempts. Risk: misses cases 
  where the company posts to Greenhouse a few days later — acceptable for the 
  long-tail ghost cases.
- **UI signal for ghost postings** — When `url_direct` is empty AND discovery 
  returns `failed`, surface "We couldn't find an external apply page for this 
  job" on the apply screen.
- **Gate 1 `additional_uploads` plumbing on backend** — The serializer accepts 
  `additional_uploads: List[FileField]` but the adapter doesn't yet save those 
  files anywhere or thread the extracted text into the resume payload's 
  `content` fields. Today users can confirm the default mapping or set 
  `tailoring_depth='none'` for stored docs but cannot upload a fresh document 
  at gate 1.
- **Polymorphic `StudentDocument` store** — v1 only supports CV from 
  `StudentCV`. Cover Letter / SOP / Personal Statement are always *generated*, 
  never read from a stored user file. A generic `StudentDocument` table would 
  let users save and reuse these.
- **Gate 2 "edit and re-tailor" loop** — v1 hard-cancels on rejection at 
  gate 2. A loop back to `application_tailoring` with per-document feedback as 
  edit instructions would let the user iterate without starting fresh.
- **External form auto-submission** — Browser automation / Playwright path 
  for external jobs where discovery succeeded with high confidence. Would 
  create real `Application` rows for external jobs too.
- **Email handoff delivery** — Currently handoff is in-app only.
- **Per-session LLM cost budgeting** — Worst-case ~15-20 LLM calls per session; 
  no token cap today.
- **GDPR / retention policy** — Tailored documents and scraped content live 
  indefinitely in `ApplicationSession` rows + PostgresSaver checkpoints.
- **Cooperative cancel inside nodes** — Currently the cancel marker is observed 
  only at the next node boundary; the currently-executing node finishes 
  (worst case ~30-60s of LLM cost wasted).

**Verification quality (Discovery v2)**
- **Description-keyword extraction is purely frequency-based after stopword 
  filtering.** TF-IDF against a corpus of similar job descriptions, or named-
  entity extraction, would produce better corroborators but adds complexity. 
  Acceptable for v1; revisit if the live false-negative rate is high.
- **Verification still relies on raw HTML.** Greenhouse pages contain JSON-LD 
  structured data (`<script type="application/ld+json">`) with title / location 
  / etc. Parsing those would be more robust than regex on body text. Not urgent.
