# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

UppGrad is an AI-powered platform helping students find and apply to jobs, 
graduate programs, and scholarships. It has two core agentic workflows: 
document feedback and auto-apply. Document feedback analyzes uploaded CVs, 
SOPs, and cover letters and proposes structured reviewable edits. Auto-apply assesses eligibility, generates tailored application materials, and attempts submission. Both workflows use human-in-the-loop approval before any consequential action.

---

## Priority TODO

This section captures the next concrete agentic-side change to make, with 
enough detail that the change can be implemented without re-deriving the 
plan. Items are ordered by priority. Strike through (or delete) once shipped.

### 1. Auto-apply gate-1 remodel + remove `human_gate_0` (single PR)

**Goal**: replace the `asset_mapping → human_gate_1 → application_tailoring 
→ application_evaluation → human_gate_2` chain with a form-field-driven 
requirement model, and remove gate 0 in the same PR. Asset mapping stops 
trying to map requirements to user-uploaded documents (only CV is ever 
stored — `accounts_studentcv` is one row per student, no `uploaded_document` 
table exists, the `available=True / tailoring_depth='light'` branches of 
today's heuristic are dead in production). It instead produces a categorized 
requirement list derived from `extract_form_fields`. Gate 1 collects 
per-requirement choices (Upload / Auto-generate / Ignore for now / Skip) and 
an optional 200-char user prompt per document. Tailoring becomes two-pass 
for uploaded docs and single-pass for auto-generated. Evaluation drops the 
retry loop and surfaces issues as warnings at gate 2.


**Subchanges**:

A. **Remove `human_gate_0`**.
   - Delete `nodes/human_gate_0.py`.
   - In `eligibility_and_readiness.py`: delete `_check_profile_completeness` 
     and the lazy `_GENERATABLE` import; delete the `if missing_fields:` 
     pending branch. Eligibility reduces to deadline check (hard-block) + 
     compatibility warnings (non-blocking). Always pass `missing_fields=[]`.
   - In `graph.py`: remove `human_gate_0` node + `_route_after_gate_0` + 
     conditional edges. Simplify `_route_after_eligibility` to two 
     destinations (`end_with_explanation`, `asset_mapping`).
   - In `state.py`: drop `gate_0_iteration_count`.
   - In `schemas.py`: narrow `EligibilityDecision` to 
     `Literal["ready", "ineligible"]`.
   - Delete `tests/workflows/auto_apply/test_human_gate_0.py` and 
     `test_graph_gate_0_loop.py`.

B. **Internal-jobs short-circuit**.
   - In `graph.py`: new conditional after `load_opportunity` — if 
     `opportunity_type=='job' AND opportunity_data.employer_id==1`, route 
     straight to `determine_requirements`, skipping discover/scrape/
     evaluate-scrape/extract-form-fields entirely.
   - In `determine_requirements.py`: when this short-circuit fires, emit 
     `[CV, Cover Letter]` as document requirements directly. (Match 
     `jobs_application` non-system fields: `resume_file` + `cover_letter`.)

C. **Canonical document-type tagging on FormFields**.
   - `schemas.FormField`: add `canonical_document_type: str = ""`, 
     populated only for `field_type='file'`.
   - New `tools/canonical_doc_types.py`: heuristic keyword table mapping 
     common label phrases to `_GENERATABLE ∪ _USER_SUPPLIED` canonical 
     types ("resume"/"cv" → CV; "cover letter"/"motivation letter" → 
     Cover Letter; "transcript" → Transcript; "portfolio" → Portfolio; 
     etc.).
   - In `determine_requirements.py` (jobs path only): after form_fields 
     are present, walk each `field_type='file'` field — apply heuristic 
     first, fall back to LLM classification for unmatched labels (LLM 
     prompt enumerates the valid canonical-type set). Write the result 
     back onto each FormField in `state['form_fields']`.

D. **New requirement schema** — replaces `AssetMap`/`AssetMappingOutput`.
   - In `schemas.py`:
     ```python
     class RequirementItem(BaseModel):
         id: int
         category: Literal["document", "text", "misc"]
         label: str
         description: str
         field_type: Optional[str]            # FormFieldType for items derived from form_fields
         required: bool
         document_type: Optional[str]         # canonical doc_type when category=document
         question: Optional[str]              # text-category items: drives generation
         form_field_index: Optional[int]      # back-pointer into state['form_fields']
     ```
   - Drop `tailoring_depth='none'` everywhere it appears: 
     `TailoringDepth` literal, `_VALID_DEPTHS` in `human_gate_1`, the 
     `none` branch in `application_tailoring` (line 334), the 
     `none`-skip in `application_evaluation` (line 91), the `none` filter 
     in `package_and_handoff` (line 54). New `TailoringDepth = 
     Literal["light", "deep", "generate"]`.

E. **Asset mapping rewrite** (`nodes/asset_mapping.py`).
   - Inputs: `state['form_fields']` (jobs with non-empty extraction) OR 
     `state['normalized_requirements']` (non-jobs and form-failed jobs).
   - Output: `state['requirement_items']: List[RequirementItem]`. (Keep 
     the JSONB column name `asset_mapping` on `ApplicationSession` for 
     stability — only the dict shape changes.)
   - Logic:
     - Jobs with form_fields → group by `field_type` into document/text/misc; 
       documents get a dedupe pass on `canonical_document_type` (keep the 
       required one if any, else first); text items take their FormField 
       label as `question`; misc collapses to a single virtual item.
     - All other paths → emit document-only items from 
       `normalized_requirements` (no text/misc groups).
     - Floor: if both lists are empty, fall back to 
       `_DEFAULTS[opportunity_type]` (per-type default doc set).

F. **Two-pass tailoring (uploaded) / single-pass (auto-generated)**.
   - New schemas in `schemas.py`:
     ```python
     class UploadedDocPreAnalysis(BaseModel):
         completeness: str
         relevance: str
         correctness: str
         overall_quality: Literal["needs_major_work", "needs_revision", "ready_for_polish"]
         top_priorities: List[str]                 # cap 3

     class UploadedDocLightPostAnalysis(BaseModel):
         structure_issues: List[str]               # cap 3
         content_gap_vs_opportunity: List[str]     # cap 3
         content_gap_vs_profile: List[str]         # cap 3
     ```
   - New `nodes/upload_pre_analysis.py` and `nodes/upload_light_post_analysis.py` 
     (LLM calls returning the schemas above; no heuristic fallback — emit 
     empty/safe analysis if LLM unavailable).
   - Rewrite `nodes/application_tailoring.py`:
     - **Upload path**: PreA → T1 (uses PreA + opportunity + profile + CV + 
       per-doc user_prompt) → LA → T2 (uses T1 + LA + same context). 
       Always 2-pass; do NOT short-circuit on `ready_for_polish`.
     - **Auto-generate document**: single tailoring call (CV + profile + 
       opportunity + user_prompt + canonical_document_type). No T2.
     - **Auto-generate text**: single LLM call with FormField label as the 
       question, profile + CV + opportunity as context. Output goes into 
       new state field `tailored_answers: Dict[str, dict]` keyed by 
       `form_field_index` (NOT `tailored_documents`).
   - Per-doc-type caps unchanged. Add a 1500-char cap per text answer.

G. **Drop the evaluation retry loop** (`nodes/application_evaluation.py`).
   - Rewrite as informational only: length / placeholder / 
     keyword-coverage check across `tailored_documents` and 
     `tailored_answers`. Output: `evaluation_result.warnings: List[str]`.
   - In `graph.py`: remove the `application_evaluation → 
     application_tailoring` retry edge; always proceed to `human_gate_2`.
   - In `state.py`: drop `iteration_count` (unused after loop removal).

H. **Gate 1 rewrite** (`nodes/human_gate_1.py`).
   - Interrupt payload: `{requirement_items, opportunity_title, 
     opportunity_type}`.
   - Resume value:
     ```python
     {
       "requirements": {
         "<id>": {
           "choice": "upload" | "auto_generate" | "ignore_for_now" | "skip",
           "uploaded_text": "<extracted text>" | null,
           "user_prompt": "<≤200 chars>" | null      # documents only
         }
       },
       "misc_strategy": "auto_fill" | "ignore"
     }
     ```
   - Validation: required document items reject `skip` and 
     `ignore_for_now`; required text items reject `skip`; USER_SUPPLIED 
     canonical doc types reject `auto_generate` (only `upload` / `skip` 
     when not required). On invalid resume, return 400 from the backend 
     view; the graph stays paused at gate 1 until a valid payload arrives.
   - Compute `auto_submit_feasible_at_gate_1: bool` from the choices and 
     stash on state.

I. **Gate 2 rewrite** (`nodes/human_gate_2.py`).
   - Interrupt payload: previews of `tailored_documents` AND 
     `tailored_answers`, `evaluation_result.warnings`, `posting_closed`, 
     and the recomputed `auto_submit_feasible` (per P1: false when any 
     required document's auto-generation produced no content, or any 
     required upload is missing).
   - Resume value: `{approved, attempt_auto_submit, feedback}`.
   - Routing: `attempt_auto_submit=True` is recorded on 
     `gate_2_response` regardless of feasibility (intent only — auto-submit 
     itself isn't implemented yet). Today's branching is preserved: 
     `employer_id==1 → submit_internal`, else `package_and_handoff`.

J. **Backend integration**.
   - `auto_apply_adapter.py`: delete `resume_session_gate_0`, the 
     gate-0 branch in `_pending_node_after_invoke`, the 
     `AWAITING_PROFILE_COMPLETION` mapping in `_persist_state_after_phase`, 
     and the post-gate-0 profile-snapshot reinjection. New: persist gate 
     1 uploaded files (Railway Volume — backend sets `cv_file` / future 
     fields to `FileField` paths and extracts text via the 
     document-feedback OCR fallback before passing into the resume payload 
     as `uploaded_text` per requirement). Closes the long-standing 
     "additional_uploads not plumbed" gap.
   - `views.py` + `urls.py`: remove the `resume-gate-0/` view and route.
   - `serializers.py`: rewrite `Gate1ResumeSerializer` to the new shape 
     (per-requirement choice dict, `misc_strategy`, optional 
     `FileField`s). Enforce `max_length=200` on each `user_prompt`. 
     Rewrite `Gate2ResumeSerializer` to add 
     `attempt_auto_submit: bool` and a per-doc/answer feedback dict.
   - `models.py` + new Django migration: drop `gate_0_response`; add 
     `tailored_answers: JSONField` and `requirement_items: JSONField`; 
     remove `AWAITING_PROFILE_COMPLETION` from the status enum AND from 
     the `one_active_application_session_per_student_opportunity` partial 
     unique-index condition.
   - `janitor.py`: drop the `AWAITING_PROFILE_COMPLETION` TTL row.
   - Pre-deploy step: cancel all in-flight sessions in 
     `AWAITING_PROFILE_COMPLETION` and `AWAITING_DOCUMENT_MAPPING` — old 
     resume payload shapes are incompatible with the new serializers.

**Things to preserve**:
- Deadline check + `INELIGIBLE` termination + `end_with_explanation`.
- `compatibility_warnings` flow (non-blocking).
- Discovery cache + 14-day staleness gate.
- `posting_closed` warning surface in handoff package.
- `form_fields` in package (auto-submit foundation).
- `_GENERATABLE` / `_USER_SUPPLIED` sets — now drive canonical-type 
  classification + auto-generate-button visibility.
- PostgresSaver checkpointer.

**Out of scope (deferred follow-ups)**:
- External-job auto-submit implementation (this PR records intent only).
- Gate 2 "edit and re-tailor" loop.
- Polymorphic `StudentDocument` table (CV is still the only stored doc).
- ZIP download endpoint at `GET /api/ai/application-sessions/<id>/package.zip` 
  (intent recorded; render docs to PDF via `document_renderer`, text 
  answers as `.txt`, on-demand build, no disk persistence).
- `expected_source` classification quality tuning.
- `jobs_application.cover_letter` text → file-path migration.

**Smoke tests**:
- Internal job (`employer_id=1`, complete profile): 
  load → internal-route → determine_requirements emits `[CV, Cover Letter]` 
  → eligibility ready → asset_mapping → gate 1 with 2 document items 
  (Upload/Auto-generate, no Skip since required) and per-doc prompt 
  fields → user picks Auto-generate for both → tailoring single-pass 
  each → evaluation informational → gate 2 → submit_internal writes 
  `Application` row.
- External Greenhouse job, full discovery success: full discover/scrape/
  evaluate/extract → asset_mapping derives item list from form_fields, 
  dedupes duplicate file fields, classifies canonical types → gate 1 mixes 
  documents, texts, and one Misc line → user uploads CV (text extracted 
  server-side), Auto-generate Cover Letter, Auto-generate "Why us?" text, 
  Ignore "Salary expectations", Auto-fill misc → tailoring 
  PreA→T1→LA→T2 for CV + single-pass for CL + single-pass text-answer 
  for "Why us?" → evaluation informational → gate 2 with 
  `auto_submit_feasible=False` (one required text ignored) → user clicks 
  Get package only → handoff.
- Job with empty form_fields (Workday auth wall): falls back to 
  `_DEFAULTS["job"]` → gate 1 has CV + Cover Letter as documents only, no 
  text/misc.
- Masters program with Transcript required (parsed from `data.json`): 
  determine_requirements emits CV + SOP + Transcript → gate 1 with 
  Transcript as USER_SUPPLIED (only Upload offered) → user submits gate 
  1 without Transcript → backend returns 400, graph stays paused → user 
  re-submits with Transcript file → tailoring proceeds.
- Past-deadline opportunity: eligibility ineligible → 
  end_with_explanation with `error_code='INELIGIBLE'` (unchanged).

---

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
scrape_application_page.py, evaluate_scrape.py, extract_form_fields.py, 
determine_requirements.py)
Loads the opportunity record from the correct DB table based on opportunity_type 
(short-circuits when the backend adapter has pre-loaded `opportunity_data`).
For jobs only: discovers the real apply URL via `discover_apply_url` (Brave 
search through ATS / company-careers / generic tiers, with multi-factor 
verification — see Discovery v2 below). Discovery also resolves the 
apply-form URL via per-ATS rules (Ashby `<overview>/application`, Lever 
`<overview>/apply`, SmartRecruiters `<overview>/apply`, Greenhouse / Workable 
same URL, Workday returns `None` for the auth wall) and stores it in 
`state['discovered_form_url']`. Scrapes the application page using the 
verified content from discovery (no double-fetch). Evaluates scrape quality 
as full, partial, or failed and normalizes scraped content into a structured 
requirements list. Then `extract_form_fields` parses the rendered form HTML 
into a list of `FormField` records (label, field_type, name, required, 
options, accepts_file, expected_source) so a future auto-submit step has a 
complete map of every input on the form. Field extraction follows a 3-tier 
strategy: in-state HTML from discovery → forced browser fetch via Crawl4AI 
(for company-direct careers pages with client-side hydrated forms) → ATS 
iframe-follow (for `mongodb.com/careers/<id>` style pages that embed a 
third-party Greenhouse/Lever/etc. form via cross-origin iframe).
For programs and scholarships: skips scraping AND form-field extraction 
entirely and parses requirements directly from the data json field in the 
DB record.
If scraping fails or is partial, falls back to assumed default requirements based 
on opportunity type. Never blocks on a failed scrape.
Stores scrape_status, scrape_confidence, and form_fields in state so downstream 
agents and the user are aware of whether requirements are real or assumed.

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
→ discover_apply_url (3-tier Brave search → verified URL + raw_html + per-ATS form_url)
→ scrape_application_page (uses pre-fetched discovery content; falls back to fresh httpx)
→ evaluate_scrape (assess quality: full | partial | failed)
→ extract_form_fields (LLM → FormSchema; 3-tier: in-state HTML → forced browser → ATS iframe-follow)
→ determine_requirements (full → use scraped, partial → merge with defaults, failed → defaults)
if opportunity_type == masters or phd:
→ skip scraping AND form-field extraction
→ determine_requirements (parse from data json; assume [CV, SOP] as baseline)
if opportunity_type == scholarship:
→ skip scraping AND form-field extraction
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
  to avoid double-fetching the same URL. `discovered_page_content` (markdown for 
  the browser path / HTML for httpx) feeds prose extraction; 
  `discovered_raw_html` (always actual HTML) feeds form-field extraction.
- Form-field extraction is job-only and short-circuits internally on error / no 
  resolved form URL — `discovered_form_url=None` (Workday auth wall, etc.) yields 
  `form_fields=[]` and the graph proceeds normally.
- Never block the workflow on a failed scrape or empty form_fields, always degrade 
  to assumed requirements / empty field list
- Store scrape_status, scrape_confidence, and form_fields in state throughout so 
  the user is always informed whether requirements were scraped or assumed
- Internal vs external is determined by employer_id in linkedin_jobs, not site column
- human_gate_0 is conditional and only triggered when eligibility check finds 
  missing user-supplied documents or required profile fields
- For now internal submission only requires CV and Cover Letter fields
- All three opportunity types share the same pipeline after determine_requirements; 
  only the scraping + form-field-extraction steps and the final routing differ

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
  (`discovered_page_content` = markdown for browser / HTML for httpx — feeds prose extraction)
- discovered_raw_html: actual HTML always (httpx response body or browser-rendered DOM
  after JS hydration); separate from discovered_page_content because the browser path's
  text is markdown but form extraction needs real `<input>/<select>/<textarea>` tags
- discovered_form_url: Optional[str] — apply-form URL from per-ATS rules
  (`tools/ats_form_urls.py`). Equals discovered_apply_url for ATSes that keep the form
  on the same URL (Greenhouse, Workable, company-direct careers); differs for split-URL
  ATSes (Ashby `/application`, Lever `/apply`, SmartRecruiters `/apply`); None for
  Workday auth wall and when no apply URL was found.
- posting_closed: bool (true when discovery found a real listing that says the
  posting is closed; surfaced in handoff package)
- scraped_requirements: dict (status, requirements list, confidence, source, raw_content,
  raw_html, http_status; `evaluate_scrape` rewrites this dict — top-level
  `discovered_raw_html` is the durable source for form extraction)
- form_fields: List[Dict] — one FormField dict per visible <input>/<select>/<textarea>
  on the application form. Empty list when discovery couldn't reach the form or no LLM
  is configured. Surfaced in `application_package.form_fields` for future auto-submit.
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

### Form-field extraction (Phase 2 of auto-submit foundation)

`extract_form_fields` runs after `evaluate_scrape` and before 
`determine_requirements` for jobs only. Captures every input on the rendered 
application form as a structured `FormField` record so a future auto-submit 
node can fill the form without re-extracting structure. Output is surfaced 
in the handoff package today (`application_package.form_fields`) but no 
node consumes it yet.

**Modules**:
- `tools/ats_form_urls.py:resolve_application_form_url(overview_url)` — per-ATS 
  rules that map an overview URL to the apply-form URL. Returns the original 
  URL for ATSes that keep the form on the same page; appends `/application` 
  (Ashby), `/apply` (Lever, SmartRecruiters); returns `None` for Workday 
  (auth wall) and unknown / malformed inputs.
- `tools/form_extractor.py`:
  - `extract_form_html(html)` — Strategy 1: pick the `<form>` element with 
    the most `<input>/<select>/<textarea>` descendants (Greenhouse, Lever, 
    SmartRecruiters, anything that uses native form markup). Strategy 2 
    fallback: when no `<form>` exists (Ashby, modern Workday, other 
    React-driven ATSes that put inputs in plain `<div>`s and submit via 
    fetch), return the body with `<script>/<style>/<meta>/<link>/<noscript>` 
    and hidden inputs stripped. Empty string when neither yields inputs.
  - `extract_ats_iframe_src(html)` — finds the src of the first iframe 
    pointing at a known ATS host (`_ATS_IFRAME_HOSTS`: greenhouse.io, 
    lever.co, ashbyhq.com, workable.com, smartrecruiters.com, 
    myworkdayjobs.com, bamboohr.com, jobvite.com, recruitee.com). Used to 
    follow company-direct careers pages that embed third-party ATS forms 
    via cross-origin iframe (mongodb.com → boards.greenhouse.io/embed/...). 
    Cross-origin iframes can't be parsed in-place — caller fetches the src 
    directly.
- `tools/web_fetcher.force_browser_fetch(url)` — bypasses the thin gate to 
  render a URL with Crawl4AI/Playwright regardless of httpx's verdict. Used 
  for company-direct careers pages that return non-thin server-rendered HTML 
  but render the form area client-side (mongodb.com/careers/<id>, Anthropic 
  careers index, any Ashby/Workday SPA). Returns `None` when 
  `UPPGRAD_BROWSER_SCRAPE_ENABLED=false` or crawl4ai isn't installed.
- `_build_crawler_run_config` passes `wait_for: 'js:() => document.body.innerText.length > 1000'` 
  so Crawl4AI defers extraction until React/SPA hydration completes. Verified 
  live on Notion's Ashby `/application` URL: text_len went from 1 char to 7021.

**FormSchema** (LLM structured output target, in `workflows/auto_apply/schemas.py`):
- `FormSchema.fields: List[FormField]` — one entry per visible input, in document order.
- `FormSchema.form_action: str` — form's `action` attribute (useful for direct POST).
- `FormSchema.form_method: str` — POST/GET, defaults to POST.

**FormField** (one record per input):
- `label: str` — human-readable label from `<label>`, surrounding text, 
  `aria-label`, or `placeholder`.
- `field_type: FormFieldType` — see flag list below.
- `name: str` — DOM `name` attribute, used at actual submission time. Empty 
  when not in markup.
- `required: bool` — true when `required` attr is set OR the label/legend 
  ends with `*` / contains "(required)".
- `options: List[str]` — for `select` and grouped `radio`: option labels in 
  document order. Empty for non-choice fields.
- `accepts_file: List[str]` — for `file` inputs only: values of the `accept` 
  attribute split on commas (`[".pdf", ".docx"]`). Empty for non-file fields.
- `expected_source: FormFieldValueSource` — see flag list below.

**FormFieldType flags** (one per input element, in 
`schemas.FormFieldType`):
- `file` — file upload input. The system will populate from `user_document` 
  (CV/Cover Letter/etc.) at auto-submit time.
- `text` — generic single-line text input.
- `textarea` — multi-line text. Long-form textareas are typically free-form 
  questions classified as `user_answer`.
- `select` — `<select>` dropdown; `options` lists every visible choice.
- `checkbox` — single checkbox or one of a multi-select group.
- `radio` — radio button; grouped radios share the same `name` and the LLM 
  collapses them into one FormField with `options`.
- `number` — `<input type="number">`.
- `email` — `<input type="email">` (typically `expected_source=user_profile`).
- `url` — `<input type="url">` (LinkedIn / GitHub / portfolio links).
- `date` — `<input type="date">`.
- `tel` — `<input type="tel">` (typically `expected_source=user_profile`).

**FormFieldValueSource flags** (where the value should come from when 
auto-filling, in `schemas.FormFieldValueSource`):
- `user_profile` — fields whose label maps to a profile attribute (name, 
  email, phone, country, LinkedIn URL, GitHub URL, location, work auth).
- `user_document` — file inputs whose label suggests an uploadable document 
  (resume/CV, cover letter, portfolio, transcript). Drawn from 
  `tailored_documents` (output of `application_tailoring`) or stored user 
  files (`StudentCV` etc.) at auto-submit time.
- `user_answer` — free-form questions like "Why do you want to join us?" 
  textareas, screening multiple-choice. Will be LLM-drafted at auto-submit 
  time from profile + opportunity + the question prompt.
- `computed` — fields the system can derive without the user (today's 
  date, etc.).
- `unknown` — none of the above clearly apply; LLM couldn't classify. 
  Auto-submit will surface these to the user for manual entry.

**ScrapeStatus flag** (existing, on `scraped_requirements.status`):
- `full` — scrape returned a structured requirements list with high confidence.
- `partial` — scrape returned content but couldn't extract structured 
  requirements; fall back to type-defaults merged with what we have.
- `failed` — scrape couldn't fetch / page was thin; fall back to 
  type-defaults entirely.

**Discovery method flag** (on `discovery_method` / `DiscoveryResult.method`):
- `url_direct` — `linkedin_jobs.url_direct` was populated; no search needed.
- `ats` — Tier 1 search hit, verified via slug + corroborators.
- `careers` — Tier 2 search hit (`site:<company-domain>`), verified.
- `generic` — Tier 3 search hit (`"<title>" "<company>" apply`), verified.
- `closed` — A page verified on title + company but contains 
  closed-posting phrases ("no longer accepting applications" etc.). Surfaced 
  in handoff package as a warning instead of skipped silently.
- `failed` — No tier produced a verified hit.
- `skipped_internal` — Internal job (employer_id == 1); no external apply URL needed.

**FetchResult flags** (`tools/web_fetcher.py:FetchResult`):
- `success: bool` — HTTP 2xx (or browser equivalent).
- `thin: bool` — anti-bot wall, JS shell, short body, or 4xx; gates browser 
  escalation in `fetch_url_with_fallback`.
- `text: str` — best-effort readable content (httpx: HTML; browser: markdown).
- `raw_html: str` — actual HTML (httpx: response body; browser: result.html 
  raw rendered DOM after JS hydration). Used by form-field extraction.
- `http_status: int` / `final_url: str` / `error: str` / `thin_signals: List[str]`.
- `used_browser: bool` — true when escalated to Playwright/Crawl4AI.

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
automation (Playwright or equivalent). The structured `form_fields` list 
(from `extract_form_fields`) is already captured per session and surfaced in 
`application_package.form_fields`, so the auto-submit node can map values via 
`expected_source` (user_profile → Student row, user_document → tailored doc, 
user_answer → LLM-drafted, computed → derived). This step still has to handle 
multi-step forms, file upload fields, captcha presence detection, and 
graceful fallback to handoff when anti-bot mechanisms are encountered.

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

**Auto-Apply Workflow** — all 16 nodes wired end-to-end, smoke tested across 
job/masters/phd/scholarship, routing verified at every conditional edge.
- Opportunity Intelligence: load_opportunity (short-circuits on pre-loaded 
  opportunity_data, falls back to in-repo stubs in CLI), discover_apply_url 
  (Brave + 3-tier search + multi-factor verification + per-ATS form URL 
  resolution + raw_html propagation), 
  scrape_application_page (consumes pre-fetched discovery content; falls back 
  to fresh httpx fetch via the tiered fetcher; surfaces raw_html alongside 
  raw_content), evaluate_scrape (LLM structured output + heuristic fallback), 
  extract_form_fields (LLM with FormSchema structured output; 3-tier 
  in-state-HTML → forced-browser → ATS-iframe-follow strategy; LLM-only — no 
  useful heuristic fallback so emits empty list when LLM unavailable), 
  determine_requirements (scrape → parse data json → assumed defaults).
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
  provenance, compatibility warnings, posting_closed flag, form_fields, and 
  opportunity.form_url for jobs); record_application (logs outcome, 
  timestamp, doc types, scrape metadata).
- graph.py fully wired end-to-end; run.py CLI entry point with `--thread-id`.
- Frontend progress tracking: `current_step` and `step_history` on 
  AutoApplyState; all 16 nodes set both fields. Auto-apply has no parallel 
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
  ≥2 anti-bot keywords. `FetchResult.raw_html` carries actual HTML for both 
  paths (httpx response body / browser-rendered DOM) so form extraction can 
  read real DOM tags. `force_browser_fetch(url)` bypasses the thin gate when 
  the caller knows it needs JS-rendered content (form area on a non-thin 
  server-rendered page).
- `url_discovery.py` — Multi-factor verification + 3-tier orchestration. 
  `DiscoveryResult` now also carries `raw_html`, `form_url` (resolved by 
  `ats_form_urls`), and `posting_closed`.
- `ats_form_urls.py` — Per-ATS rules to map an overview URL to the apply-form 
  URL. Ashby `/application`, Lever `/apply`, SmartRecruiters `/apply`, 
  Greenhouse / Workable / company-direct return as-is, Workday returns 
  `None`. Used at every `DiscoveryResult` construction site.
- `form_extractor.py` — Pulls the form area out of rendered HTML; cleans 
  `<script>/<style>/<meta>/<link>/<noscript>` and hidden inputs; supports 
  cross-origin ATS-iframe follow (`extract_ats_iframe_src`).

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

**Single-fetch architecture**: `DiscoveryResult.text` and `.raw_html` carry 
verified content forward via `state['discovered_page_content']` and 
`state['discovered_raw_html']`; `scrape_application_page` consumes the prose 
content instead of re-fetching the same URL, and `extract_form_fields` 
consumes the raw HTML for form parsing. Halves the request count and reduces 
ban risk.

**Graph wiring**: `discover_apply_url` → `scrape_application_page` → 
`evaluate_scrape` → `extract_form_fields` → `determine_requirements` for jobs 
(extract_form_fields skipped via internal short-circuit when 
`opportunity_type != 'job'` or `discovered_form_url` is None). Cache-hit 
short-circuit honored at the discovery node.

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

**Form-field extraction (auto-submit Phase 2 foundation)**
- **Apply-form URL resolution** — `tools/ats_form_urls.py` ships per-ATS 
  rules so split-URL ATSes (Ashby `/application`, Lever `/apply`, 
  SmartRecruiters `/apply`) are followed correctly; Workday returns None to 
  signal the auth wall is unreachable. Wired at every `DiscoveryResult` 
  construction site. State carries the result via `discovered_form_url`.
- **Raw HTML propagation** — `FetchResult.raw_html` populated for httpx 
  (response body) and browser (rendered DOM after JS hydration); 
  `discovered_raw_html` carried at the top level of state so it survives 
  `evaluate_scrape` rewriting `scraped_requirements`.
- **`force_browser_fetch`** — Bypasses the thin gate for callers that know 
  they need JS-rendered content (company-direct careers pages with 
  client-side hydrated forms; non-thin server HTML where the form area is 
  React-rendered).
- **`extract_form_fields` node** — Wired between `evaluate_scrape` and 
  `determine_requirements`. LLM with `FormSchema` structured output. Three 
  tiers: in-state HTML → forced browser fetch → ATS iframe-follow 
  (mongodb.com → Greenhouse pattern). Surfaces `form_fields` in 
  `application_package.form_fields` for future auto-submit.
- **JS hydration wait condition** — `_build_crawler_run_config` passes 
  `wait_for: 'js:() => document.body && document.body.innerText.length > 1000'` 
  so Crawl4AI defers extraction until React/SPA hydration completes. 
  Verified live on Notion's Ashby `/application`: text_len went from 1 char 
  to 7021.

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
  create real `Application` rows for external jobs too. Foundation is in 
  place: `extract_form_fields` already captures the structured 
  `form_fields` list with `expected_source` classification per input; 
  the missing piece is a `submit_external` node that drives a Playwright 
  page through the form and a gate-2 split between "approve and submit" 
  vs "approve and handoff".
- **Crawl4AI → direct Playwright consolidation** — Crawl4AI's value today 
  is HTML→markdown + JS hydration wait; both are ~30 lines of native 
  Playwright once auto-submit makes Playwright a committed dependency. 
  The auto-submit work is the natural moment to migrate, and unblocks 
  iframe-embedded ATS forms (MongoDB → Greenhouse, others) where 
  `page.frame_locator("#grnhse_iframe")` traverses cross-origin iframes 
  natively. Crawl4AI cannot capture late-injected iframes in its 
  rendering window (verified live with 30s `wait_for` timeout on 
  `#grnhse_iframe`).
- **`expected_source` classification quality** — Live tests showed noise: 
  Anthropic's `Country` got `unknown`, Notion's `Location` got 
  `user_profile` despite being free-text. The 
  `extract_form_fields._SYSTEM` prompt needs few-shot examples to tighten 
  before auto-fill is reliable.
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
