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
| `UPPGRAD_OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| `OPENAI_API_KEY` | Required when provider is `openai` | _(none)_ |

## Architecture

### Package layout

```
src/uppgrad_agentic/
  common/          # Shared utilities: LLM factory (llm.py), logging, guardrails, error types
  config/          # Settings (currently stub)
  tools/           # File-level tools: documents.py (PDF/DOCX/TXT extraction), opportunities.py
  workflows/
    document_feedback/   # The only implemented workflow so far
      state.py     # DocFeedbackState TypedDict — the single source of truth for graph state
      schemas.py   # Pydantic models used for LLM structured output (DocTypeClassification)
      graph.py     # build_graph() — assembles and compiles the LangGraph StateGraph
      run.py       # CLI entry point: python -m uppgrad_agentic.workflows.document_feedback.run
      nodes/       # One file per node function
      prompts.py   # System/human prompt strings
      tests/       # Smoke test + unit test stubs (currently empty)
    auto_apply/          # Fully implemented end-to-end
      state.py     # AutoApplyState TypedDict
      schemas.py   # Pydantic models (NormalizedRequirement, AssetMap, EligibilityResult, etc.)
      graph.py     # build_graph() — assembles and compiles the LangGraph StateGraph
      run.py       # CLI entry point: python -m uppgrad_agentic.workflows.auto_apply.run
      nodes/       # One file per node function
```

### General patterns

- **State pattern**: each node receives the full State TypedDict and returns a partial 
  dict of keys to merge. Errors are signalled by setting `result.status = "error"` 
  in state. Downstream nodes check this and short-circuit with `return {}`.
- **LLM pattern**: `get_llm()` in `common/llm.py` returns `None` when no provider is 
  configured. Every node that calls an LLM must handle the `None` case with a 
  heuristic fallback. See `detect_doc_type.py` for the reference implementation.
- **Prompt pattern**: prompts live inline inside each node file, not in prompts.py. 
  This keeps each prompt next to its logic. prompts.py is unused and can be ignored.
- **Human-in-the-loop**: any workflow that triggers external actions must include a 
  human_gate node using LangGraph interrupt() before the action.
- **Checkpointer**: MemorySaver is used for now. Will be replaced with 
  AsyncPostgresSaver during backend integration.

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

**Opportunity Intelligence Agent** (load_opportunity.py, scrape_application_page.py, 
evaluate_scrape.py, determine_requirements.py)
Loads the opportunity record from the correct DB table based on opportunity_type.
For jobs only: attempts to scrape the application page (url_direct if present, 
else url), evaluates scrape quality as full, partial, or failed, and normalizes 
scraped content into a structured requirements list.
For programs and scholarships: skips scraping entirely and parses requirements 
directly from the data json field in the DB record.
If scraping fails or is partial, falls back to assumed default requirements based 
on opportunity type. Never blocks on a failed scrape.
Stores scrape_status and scrape_confidence in state so downstream agents and the 
user are aware of whether requirements are real or assumed.

**Applicant Eligibility and Readiness Agent** (eligibility_and_readiness.py)
Checks hard constraints: deadline not passed, location fit, degree requirements 
from data json, profile completeness against normalized requirements.
Produces one of: ready | pending | ineligible | manual_review.
If pending, triggers human_gate_0 to ask user to complete missing profile info 
before continuing.

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

**Application Evaluation Agent** (application_evaluation.py)
Verifies full package for groundedness, requirement coverage, and hallucinations.
Triggers refinement loop back to application_tailoring, capped at 2 iterations.

**Human Review Coordinator** (human_gate_0.py, human_gate_1.py, human_gate_2.py)
Gate 0: only triggered if eligibility check finds missing profile info. Asks user 
to complete required fields before workflow continues.
Gate 1: after asset mapping, presents document mapping to user, collects document 
selections and any additional uploads from local device.
Gate 2: after evaluation, presents final tailored package for user approval before 
submission or handoff. No materials are submitted without passing this gate.
All three use LangGraph interrupt() and require MemorySaver checkpointer.

**Submission Agent** (route_by_source.py, submit_internal.py, package_and_handoff.py, 
record_application.py)
First determines whether the opportunity is internal or external using employer_id 
from the linkedin_jobs table. employer_id == 1 means internal, NULL means external.
For internal jobs: submits CV and Cover Letter directly to platform backend (stub 
for now, to be wired during integration).
For all external opportunities (external jobs, masters, phd, scholarships): 
assembles the final tailored package and hands it off to the user.
Records the application outcome in both cases. For jobs, also stores scrape_status 
and scrape_confidence in the application record for potential future use when 
attempting external submission automation.

### Orchestration

START
→ load_opportunity (receive opportunity_type + id from frontend, query correct table)
→ determine_requirements
if opportunity_type == job:
→ scrape_application_page (use url_direct if present, else url)
→ evaluate_scrape (assess quality: full | partial | failed)
→ if full: use scraped requirements
→ if partial or failed: fall back to assumed defaults [CV, Cover Letter]
→ store scrape_status and scrape_confidence in state
if opportunity_type == masters or phd:
→ skip scraping
→ parse requirements from data json field
→ assume [CV, SOP] as baseline
if opportunity_type == scholarship:
→ skip scraping
→ parse eligibility from data json field
→ assume [CV, Cover Letter] as baseline
→ eligibility_and_readiness
→ ineligible: end_with_explanation → END
→ pending missing profile info: human_gate_0 → (user completes profile) → continue
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
- Scraping is only attempted for job opportunities, never for programs or scholarships
- Use url_direct if present for scraping, fall back to url if not
- Never block the workflow on a failed scrape, always degrade to assumed requirements
- Store scrape_status and scrape_confidence in state throughout so the user is 
  always informed whether requirements were scraped or assumed
- Internal vs external is determined by employer_id in linkedin_jobs, not site column
- human_gate_0 is conditional and only triggered when eligibility check finds 
  missing profile information
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
- opportunity_data: dict (raw DB record)
- scraped_requirements: dict (status, requirements list, confidence, source)
- normalized_requirements: list (final requirements list after scrape or assumption)
- eligibility_result: dict (decision, reasons, missing_fields)
- asset_mapping: dict (requirement to document mapping with tailoring depth)
- human_review_0: dict (user response to missing profile info gate, if triggered)
- human_review_1: dict (user selections from gate 1)
- tailored_documents: dict (document type to tailored content)
- evaluation_result: dict
- human_review_2: dict (user approval from gate 2)
- application_package: dict (final documents ready for handoff or submission)
- application_record: dict (logged outcome, includes scrape_status and scrape_confidence for jobs)
- iteration_count: int
- result: dict (status, error_code, user_message)

### Future implementation steps (post integration)

These are intentionally out of scope for the current implementation but 
should be tracked for future development:

**Requirements review human gate**
After determine_requirements, surface the normalized requirements list to the 
user as a reviewable checklist before asset mapping begins. Similar to the 
ChangeProposal review in document feedback. User can confirm what they can 
provide, flag items they cannot provide right now (e.g. financial documents, 
transcripts), and upload additional documents on the spot. Anything flagged 
as unavailable is noted in the final package rather than blocking the workflow. 
This makes asset mapping cleaner since it works with a confirmed set of assets.

**External application form submission**
For external job opportunities where scraping was fully successful, attempt to 
automatically fill and submit the application form using browser automation 
(Playwright or equivalent). This requires handling multi-step forms, file upload 
fields, and graceful fallback when anti-bot mechanisms are encountered. 
Scrape confidence and scraped field structure should be stored in the application 
record during current implementation to make this step easier to add later.

### Scraping status

Current implementation in scrape_application_page.py makes a plain requests.get()
call which has effectively zero success rate against LinkedIn pages due to
bot detection, JavaScript rendering, and login walls. The fallback to assumed
default requirements [CV, Cover Letter] works correctly and gracefully.

Planned improvement (next implementation step):
Replace requests.get() with a Playwright-based scraper that:
1. Navigates to url_direct if present in DB, else navigates to url
2. If on LinkedIn job page, finds and follows the Apply button to reach url_direct
3. Scrapes the actual application page at url_direct for required documents,
   form fields, and special instructions
4. Returns structured requirements with high confidence if successful

Since scraping only triggers once per user auto-apply action (not bulk),
LinkedIn ban risk is low. Playwright with stealth patches is the recommended
approach. This is a planned improvement, not yet implemented.

Also note: LLM-powered scraping libraries like Firecrawl, Crawl4AI, and
Scrapegraph-ai are worth evaluating as alternatives or complements to Playwright
for structured requirement extraction. Crawl4AI is open source and free.
Evaluate these before implementing the Playwright approach.

---

## Implementation Status

### Completed

**Document Feedback Workflow** — fully implemented, smoke tested across all three
doc types (CV, SOP, Cover Letter), bug audit completed.
- Phase 0: load_document.py, detect_doc_type.py, end_with_error.py, graph.py routing
- Phase 1: Context Assembly nodes (fetch_profile_snapshot, extract_doc_sections, 
  parse_user_instructions, get_opportunity_context, build_context_pack); 
  state.py and schemas.py extended with all Phase 1+ fields and 
  ChangeProposal/EvaluationResult schemas
- Phase 2: Parallel analysis nodes (analyze_structure, analyze_style, 
  analyze_content_gaps, analyze_ats, analyze_opportunity_alignment); 
  graph wired with LangGraph Send fan-out from build_context_pack
- Phase 3: Synthesis & Planning (synthesize_feedback); graph wired with 
  MemorySaver checkpointer
- Phase 4: Evaluation loop (evaluate_output); conditional routing back to 
  synthesize_feedback on failure, capped at MAX_EVAL_ITERATIONS=2
- Phase 5: Human gate (human_gate); interrupt() suspends graph and surfaces 
  proposals to frontend; resume via Command(resume=decisions)
- Phase 6: Rewrite (finalize); applies accepted proposals right-to-left, 
  resolves overlapping spans by confidence, LLM coherence smoothing pass, 
  produces diff summary

**Auto-Apply Workflow** — fully implemented end-to-end, smoke tested across all four
opportunity types (job, masters, phd, scholarship), routing verified at every
conditional edge, bug audit completed.
- Opportunity Intelligence: load_opportunity.py (stub DB lookup for all four types),
  scrape_application_page.py (requests-based fetch with graceful failure),
  evaluate_scrape.py (LLM structured output + heuristic fallback),
  determine_requirements.py (scrape → parse data json → assumed defaults)
- Eligibility: eligibility_and_readiness.py (deadline, hard constraints, profile
  completeness checks); end_with_explanation.py; human_gate_0.py (stub interrupt)
- Asset Mapping: asset_mapping.py (LLM structured output + heuristic depth
  classification); AssetMap / AssetMappingOutput schemas; human_gate_1.py
  (full interrupt/resume with per-document override and upload support)
- Tailoring & Evaluation: application_tailoring.py (LLM apply-mode rewrite +
  heuristic fallback per depth); application_evaluation.py (length, placeholder,
  keyword-coverage checks with 2-iteration retry loop)
- Human Review: human_gate_2.py (full interrupt/resume, approve/reject with
  per-document feedback)
- Submission: submit_internal.py (stub backend POST); package_and_handoff.py
  (assembles package with scrape provenance for jobs); record_application.py
  (logs outcome, timestamp, doc types, scrape metadata)
- graph.py fully wired end-to-end; run.py CLI entry point

### In Progress
- Nothing currently in progress

### Not Started
- Nothing

---

## Integration TODO

Items currently stubbed, hardcoded, or mocked that must be replaced during
backend / frontend / database integration.

### Authentication and user identity
- **state.py** — No user_id field in DocFeedbackState. API layer must inject 
  authenticated user ID into state at invocation time.
- **fetch_profile_snapshot.py** — Returns hardcoded stub profile. Replace with 
  real DB lookup keyed on state["user_id"].

### Opportunity context
- **get_opportunity_context.py** — Returns hardcoded mock opportunity. Real 
  implementation must accept structured opportunity input from frontend and 
  look up or parse the opportunity properly.

### File ingestion and storage
- **load_document.py** — Reads from local filesystem path. In production files 
  will arrive as multipart uploads or from object storage.
- **tools/documents.py** — DOCX tables and images silently ignored. PDF scanned 
  pages return empty text with no OCR fallback. Wire in OCR for scanned docs.

### Graph state persistence
- **graph.py** — MemorySaver is non-durable. Replace with AsyncPostgresSaver 
  pointing at production database.
- **run.py** — No thread_id passed to graph.invoke(). API layer must generate 
  and store thread IDs for interrupt/resume to work across requests.

### LLM and configuration
- **common/llm.py** — Only OpenAI wired up. Add other providers as needed.
- **config/settings.py** — Empty stub. Wire in real settings module using 
  pydantic-settings.

### Analysis quality
- **analyze_ats.py** — Static keyword list. Should use keywords from 
  opportunity_context and user target_roles.
- **All analysis nodes** — parsed_instructions is available in context_pack 
  but not yet used to narrow or prioritize findings.
- **synthesize_feedback.py heuristic path** — before_text and after_text are 
  placeholder strings rather than actual spans from doc_sections.

### API / frontend surface
- No HTTP API layer yet. run.py is CLI only. A FastAPI service layer is needed 
  to handle authenticated file upload, graph invocation, streaming or polling 
  for intermediate state, and returning structured JSON to the frontend.

### Auto-apply workflow

**User identity and profile**
- **eligibility_and_readiness.py** — `_STUB_PROFILE` is a hardcoded dict (name,
  email, age, nationality, location, degree_level, disciplines, gpa,
  uploaded_documents, document_texts). Replace `_get_stub_profile()` with a real
  DB lookup keyed on `state["user_id"]`. `AutoApplyState` has no `user_id` field
  yet; the API layer must inject it at invocation time.
- **eligibility_and_readiness.py** — `uploaded_documents` values are plain
  booleans. Real implementation needs file references (storage keys or URLs) so
  downstream nodes can fetch actual content.
- **eligibility_and_readiness.py** — `document_texts["CV"]` is a hardcoded stub
  string. Replace with content fetched from object storage using the file
  reference stored against the user's profile.

**Opportunity database**
- **load_opportunity.py** — `_fetch_opportunity()` returns one of four hardcoded
  stub dicts (`_STUB_JOB`, `_STUB_MASTERS`, `_STUB_PHD`, `_STUB_SCHOLARSHIP`)
  regardless of `opportunity_id`. Replace with real queries:
  - job → `SELECT * FROM linkedin_jobs WHERE id = %s`
  - masters/phd → `SELECT * FROM programs WHERE id = %s AND program_type = %s`
  - scholarship → `SELECT * FROM scholarships WHERE id = %s`

**Web scraping**
- **scrape_application_page.py** — Uses a generic bot User-Agent string. Production
  scraping will need rotating proxies, session cookies, and handling of JS-rendered
  pages (Playwright or equivalent) for sites that block simple HTTP GET requests.

**Internal submission**
- **submit_internal.py** — `_post_to_backend()` is a no-op stub that logs and
  returns a fake `platform_application_id`. Replace with a real authenticated
  HTTP POST to the platform API. The required fields (endpoint URL, auth headers,
  payload schema) must be defined during backend integration.

**Graph state persistence**
- **graph.py (auto_apply)** — Uses `MemorySaver`. Replace with `AsyncPostgresSaver`
  for durable interrupt/resume across API requests. The API layer must generate
  and persist `thread_id` per workflow run.

**human_gate_0 interrupt/resume**
- **human_gate_0.py** — Currently a stub that terminates the graph. Must be
  replaced with a real `interrupt()` / `Command(resume=...)` cycle matching the
  pattern in `human_gate_1.py`. After the user completes missing profile fields,
  the graph should resume and re-run `eligibility_and_readiness` to confirm the
  gap is closed before continuing.

**Resume value contract quirk**
- **human_gate_1.py, human_gate_2.py** — LangGraph re-interrupts when
  `Command(resume=<falsy value>)` is passed (e.g. empty dict, None). The API
  layer must always send a non-empty dict as the resume value. "Confirm all
  defaults" for gate 1 uses `{"confirm": True}`; approval for gate 2 uses
  `{"approved": True}`.

**Stub DB calls — all four nodes import from the same stub**
- **load_opportunity.py** — `_fetch_opportunity()` ignores `opportunity_id`
  entirely; returns the same hardcoded record for any ID of a given type.
  The not-found error path (`OPPORTUNITY_NOT_FOUND`) is therefore untestable
  with the current stub. Replace with real table queries keyed on both type
  and ID.
- **eligibility_and_readiness.py, asset_mapping.py, application_tailoring.py** —
  all three import `_get_stub_profile()` from `eligibility_and_readiness.py`.
  Replace with a single shared profile-fetching utility keyed on `state["user_id"]`
  once the API layer injects `user_id` into state.

**human_gate_0 is a dead stub**
- **human_gate_0.py** — Does not call `interrupt()`. Returns a plain dict and
  the graph edge goes directly to END. Any workflow where eligibility=`pending`
  terminates immediately instead of suspending for user input. `record_application`
  never runs on this path, so no application is logged. Must be replaced with a
  real `interrupt()` / `Command(resume=...)` cycle matching `human_gate_1.py`,
  followed by a re-run of `eligibility_and_readiness` to confirm the gap is closed.

**Stub profile marks all documents as unavailable**
- **eligibility_and_readiness.py:_STUB_PROFILE** — `uploaded_documents` has
  `Cover Letter: False`, `SOP: False`, `References: False`, etc. This causes
  every opportunity type to trigger `eligibility=pending` and route to the dead
  `human_gate_0`. The full post-eligibility pipeline (asset_mapping → tailoring
  → evaluation → human_gate_2 → submission) is unreachable through the graph
  without patching the stub. Add a testing mode flag or a separate "all docs
  uploaded" fixture profile to make integration testing possible without a real DB.

**AssetMap.requirement_type naming is misleading**
- **asset_mapping.py, schemas.py** — `AssetMap.requirement_type` stores the
  document type name (e.g. "CV", "Cover Letter") not the requirement category
  ("document", "language", "other"). The field should be renamed `document_type`
  to match its actual content. Verify downstream consumers (`human_gate_1.py`,
  `application_tailoring.py`) before renaming, as both key confirmed_mappings
  by this field value.

**Heuristic SOP/Personal Statement produces placeholder text**
- **application_tailoring.py:_heuristic_tailor** — When LLM is unavailable,
  the heuristic generator for SOP and Personal Statement embeds the literal
  string `[Source material summary — real generation requires LLM]` in the
  output. `application_evaluation` correctly flags this as an unfilled placeholder,
  causing the evaluation loop to retry (then proceed at MAX_EVAL_ITERATIONS).
  The final document handed to the user contains this placeholder string.
  Either remove the placeholder and generate minimal coherent prose without an
  LLM, or gate SOP/Personal Statement generation on LLM availability and surface
  a clear message to the user when it is not configured.

**submit_internal is a stub POST**
- **submit_internal.py** — `_post_to_backend()` is a no-op that logs and returns
  a fake `platform_application_id`. Replace with a real authenticated HTTP POST
  to the platform API once the endpoint URL, auth headers, and payload schema
  are defined during backend integration.

**Graph state persistence — auto-apply**
- **graph.py (auto_apply)** — Uses `MemorySaver`. Replace with `AsyncPostgresSaver`
  for durable interrupt/resume across API requests. The API layer must generate
  and persist `thread_id` per workflow run, same as the document feedback workflow.

**No FastAPI service layer for auto-apply**
- No HTTP API layer exists for auto-apply. `run.py` is CLI only. A FastAPI service
  layer is needed to handle authenticated invocation, interrupt/resume across
  requests, streaming or polling for intermediate state (eligibility result,
  asset mapping, tailored documents), and returning structured JSON to the frontend.

