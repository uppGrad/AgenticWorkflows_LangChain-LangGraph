# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

UppGrad is an AI-powered platform helping students find and apply to jobs, graduate programs, and scholarships. Two core agentic workflows: document feedback (analyzes CVs/SOPs/cover letters, proposes reviewable edits) and auto-apply (assesses eligibility, generates tailored materials, attempts submission). Both use human-in-the-loop approval before any consequential action.

## Commands

This project uses [uv](https://github.com/astral-sh/uv) for environment and dependency management.

```bash
# Install dependencies
uv sync

# Run the document feedback workflow against a file
uv run python -m uppgrad_agentic.workflows.document_feedback.run --file path/to/cv.pdf --instructions "Focus on clarity"

# Run tests
uv run pytest

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

## Architecture

### Package layout

```
src/uppgrad_agentic/
  common/          # Shared utilities: LLM factory (llm.py), logging, guardrails, error types
  config/          # Settings (currently stub)
  tools/           # File-level tools: documents.py (PDF/DOCX/TXT extraction), opportunities.py
  workflows/
    document_feedback/
      state.py     # DocFeedbackState TypedDict — source of truth for graph state
      schemas.py   # Pydantic models for LLM structured output
      graph.py     # build_graph() — assembles and compiles the LangGraph StateGraph
      run.py       # CLI entry point
      nodes/       # One file per node function
    auto_apply/
      state.py     # AutoApplyState TypedDict
      schemas.py   # Pydantic models (NormalizedRequirement, AssetMap, EligibilityResult, etc.)
      graph.py     # build_graph()
      run.py       # CLI entry point
      nodes/       # One file per node function
```

### General patterns

- **State pattern**: each node receives the full State TypedDict and returns a partial dict of keys to merge. Errors are signalled by setting `result.status = "error"` in state. Downstream nodes check this and short-circuit with `return {}`.
- **LLM pattern**: `get_llm()` in `common/llm.py` returns `None` when no provider is configured. Every node that calls an LLM must handle the `None` case with a heuristic fallback. See `detect_doc_type.py` for the reference implementation.
- **Prompt pattern**: prompts live inline inside each node file, not in prompts.py. prompts.py is unused and can be ignored.
- **Human-in-the-loop**: any workflow that triggers external actions must include a human_gate node using LangGraph interrupt() before the action.
- **Checkpointer**: MemorySaver is used for now. Will be replaced with AsyncPostgresSaver during backend integration.

### common/state.py
Intentionally reserved as a shared base state for future workflows. Currently empty. Do not delete or use for workflow-specific state.

### Adding a new workflow
1. Create `src/uppgrad_agentic/workflows/<name>/` mirroring the `document_feedback` layout.
2. Define a `State` TypedDict in `state.py`.
3. Put each node in `nodes/<node_name>.py` with signature `(state: YourState) -> dict`.
4. Wire the graph in `graph.py` with `build_graph() -> CompiledGraph`.
5. Add a `run.py` CLI entry point.

---

## Document Feedback Workflow

### Agent responsibilities

**Intake & Classification** (load_document.py, detect_doc_type.py) — Loads and extracts text from the uploaded file, validates minimum length, classifies as CV/SOP/COVER_LETTER, and routes accordingly.

**Context Assembly** (fetch_profile_snapshot.py, extract_doc_sections.py, parse_user_instructions.py, get_opportunity_context.py, build_context_pack.py) — Fetches user profile, extracts doc sections, parses instructions, optionally retrieves opportunity context, and assembles a unified context pack for all downstream agents.

**Document Analysis** (analyze_structure.py, analyze_style.py, analyze_content_gaps.py, analyze_ats.py, analyze_opportunity_alignment.py) — Runs parallel analyses (structure, style/grammar, content gaps, ATS compatibility for CVs, opportunity alignment if context provided); all parameterized by doc_type.

**Synthesis & Planning** (synthesize_feedback.py) — Merges parallel analysis outputs and generates a structured ChangeProposal list (section, rationale, before/after text, confidence, confirmation flag).

**Evaluation** (evaluate_output.py) — Checks proposals for groundedness, hallucinations, and format compliance; triggers refinement loop capped at 2 iterations.

**Human Review Coordinator** (human_gate.py) — LangGraph interrupt point; presents proposals as a reviewable checklist and holds the workflow until explicit user approval.

**Rewrite** (finalize.py) — Applies approved edits right-to-left, resolves overlapping spans by confidence, runs coherence smoothing, and produces the final document and diff.

### Orchestration

```
START → load_document → detect_doc_type
→ [route: cv_route / sop_route / cover_route / end_with_error]
→ fetch_profile_snapshot → extract_doc_sections → parse_user_instructions
→ get_opportunity_context → build_context_pack
→ [Send fan-out] analyze_structure, analyze_style, analyze_content_gaps,
                 analyze_ats (CV only), analyze_opportunity_alignment (if opportunity)
→ synthesize_feedback → evaluate_output
→ [loop to synthesize_feedback if failed, max 2] → human_gate → finalize → END
```

Key orchestration rules:
- Parallel analysis nodes fan out from build_context_pack via LangGraph `Send`; all five merge into synthesize_feedback
- All three doc types share the same nodes after routing; doc_type in state parameterizes behavior inside each node

---

## Auto-Apply Workflow

### Opportunity database tables

**linkedin_jobs** (jobs) — Key columns: id, title, company, location, description, is_remote, is_closed, url, url_direct, company_url, employer_id (NULL=external, 1=internal UppGrad job), posted_time, salary.

**programs** (masters/phd) — Key columns: id, url, title, university, location, degree_type, study_mode, tuition_fee. `data` (json): description, requirements (academic/english/other), curriculum, funding, start_dates — primary eligibility source.

**scholarships** — Key columns: id, url, title, provider_name, disciplines, deadline, coverage, eligibility_text, req_disciplines, req_locations, req_nationality, req_age. `data` (json): all of the above in structured form — use as primary source.

### Agent responsibilities

**Opportunity Intelligence** (load_opportunity.py, discover_apply_url.py, scrape_application_page.py, evaluate_scrape.py, determine_requirements.py) — Loads the opportunity record and, for jobs, runs the 3-tier URL discovery pipeline (Brave search + httpx + optional Crawl4AI) then normalizes requirements. Programs and scholarships skip scraping and parse requirements from the `data` json field directly.

**Eligibility and Readiness** (eligibility_and_readiness.py) — Hard-blocks only on deadline-passed or missing user-supplied (non-generatable) documents; location/age/degree-level/nationality issues become non-blocking compatibility warnings surfaced in the handoff package. Triggers human_gate_0 when blocking fields are missing.

**Asset Mapping** (asset_mapping.py) — Maps each normalized requirement to the best available user document and assigns tailoring depth: none | light | deep | generate.

**Application Tailoring** (application_tailoring.py) — Generates a tailored version of each required document in apply-mode (changes applied directly, no proposal review). Output capped per doc type: CV ≤ 8000, Cover Letter ≤ 3000, SOP/Personal Statement ≤ 6000 chars.

**Application Evaluation** (application_evaluation.py) — Verifies the full package for groundedness, requirement coverage, and placeholder text; triggers refinement loop capped at 2 iterations.

**Human Review Coordinator** (human_gate_0.py, human_gate_1.py, human_gate_2.py) — Gate 0 (profile completion), Gate 1 (document mapping review), Gate 2 (final package approval). All use LangGraph interrupt(); gate 0 loops back to eligibility_and_readiness after resume, capped at 2 retries.

**Submission** (submit_internal.py, package_and_handoff.py, record_application.py) — Internal jobs (employer_id=1) record submission intent for the backend ORM write; all other types assemble a handoff package including compatibility warnings and posting_closed flag. Both paths call record_application.

### Orchestration

```
START → load_opportunity
→ [job] discover_apply_url → scrape_application_page → evaluate_scrape → determine_requirements
→ [masters/phd/scholarship] determine_requirements (parse data json, no scraping)
→ eligibility_and_readiness
  → ineligible: end_with_explanation → END
  → pending: human_gate_0 → eligibility_and_readiness (loop, cap 2) or end_with_error
  → ready: asset_mapping → human_gate_1 → application_tailoring → application_evaluation
    → [failed and count < 2] loop to application_tailoring
    → human_gate_2
      → [approved, employer_id=1] submit_internal → record_application → END
      → [approved, other] package_and_handoff → record_application → END
      → [rejected] end_with_error → END
```

Key orchestration rules:
- Internal vs external routing uses `employer_id` from linkedin_jobs — **not** the `site` column
- Never block on a failed scrape; always degrade gracefully to assumed default requirements
- `url_direct` takes precedence over `url` for scraping; discovery uses it as a cache-hit short-circuit (skips Brave search entirely)
- After gate 0 resume, the backend adapter must inject a fresh `profile_snapshot` via `graph.update_state()` before resuming or the gate re-fires on stale data (see backend integration bug #2)

### Future implementation steps (post integration)

**Requirements review gate** — After determine_requirements, surface the normalized requirements list to the user as a reviewable checklist before asset mapping. User can flag items they can't provide; those are noted in the final package rather than blocking.

**External application form submission** — For external jobs where discovery succeeded with high confidence, attempt to auto-fill and submit via browser automation. Scrape confidence and field structure are already stored in the application record to ease this addition later.

### Scraping status

Discovery v2 (shipped in `feature/discovery-v2`) replaced the old `requests.get()` path. Current architecture:
1. `discover_apply_url` runs a 3-tier Brave search (ATS → company careers → generic) with multi-factor verification (title fuzzy ≥85 + ≥2 corroborators for ATS/generic, ≥1 for careers-tier with domain proof).
2. `scrape_application_page` consumes the verified page content from discovery (single-fetch — no re-fetch of the same URL).
3. `evaluate_scrape` + `determine_requirements` normalize scraped content or fall back to assumed defaults.

With `UPPGRAD_BROWSER_SCRAPE_ENABLED=false` (current prod default), Workday-hosted careers pages return thin/JS-shell responses and discovery degrades gracefully to assumed defaults. Enabling the browser fallback requires Railway Chromium setup (~1GB per instance); the env gate and lazy Crawl4AI import are already in place in `tools/web_fetcher.py`.

---

## Implementation Status

### Completed

**Document Feedback Workflow** — fully implemented, smoke tested across all three doc types (CV, SOP, Cover Letter), bug audit completed.
- All 6 phases: load/detect → context assembly → parallel analysis (LangGraph Send fan-out) → synthesize → evaluation loop (MAX_EVAL_ITERATIONS=2) → human gate → finalize
- Bug fix: `json.dumps` crash in `run.py` when graph suspends at `human_gate` (Interrupt object not JSON serializable); fixed with `default=str`
- Frontend progress tracking: `current_step` + `step_history` on all 17 nodes. The 5 parallel analysis nodes set only `step_history` (concurrent writes would conflict); `build_context_pack` sets `current_step="parallel_analysis"` as the fan-out indicator.

**Auto-Apply Workflow** — fully implemented end-to-end, smoke tested across all four opportunity types, all conditional edge routings verified.
- Full pipeline: opportunity intelligence (with discovery v2) → eligibility → asset mapping → tailoring → evaluation → human review → submission
- All three human gates use real interrupt/resume cycles; gate 0 iteration cap (2 retries) wired in graph
- `resolve_profile(state)` in `_profile.py` returns injected `profile_snapshot` or falls back to in-repo stub with a WARNING log
- `cancel_session(thread_id, checkpointer)` in `control.py` writes `CANCELLED` marker via `graph.update_state()`; existing short-circuit pattern drains the graph to END

**Backend Integration (2026-04-26)** — Django adapter in `backend/ai_services/` drives auto_apply end-to-end against Neon. No FastAPI layer needed in the agentic repo.
- `auto_apply_adapter.py`: `start_session`, `resume_session_gate_{0,1,2}`, `finalize_internal_submission`, `cancel_session`
- `ApplicationSession` model with partial unique constraint (one active session per student+opportunity)
- Per-status TTL janitor: PROCESSING/15m, FINALIZING/30m, AWAITING_*/7 days
- ReportLab text→PDF renderer for internal-submit resume files

**Discovery v2 (2026-04-27)** — Replaced old requests.get() scrape with 3-tier Brave search + httpx + optional Crawl4AI. See Scraping Status section above.

### In Progress
- Nothing

### Not Started
- Nothing

---

## Integration TODO

Items currently stubbed, hardcoded, or mocked that must be replaced during backend / frontend / database integration.

### Document feedback

- **No `user_id` in DocFeedbackState** — API layer must inject authenticated user ID at invocation time; `fetch_profile_snapshot.py` and `get_opportunity_context.py` are both hardcoded stubs keyed on nothing.
- **`tools/documents.py`** — DOCX tables and images are silently ignored; PDF scanned pages return empty text with no OCR fallback. Wire in OCR for scanned docs — this is a silent data-loss path.
- **`run.py` passes no `thread_id`** — API layer must generate and store thread IDs for interrupt/resume to work across requests. MemorySaver must also be replaced with AsyncPostgresSaver.
- **`analyze_ats.py`** — uses a static keyword list. Should pull keywords from `opportunity_context` and user `target_roles`.
- **All analysis nodes** — `parsed_instructions` is available in `context_pack` but not yet used to narrow or prioritize findings.
- **`synthesize_feedback.py` heuristic path** — `before_text` and `after_text` are placeholder strings, not actual spans from `doc_sections`.
- **No HTTP API** — `run.py` is CLI only. A service layer is needed for authenticated file upload, graph invocation, streaming/polling, and structured JSON responses.

### Auto-apply

- **Stub profile** (`eligibility_and_readiness.py:_STUB_PROFILE`) — returns a hardcoded dict with `uploaded_documents` all `False` except CV. This causes every opportunity to trigger `eligibility=pending` and gate at human_gate_0, making the post-eligibility pipeline unreachable in testing without patching the stub. Replace with real DB lookup keyed on `state["user_id"]`; `uploaded_documents` values must be file references (storage keys/URLs), not booleans.
- **`load_opportunity.py`** — `_fetch_opportunity()` ignores `opportunity_id` and returns the same stub dict for any ID of a given type. The `OPPORTUNITY_NOT_FOUND` error path is therefore untestable. Replace with real DB queries: job→`linkedin_jobs`, masters/phd→`programs`, scholarship→`scholarships`.
- **Resume value contract quirk** — LangGraph re-interrupts when `Command(resume=<falsy value>)` is passed (e.g. empty dict, None). API layer must always send a non-empty dict: gate 1 confirmation uses `{"confirm": True}`, gate 2 approval uses `{"approved": True}`.
- **`AssetMap.requirement_type`** stores the document type name (e.g. "CV"), not the requirement category. Should be renamed `document_type`. Verify `human_gate_1.py` and `application_tailoring.py` (both key `confirmed_mappings` by this field value) before renaming.
- **Heuristic SOP/Personal Statement** — `application_tailoring.py:_heuristic_tailor` embeds the literal string `[Source material summary — real generation requires LLM]` when LLM is unavailable. `application_evaluation` correctly flags it, exhausting the retry loop before the placeholder reaches the user. Either generate minimal coherent prose without an LLM or gate SOP generation on LLM availability.
- **`web_fetcher.py`** — uses a bot User-Agent string (`UppGrad-Bot/1.0`). Production scraping will need realistic browser headers, rotating proxies, and session cookies for sites that block bots at the HTTP level (separate concern from the Crawl4AI browser fallback, which handles JS rendering).
- **`graph.py (auto_apply)`** — uses MemorySaver; replace with AsyncPostgresSaver. API layer must generate and persist `thread_id` per run.
- **`submit_internal.py`** — records intent only; the backend ORM write lives in `auto_apply_adapter.finalize_internal_submission`. No HTTP loopback from agentic → backend.

---

## 2026-04-26 — Backend Integration (auto-apply)

**Spec/plan refs:** `docs/superpowers/specs/2026-04-26-auto-apply-backend-integration.md`, `docs/superpowers/plans/2026-04-26-auto-apply-backend-integration.md`

The backend (`backend/ai_services`) drives `auto_apply` end-to-end against Neon via the Django adapter pattern. The agentic repo needs no FastAPI layer of its own.

### Key bugs found during live integration (fixes in backend repo commit `417e614`)

1. **Gate-suspension detection lagged.** Checking `current_step` to find which gate is held is wrong — `interrupt()` suspends *before* the node returns, so `current_step` reflects the previous node. Fix: read the pending node from `graph.get_state(config).next`. Implemented in `_pending_node_after_invoke()`.
2. **Gate 0 resume kept stale `profile_snapshot`.** After the user uploaded missing docs, eligibility re-ran on the original snapshot and re-fired the gate. Fix: `resume_session_gate_0` builds a fresh snapshot and injects it via `graph.update_state(config, {"profile_snapshot": fresh})` before resuming.
3. **`INELIGIBLE` error_code unmapped.** Backend persist logic only recognised `CANCELLED` and `PROFILE_INCOMPLETE_AFTER_RETRIES`; `INELIGIBLE` fell through to generic `ERROR`. Fix: add explicit mapping.
4. **Hard-block ineligibility design (open issue).** Location/age/degree-level checks hard-terminated the workflow. These are now compatibility warnings in the eligibility node (fixed in discovery v2), but the backend display layer still needs to surface them on the apply screen.
5. **Eligibility blocked on generatable docs (fixed in `47753d1`).** `_check_profile_completeness` flagged Cover Letter, SOP, etc. as `pending` even though the system writes them. Fix: skip docs listed in `_GENERATABLE` (defined in `asset_mapping.py`). Generatable (do NOT block at gate 0): Cover Letter, SOP, Personal Statement, Research Proposal, Writing Sample, Motivation Letter, References. User-supplied (still block at gate 0): Transcript, English Proficiency Test.

### Verification (live Neon dev, project `summer-math-90128942`)

| Path | Result |
|---|---|
| External job → completed_handoff | ✅ (scrape failed → assumed defaults) |
| External job with real Koray CV + gpt-4o-mini | ✅ (16 graph steps, properly formatted) |
| Internal job (employer_id=1) → completed_submitted | ✅ (real Application row via ORM) |
| Gate 0 retry loop | ✅ (fresh snapshot → eligibility re-ran → ready) |
| Cancel mid-flight | ✅ (CANCELLED marker in PostgresSaver) |
| Hard-ineligibility path | ✅ (INELIGIBLE status after fix #3) |

### Future work

**Critical:**
- **Browser fallback deployment.** Crawl4AI/Playwright path is env-gated but Railway Chromium setup (~1GB per instance) is not done. Workday-hosted jobs fail discovery until this ships.
- **LinkedIn ghost-posting detection upstream.** Fix in scraper repo (`bitirme/linkedin_jobspy/scraper.py`): detect apply-flow type at ingestion, store as `apply_type`, filter tracker-only jobs out of auto-apply candidates. Until then, consider negative caching (7-day TTL on `failed` discoveries).

**Important but not blocking:**
- Gate 1 `additional_uploads`: serializer accepts files but adapter doesn't save or thread them into resume payload. Users can't upload a fresh doc at gate 1 today.
- Gate 2 edit-and-re-tailor loop: v1 hard-cancels on rejection. A loop back to `application_tailoring` with per-document feedback would let users iterate without starting fresh.
- Polymorphic `StudentDocument` store (Spec §11.3): Cover Letter/SOP are always generated; a generic table would let users save and reuse them.
- External form auto-submission (Spec §11.2): browser automation for high-confidence discovery hits. Would create real `Application` rows for external jobs.
- Cooperative cancel inside nodes: cancel marker is observed only at the next node boundary — worst case ~30-60s of LLM cost wasted per cancellation.
- Per-session LLM cost budgeting (Spec T3): worst-case ~15-20 LLM calls per session; no token cap today.
- GDPR/retention: tailored documents live forever in ApplicationSession rows + PostgresSaver checkpoints.

---

## 2026-04-27 — Discovery v2 + Eligibility Cleanup

**Plan refs:** `docs/superpowers/plans/2026-04-26-discovery-v2.md` (supersedes `2026-04-26-apply-url-discovery.md`)

### What shipped

**`tools/search.py`** — `SearchProvider` ABC + `BraveSearchProvider`. Opt-in via `UPPGRAD_SEARCH_PROVIDER=brave` + `BRAVE_SEARCH_API_KEY`. Mirrors `get_llm()` factory pattern.

**`tools/web_fetcher.py`** — httpx-first fetch with optional Crawl4AI browser fallback (`UPPGRAD_BROWSER_SCRAPE_ENABLED=true`). `_detect_thin` uses strong multi-word phrases (single hit = thin) + weak word-boundary tokens (≥2 hits = thin) to avoid false positives on legitimate Greenhouse pages that incidentally contain "404"/"captcha" in JS paths.

**`tools/url_discovery.py`** — 3-tier orchestration with multi-factor verification:
- Title fuzzy ≥85 is a hard prerequisite; ATS slug must also match the queried company (prevents same-title/different-company false positives — see bug #1 below)
- Corroborators: {company-in-text, location-match, posted-time-match, description-keyword-overlap}; ATS/generic need 2, careers-tier needs 1 (site: constraint proves company)
- Thin pages rejected at the verification gate, not during scrape (prevents thin-detector conflict — see bug #3)
- `DiscoveryResult.text` carries verified content forward; `scrape_application_page` consumes it without re-fetching

**`nodes/discover_apply_url.py`** — new node between `load_opportunity` and `scrape_application_page` for jobs only; skipped for internal jobs. Honors backend cache-hit via pre-populated state fields.

**Eligibility refactor** — Hard-blocks only for deadline-passed + missing user-supplied (non-generatable) docs. Location/age/degree-level/nationality are now `compatibility_warnings` in state, surfaced into `application_package.warnings`.

**Closed-posting detection** — Discovery returns `method='closed'` + `posting_closed=True` when a verified page contains phrases like "no longer accepting applications". The handoff package includes a user-facing warning.

**Backend additions** — `JobApplyUrlDiscovery` model (cross-user URL cache, 14-day staleness gate); `_lookup_discovery_cache` + `_persist_discovery_to_cache` in adapter (skips `failed` and `skipped_internal`); `ApplicationSession.compatibility_warnings` JSONField.

### Bugs detected during live testing (and fixed)

1. **False-positive on title+company alone.** Celonis CVP (Schwyz, Switzerland) matched a Greenhouse URL for the same role in Cleveland, Ohio purely on title fuzz + same-company. Fixed by requiring ≥2 corroborators AND an ATS slug match.
2. **Double httpx fetch.** Verification fetched a candidate; `scrape_application_page` re-fetched the same URL. Fixed by propagating verified text through state.
3. **Thin-detector second-look conflict.** `_detect_thin` could veto pages that verification had already accepted (Greenhouse pages contain "404"/"captcha" in JS paths/hidden fields). Fixed by rejecting thin pages only at the verification gate.
4. **`site:linkedin.com` careers tier waste.** `company_url` in linkedin_jobs is usually a LinkedIn company-page URL. Fixed via `_CAREERS_DOMAIN_BLOCKLIST`.

### Live verification results

| Job | Location | Result | Cross-check |
|---|---|---|---|
| Celonis CVP (id 202599) | Schwyz, Switzerland | `failed` | ✅ Greenhouse only had Cleveland role for that title |
| GitHub Senior SE (id 199838) | Germany | `failed` | ✅ GitHub uses Workday (needs browser fallback) |
| Anthropic SA Munich (id 200082) | Munich | `failed` | ✅ Greenhouse had Paris/NYC only; confirmed ghost posting |

Cache rows after 3 failed runs: 0 ✅ (cache only stores successful matches)

### Future work

- **Browser fallback in prod** (critical) — see Backend Integration future work above.
- **Negative caching for `failed` discoveries** — 7-day TTL to save Brave budget on repeat ghost-posting attempts. Risk: misses companies that post to Greenhouse days later, but acceptable for long-tail ghost cases.
- **UI signal for ghost postings** — surface "couldn't find an external apply page" when `url_direct` is empty and discovery returns `failed`.
- **JSON-LD structured data** — Greenhouse pages have `<script type="application/ld+json">` with title/location; parsing these would be more robust than regex on body text.
