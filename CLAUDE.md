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

The document feedback workflow is the active development focus. Phase 0 
(load, classify, route) is complete. The remaining phases follow the node 
files already scaffolded in nodes/.

## Agent responsibilities (document feedback workflow)

Each node file maps to one of these agents:

**Intake & Classification Agent** (load_document.py, detect_doc_type.py)
Loads and extracts text from uploaded file, validates minimum length, 
classifies document as CV/SOP/COVER_LETTER, routes accordingly.

**Context Assembly Agent** (parse_user_prompt.py + new nodes)
Fetches user profile snapshot, extracts document sections, parses user 
instructions, optionally retrieves opportunity context, builds unified 
context pack for all downstream agents.

**Document Analysis Agent** (analyze_document.py)
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
preserves rejected segments, produces final rewritten document and diff.

## Document feedback workflow orchestration

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
```
### Prompt pattern decision
Prompts live inline inside each node file, not in prompts.py. 
This keeps each prompt next to its logic and is easier to maintain 
at this project scale. prompts.py is unused and can be ignored.

### common/state.py
This file is intentionally reserved as a shared base state for 
future workflows (auto-apply etc.). It is currently empty. 
Do not delete it or use it for document_feedback specific state.

### Human-in-the-loop requirement

Any workflow that can trigger external actions (e.g., submitting an application) **must** include a `human_gate` node before the action. `nodes/human_gate.py` exists as a placeholder. See `HumanGate` in that file for the intended pattern using LangGraph's interrupt mechanism.

### Adding a new workflow

1. Create `src/uppgrad_agentic/workflows/<name>/` mirroring the `document_feedback` layout.
2. Define a `State` TypedDict in `state.py`.
3. Put each node in `nodes/<node_name>.py` with signature `(state: YourState) -> dict`.
4. Wire the graph in `graph.py` with `build_graph() -> CompiledGraph`.
5. Add a `run.py` CLI entry point.

## Implementation status

### Completed
- Phase 0: load_document.py, detect_doc_type.py, end_with_error.py, graph.py routing
- Phase 1: Context Assembly nodes (fetch_profile_snapshot, extract_doc_sections, parse_user_instructions, get_opportunity_context, build_context_pack); state.py and schemas.py extended with all Phase 1+ fields and ChangeProposal/EvaluationResult schemas
- Phase 2: Parallel analysis nodes (analyze_structure, analyze_style, analyze_content_gaps, analyze_ats, analyze_opportunity_alignment); graph wired with LangGraph Send fan-out from build_context_pack, converging at synthesize_feedback
- Phase 3: Synthesis & Planning (synthesize_feedback); graph wired with MemorySaver checkpointer for human_gate interrupt support
- Phase 4: Evaluation loop (evaluate_output); conditional routing back to synthesize_feedback on failure, capped at MAX_EVAL_ITERATIONS=2; synthesize_feedback passes prior evaluation issues to LLM on retry
- Phase 5: Human gate (human_gate); interrupt() suspends graph and surfaces proposals to frontend; resume via Command(resume=decisions) where decisions maps string proposal IDs to {"action": "accept"|"reject", "comment": "..."}
- Phase 6: Rewrite (finalize); applies accepted proposals right-to-left to preserve positions, resolves overlapping spans by confidence, LLM coherence smoothing pass, produces diff summary; result.details contains final_document and diff for frontend

### In progress
- Nothing currently in progress

### Not started

- Auto-apply workflow (not started)

### Known issues
- common/state.py is intentionally a shared base for future workflows, currently empty, do not delete

## Integration TODO

Items that are currently stubbed, hardcoded, or mocked and must be replaced during
backend / frontend / database integration. Grouped by area.

### Authentication and user identity

- **`state.py`** — No `user_id` field in `DocFeedbackState`. The API layer must inject
  the authenticated user's ID into state at invocation time so nodes can scope DB queries.
- **`fetch_profile_snapshot.py`** — Returns a hardcoded `_STUB_PROFILE` dict (user
  "Alex Johnson"). Replace with a real DB lookup keyed on `state["user_id"]`.

### Opportunity context

- **`get_opportunity_context.py`** — Returns a hardcoded `_MOCK_OPPORTUNITY` whenever
  the user's instructions contain job-related keywords. Real implementation must:
  - Accept a structured opportunity input from the frontend (job URL, paste of JD, or
    saved opportunity ID) rather than inferring intent from free text.
  - Look up or scrape/parse the opportunity and return structured data
    (title, org, description, requirements, keywords).

### File ingestion and storage

- **`load_document.py`** — Reads the file from a local filesystem path via
  `file.get("path")`. In production the file will arrive as a multipart upload or
  be stored in object storage (S3/GCS). Options: materialise bytes to a temp path in
  the API layer before invoking the graph, or extend `extract_text_from_file` to
  accept `bytes` directly (the `bytes` field already exists in `FileInput` but is
  unused).
- **`tools/documents.py`** — Several known extraction gaps to address:
  - DOCX: tables and images are silently ignored; `page_count` is always `None`.
  - PDF: scanned / image-based pages return empty text with no OCR fallback. Wire in
    an OCR library (e.g. pytesseract, AWS Textract, Google Document AI) for scanned docs.

### Graph state persistence and checkpointing

- **`graph.py`** — `MemorySaver` is an in-process, non-durable checkpointer. Replace
  with `AsyncPostgresSaver` (from `langgraph-checkpoint-postgres`) pointing at the
  production database for persistence across API requests and process restarts.
- **`run.py`** — The CLI calls `graph.invoke(...)` without a `config` dict. Now that a
  checkpointer is active, every invocation requires
  `config={"configurable": {"thread_id": "<unique-run-id>"}}` or LangGraph will
  generate a new thread on every call. The API layer must generate and store thread IDs
  so interrupted runs (human gate) can be resumed.

### LLM provider and configuration

- **`common/llm.py`** — Only OpenAI (`langchain-openai`) is wired up. Add support for
  other providers as needed (Anthropic, Azure OpenAI, etc.).
- **`config/settings.py`** — Currently an empty stub. Wire in a real settings module
  (e.g. pydantic-settings reading from environment / secrets manager) and route all
  configuration through it instead of scattered `os.getenv` calls.

### Analysis quality

- **`analyze_ats.py`** — `_ATS_KEYWORDS` is a static, role-agnostic list. It should be
  supplemented or replaced by keywords extracted from `opportunity_context.keywords`
  (already in `context_pack`) and the user's `target_roles` from the profile snapshot.
- **All analysis nodes** — `parsed_instructions` (including `focus_areas`,
  `target_role`, and `explicit_constraints`) is assembled in `context_pack` and passed
  to every node, but none of the analysis nodes currently read it to narrow or
  prioritise their output. Each node should filter or weight findings based on the
  user's stated intent and constraints.
- **`synthesize_feedback.py` (heuristic path)** — `before_text` and `after_text` for
  most heuristic proposals are placeholder strings (e.g. `"[Current Skills section]"`)
  rather than actual text spans extracted from `doc_sections`. The LLM path handles
  this correctly. If the heuristic path needs to remain accurate without an LLM, each
  proposal generator should locate and quote the relevant span from `doc_sections`.

### Not-yet-implemented nodes (empty stubs)

- **`run.py`** — No `thread_id` passed to `graph.invoke()`; required now that checkpointer is active. Callers must pass `config={"configurable": {"thread_id": "<id>"}}` for interrupt/resume to work correctly.

### API / frontend surface

- **No HTTP API layer yet.** `run.py` is a CLI-only entry point. A FastAPI (or
  equivalent) service layer is needed to handle: authenticated file upload, graph
  invocation with a scoped thread ID, streaming or polling for intermediate state,
  and returning the final `result` / `proposals` as structured JSON to the frontend.



