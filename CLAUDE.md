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
- Phase 0: load_document.py, detect_doc_type.py, end_with_error.py, graph.py routing (stub endpoints only)

### In progress
- Nothing currently in progress

### Not started
- Phase 1: Context Assembly nodes (fetch_profile_snapshot, extract_doc_sections, parse_user_instructions, get_opportunity_context, build_context_pack)
- Phase 2: Parallel analysis nodes (analyze_structure, analyze_style, analyze_content_gaps, analyze_ats, analyze_opportunity_alignment)
- Phase 3: Synthesis & Planning (synthesize_feedback)
- Phase 4: Evaluation loop (evaluate_output, iteration counter)
- Phase 5: Human gate (human_gate with LangGraph interrupt)
- Phase 6: Rewrite (finalize)
- Auto-apply workflow (not started)

### Known issues to address before Phase 1
- state.py missing Phase 1+ fields: context_pack, parsed_instructions, proposals, iteration_count, final_document
- schemas.py missing ChangeProposal and evaluation output schemas
- common/state.py is intentionally a shared base for future workflows, currently empty, do not delete



