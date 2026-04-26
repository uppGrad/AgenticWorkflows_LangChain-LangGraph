# Auto-Apply Backend Integration Spec

**Status:** Draft for review — not yet a plan.
**Date:** 2026-04-26
**Scope:** Wire the `auto_apply` LangGraph workflow into the Django backend (`backend/`) so users can run it end-to-end against real DB data, real profile data, and real document storage. Mirrors the pattern already established for `document_feedback` in `backend/ai_services/graph_adapter.py`.

This is a **specification**, not an implementation plan. It locks in architecture, contracts, and known trade-offs. An implementation plan follows in a separate doc once this spec is reviewed.

---

## 1. Goal

Replace every stub the auto-apply workflow currently relies on (`_fetch_opportunity`, `_get_stub_profile`, `MemorySaver`, `human_gate_0` dead stub, `submit_internal._post_to_backend`) with real integrations against the existing Django backend, and expose the workflow through DRF endpoints structured like the existing `FeedbackSession` views.

After this spec is implemented, a student should be able to:

1. Pick any opportunity (job / masters / phd / scholarship) from the platform.
2. Click "Apply with UppGrad" to start an `ApplicationSession`.
3. Watch progress through three human gates (profile completion, document mapping, final approval).
4. Receive either:
   - A submitted `Application` row recorded against the platform (internal jobs only, today; external auto-apply later as a v2 hook).
   - An in-app **handoff package** (every other path: external jobs, masters, phd, scholarship, and any internal path that didn't actually submit).

Anything beyond this — discovery, external form submission, multi-region splits, polymorphic document storage, OCR for non-CV docs — is explicitly out of scope and listed in §11.

---

## 2. Architectural decisions (locked)

These were resolved in pre-spec brainstorming. Not up for re-litigation in implementation:

| ID | Decision |
|---|---|
| **A1** | Backend pre-loads opportunity data into initial graph state (mirrors `document_feedback`'s `build_opportunity_context`). Agentic repo's `_fetch_opportunity` stub stays for CLI mode and is never reached when invoked from the backend. |
| **A2** | New Django model `ApplicationSession` (sibling to `FeedbackSession`). One active session per `(student, opportunity)`. |
| **A3** | `Application` rows are **created server-side via Django ORM** by the adapter on completion of an internal-submission path. No HTTP loopback from agentic → backend. |
| **A4** | All non-submission terminations end in an **in-app handoff package** rendered by the frontend. This is the universal terminal for masters / phd / scholarship / external jobs / internal jobs that bailed before submission. |
| **A5** | `ApplicationSession.status` enum has two distinct success terminals: `COMPLETED_SUBMITTED` (internal submit) and `COMPLETED_HANDOFF` (everything else). Frontend renders different screens for each. |
| **A6** | `discovered_apply_url`, `discovery_method`, `discovery_confidence` are added to `AutoApplyState` and `ApplicationSession` schema **now**, even though discovery is deferred. The discovery node is wired as a no-op pass-through; future discovery work doesn't need a schema migration. |
| **A7** | Cancel uses `graph.update_state()` to inject `result.status = "error"` out-of-band. Existing error-propagation pattern drains the graph to END. The currently-executing node finishes; subsequent nodes short-circuit. The agentic repo exposes a `cancel_session(thread_id, checkpointer)` helper; the backend's cancel view calls it. |
| **A8** | Auto-apply integration follows the existing `document_feedback` integration pattern verbatim: PostgresSaver checkpointer, `psycopg.connect(autocommit=True)`, background `threading.Thread` per phase, `django_db.close_old_connections()` at thread boundaries, polling-based status reads from the frontend. |

---

## 3. Data model

### 3.1 `ApplicationSession` (new)

```python
class ApplicationSession(models.Model):
    """Tracks an agentic auto-apply workflow run."""

    class Status(models.TextChoices):
        PROCESSING                 = "processing", "Processing"
        AWAITING_PROFILE_COMPLETION = "awaiting_profile_completion", "Awaiting Profile Completion"   # gate 0
        AWAITING_DOCUMENT_MAPPING   = "awaiting_document_mapping",   "Awaiting Document Mapping"     # gate 1
        AWAITING_FINAL_APPROVAL     = "awaiting_final_approval",     "Awaiting Final Approval"       # gate 2
        FINALIZING                  = "finalizing",                  "Finalizing"
        COMPLETED_SUBMITTED         = "completed_submitted",         "Completed (Submitted)"
        COMPLETED_HANDOFF           = "completed_handoff",           "Completed (Handoff)"
        INELIGIBLE                  = "ineligible",                  "Ineligible"
        ERROR                       = "error",                       "Error"
        CANCELLED                   = "cancelled",                   "Cancelled"

    OPPORTUNITY_TYPE_CHOICES = [
        ("job", "Job"), ("masters", "Masters"), ("phd", "PhD"), ("scholarship", "Scholarship"),
    ]

    student          = models.ForeignKey(Student, on_delete=models.CASCADE, related_name="application_sessions")
    thread_id        = models.UUIDField(unique=True)
    status           = models.CharField(max_length=40, choices=Status.choices, default=Status.PROCESSING)

    # Inputs
    opportunity_type = models.CharField(max_length=20, choices=OPPORTUNITY_TYPE_CHOICES)
    opportunity_id   = models.BigIntegerField()

    # Snapshot of opportunity at session start (frozen — see §6.4)
    opportunity_snapshot = models.JSONField()

    # Eligibility outcome (gate 0 trigger)
    eligibility_result   = models.JSONField(null=True, blank=True)   # decision, reasons, missing_fields

    # Asset mapping (gate 1 input)
    asset_mapping        = models.JSONField(null=True, blank=True)   # list of AssetMap dicts

    # User decisions captured at each gate
    gate_0_response      = models.JSONField(null=True, blank=True)
    gate_1_response      = models.JSONField(null=True, blank=True)
    gate_2_response      = models.JSONField(null=True, blank=True)

    # Tailored document outputs
    tailored_documents   = models.JSONField(null=True, blank=True)   # {doc_type: {content, tailoring_depth, char_count}}

    # Final package (handoff or submission record)
    application_package  = models.JSONField(null=True, blank=True)
    submission_type      = models.CharField(max_length=20, blank=True, default="")   # 'internal' | 'handoff'

    # Discovery metadata (added now, populated when discovery ships — see A6)
    discovered_apply_url   = models.TextField(blank=True, default="")
    discovery_method       = models.CharField(max_length=20, blank=True, default="")
    discovery_confidence   = models.FloatField(null=True, blank=True)

    # Outcome / error
    application_record   = models.JSONField(null=True, blank=True)
    error_message        = models.TextField(blank=True, default="")

    # Frontend progress indicator (mirrors AutoApplyState.current_step)
    current_step         = models.CharField(max_length=50, blank=True, default="")
    step_history         = models.JSONField(default=list, blank=True)

    # Link to the resulting Application row (only when submission_type == 'internal')
    application          = models.ForeignKey(
        "jobs.Application", on_delete=models.SET_NULL, null=True, blank=True,
        related_name="application_session",
    )

    created_at   = models.DateTimeField(auto_now_add=True)
    updated_at   = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["student", "opportunity_type", "opportunity_id"],
                condition=models.Q(status__in=[
                    "processing", "awaiting_profile_completion",
                    "awaiting_document_mapping", "awaiting_final_approval", "finalizing",
                ]),
                name="one_active_application_session_per_student_opportunity",
            ),
        ]
```

Notes:
- The unique constraint is **partial** (only on active statuses). A student can apply, finish, and start a fresh session against the same opportunity if they want to re-tailor. Whether the UI exposes this is a frontend choice.
- `opportunity_snapshot` is the output of the adapter's pre-load step (A1). It's the dict the graph runs against. It's frozen at session start; cf. §6.4 for the freshness re-check.
- `gate_*_response` fields hold the raw resume payload the user submitted. Useful for audit and debugging; not part of the public API response.
- The `application` FK is `SET_NULL` so we don't lose `ApplicationSession` history if an `Application` is later deleted.

### 3.2 `StudentDocument` (new — see §11.3)

**Out of scope for this spec** (would expand to a polymorphic document store). For v1 we cap auto-apply to opportunities whose required-document set is reachable from `StudentCV` only. Cover Letter / SOP / Personal Statement / Portfolio are *generated* (tailoring depth = `generate`) using the CV + profile as source material, never read from a stored user file. Open question §10.Q1.

### 3.3 `Application` (existing, no changes)

`Application` is in `backend/jobs/models.py:123`. The adapter writes a row with:
- `student = session.student`
- `job = JobOpportunity.objects.get(id=session.opportunity_id)`
- `cover_letter = tailored_documents["Cover Letter"]["content"]`
- `resume_file = <PDF-rendered tailored CV, see §6.5>`
- `status = Application.Status.SUBMITTED`

This only happens on the internal-submit path (employer_id == 1).

---

## 4. API surface

Mirrors `FeedbackSession` views (`backend/ai_services/views.py:136-318`) and lives in the same Django app (`ai_services`) for cohesion with existing agentic infrastructure.

### 4.1 Routes

```
POST   /api/ai/application-sessions/                              start
GET    /api/ai/application-sessions/                              list (current user's sessions)
GET    /api/ai/application-sessions/<id>/                         poll
POST   /api/ai/application-sessions/<id>/resume-gate-0/           resume after profile completion
POST   /api/ai/application-sessions/<id>/resume-gate-1/           resume after document mapping
POST   /api/ai/application-sessions/<id>/resume-gate-2/           resume after final approval
POST   /api/ai/application-sessions/<id>/cancel/                  cancel
```

Each gate has its own resume endpoint because the payload shapes are genuinely different (see §4.3). One generic `/resume/` endpoint with a discriminator is also viable but adds ambiguity that the frontend doesn't need.

### 4.2 Start (POST /api/ai/application-sessions/)

**Request:**
```json
{
  "opportunity_type": "job",   // 'job' | 'masters' | 'phd' | 'scholarship'
  "opportunity_id": 12345
}
```

**Response (201):**
```json
{
  "id": 7,
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "opportunity_type": "job",
  "opportunity_id": 12345,
  "current_step": "load_opportunity",
  "step_history": [],
  "created_at": "2026-04-26T..."
}
```

**Errors:**
- `409 Conflict` — active session already exists for `(student, opportunity_type, opportunity_id)`.
- `404 Not Found` — opportunity ID doesn't exist or `is_closed = True`.
- `400 Bad Request` — invalid opportunity_type, missing fields.

The view enforces the `is_closed` check at start so we don't burn a session on a dead posting.

### 4.3 Resume payloads

Each gate's payload is locked in here so frontend and backend can be built in parallel.

**Gate 0 (resume-gate-0/) — profile completion**

The user has filled in the missing profile fields (or uploaded the missing CV). The frontend's responsibility: collect the values and POST them. The backend's responsibility: write them to `Student` / `StudentCV` *before* resuming the graph, then resume.

```json
{
  "profile_updates": {
    "name": "Alex Johnson",                 // optional, only if missing
    "nationality": "Turkish",               // optional
    "location": "Istanbul, TR",             // optional
    "degree_level": "BSc",                  // optional
    "disciplines": ["Computer Science"]     // optional
  },
  "uploaded_cv": "<file or null>"           // multipart if user is uploading their CV here
}
```

After writing updates to the student profile, the adapter resumes the graph with `Command(resume={"profile_completed": True})`. Gate 0's node body re-runs `eligibility_and_readiness` which now sees the updated profile.

**Gate 1 (resume-gate-1/) — document mapping**

The user has reviewed the asset mapping and either confirmed it as-is or overridden specific document choices.

```json
{
  "confirm": true,                          // required, must be non-falsy per existing gate contract
  "overrides": [
    {"document_type": "CV", "use_default": true},
    {"document_type": "Cover Letter", "uploaded_file_ref": "session-uploads/cl-ab12.pdf"}
  ],
  "additional_uploads": [
    // multipart files; the backend stores them under session-uploads/<session_id>/
    // and rewrites the override entries to reference the stored path
  ]
}
```

The backend stores uploads, rewrites the override list to point at storage paths, then resumes with the rewritten payload. Whether session-scoped uploads persist back to a permanent `StudentDocument` store is §10.Q1.

**Gate 2 (resume-gate-2/) — final approval**

```json
{
  "approved": true,                         // required, true to submit/handoff, false to cancel
  "feedback": [
    {"document_type": "CV", "comment": "..."},
    {"document_type": "Cover Letter", "comment": "..."}
  ]
}
```

If `approved == false`, the graph terminates and the session moves to `CANCELLED` with the user's feedback recorded. (For v1, "approved=false" is hard-cancel; loop-back-to-tailoring with edit instructions is §10.Q2.)

### 4.4 Cancel (POST .../cancel/)

Calls `cancel_session(thread_id, checkpointer)` from the agentic repo (see A7). Response is immediate; the background thread drains naturally as nodes short-circuit.

```json
{ "message": "Session cancelled. The workflow will terminate after the current step completes." }
```

The status flips to `CANCELLED` immediately; downstream `step_history` updates from the draining graph are still written to the same row but don't change the terminal status.

### 4.5 Poll response shape (GET .../<id>/)

Stable across all statuses; clients filter by status:

```json
{
  "id": 7,
  "thread_id": "...",
  "status": "awaiting_document_mapping",
  "opportunity_type": "job",
  "opportunity_id": 12345,
  "opportunity_snapshot": { ... },                  // for rendering the opportunity in the gate UI
  "current_step": "human_gate_1",
  "step_history": [...],
  "eligibility_result": { ... },                    // populated after eligibility_and_readiness runs
  "asset_mapping": [ ... ],                         // populated for gate 1
  "tailored_documents": null,                       // populated for gate 2 onward
  "application_package": null,                      // populated at terminal status
  "submission_type": "",
  "discovery_method": "",                           // populated when discovery ships
  "error_message": "",
  "created_at": "...",
  "updated_at": "...",
  "completed_at": null
}
```

---

## 5. Adapter design

New module: `backend/ai_services/auto_apply_adapter.py`. Mirrors `graph_adapter.py:1-468`. Key responsibilities:

1. **Pre-load opportunity data** (`build_auto_apply_opportunity_snapshot`) — switches on `opportunity_type` and queries `JobOpportunity` / `ProgramOpportunity` / `ScholarshipOpportunity`. Returns the dict shape `auto_apply` expects. Reuses the dispatch logic already in `graph_adapter.build_opportunity_context:156` but extends it for auto-apply's richer needs (e.g., includes `data` json verbatim for programs/scholarships).

2. **Pre-load profile snapshot** (`build_auto_apply_profile_snapshot`) — extends `build_profile_snapshot:103`. Adds:
   - `uploaded_documents`: maps doc_type → bool from `StudentCV` existence (CV only for v1; cf. §11.3).
   - `document_texts["CV"]`: extracted text from `StudentCV.cv_file` using `accounts.cv_parser._extract_text` + OCR fallback (already implemented at `graph_adapter.py:216`).
   - All other doc_types in `uploaded_documents` are `False` for v1.

3. **Build initial state** (`build_initial_state`) — wires all the above into `AutoApplyState`, including `discovered_apply_url`/`discovery_method` (empty for v1).

4. **`_build_checkpointed_graph_for_auto_apply()`** — duplicates `_build_checkpointed_graph` from document_feedback but imports `auto_apply.graph.build_graph`. The PostgresSaver `setup()` global flag (`_checkpointer_setup_done`) is shared across both workflows because the underlying tables are workflow-agnostic.

5. **`start_session(session)`, `_run_graph_phase_initial(session_id)`** — runs the graph until the first interrupt or terminal. Branches:
   - `eligibility_result.decision == "ineligible"` → status = `INELIGIBLE`, terminate.
   - `eligibility_result.decision == "pending"` → status = `AWAITING_PROFILE_COMPLETION`. (Requires gate 0 to actually call `interrupt()` — this is one of the agentic-repo changes in §6.)
   - Otherwise, the graph runs through to `human_gate_1`'s interrupt → status = `AWAITING_DOCUMENT_MAPPING`.

6. **`resume_session_gate_n(session, payload)`** — three functions, one per gate. Each:
   - Writes payload artefacts (uploaded files, profile updates) to the appropriate Django models.
   - Translates the payload into the graph's resume format.
   - Resumes via `graph.invoke(Command(resume=...), config)`.
   - On the next interrupt/terminal, persists state back to `ApplicationSession` and updates status.

7. **`finalize_internal_submission(session)`** — called from the gate 2 resume handler when the graph terminates with `submission_type == "internal"`. Creates the `Application` row via Django ORM. The `submit_internal` graph node becomes a marker (records intent into state); the actual ORM write happens here, in the adapter, after the graph returns. This honors A3 (no HTTP loopback) cleanly.

8. **`cancel_session(session)`** — calls into the agentic repo (see A7 + §6.6).

---

## 6. Required changes inside the agentic repo

Even with A1 keeping the agentic repo DB-free, four files need changes for the integration to work cleanly. Each is small and doesn't break CLI mode.

### 6.1 `eligibility_and_readiness.py` — accept profile from state

Replace `_get_stub_profile()` calls with a helper that reads `state["profile_snapshot"]` first, falls back to the stub when absent (CLI mode).

```python
def _resolve_profile(state: AutoApplyState) -> Dict[str, Any]:
    return state.get("profile_snapshot") or _get_stub_profile()
```

This same helper is reused in `asset_mapping.py` and `application_tailoring.py`, both of which currently import `_get_stub_profile` directly. Move the helper into a shared utility (e.g. `workflows/auto_apply/_profile.py`) so we don't fan out three imports.

### 6.2 `human_gate_0.py` — implement the real interrupt cycle

Currently a dead stub (`nodes/human_gate_0.py:11`). Replace with the same `interrupt()` / `Command(resume=...)` pattern as `human_gate_1.py`, then route to a re-run of `eligibility_and_readiness` (the existing graph edge goes to END — needs to change to point at `eligibility_and_readiness`).

The re-eligibility loop needs a max-iteration guard so a user with a chronically incomplete profile doesn't loop forever. Cap at 2 retries; if eligibility still says `pending` after 2 gate-0 cycles, terminate as `INELIGIBLE` with a "could not complete profile" message.

### 6.3 `submit_internal.py` — record intent, don't pretend to POST

The current `_post_to_backend` returns a fake `platform_application_id`. Replace with a no-op that just records `submission_type = "internal"` in `application_package` and stores enough metadata for the adapter's `finalize_internal_submission` to do the real ORM write. No HTTP, no fake IDs.

### 6.4 `load_opportunity.py` — accept pre-loaded data, fall back to stub

```python
def load_opportunity(state: AutoApplyState) -> dict:
    if state.get("opportunity_data"):
        # Pre-loaded by adapter (A1) — just pass through with validation
        return {"current_step": "load_opportunity", "step_history": ["load_opportunity"]}
    # CLI / stub mode (existing logic)
    ...
```

When invoked from the backend, `opportunity_data` is already in state and this node is a near no-op. The CLI path is unchanged.

**Freshness re-check (decision):** at gate 2 resume time, the adapter re-queries `JobOpportunity.is_closed` (or scholarship `deadline`) one more time before resuming. If the posting closed mid-session, status flips to `INELIGIBLE` with a "this opportunity has just closed" message and the graph isn't resumed. Cheap query, big UX win.

### 6.5 New `tools/document_renderer.py` — CV-to-PDF

For the internal-submit path's `Application.resume_file`, we need a PDF, not a text blob. Reuse the existing `tools/latex_compiler.py` from document_feedback if its output shape suits us; otherwise, simplest path is plain-text → ReportLab → PDF. **This is the one piece that needs a real new module, not just adapter glue.**

For the handoff path, no PDF rendering is required — the frontend renders tailored documents inline. That's a v1 simplification; we revisit if users actually want downloadable PDFs of their tailored docs.

### 6.6 New `workflows/auto_apply/control.py` — `cancel_session`

Per A7:

```python
def cancel_session(thread_id: str, checkpointer) -> None:
    """Inject a cancel marker into the graph's checkpoint.

    The currently-executing node finishes normally; subsequent nodes see
    result.status == 'error' and short-circuit through to END. The thread
    running the graph terminates naturally.
    """
    from langgraph.types import Command  # noqa
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    graph.update_state(config, {
        "result": {
            "status": "error",
            "error_code": "CANCELLED",
            "user_message": "Cancelled by user.",
        },
    })
```

The backend's `cancel` view calls this, then flips `ApplicationSession.status = CANCELLED` synchronously. The background thread will eventually write its draining-step updates to the same row, but since the terminal status is already `CANCELLED`, those writes only update `step_history` (idempotent).

---

## 7. Sequencing recommendation

Three scopes, shippable independently:

**Scope 1 — Core integration (this spec).** Everything above. Caps at:
- CV-only document support (Cover Letter / SOP / Personal Statement are *generated*, not loaded from user files).
- Discovery deferred (`discover_apply_url` is a no-op pass-through; every job goes to handoff).
- External form auto-submission deferred.

Outcome: students can run auto-apply against any opportunity type. Internal jobs get an `Application` row. Everything else gets a handoff package. Gate 0 works for missing profile fields but not missing documents (since v1 only requires CV, which is always uploadable via `StudentCV`).

**Scope 2 — `StudentDocument` polymorphic store.** Adds a generic document table keyed on `(student, doc_type)`. Lets users save Cover Letters / SOPs / Portfolios. Auto-apply's `application_tailoring` switches from "generate" to "deep-tailor a stored document" when one exists. `human_gate_0` starts firing for missing-document scenarios beyond CV.

**Scope 3 — Discovery + external auto-submission.** Re-executes the discovery plan against real DB. Adds a "fully scrape and verify" → "attempt form auto-submit" → "create `Application` row on success" branch alongside the existing handoff terminal. Routes external jobs based on `discovery_method` and `discovery_confidence`.

This spec covers Scope 1. The other two are noted as follow-ups, not requirements.

---

## 8. Testing strategy

Following the existing pattern (document_feedback tests are ~~thin~~ inline-script-style, not pytest fixtures), the bar is:

1. **Adapter unit tests** (`backend/ai_services/tests/test_auto_apply_adapter.py`):
   - `build_auto_apply_opportunity_snapshot` for each of the 4 opportunity types.
   - `build_auto_apply_profile_snapshot` against a fixtured `Student` + `StudentCV`.
   - `finalize_internal_submission` writing a correct `Application` row.

2. **End-to-end view tests** (`backend/ai_services/tests/test_auto_apply_views.py`): one per status transition. Use Django's `TestCase` with mocked LLM (`UPPGRAD_LLM_PROVIDER` unset → heuristic fallback).

3. **Cancel test**: confirm that calling `cancel_session` mid-graph terminates cleanly without raising.

4. **One-active-session test**: confirm the partial unique constraint prevents duplicates and allows post-completion re-runs.

5. **Manual smoke tests** against staging Neon — at least one of each opportunity type, end-to-end through gate 2.

The agentic-repo changes (§6) get TDD'd in the implementation plan as before.

---

## 9. Open trade-offs known going in

These are explicit accepted-cost decisions, not bugs to fix later:

- **T1: Cancel is best-effort, not instantaneous.** The currently-executing node runs to completion before the cancel marker is observed. Worst case ~30-60s of LLM cost wasted on a single cancelled tailoring call. Acceptable for v1; revisit if cost data shows it matters.
- **T2: Stale-session cleanup must be per-status.** The `cleanup_stale_sessions` pattern from `graph_adapter.py:452` uses a flat 15-min cap which is wrong for human-gate states that legitimately wait days. Auto-apply integration will introduce a per-status TTL map: `PROCESSING → 15min`, `FINALIZING → 30min`, `AWAITING_PROFILE_COMPLETION / AWAITING_DOCUMENT_MAPPING / AWAITING_FINAL_APPROVAL → 7 days each`. Sessions that exceed their TTL are flipped to `ERROR` with a "session timed out — please start fresh" message. Document_feedback can adopt the same per-status pattern later.
- **T3: Cost ceiling per session is uncapped.** Worst-case ~15-20 LLM calls per session (eval + map + tailoring × N + tailoring evals × N + smoothing). Heuristic fallback is the implicit cap (no LLM provider = no cost), but with `UPPGRAD_LLM_PROVIDER=openai` set, there's no per-session token budget. If cost becomes a concern we add a per-session ceiling later, not now.
- **T4: GDPR / retention of tailored documents and scraped content.** `ApplicationSession` rows live forever. PostgresSaver checkpoints (which embed full graph state, including extracted CV text) live forever. A retention policy is a separate piece of work tracked outside this spec.
- **T5: Connection pool pressure.** Each background thread holds a Postgres connection while running. With one-per-`(student, opportunity)` and active gates, a power user with 10 paused sessions consumes 10 idle connections. Mitigation: connections are closed at every node-batch boundary (existing pattern at `graph_adapter.py:357`); idle sessions at gates hold no connection. Should be a non-issue in practice but worth load-testing before launch.

---

## 10. Resolved sub-decisions

These were raised as open questions during spec review and have been resolved. Recorded here so the implementation plan inherits them without re-litigation.

**Q1: Where do session-uploaded documents live? — decided (a).**
Session uploads at gate 1 are stored under `media/session-uploads/<session_id>/`, scoped to this session, and are *not* promoted to the user's permanent profile. Cleaned up after completion (or when the session is flipped to `ERROR` by the stale-session janitor). The "promote to permanent profile" behavior is deferred to Scope 2 alongside the polymorphic `StudentDocument` store.

**Q2: Gate 2 "rejected" — terminate (v1) → loop back (future work).**
**Decided:** v1 terminates the session with `CANCELLED` status when the user rejects at gate 2. The user must start a fresh `ApplicationSession` if they want a second iteration with edit instructions.
**Future work (tracked here so it's not lost):** add an "edit and re-tailor" loop where rejected feedback at gate 2 routes back to `application_tailoring` with the user's per-document comments threaded into the tailoring prompts, capped at 2 retries. This requires both an agentic-repo change (new conditional edge from `human_gate_2`) and a frontend change (per-document comment fields on the rejection form). Listed alongside Scope 2 / Scope 3 work in §7.

**Q3: Stale-session per-status TTLs — decided.**
- `PROCESSING`: 15 minutes (matches existing document_feedback).
- `FINALIZING`: 30 minutes.
- `AWAITING_PROFILE_COMPLETION`: 7 days.
- `AWAITING_DOCUMENT_MAPPING`: 7 days.
- `AWAITING_FINAL_APPROVAL`: 7 days.

Anything older → flip to `ERROR` with a "session timed out — please start fresh" message. Frontend shows a "start fresh" button. The 7-day uniform cap for human-gate states aligns with opportunity-freshness decay (job postings get stale within a week or two regardless) and leaves room for an email-nudge feature at day 5–6 in the future.

**Q4: Frontend tab — decided: replace wholesale.**
The mock auto-apply page in `ui latest/` is decoration. We replace it wholesale and design fresh screens around the data shapes in §4.5. Frontend co-design is a separate workstream; this spec just locks the API contract.

**Q5: Tailored-document character cap — decided.**
Per-doc_type output caps applied at the end of `application_tailoring`:
- CV: 8000 chars
- Cover Letter: 3000 chars
- SOP / Personal Statement: 6000 chars
- Other doc types: 5000 chars (default)

Output exceeding the cap is truncated at the last paragraph boundary that fits, with a warning logged. Caps apply to LLM-mode output only; heuristic-mode output is naturally shorter. Prevents PDF rendering and frontend display issues.

---

## 11. Out of scope (explicit)

- **§11.1 Discovery + ATS / careers-site URL discovery.** Tracked in `2026-04-26-apply-url-discovery.md`. The integration spec adds the *fields* (`discovered_apply_url` etc.) and the *no-op node*, nothing more.
- **§11.2 External form auto-submission.** No browser automation, no Playwright, no form-filling. Every external opportunity terminates in handoff for v1.
- **§11.3 Polymorphic `StudentDocument` store.** Cover Letter / SOP / Personal Statement / Portfolio are tailored from CV + profile, never read from user-stored files. Scope 2 territory.
- **§11.4 OCR for non-CV documents.** Existing `_extract_text_with_ocr_fallback` covers CVs only. Other doc types don't have a stored-file path in v1, so OCR doesn't apply.
- **§11.5 Email handoff delivery.** Handoff is in-app only. No "email me my package" feature.
- **§11.6 Bulk auto-apply across multiple opportunities.** One opportunity per session.
- **§11.7 Per-session cost budgeting.** No token cap. T3.
- **§11.8 GDPR retention policy.** T4.
- **§11.9 Session-cancel cooperative checks inside nodes.** Best-effort cancel is the v1 contract. T1.
- **§11.10 Gate 2 "edit and re-tailor" loop.** v1 terminates on rejection at gate 2. Future work: route rejection back to `application_tailoring` with the user's per-document feedback as edit instructions, capped at 2 retries. Requires a new conditional edge from `human_gate_2` and matching frontend support for per-document comment fields. See §10.Q2.

---

## 12. Ready for implementation plan

All open questions from §10 are resolved. T1–T5 are accepted trade-offs. The next step is a TDD-shaped implementation plan covering Scope 1 in sub-phases:
1. DB migration: `ApplicationSession` model + partial unique constraint.
2. Agentic-repo surgical changes (§6.1–§6.6).
3. Adapter scaffold (`auto_apply_adapter.py`).
4. DRF views (start, poll, three resume endpoints, cancel).
5. Stale-session janitor with per-status TTL map.
6. Tests (adapter unit, view integration, cancel, one-active-session constraint).
7. Manual smoke pass against staging Neon.
