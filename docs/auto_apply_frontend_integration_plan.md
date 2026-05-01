# Auto-Apply — Frontend Integration Plan

**Status:** Draft pending implementation
**Spans:** `ui latest/` · `backend/` · `AgenticWorkflows_LangChain-LangGraph/`
**Replaces:** `AgenticApplyPage.tsx` (mock config screen) and `JobApplicationModal.tsx` (legacy internal-only path)

---

## 1. Vision

A single `/auto-apply` workspace, structured like `/documental-feedback`, that handles the complete auto-apply flow for **all four opportunity types** (job/masters/phd/scholarship) and **both internal and external jobs**. The "Apply" button on `/opportunities/browse` becomes a redirect into this page, carrying opportunity context. Internal jobs benefit from tailoring (currently they get raw uploads via `JobApplicationModal`); external jobs add discovery + scrape + form-fill on top.

Mental model:
- **Page = workspace** (active session monitoring + history + new-session form)
- **Modal = gate interaction** (focused per-gate UI, no two-pane layout)

This mirrors doc-feedback's logistics (top-of-page form → active session card → history grid below) while keeping the human-gate interactions in modals.

---

## 2. Page layout

```
/auto-apply
┌─────────────────────────────────────────────────────────────┐
│  Auto-Apply                                                 │
├─────────────────────────────────────────────────────────────┤
│  ╭─ Start a new session ──────────────────────────╮         │
│  │ Opportunity: [Selector or pre-filled chip]     │         │
│  │ Custom instructions (optional, ≤2000 chars):   │         │
│  │ [textarea]                                     │         │
│  │                          [ Start session ]     │         │
│  ╰────────────────────────────────────────────────╯         │
│                                                             │
│  ╭─ Active session (when present) ────────────────╮         │
│  │ ⏳ <opportunity title> · <status pill>         │         │
│  │ Step: <current_step> · Updated <relative_time> │         │
│  │ ▸ load → discover → … → gate_2                 │         │
│  │   [ Open ]  [ Cancel ]                         │         │
│  ╰────────────────────────────────────────────────╯         │
│                                                             │
│  ╭─ Past sessions ────────────────────────────────╮         │
│  │ Card grid: title · type badge · status pill ·  │         │
│  │ outcome (handoff / submitted) · X/Y filled ·   │         │
│  │ completed_at · "Open" → read-only modal        │         │
│  ╰────────────────────────────────────────────────╯         │
└─────────────────────────────────────────────────────────────┘
```

The "Open" buttons all open `<AutoApplyModal>` which routes to the right step based on `session.status`.

---

## 3. Routing migration

| Current | New |
|---|---|
| `/agentic-apply` (mock config page) | **Replaced by `/auto-apply`**. Old route becomes a redirect for one cycle then deletes. |
| `/opportunities/browse` "Apply Now" → `JobApplicationModal` (internal jobs only) | `/opportunities/browse` "Apply" → navigate to `/auto-apply?oppId=X&oppType=Y&oppTitle=Z` (all types) |
| Doc-feedback's "Get Feedback" → `/documental-feedback?oppId=...` | Unchanged — mirrors this pattern |

**`JobApplicationModal.tsx` retired** in this migration; its file deleted.

---

## 4. Cross-type support — all four from day one

| Type | Path through agentic graph |
|---|---|
| **job (external)** | discover → scrape → extract → asset_mapping → … → gate-2 → handoff (auto-fill optional) |
| **job (internal, employer_id=1)** | short-circuit → asset_mapping → … → gate-2 → submit_internal (Application row written) |
| **masters / phd** | parse `programs.data.requirements` → … → gate-2 → handoff |
| **scholarship** | parse `scholarships.data.required_documents` → … → gate-2 → handoff |

UI is type-agnostic: type renders as a badge in the active session card; gate UIs are uniform across types.

---

## 5. Internal-jobs critical constraint — respect employer-added fields

### The problem

Today's agentic internal short-circuit (`determine_requirements.py:278–280`) emits a hardcoded `[CV, Cover Letter]` requirement set for any opportunity with `employer_id == 1`:

```python
if opportunity_data.get("employer_id") == 1:
    return {**updates, "normalized_requirements": [r.model_dump() for r in _DEFAULTS["job"]]}
```

This matches `jobs.Application`'s `resume_file + cover_letter` (text), but **silently ignores**:

1. **Existing model fields** the user could fill via the application form:
   - `Application.cover_letter_file` (FileField — alternative to text-only `cover_letter`)
   - `Application.availability_start_date` (DateField)
   - `Application.additional_information` (TextField)
2. **Future per-posting custom fields** that employers will be able to add (currently no model for this; see §10).

### The rule

> Auto-apply MUST NOT decide on its own to default internal jobs to `[CV, Cover Letter]` when the employer has defined an explicit set of fields for the posting. The employer's spec is the source of truth.

### What changes

**Phase 0 (immediate, lands with v1):**
- Update `determine_requirements` internal short-circuit to **read from `Application` model fields** instead of hardcoded `_DEFAULTS["job"]`. Surface as `RequirementItem`s:
  - `resume_file` → CV (document, required)
  - `cover_letter_file` OR `cover_letter` text → Cover Letter (document if FileField is preferred, text if text is preferred)
  - `availability_start_date` → "Earliest start date" (text, optional, type=date)
  - `additional_information` → "Additional information" (text, optional)
- The `RequirementItem.form_field_index` for these synthesized items uses negative indices (`-1`, `-2`, ...) to indicate "no source FormField; this is from the Application model schema".
- `submit_internal` → `finalize_internal_submission` reads tailored content + user uploads + per-id answers and populates the right Application field.

**Phase 1 (future, separate rollout — see §10):**
- A new `EmployerJobField` model (or JSONB column on `JobOpportunity`) lets employers add custom fields per posting (e.g. "Why are you a fit?"). When present, `determine_requirements` builds requirement_items from THIS spec, not the model defaults.

### Backend impact

- `auto_apply_adapter.finalize_internal_submission`: extend to read `gate_1_response.requirements` + `tailored_documents` + `tailored_answers` and populate Application fields beyond CV/CL — `availability_start_date`, `additional_information`, `cover_letter_file`.
- New helper `_build_internal_requirement_items(opportunity_data) -> List[RequirementItem]` that generates the per-Application-field item list. Lives in agentic side or backend depending on coupling preference; agentic side keeps the graph self-contained.

### Frontend impact

None for the page shell — the gate-1 step renders whatever requirement_items the graph produces. The internal-job request_items just have more variety than the current 2-item default.

---

## 6. Custom instructions threading (session-level)

Doc-feedback has `user_instructions`; auto-apply needs the equivalent. A single text field on the new-session form, applied to ALL tailoring passes.

- Agentic state: `AutoApplyState.user_instructions: str = ""`.
- Backend `start_session`: accept `user_instructions: str` in the start payload, inject into `initial_state`.
- `application_tailoring`: prepend the string to every tailoring prompt's user-message context. Per-doc gate-1 `user_prompt` (200-char) overrides session-level for that doc only.

~30 lines across 3 files.

---

## 7. LaTeX/PDF document rendering

User wants tailored docs downloadable as **PDF (LaTeX-rendered) AND `.tex` source**, at gate 2 AND on the result page. Treat as parallel sub-rollout — page can ship without it (plain-text + copy buttons) and PDFs can backfill via follow-up.

### Sub-PR A — Agentic: tailoring outputs LaTeX

- `application_tailoring` prompt updated per-doc-type:
  - CV → CV-friendly template (sections + bullets)
  - Cover Letter → letter template
  - SOP / Personal Statement / Research Proposal → article with sections
- Output stored on `tailored_documents[<doc_type>].latex` (new field next to `content`).
- Plain-text `content` retained for previews + value_planner's mock fallback.
- Schema: `TailoredDocument` adds `latex: str = ""`.

### Sub-PR B — Backend: render service + storage

- `document_renderer.render_latex_to_pdf(latex: str) -> bytes` using `pdflatex` (already on Railway image).
- At gate-2 surface time (in `_persist_state_after_phase`): for each `tailored_documents[<doc>].latex`, render to PDF and save under `MEDIA_ROOT/applications/<thread_id>/tailored/<doc>.pdf` and `.tex`. Save URLs into `tailored_documents[<doc>].pdf_url` and `.tex_url`.
- Same materialization at `attempt_auto_fill` time — extend `_materialise_tailored_documents_to_paths` to add the LaTeX render path.

### Sub-PR C — Frontend: download UI

Per tailored doc card at gate 2 + result page:
- "Download PDF" → `tailored_documents[<doc>].pdf_url`
- "Download .tex" → `tailored_documents[<doc>].tex_url`
- "Copy text" → `tailored_documents[<doc>].content`

Tailored answers (text answers): plain text only, copy-button. Answers aren't documents.

---

## 8. Frontend file map

```
src/
├── lib/api.ts                                    # +applicationSessionApi
├── hooks/useQueries.ts                           # +5 hooks (start/get/cancel/resumeGate1/resumeGate2)
├── types/application-session.ts                  # NEW — all contract types
└── app/
    ├── App.tsx                                   # /auto-apply route added; remove /agentic-apply
    └── components/
        ├── AutoApplyPage.tsx                     # NEW — replaces AgenticApplyPage
        ├── auto-apply/                           # NEW dir
        │   ├── NewSessionForm.tsx                # opportunity selector + custom prompt + start
        │   ├── ActiveSessionCard.tsx             # status pill, step breadcrumb, Open / Cancel
        │   ├── SessionHistoryGrid.tsx            # past sessions card grid
        │   ├── AutoApplyModal.tsx                # Radix Dialog; routes by status
        │   ├── steps/
        │   │   ├── StepProcessing.tsx            # processing | finalizing
        │   │   ├── StepGate1.tsx                 # awaiting_document_mapping
        │   │   ├── StepGate2.tsx                 # awaiting_final_approval
        │   │   ├── StepIneligible.tsx
        │   │   └── StepResult.tsx                # terminal
        │   └── ui/
        │       ├── RequirementItemCard.tsx       # per-id document/text card
        │       ├── TailoredDocCard.tsx           # PDF/.tex/copy download buttons
        │       ├── ClarifyingQuestionInput.tsx   # answer/skip/ignore_for_now
        │       ├── AutoFillResultPanel.tsx
        │       └── StatusPill.tsx
        ├── OpportunitiesPage.tsx                 # "Apply" → redirect to /auto-apply
        ├── AgenticApplyPage.tsx                  # DELETE
        └── JobApplicationModal.tsx               # DELETE
```

---

## 9. Implementation phases

| # | Scope | Effort | Repos |
|---|---|---|---|
| **P1** | Plumbing — types, applicationSessionApi, 5 TanStack hooks | ~½ day | ui |
| **P2** | Page shell — AutoApplyPage with NewSessionForm + ActiveSessionCard + SessionHistoryGrid (no modal) | ~½ day | ui |
| **P3** | Modal skeleton — StepProcessing, StepIneligible, StepResult | ~½ day | ui |
| **P4** | StepGate1 — RequirementItemCard, file uploads via FormData, multipart resume | ~1 day | ui |
| **P5** | StepGate2 — TailoredDocCard (plain-text v1), ClarifyingQuestionInput, attempt_auto_submit toggle, kill-switch banner | ~1 day | ui |
| **P6** | Routing migration — Apply button redirect, retire JobApplicationModal, redirect /agentic-apply | ~½ day | ui |
| **P7** | Custom instructions threading — agentic state + backend adapter + tailoring prompt | ~½ day | agentic + backend |
| **P8** | **Internal-jobs employer-fields respect** — `_build_internal_requirement_items` from Application model schema; finalize_internal_submission populates all fields | ~1 day | agentic + backend |
| **Sub-PR A** | Agentic LaTeX tailoring output | ~½ day | agentic |
| **Sub-PR B** | Backend LaTeX render + materialization | ~½ day | backend |
| **Sub-PR C** | Frontend PDF/.tex downloads | ~2 hrs | ui |

**v1 minimum viable** = P1–P8 (excludes LaTeX). Total: ~6 days. LaTeX rollout adds ~1.5 days.

---

## 10. Future work — explicit deferrals

### 10.1 Employer-defined custom fields per posting

Today there is no model for "fields an employer added to their posting" (e.g. screening questions). This is a separate feature that will:

- Add `EmployerJobField` model OR `JobOpportunity.application_form_spec: JSONField`.
- Update `_build_internal_requirement_items` to read this spec when present, falling back to Application-model defaults when absent.
- Frontend: an employer-side UI to define those fields when creating/editing a posting.

When this lands, our auto-apply flow will already be reading from a structured spec source (per §5 Phase 0), so the only change is to add the new spec source to the read path.

### 10.2 Soft-deleting / paging old sessions

History grid grows unbounded. Match doc-feedback (no pagination) for v1; add pagination or 30-day soft-cutoff later.

### 10.3 Resumeable session start

If the user closes the browser between session start and gate-1 reach, the polling stops. They need to land back on `/auto-apply` to resume polling. Add a "session in progress" notification badge in nav for awareness — follow-up.

### 10.4 Gate-1 user_prompt vs session-level user_instructions cascade

Both are honored in P7. Behavior: per-doc `user_prompt` overrides session-level for that doc; session-level used as base context for all tailoring passes. Document this explicitly in tailoring node comments.

### 10.5 ZIP package download

Doc-feedback CLAUDE.md flags this; carry over here. Single download bundling all tailored docs (.pdf + .tex) + text answers (.txt). Trivial follow-up once Sub-PR B's per-doc PDFs exist.

---

## 11. Open questions to resolve before P1

| # | Question | Default (proposed) |
|---|---|---|
| 1 | Custom instructions field location — session-level only, or also per-doc at gate 1? | Both: session-level always honored; per-doc `user_prompt` (200 chars) overrides for that doc when present. |
| 2 | Internal-job result-step UX | Mirror doc-feedback's "completed" — show "Submitted as Application #N" with link to `/applications`. |
| 3 | Cross-type opportunity selector in NewSessionForm | Same pattern as doc-feedback: query each type endpoint (jobs/programs/scholarships) and show grouped results. |
| 4 | History grid pagination | None for v1; show all (matches doc-feedback). |
| 5 | LaTeX rollout sequencing | Ship v1 with plain-text + .txt downloads; add Sub-PR A/B/C as a polish PR. |
| 6 | Internal-jobs Phase 0 (§5) sequencing | Land alongside v1 (P8). Without it, internal jobs lose `availability_start_date` etc. that today's `JobApplicationModal` does collect. |

---

## 12. Cross-repo dependency graph

```
Agentic                  Backend                Frontend (ui latest)
─────────                ─────────              ─────────
P7  user_instructions ──→ P7  start_session
                           accepts it
                                                    ↑
                                             P1  applicationSessionApi
                                                    + types
                                                    ↑
P8  internal req_items ──→ P8  finalize_internal
    from Application         populates all
    schema                   Application fields
                                                    ↑
                                             P2-P6  Page + modal + steps

Sub-A  LaTeX output  ──→  Sub-B  render + URLs
                                                    ↑
                                             Sub-C  download buttons
```

Merge order: agentic-side P7+P8 → backend P7+P8 → frontend P1–P6 (any order). LaTeX A → B → C strict.

---

## 13. References

- Backend API contract: `backend/ai_services/views.py`, `backend/ai_services/serializers.py`, `backend/ai_services/auto_apply_adapter.py`.
- Agentic gate contracts: `AgenticWorkflows_LangChain-LangGraph/src/uppgrad_agentic/workflows/auto_apply/nodes/human_gate_1.py`, `human_gate_2.py`.
- Doc-feedback frontend pattern (template): `ui latest/src/app/components/DocumentalFeedbackPage.tsx`, `ui latest/src/hooks/useQueries.ts`.
- Existing auto-apply scaffold (to be replaced): `ui latest/src/app/components/AgenticApplyPage.tsx`.
- Open questions and architectural deferrals: `AgenticWorkflows_LangChain-LangGraph/open_questions_remodel.md` (the gate-1 remodel doc — items #15-17 cover Crawl4AI consolidation, PR A rebase, custom-React-control extraction).
