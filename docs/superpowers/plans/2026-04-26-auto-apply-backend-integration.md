# Auto-Apply Backend Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the `auto_apply` LangGraph workflow into the Django backend so users can run real auto-apply sessions end-to-end against real DB/profile data, terminating in either a submitted `Application` row (internal jobs) or an in-app handoff package (everything else).

**Architecture:** Phase A makes surgical changes inside the agentic repo so its nodes accept pre-loaded state and the dead `human_gate_0` actually interrupts. Phase B adds an `ApplicationSession` Django model, an `auto_apply_adapter.py` mirroring the existing `graph_adapter.py` pattern, DRF views with one resume endpoint per gate, and a per-status TTL janitor. Internal submissions become Django ORM writes from the adapter (no HTTP loopback); everything else terminates in handoff. Discovery state fields are added now as no-ops so the deferred discovery plan re-executes without a schema migration.

**Tech Stack:** Python 3.11, Django 4.2, DRF, psycopg 3.2 (autocommit), LangGraph 1.0+, langgraph-checkpoint-postgres, ReportLab (new — backend dep for PDF rendering on internal-submit), pytest, pytest-asyncio, respx (already added in discovery plan), Postgres (Neon).

**Spec:** `docs/superpowers/specs/2026-04-26-auto-apply-backend-integration.md`

**Repos involved:**
- `AgenticWorkflows_LangChain-LangGraph/` (Phase A)
- `backend/` (Phase B)

Per repo memory, both are independent git repos under the parent `cs491-2/` folder. Never branch/commit in the parent.

---

## File Structure

### Phase A — Agentic repo

| Path | Responsibility | Status |
|---|---|---|
| `src/uppgrad_agentic/workflows/auto_apply/state.py` | Add `profile_snapshot`, `discovered_apply_url`, `discovery_method`, `discovery_confidence`, `gate_0_iteration_count` | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/_profile.py` | `resolve_profile(state)` helper — read `state["profile_snapshot"]` first, fall back to stub | Create |
| `src/uppgrad_agentic/workflows/auto_apply/nodes/eligibility_and_readiness.py` | Use `resolve_profile`; export stub for CLI | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/nodes/asset_mapping.py` | Switch import to `resolve_profile` | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/nodes/application_tailoring.py` | Switch import; add per-doc-type output cap | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/nodes/load_opportunity.py` | Short-circuit when `state["opportunity_data"]` already present | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/nodes/human_gate_0.py` | Real `interrupt()` cycle; iteration cap | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/nodes/submit_internal.py` | Drop fake POST; record intent only | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/graph.py` | Wire gate 0 → eligibility loop; iteration-cap router | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/control.py` | `cancel_session(thread_id, checkpointer)` helper | Create |

### Phase B — Backend

| Path | Responsibility | Status |
|---|---|---|
| `backend/ai_services/models.py` | Add `ApplicationSession` model | Modify |
| `backend/ai_services/migrations/0007_application_session.py` | Django migration (auto-generated, then add partial unique constraint manually) | Create |
| `backend/ai_services/serializers.py` | `ApplicationSessionSerializer`, `ApplicationSessionStartSerializer`, three gate-resume serializers | Modify |
| `backend/ai_services/document_renderer.py` | `render_text_to_pdf(text, title) -> bytes` using ReportLab | Create |
| `backend/ai_services/auto_apply_adapter.py` | Opportunity/profile snapshot builders, graph runner, gate resumers, finalize_internal_submission, cancel_session | Create |
| `backend/ai_services/janitor.py` | Per-status stale-session cleanup for both `FeedbackSession` and `ApplicationSession` | Create |
| `backend/ai_services/views.py` | 6 new `ApplicationSession*View` classes | Modify |
| `backend/ai_services/urls.py` | 6 new URL patterns | Modify |
| `backend/requirements.txt` | Add `reportlab>=4.0.0` | Modify |
| `backend/ai_services/tests/__init__.py` | Test package | Create |
| `backend/ai_services/tests/test_auto_apply_adapter.py` | Adapter unit tests | Create |
| `backend/ai_services/tests/test_application_session_views.py` | View integration tests | Create |
| `backend/ai_services/tests/test_janitor.py` | Per-status TTL tests | Create |

---

# PHASE A — Agentic Repo Changes

All Phase A tasks run inside `AgenticWorkflows_LangChain-LangGraph/`.

## Task A1: Add new state fields

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/state.py`
- Test: `tests/workflows/auto_apply/test_state_fields.py`

- [ ] **Step 1: Write failing test**

```python
# tests/workflows/auto_apply/test_state_fields.py
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState


def test_state_accepts_profile_snapshot():
    s: AutoApplyState = {"profile_snapshot": {"name": "X"}}
    assert s["profile_snapshot"]["name"] == "X"


def test_state_accepts_discovery_fields():
    s: AutoApplyState = {
        "discovered_apply_url": "https://x.com/job/1",
        "discovery_method": "ats",
        "discovery_confidence": 0.9,
    }
    assert s["discovered_apply_url"] == "https://x.com/job/1"
    assert s["discovery_method"] == "ats"
    assert s["discovery_confidence"] == 0.9


def test_state_accepts_gate_0_iteration_count():
    s: AutoApplyState = {"gate_0_iteration_count": 1}
    assert s["gate_0_iteration_count"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/workflows/auto_apply/test_state_fields.py -v`
Expected: FAIL — TypedDict keys not declared (mypy/pydantic-style runtime checks pass with TypedDict total=False, so the test verifies presence; if total=False is preserved, this passes trivially — write the test to confirm structural shape via dict ops).

If the existing `total=False` makes the test pass without changes, swap the test to assert `AutoApplyState.__annotations__` contains the new keys:

```python
def test_state_declares_new_keys():
    keys = AutoApplyState.__annotations__
    assert "profile_snapshot" in keys
    assert "discovered_apply_url" in keys
    assert "discovery_method" in keys
    assert "discovery_confidence" in keys
    assert "gate_0_iteration_count" in keys
```

- [ ] **Step 3: Add fields to state.py**

In `src/uppgrad_agentic/workflows/auto_apply/state.py`, inside `class AutoApplyState(TypedDict, total=False):`, add immediately after `opportunity_data`:

```python
    # injected by backend adapter (Spec A1) — replaces _get_stub_profile lookups
    profile_snapshot: Dict[str, Any]

    # apply-URL discovery (Spec A6) — populated when discovery feature ships
    discovered_apply_url: Optional[str]
    discovery_method: Optional[str]
    discovery_confidence: Optional[float]

    # human_gate_0 retry counter (Spec §6.2) — caps the eligibility re-check loop
    gate_0_iteration_count: int
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/workflows/auto_apply/test_state_fields.py -v`
Expected: all 4 pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/state.py tests/workflows/auto_apply/test_state_fields.py
git commit -m "feat(auto_apply): add profile_snapshot, discovery_*, gate_0_iteration_count to state"
```

---

## Task A2: `resolve_profile` helper + use in eligibility_and_readiness

**Files:**
- Create: `src/uppgrad_agentic/workflows/auto_apply/_profile.py`
- Modify: `src/uppgrad_agentic/workflows/auto_apply/nodes/eligibility_and_readiness.py`
- Test: `tests/workflows/auto_apply/test_resolve_profile.py`

- [ ] **Step 1: Write failing test**

```python
# tests/workflows/auto_apply/test_resolve_profile.py
from uppgrad_agentic.workflows.auto_apply._profile import resolve_profile


def test_returns_profile_snapshot_when_present():
    state = {"profile_snapshot": {"name": "Real User", "email": "real@x.com"}}
    profile = resolve_profile(state)
    assert profile["name"] == "Real User"


def test_falls_back_to_stub_when_snapshot_absent():
    profile = resolve_profile({})
    assert profile["name"] == "Alex Johnson"   # the stub's name


def test_falls_back_to_stub_when_snapshot_empty_dict():
    profile = resolve_profile({"profile_snapshot": {}})
    # Empty dict counts as absent — fall through to stub
    assert profile["name"] == "Alex Johnson"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/workflows/auto_apply/test_resolve_profile.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Create the helper**

```python
# src/uppgrad_agentic/workflows/auto_apply/_profile.py
from __future__ import annotations

from typing import Any, Dict


def resolve_profile(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return the user profile dict for this graph run.

    Prefers state['profile_snapshot'] (injected by backend adapter); falls back
    to the in-repo stub for CLI / local-dev mode.
    """
    snapshot = state.get("profile_snapshot")
    if snapshot:
        return snapshot
    from uppgrad_agentic.workflows.auto_apply.nodes.eligibility_and_readiness import _get_stub_profile
    return _get_stub_profile()
```

- [ ] **Step 4: Switch eligibility_and_readiness to use the helper**

In `src/uppgrad_agentic/workflows/auto_apply/nodes/eligibility_and_readiness.py`, replace line 243:

```python
    profile = _get_stub_profile()
```

with:

```python
    from uppgrad_agentic.workflows.auto_apply._profile import resolve_profile
    profile = resolve_profile(state)
```

(`_get_stub_profile` stays defined in this file — `_profile.py` imports it lazily, and CLI mode still works via the fallback.)

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_resolve_profile.py -v`
Expected: all 3 pass.

- [ ] **Step 6: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/_profile.py src/uppgrad_agentic/workflows/auto_apply/nodes/eligibility_and_readiness.py tests/workflows/auto_apply/test_resolve_profile.py
git commit -m "feat(auto_apply): add resolve_profile helper, wire into eligibility_and_readiness"
```

---

## Task A3: Wire `resolve_profile` into asset_mapping and application_tailoring

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/nodes/asset_mapping.py:18,245`
- Modify: `src/uppgrad_agentic/workflows/auto_apply/nodes/application_tailoring.py:10,296`
- Test: `tests/workflows/auto_apply/test_resolve_profile_integration.py`

- [ ] **Step 1: Write failing test**

```python
# tests/workflows/auto_apply/test_resolve_profile_integration.py
def test_asset_mapping_uses_resolve_profile():
    """When state has profile_snapshot, asset_mapping must use it (not the stub)."""
    from uppgrad_agentic.workflows.auto_apply.nodes import asset_mapping as am_module

    captured = {}
    real_resolve = am_module.resolve_profile if hasattr(am_module, "resolve_profile") else None

    def spy_resolve(state):
        captured["state_passed"] = state
        return {"name": "Snapshot User", "email": "s@x.com",
                "uploaded_documents": {"CV": True}, "document_texts": {"CV": "hi"}}

    am_module.resolve_profile = spy_resolve
    try:
        am_module.asset_mapping({
            "profile_snapshot": {"x": 1},
            "normalized_requirements": [{"requirement_type": "document", "document_type": "CV"}],
        })
        assert captured["state_passed"]["profile_snapshot"] == {"x": 1}
    finally:
        if real_resolve:
            am_module.resolve_profile = real_resolve


def test_application_tailoring_uses_resolve_profile():
    from uppgrad_agentic.workflows.auto_apply.nodes import application_tailoring as at_module

    captured = {}

    def spy_resolve(state):
        captured["state_passed"] = state
        return {"name": "Snapshot User", "email": "s@x.com",
                "uploaded_documents": {"CV": True}, "document_texts": {"CV": "hi"}}

    real = at_module.resolve_profile if hasattr(at_module, "resolve_profile") else None
    at_module.resolve_profile = spy_resolve
    try:
        at_module.application_tailoring({
            "profile_snapshot": {"y": 2},
            "asset_mapping": [],
            "normalized_requirements": [],
            "opportunity_data": {},
            "opportunity_type": "job",
        })
        assert captured["state_passed"]["profile_snapshot"] == {"y": 2}
    finally:
        if real:
            at_module.resolve_profile = real
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/workflows/auto_apply/test_resolve_profile_integration.py -v`
Expected: FAIL — `am_module.resolve_profile` AttributeError (these modules don't import it yet).

- [ ] **Step 3: Modify asset_mapping.py**

In `src/uppgrad_agentic/workflows/auto_apply/nodes/asset_mapping.py`, replace line 18:

```python
from uppgrad_agentic.workflows.auto_apply.nodes.eligibility_and_readiness import _get_stub_profile
```

with:

```python
from uppgrad_agentic.workflows.auto_apply._profile import resolve_profile
```

Replace line 245:

```python
    profile = _get_stub_profile()
```

with:

```python
    profile = resolve_profile(state)
```

- [ ] **Step 4: Modify application_tailoring.py**

Same changes in `src/uppgrad_agentic/workflows/auto_apply/nodes/application_tailoring.py`:

Line 10:
```python
from uppgrad_agentic.workflows.auto_apply._profile import resolve_profile
```

Line 296:
```python
    profile = resolve_profile(state)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_resolve_profile_integration.py tests/workflows/auto_apply/test_resolve_profile.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/nodes/asset_mapping.py src/uppgrad_agentic/workflows/auto_apply/nodes/application_tailoring.py tests/workflows/auto_apply/test_resolve_profile_integration.py
git commit -m "refactor(auto_apply): asset_mapping and application_tailoring use resolve_profile"
```

---

## Task A4: `load_opportunity` short-circuit when pre-loaded

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/nodes/load_opportunity.py:141-192`
- Test: `tests/workflows/auto_apply/test_load_opportunity_preloaded.py`

- [ ] **Step 1: Write failing test**

```python
# tests/workflows/auto_apply/test_load_opportunity_preloaded.py
from uppgrad_agentic.workflows.auto_apply.nodes.load_opportunity import load_opportunity


def test_short_circuits_when_opportunity_data_preloaded():
    state = {
        "opportunity_type": "job",
        "opportunity_id": "real-123",
        "opportunity_data": {"id": 999, "title": "Real Job", "company": "RealCorp"},
    }
    out = load_opportunity(state)
    # Must NOT overwrite the pre-loaded data with the stub
    assert "opportunity_data" not in out  # node didn't return a new value
    assert out["current_step"] == "load_opportunity"
    assert out["step_history"] == ["load_opportunity"]


def test_falls_back_to_stub_when_no_opportunity_data():
    state = {"opportunity_type": "job", "opportunity_id": "job-001"}
    out = load_opportunity(state)
    # CLI / stub mode — node loads from _STUB_RECORDS
    assert out["opportunity_data"]["title"] == "Software Engineer"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/workflows/auto_apply/test_load_opportunity_preloaded.py -v`
Expected: FAIL on the first test — current code unconditionally overwrites with stub.

- [ ] **Step 3: Patch `load_opportunity`**

In `src/uppgrad_agentic/workflows/auto_apply/nodes/load_opportunity.py`, after the `updates` dict and the error short-circuit (line 142-144), add the pre-loaded short-circuit:

```python
def load_opportunity(state: AutoApplyState) -> dict:
    updates = {"current_step": "load_opportunity", "step_history": ["load_opportunity"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    # Pre-loaded by backend adapter (Spec A1) — pass through with no DB hit.
    if state.get("opportunity_data"):
        return updates

    # ... existing CLI/stub logic continues unchanged below ...
```

(Leave lines 146-192 untouched.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_load_opportunity_preloaded.py -v`
Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/nodes/load_opportunity.py tests/workflows/auto_apply/test_load_opportunity_preloaded.py
git commit -m "feat(auto_apply): load_opportunity short-circuits when state pre-loaded"
```

---

## Task A5: Real `human_gate_0` interrupt cycle with iteration cap

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/nodes/human_gate_0.py`
- Test: `tests/workflows/auto_apply/test_human_gate_0.py`

- [ ] **Step 1: Write failing test**

```python
# tests/workflows/auto_apply/test_human_gate_0.py
import pytest
from unittest.mock import patch

from uppgrad_agentic.workflows.auto_apply.nodes.human_gate_0 import human_gate_0


def test_calls_interrupt_with_missing_fields():
    state = {
        "eligibility_result": {"decision": "pending", "missing_fields": ["email", "document:CV"]},
        "gate_0_iteration_count": 0,
    }
    with patch("uppgrad_agentic.workflows.auto_apply.nodes.human_gate_0.interrupt") as fake_int:
        fake_int.return_value = {"profile_completed": True}
        out = human_gate_0(state)
    fake_int.assert_called_once()
    payload = fake_int.call_args.args[0]
    assert payload["missing_fields"] == ["email", "document:CV"]
    assert out["gate_0_iteration_count"] == 1
    assert out.get("human_review_0") == {"profile_completed": True}


def test_returns_error_when_iteration_cap_exceeded():
    state = {
        "eligibility_result": {"decision": "pending", "missing_fields": ["email"]},
        "gate_0_iteration_count": 2,   # already retried twice
    }
    out = human_gate_0(state)
    assert out["result"]["status"] == "error"
    assert out["result"]["error_code"] == "PROFILE_INCOMPLETE_AFTER_RETRIES"


def test_short_circuits_on_upstream_error():
    state = {"result": {"status": "error", "error_code": "X"}}
    out = human_gate_0(state)
    assert out == {"current_step": "human_gate_0", "step_history": ["human_gate_0"]}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/workflows/auto_apply/test_human_gate_0.py -v`
Expected: FAIL — current node is a dead stub.

- [ ] **Step 3: Replace the node**

Overwrite `src/uppgrad_agentic/workflows/auto_apply/nodes/human_gate_0.py`:

```python
from __future__ import annotations

from langgraph.types import interrupt

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

MAX_GATE_0_ITERATIONS = 2


def human_gate_0(state: AutoApplyState) -> dict:
    updates = {"current_step": "human_gate_0", "step_history": ["human_gate_0"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    iteration = state.get("gate_0_iteration_count", 0)
    if iteration >= MAX_GATE_0_ITERATIONS:
        return {
            **updates,
            "result": {
                "status": "error",
                "error_code": "PROFILE_INCOMPLETE_AFTER_RETRIES",
                "user_message": (
                    "Profile is still incomplete after the maximum number of completion attempts. "
                    "Please update your profile and start a new application session."
                ),
            },
        }

    eligibility = state.get("eligibility_result") or {}
    payload = {
        "type": "profile_completion",
        "missing_fields": eligibility.get("missing_fields", []),
        "reasons": eligibility.get("reasons", []),
        "iteration": iteration,
    }
    response = interrupt(payload)

    return {
        **updates,
        "human_review_0": response or {},
        "gate_0_iteration_count": iteration + 1,
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_human_gate_0.py -v`
Expected: all 3 pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/nodes/human_gate_0.py tests/workflows/auto_apply/test_human_gate_0.py
git commit -m "feat(auto_apply): human_gate_0 implements real interrupt cycle with retry cap"
```

---

## Task A6: Wire gate 0 → eligibility loop in graph.py

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/graph.py`
- Test: `tests/workflows/auto_apply/test_graph_gate_0_loop.py`

- [ ] **Step 1: Write failing test**

```python
# tests/workflows/auto_apply/test_graph_gate_0_loop.py
from uppgrad_agentic.workflows.auto_apply.graph import build_graph


def test_human_gate_0_routes_back_to_eligibility():
    """human_gate_0 must have an outgoing edge to eligibility_and_readiness, not END."""
    graph = build_graph()
    g = graph.get_graph()
    edges_from_gate_0 = [e for e in g.edges if e.source == "human_gate_0"]
    targets = {e.target for e in edges_from_gate_0}
    # Must route back to eligibility_and_readiness (re-check after profile update)
    # OR to end_with_error when iteration cap exceeded
    assert "eligibility_and_readiness" in targets or any(
        "eligibility" in t for t in targets
    ), f"human_gate_0 routes to {targets}, expected eligibility loop"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/workflows/auto_apply/test_graph_gate_0_loop.py -v`
Expected: FAIL — current edge goes to END.

- [ ] **Step 3: Update graph.py**

In `src/uppgrad_agentic/workflows/auto_apply/graph.py`, find:

```python
    # human_gate_0: interrupt/resume wired in human-in-the-loop phase
    g.add_edge("human_gate_0", END)
```

Replace with:

```python
    # human_gate_0 routes back to eligibility re-check after profile update,
    # or to end_with_error when the iteration cap fires inside the node.
    g.add_conditional_edges(
        "human_gate_0",
        _route_after_gate_0,
        {
            "eligibility_and_readiness": "eligibility_and_readiness",
            "end_with_error": "end_with_error",
        },
    )
```

Add the router function near the other `_route_after_*` helpers:

```python
def _route_after_gate_0(state: AutoApplyState) -> str:
    if state.get("result", {}).get("status") == "error":
        return "end_with_error"
    return "eligibility_and_readiness"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_graph_gate_0_loop.py tests/workflows/auto_apply/test_human_gate_0.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/graph.py tests/workflows/auto_apply/test_graph_gate_0_loop.py
git commit -m "feat(auto_apply): route human_gate_0 back to eligibility_and_readiness"
```

---

## Task A7: `submit_internal` records intent only (no fake POST)

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/nodes/submit_internal.py`
- Test: `tests/workflows/auto_apply/test_submit_internal_intent.py`

- [ ] **Step 1: Write failing test**

```python
# tests/workflows/auto_apply/test_submit_internal_intent.py
from uppgrad_agentic.workflows.auto_apply.nodes.submit_internal import submit_internal


def test_records_submission_intent_without_fake_id():
    state = {
        "opportunity_id": "job-42",
        "opportunity_data": {"id": 42, "title": "SWE", "company": "Acme"},
        "tailored_documents": {
            "CV": {"content": "the CV"},
            "Cover Letter": {"content": "the CL"},
        },
    }
    out = submit_internal(state)
    pkg = out["application_package"]
    assert pkg["submission_type"] == "internal"
    assert pkg["CV"] == "the CV"
    assert pkg["Cover Letter"] == "the CL"
    # No fake platform_application_id — adapter creates the real Application row
    assert "platform_application_id" not in pkg
    assert out["result"]["status"] == "ok"


def test_short_circuits_on_upstream_error():
    state = {"result": {"status": "error", "error_code": "X"}}
    out = submit_internal(state)
    # Always returns step indicators even on error
    assert out.get("current_step") == "submit_internal"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/workflows/auto_apply/test_submit_internal_intent.py -v`
Expected: FAIL — current code injects fake `platform_application_id`.

- [ ] **Step 3: Replace `submit_internal`**

Overwrite `src/uppgrad_agentic/workflows/auto_apply/nodes/submit_internal.py`:

```python
from __future__ import annotations

import logging
from typing import Any, Dict

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)


def submit_internal(state: AutoApplyState) -> dict:
    """Record submission intent for internal jobs.

    The actual Application row is created server-side by the backend adapter
    after the graph terminates (Spec A3). This node is a marker, not a writer.
    """
    updates = {"current_step": "submit_internal", "step_history": ["submit_internal"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    opportunity_data = state.get("opportunity_data") or {}
    tailored: Dict[str, Any] = state.get("tailored_documents") or {}

    cv_content = (tailored.get("CV") or {}).get("content", "")
    cl_content = (tailored.get("Cover Letter") or {}).get("content", "")

    package: Dict[str, Any] = {
        "CV": cv_content,
        "Cover Letter": cl_content,
        "submission_type": "internal",
    }

    logger.info(
        "submit_internal: recorded internal submission intent for opportunity_id=%s",
        state.get("opportunity_id", "unknown"),
    )

    return {
        **updates,
        "application_package": package,
        "result": {
            "status": "ok",
            "user_message": (
                f"Your application for {opportunity_data.get('title', 'the role')} "
                f"at {opportunity_data.get('company', 'the company')} is ready for submission."
            ),
            "details": {"submission_type": "internal"},
        },
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_submit_internal_intent.py -v`
Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/nodes/submit_internal.py tests/workflows/auto_apply/test_submit_internal_intent.py
git commit -m "refactor(auto_apply): submit_internal records intent only (ORM write moves to adapter)"
```

---

## Task A8: Per-doc-type output cap in `application_tailoring`

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/nodes/application_tailoring.py`
- Test: `tests/workflows/auto_apply/test_application_tailoring_caps.py`

- [ ] **Step 1: Write failing test**

```python
# tests/workflows/auto_apply/test_application_tailoring_caps.py
from uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring import _truncate_to_cap


def test_cv_capped_at_8000():
    long = "X" * 12000
    out = _truncate_to_cap(long, "CV")
    assert len(out) <= 8000


def test_cover_letter_capped_at_3000():
    long = "Y" * 5000
    out = _truncate_to_cap(long, "Cover Letter")
    assert len(out) <= 3000


def test_sop_capped_at_6000():
    long = "Z" * 10000
    out = _truncate_to_cap(long, "SOP")
    assert len(out) <= 6000


def test_unknown_doc_type_uses_default_5000():
    long = "Q" * 8000
    out = _truncate_to_cap(long, "Reference Letter")
    assert len(out) <= 5000


def test_does_not_truncate_short_content():
    out = _truncate_to_cap("short", "CV")
    assert out == "short"


def test_truncates_at_paragraph_boundary_when_possible():
    # Crafted so the last paragraph break lands before the cap
    text = ("para one. " * 100) + "\n\n" + ("para two. " * 1000)
    out = _truncate_to_cap(text, "Cover Letter")  # cap=3000
    # If a paragraph boundary exists within [0, 3000], output ends at it
    assert len(out) <= 3000
    # Should end at the paragraph break, not mid-word
    assert not out.endswith("para two")  # we'd see "para two." or boundary
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/workflows/auto_apply/test_application_tailoring_caps.py -v`
Expected: FAIL — `_truncate_to_cap` not defined.

- [ ] **Step 3: Add cap logic to application_tailoring.py**

In `src/uppgrad_agentic/workflows/auto_apply/nodes/application_tailoring.py`, add near the top (after imports, before any node logic):

```python
_DOC_TYPE_CAPS = {
    "CV": 8000,
    "Cover Letter": 3000,
    "SOP": 6000,
    "Personal Statement": 6000,
}
_DEFAULT_CAP = 5000


def _truncate_to_cap(content: str, doc_type: str) -> str:
    cap = _DOC_TYPE_CAPS.get(doc_type, _DEFAULT_CAP)
    if len(content) <= cap:
        return content
    # Try to truncate at the last paragraph boundary within the cap
    boundary = content.rfind("\n\n", 0, cap)
    if boundary > cap // 2:   # avoid truncating to almost nothing
        return content[:boundary]
    return content[:cap]
```

In the same file, find the place where tailored content is written into `tailored_documents` (search for `"content":` near the end of the function). Wrap each content assignment with `_truncate_to_cap`:

```python
tailored_documents[doc_type] = {
    "content": _truncate_to_cap(content_str, doc_type),
    "tailoring_depth": depth,
    ...
}
```

(Apply to both LLM and heuristic branches.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_application_tailoring_caps.py -v`
Expected: all 6 pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/nodes/application_tailoring.py tests/workflows/auto_apply/test_application_tailoring_caps.py
git commit -m "feat(auto_apply): cap tailored output per doc type to prevent runaway content"
```

---

## Task A9: `cancel_session` helper (out-of-band state injection)

**Files:**
- Create: `src/uppgrad_agentic/workflows/auto_apply/control.py`
- Test: `tests/workflows/auto_apply/test_control_cancel.py`

- [ ] **Step 1: Write failing test**

```python
# tests/workflows/auto_apply/test_control_cancel.py
from langgraph.checkpoint.memory import MemorySaver
from uppgrad_agentic.workflows.auto_apply.control import cancel_session
from uppgrad_agentic.workflows.auto_apply.graph import build_graph


def test_cancel_writes_error_marker_to_state():
    """After cancel, the next graph stream/invoke sees result.status == 'error'."""
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test-cancel-1"}}

    # Run the graph until it interrupts at a gate (or to completion).
    # Then cancel and confirm the state reflects the cancel marker.
    initial = {"opportunity_type": "job", "opportunity_id": "job-001"}
    try:
        for _ in graph.stream(initial, config=config, stream_mode="values"):
            pass
    except Exception:
        pass

    # Cancel
    cancel_session("test-cancel-1", checkpointer)

    # Read back the state — result.status must now be 'error' with code CANCELLED
    final_state = graph.get_state(config).values
    assert final_state.get("result", {}).get("status") == "error"
    assert final_state["result"]["error_code"] == "CANCELLED"


def test_cancel_is_idempotent():
    """Calling cancel twice on the same thread should not raise."""
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test-cancel-2"}}
    try:
        for _ in graph.stream({"opportunity_type": "job", "opportunity_id": "job-001"},
                              config=config, stream_mode="values"):
            pass
    except Exception:
        pass

    cancel_session("test-cancel-2", checkpointer)
    cancel_session("test-cancel-2", checkpointer)  # second call must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/workflows/auto_apply/test_control_cancel.py -v`
Expected: FAIL — `control` module not found.

- [ ] **Step 3: Implement the helper**

```python
# src/uppgrad_agentic/workflows/auto_apply/control.py
from __future__ import annotations

import logging
from typing import Any

from uppgrad_agentic.workflows.auto_apply.graph import build_graph

logger = logging.getLogger(__name__)


def cancel_session(thread_id: str, checkpointer: Any) -> None:
    """Inject a cancel marker into the graph's checkpoint for the given thread_id.

    Uses LangGraph's update_state API. The currently-executing node finishes
    normally; subsequent nodes see result.status == 'error' (error_code='CANCELLED')
    via the existing top-of-node short-circuit pattern, draining the graph to END.
    The background thread running the graph terminates naturally as a result.

    Idempotent: calling this on a thread that is already cancelled (or has
    already terminated) is safe.
    """
    try:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        graph.update_state(config, {
            "result": {
                "status": "error",
                "error_code": "CANCELLED",
                "user_message": "Cancelled by user.",
            },
        })
        logger.info("cancel_session: marker injected for thread_id=%s", thread_id)
    except Exception as exc:
        # Don't propagate — cancel is best-effort by design (Spec T1).
        logger.warning("cancel_session: failed to inject marker for %s — %s", thread_id, exc)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_control_cancel.py -v`
Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/control.py tests/workflows/auto_apply/test_control_cancel.py
git commit -m "feat(auto_apply): add cancel_session helper using graph.update_state"
```

---

## Task A10: Run full agentic-repo test suite + tag release

**Files:** none (verification + version bump)

- [ ] **Step 1: Run full auto_apply test suite**

Run: `uv run pytest tests/workflows/auto_apply/ -v`
Expected: every test added in Tasks A1–A9 passes; existing tests untouched.

- [ ] **Step 2: Bump version in pyproject.toml**

Edit `pyproject.toml`:

```toml
version = "0.2.0"
```

- [ ] **Step 3: Commit + tag**

```bash
git add pyproject.toml
git commit -m "chore: bump version to 0.2.0 for backend integration"
git tag v0.2.0
git push origin main --tags
```

The backend's `requirements.txt` currently pins `@main` so no requirements.txt update is strictly needed, but Phase B will document the recommended pin.

---

# PHASE B — Backend Integration

All Phase B tasks run inside `backend/`.

## Task B1: Install dependencies

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add reportlab**

Append to `backend/requirements.txt`:

```
# PDF rendering for internal-submit Application.resume_file
reportlab>=4.0.0
```

- [ ] **Step 2: Pull updated agentic repo**

If running locally, ensure the agentic repo points at v0.2.0:

```bash
pip install --upgrade --force-reinstall "uppgrad-agentic @ git+https://github.com/uppGrad/AgenticWorkflows_LangChain-LangGraph.git@v0.2.0"
```

For local-dev with editable install (recommended during integration):

```bash
pip install -e ../AgenticWorkflows_LangChain-LangGraph
```

- [ ] **Step 3: Verify imports work**

Run: `python -c "from uppgrad_agentic.workflows.auto_apply.graph import build_graph; from uppgrad_agentic.workflows.auto_apply.control import cancel_session; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore(deps): add reportlab; bump uppgrad-agentic to v0.2.0"
```

---

## Task B2: `ApplicationSession` model

**Files:**
- Modify: `backend/ai_services/models.py`
- Test: `backend/ai_services/tests/__init__.py` (create), `backend/ai_services/tests/test_application_session_model.py`

- [ ] **Step 1: Create test package**

```bash
mkdir -p backend/ai_services/tests
touch backend/ai_services/tests/__init__.py
```

- [ ] **Step 2: Write failing test**

```python
# backend/ai_services/tests/test_application_session_model.py
import uuid
from django.contrib.auth.models import User
from django.db.utils import IntegrityError
from django.test import TestCase

from accounts.models import Student
from ai_services.models import ApplicationSession


class ApplicationSessionModelTests(TestCase):
    def setUp(self):
        user = User.objects.create_user(username="u1", email="u1@x.com", password="x")
        self.student = Student.objects.create(user=user)

    def test_can_create_session(self):
        s = ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=42,
            opportunity_snapshot={"title": "X"},
        )
        assert s.status == ApplicationSession.Status.PROCESSING

    def test_one_active_session_per_student_opportunity(self):
        ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=42, opportunity_snapshot={},
        )
        with self.assertRaises(IntegrityError):
            ApplicationSession.objects.create(
                student=self.student, thread_id=uuid.uuid4(),
                opportunity_type="job", opportunity_id=42, opportunity_snapshot={},
            )

    def test_completed_session_does_not_block_new(self):
        old = ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=42, opportunity_snapshot={},
        )
        old.status = ApplicationSession.Status.COMPLETED_HANDOFF
        old.save()
        # Must not raise — old is no longer active
        ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=42, opportunity_snapshot={},
        )

    def test_different_opportunities_dont_block(self):
        ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=42, opportunity_snapshot={},
        )
        # Different opp_id is fine
        ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=43, opportunity_snapshot={},
        )
        # Different opp_type with same id is fine
        ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="masters", opportunity_id=42, opportunity_snapshot={},
        )
```

- [ ] **Step 3: Run to verify it fails**

Run: `python manage.py test ai_services.tests.test_application_session_model -v 2`
Expected: FAIL — `ApplicationSession` model not found.

- [ ] **Step 4: Add the model**

Append to `backend/ai_services/models.py`:

```python
class ApplicationSession(models.Model):
    """Tracks an agentic auto-apply workflow run.

    Lifecycle:
      PROCESSING
        ↓
      [eligibility]
        ├─ ineligible → INELIGIBLE
        ├─ pending    → AWAITING_PROFILE_COMPLETION
        └─ ready/manual_review → AWAITING_DOCUMENT_MAPPING
                                  ↓
                              FINALIZING (after gate 1 resume)
                                  ↓
                              AWAITING_FINAL_APPROVAL (gate 2)
                                  ↓
                              COMPLETED_SUBMITTED | COMPLETED_HANDOFF
    """

    class Status(models.TextChoices):
        PROCESSING                  = "processing", "Processing"
        AWAITING_PROFILE_COMPLETION = "awaiting_profile_completion", "Awaiting Profile Completion"
        AWAITING_DOCUMENT_MAPPING   = "awaiting_document_mapping",   "Awaiting Document Mapping"
        AWAITING_FINAL_APPROVAL     = "awaiting_final_approval",     "Awaiting Final Approval"
        FINALIZING                  = "finalizing",                  "Finalizing"
        COMPLETED_SUBMITTED         = "completed_submitted",         "Completed (Submitted)"
        COMPLETED_HANDOFF           = "completed_handoff",           "Completed (Handoff)"
        INELIGIBLE                  = "ineligible",                  "Ineligible"
        ERROR                       = "error",                       "Error"
        CANCELLED                   = "cancelled",                   "Cancelled"

    OPPORTUNITY_TYPE_CHOICES = [
        ("job", "Job"), ("masters", "Masters"),
        ("phd", "PhD"), ("scholarship", "Scholarship"),
    ]

    ACTIVE_STATUSES = [
        Status.PROCESSING, Status.AWAITING_PROFILE_COMPLETION,
        Status.AWAITING_DOCUMENT_MAPPING, Status.AWAITING_FINAL_APPROVAL,
        Status.FINALIZING,
    ]

    student          = models.ForeignKey(
        "accounts.Student", on_delete=models.CASCADE, related_name="application_sessions",
    )
    thread_id        = models.UUIDField(unique=True)
    status           = models.CharField(max_length=40, choices=Status.choices, default=Status.PROCESSING)

    opportunity_type = models.CharField(max_length=20, choices=OPPORTUNITY_TYPE_CHOICES)
    opportunity_id   = models.BigIntegerField()
    opportunity_snapshot = models.JSONField()

    eligibility_result   = models.JSONField(null=True, blank=True)
    asset_mapping        = models.JSONField(null=True, blank=True)

    gate_0_response      = models.JSONField(null=True, blank=True)
    gate_1_response      = models.JSONField(null=True, blank=True)
    gate_2_response      = models.JSONField(null=True, blank=True)

    tailored_documents   = models.JSONField(null=True, blank=True)
    application_package  = models.JSONField(null=True, blank=True)
    submission_type      = models.CharField(max_length=20, blank=True, default="")

    discovered_apply_url = models.TextField(blank=True, default="")
    discovery_method     = models.CharField(max_length=20, blank=True, default="")
    discovery_confidence = models.FloatField(null=True, blank=True)

    application_record   = models.JSONField(null=True, blank=True)
    error_message        = models.TextField(blank=True, default="")

    current_step         = models.CharField(max_length=50, blank=True, default="")
    step_history         = models.JSONField(default=list, blank=True)

    application = models.ForeignKey(
        "jobs.Application", on_delete=models.SET_NULL,
        null=True, blank=True, related_name="application_session",
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

    def __str__(self):
        return f"ApplicationSession {self.thread_id} ({self.status})"
```

- [ ] **Step 5: Generate + apply migration**

```bash
python manage.py makemigrations ai_services -n application_session
python manage.py migrate ai_services
```

The migration filename will be `0007_application_session.py`. Open it and confirm the `UniqueConstraint` with the `condition` (partial unique) is present in the operations list. If `makemigrations` chose a different filename suffix, that's fine; just note the path.

- [ ] **Step 6: Run tests**

Run: `python manage.py test ai_services.tests.test_application_session_model -v 2`
Expected: all 4 pass.

- [ ] **Step 7: Commit**

```bash
git add ai_services/models.py ai_services/migrations/0007_application_session.py ai_services/tests/__init__.py ai_services/tests/test_application_session_model.py
git commit -m "feat(ai_services): add ApplicationSession model with partial unique constraint"
```

---

## Task B3: Document renderer (text → PDF)

**Files:**
- Create: `backend/ai_services/document_renderer.py`
- Test: `backend/ai_services/tests/test_document_renderer.py`

- [ ] **Step 1: Write failing test**

```python
# backend/ai_services/tests/test_document_renderer.py
from django.test import SimpleTestCase
from ai_services.document_renderer import render_text_to_pdf


class DocumentRendererTests(SimpleTestCase):
    def test_returns_bytes(self):
        out = render_text_to_pdf("Hello world.\nLine 2.", "Test CV")
        self.assertIsInstance(out, bytes)
        self.assertGreater(len(out), 100)
        # PDF magic bytes
        self.assertTrue(out.startswith(b"%PDF"))

    def test_handles_empty_text(self):
        out = render_text_to_pdf("", "Empty CV")
        self.assertTrue(out.startswith(b"%PDF"))

    def test_handles_long_text_multipage(self):
        long = ("Paragraph " + "x" * 200 + "\n\n") * 50
        out = render_text_to_pdf(long, "Long CV")
        self.assertTrue(out.startswith(b"%PDF"))
        # Multipage PDFs are still under a few hundred KB for plain text
        self.assertLess(len(out), 500_000)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python manage.py test ai_services.tests.test_document_renderer -v 2`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the renderer**

```python
# backend/ai_services/document_renderer.py
"""Plain-text → PDF rendering for internal-submit Application.resume_file.

Uses ReportLab for simplicity. Does not preserve formatting beyond paragraph
breaks — for richer CV layouts the caller should use the LaTeX compiler in
the agentic repo's tools/ instead. This renderer exists so that the internal
submit path always has a Working PDF without a LaTeX dependency.
"""
from __future__ import annotations

import io

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


def render_text_to_pdf(text: str, title: str = "Document") -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=LETTER,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        title=title,
    )

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontName="Helvetica", fontSize=11, leading=14, spaceAfter=6,
    )
    heading = ParagraphStyle(
        "Heading", parent=styles["Title"], fontSize=14, spaceAfter=10,
    )

    story = [Paragraph(_escape(title), heading), Spacer(1, 0.15 * inch)]

    paragraphs = (text or "").split("\n\n")
    for para in paragraphs:
        for line in para.splitlines():
            line = line.strip()
            if line:
                story.append(Paragraph(_escape(line), body))
        story.append(Spacer(1, 0.10 * inch))

    doc.build(story)
    return buf.getvalue()


def _escape(s: str) -> str:
    """Minimal escaping for ReportLab's mini-XML paragraph parser."""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )
```

- [ ] **Step 4: Run tests**

Run: `python manage.py test ai_services.tests.test_document_renderer -v 2`
Expected: all 3 pass.

- [ ] **Step 5: Commit**

```bash
git add ai_services/document_renderer.py ai_services/tests/test_document_renderer.py
git commit -m "feat(ai_services): add ReportLab-based text-to-PDF renderer for internal submit"
```

---

## Task B4: Adapter — opportunity & profile snapshot builders

**Files:**
- Create: `backend/ai_services/auto_apply_adapter.py` (incremental — first chunk)
- Test: `backend/ai_services/tests/test_auto_apply_adapter.py`

- [ ] **Step 1: Write failing test**

```python
# backend/ai_services/tests/test_auto_apply_adapter.py
from django.contrib.auth.models import User
from django.test import TestCase

from accounts.models import Student
from jobs.models import JobOpportunity


class OpportunitySnapshotTests(TestCase):
    def setUp(self):
        self.job = JobOpportunity.objects.create(
            title="Senior SWE", company="Acme", location="London, UK",
            description="Build great things.", site="linkedin", employer_id=None,
            url="https://www.linkedin.com/jobs/view/100",
            url_direct="https://acme.com/careers/100",
        )

    def test_builds_job_snapshot_with_all_fields(self):
        from ai_services.auto_apply_adapter import build_auto_apply_opportunity_snapshot
        snap = build_auto_apply_opportunity_snapshot("job", self.job.id)
        assert snap["id"] == self.job.id
        assert snap["title"] == "Senior SWE"
        assert snap["company"] == "Acme"
        assert snap["url_direct"] == "https://acme.com/careers/100"
        assert snap["employer_id"] is None
        assert snap["is_closed"] is False

    def test_returns_none_for_unknown_id(self):
        from ai_services.auto_apply_adapter import build_auto_apply_opportunity_snapshot
        snap = build_auto_apply_opportunity_snapshot("job", 99999999)
        assert snap is None

    def test_returns_none_for_closed_job(self):
        from ai_services.auto_apply_adapter import build_auto_apply_opportunity_snapshot
        self.job.is_closed = True
        self.job.save()
        snap = build_auto_apply_opportunity_snapshot("job", self.job.id)
        assert snap is None


class ProfileSnapshotTests(TestCase):
    def setUp(self):
        user = User.objects.create_user(
            username="u1", email="u1@example.com",
            first_name="Alex", last_name="Johnson", password="x",
        )
        self.student = Student.objects.create(user=user)

    def test_builds_minimal_profile_when_no_cv(self):
        from ai_services.auto_apply_adapter import build_auto_apply_profile_snapshot
        snap = build_auto_apply_profile_snapshot(self.student)
        assert snap["name"] == "Alex Johnson"
        assert snap["email"] == "u1@example.com"
        assert snap["uploaded_documents"]["CV"] is False
        assert snap["uploaded_documents"]["Cover Letter"] is False  # always False in v1
        assert snap["document_texts"] == {}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python manage.py test ai_services.tests.test_auto_apply_adapter -v 2`
Expected: FAIL — module not found.

- [ ] **Step 3: Create adapter (first chunk)**

```python
# backend/ai_services/auto_apply_adapter.py
"""Bridge between Django ORM and the auto_apply LangGraph workflow.

Mirrors the pattern in graph_adapter.py used for document_feedback. Adds:
  * Opportunity snapshot builders for all four opportunity types.
  * Per-gate resume handlers (gates 0, 1, 2).
  * finalize_internal_submission — Django ORM write of an Application row
    after the graph terminates with submission_type == 'internal'.
  * cancel_session — calls into the agentic repo's control.cancel_session.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from accounts.models import Student, StudentCV
from jobs.models import JobOpportunity
from common.models import ProgramOpportunity, ScholarshipOpportunity

logger = logging.getLogger(__name__)


# ─── Opportunity Snapshot Builder ────────────────────────────────────────────

def build_auto_apply_opportunity_snapshot(opp_type: str, opp_id: int) -> Optional[Dict[str, Any]]:
    """Pre-load opportunity data for the auto_apply graph.

    Returns None when:
      - The opportunity ID does not exist.
      - The opportunity is closed (is_closed=True for jobs).

    The dict shape matches what auto_apply nodes expect:
      - jobs:    full linkedin_jobs row including url, url_direct, employer_id
      - masters/phd: programs row including the data json verbatim
      - scholarship: scholarships row including the data json verbatim
    """
    if opp_type == "job":
        obj = JobOpportunity.objects.filter(id=opp_id).first()
        if obj is None or obj.is_closed:
            return None
        return {
            "id": obj.id,
            "title": obj.title,
            "company": obj.company,
            "company_url": obj.company_url,
            "location": obj.location,
            "description": obj.description,
            "url": obj.url,
            "url_direct": obj.url_direct,
            "site": obj.site,
            "employer_id": obj.employer_id,
            "posted_time": obj.posted_time,
            "is_closed": obj.is_closed,
            "is_remote": obj.is_remote,
            "salary": obj.salary,
            "job_type": obj.job_type,
            "job_level": obj.job_level,
        }

    if opp_type in ("masters", "phd"):
        obj = ProgramOpportunity.objects.filter(id=opp_id, program_type=opp_type).first()
        if obj is None:
            return None
        return {
            "id": obj.id,
            "title": obj.title,
            "university": obj.university,
            "url": obj.url,
            "location": obj.location,
            "duration": obj.duration,
            "degree_type": obj.degree_type,
            "study_mode": obj.study_mode,
            "program_type": obj.program_type,
            "tuition_fee": obj.tuition_fee,
            "venue": obj.venue,
            "data": obj.data or {},
        }

    if opp_type == "scholarship":
        obj = ScholarshipOpportunity.objects.filter(id=opp_id).first()
        if obj is None:
            return None
        return {
            "id": obj.id,
            "title": obj.title,
            "url": obj.url,
            "provider_name": obj.provider_name,
            "disciplines": obj.disciplines,
            "location": obj.location,
            "deadline": obj.deadline.isoformat() if obj.deadline else None,
            "scholarship_type": obj.scholarship_type,
            "coverage": obj.coverage,
            "description": obj.description,
            "benefits": obj.benefits,
            "eligibility_text": obj.eligibility_text,
            "req_disciplines": obj.req_disciplines,
            "req_locations": obj.req_locations,
            "req_nationality": obj.req_nationality,
            "req_age": obj.req_age,
            "req_study_experience": obj.req_study_experience,
            "application_info": obj.application_info,
            "data": obj.data or {},
        }

    return None


# ─── Profile Snapshot Builder ────────────────────────────────────────────────

def build_auto_apply_profile_snapshot(student: Student) -> Dict[str, Any]:
    """Build the profile dict the auto_apply graph expects.

    v1 cap: only CV is sourced from a stored user file (StudentCV). Cover Letter,
    SOP, Personal Statement, etc. are marked unavailable and will be *generated*
    by application_tailoring (depth='generate'). Spec §11.3.
    """
    user = student.user

    # Education
    education = []
    for ed in student.education_entries.all():
        education.append({
            "degree": ed.title_obtained, "institution": ed.university,
            "year": ed.end_year, "gpa": float(ed.gpa) if ed.gpa else None,
        })

    skills = list(student.skills.values_list("name", flat=True))

    # CV text extraction
    cv_text = ""
    cv_uploaded = False
    cv = StudentCV.objects.filter(student=student).first()
    if cv and getattr(cv, "cv_file", None):
        cv_uploaded = True
        try:
            from .graph_adapter import _extract_text_with_ocr_fallback
            cv_text = _extract_text_with_ocr_fallback(cv.cv_file.path)
        except Exception as exc:
            logger.warning("auto_apply: CV text extraction failed for student %s — %s", student.id, exc)

    return {
        "name": f"{user.first_name} {user.last_name}".strip() or user.username,
        "email": user.email,
        "age": getattr(student, "age", None),
        "nationality": getattr(student, "nationality", "") or "",
        "location": getattr(student, "location", "") or "",
        "degree_level": education[0]["degree"] if education else "",
        "disciplines": skills,
        "gpa": education[0]["gpa"] if education and education[0].get("gpa") else None,
        "uploaded_documents": {
            "CV": cv_uploaded,
            "Cover Letter": False,
            "SOP": False,
            "Personal Statement": False,
            "Research Proposal": False,
            "Transcript": False,
            "References": False,
            "English Proficiency Test": False,
            "Portfolio": False,
            "Writing Sample": False,
        },
        "document_texts": {"CV": cv_text} if cv_text else {},
    }
```

- [ ] **Step 4: Run tests**

Run: `python manage.py test ai_services.tests.test_auto_apply_adapter -v 2`
Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add ai_services/auto_apply_adapter.py ai_services/tests/test_auto_apply_adapter.py
git commit -m "feat(ai_services): add opportunity and profile snapshot builders for auto_apply"
```

---

## Task B5: Adapter — checkpointed graph builder + start_session

**Files:**
- Modify: `backend/ai_services/auto_apply_adapter.py` (append)
- Modify: `backend/ai_services/tests/test_auto_apply_adapter.py` (append)

- [ ] **Step 1: Add test for `start_session`**

Append to `backend/ai_services/tests/test_auto_apply_adapter.py`:

```python
import uuid
from unittest.mock import patch
from django.test import TransactionTestCase

from ai_services.models import ApplicationSession


class StartSessionTests(TransactionTestCase):
    def setUp(self):
        from django.contrib.auth.models import User
        user = User.objects.create_user(username="u2", email="u2@x.com", password="x")
        self.student = Student.objects.create(user=user)
        self.job = JobOpportunity.objects.create(
            title="SWE", company="Acme", site="linkedin",
            url="https://x.com", employer_id=None,
        )

    def test_start_session_writes_processing_status_and_launches_thread(self):
        session = ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=self.job.id,
            opportunity_snapshot={"id": self.job.id, "title": "SWE", "company": "Acme",
                                  "is_closed": False, "employer_id": None},
        )
        from ai_services.auto_apply_adapter import start_session

        with patch("ai_services.auto_apply_adapter.threading.Thread") as fake_thread:
            start_session(session)
            fake_thread.assert_called_once()
            # Confirm the target points at our phase function
            target = fake_thread.call_args.kwargs["target"]
            assert target.__name__ == "_run_graph_initial_phase"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python manage.py test ai_services.tests.test_auto_apply_adapter.StartSessionTests -v 2`
Expected: FAIL — `start_session` not defined.

- [ ] **Step 3: Append to adapter**

Append to `backend/ai_services/auto_apply_adapter.py`:

```python
import threading
import psycopg
from django import db as django_db
from django.utils import timezone

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from uppgrad_agentic.workflows.auto_apply.graph import build_graph as build_auto_apply_graph
from uppgrad_agentic.workflows.auto_apply.control import cancel_session as agentic_cancel_session

from .graph_adapter import _get_db_url, _get_direct_db_url

# Shared with graph_adapter via the global flag — both adapters use the same
# checkpoint tables, so setup() must run only once per process.
from . import graph_adapter as _gd_module


def _build_checkpointed_auto_apply_graph():
    """Build the auto_apply graph with a Postgres-backed checkpointer."""
    if not _gd_module._checkpointer_setup_done:
        setup_conn = psycopg.connect(_get_direct_db_url(), autocommit=True)
        try:
            PostgresSaver(setup_conn).setup()
            _gd_module._checkpointer_setup_done = True
        finally:
            setup_conn.close()

    conn = psycopg.connect(_get_db_url(), autocommit=True)
    checkpointer = PostgresSaver(conn)
    return build_auto_apply_graph(checkpointer=checkpointer), conn


# ─── Initial Phase ───────────────────────────────────────────────────────────

def start_session(session):
    """Launch the auto_apply graph in a background thread for the given session."""
    thread = threading.Thread(
        target=_run_graph_initial_phase,
        args=(session.id,),
        daemon=True,
    )
    thread.start()


def _run_graph_initial_phase(session_id: int) -> None:
    """Execute the graph from start until the first interrupt or terminal."""
    conn = None
    try:
        django_db.close_old_connections()

        session = ApplicationSession.objects.select_related("student__user").get(id=session_id)
        student = session.student

        profile_snapshot = build_auto_apply_profile_snapshot(student)
        opportunity_snapshot = session.opportunity_snapshot   # already pre-loaded at create-time

        graph, conn = _build_checkpointed_auto_apply_graph()
        config = {"configurable": {"thread_id": str(session.thread_id)}}

        initial_state = {
            "opportunity_type": session.opportunity_type,
            "opportunity_id": str(session.opportunity_id),
            "opportunity_data": opportunity_snapshot,
            "profile_snapshot": profile_snapshot,
            "gate_0_iteration_count": 0,
        }

        logger.info("auto_apply: starting graph for session %s (thread %s)", session_id, session.thread_id)
        result_state = graph.invoke(initial_state, config)

        _persist_state_after_phase(session, result_state)

    except Exception as exc:
        logger.exception("auto_apply: graph initial phase failed for session %s", session_id)
        try:
            session = ApplicationSession.objects.get(id=session_id)
            session.status = ApplicationSession.Status.ERROR
            session.error_message = str(exc)[:2000]
            session.save(update_fields=["status", "error_message", "updated_at"])
        except Exception:
            pass
    finally:
        if conn:
            try: conn.close()
            except Exception: pass
        django_db.close_old_connections()


def _persist_state_after_phase(session, result_state: Dict[str, Any]) -> None:
    """Map graph state back to the ApplicationSession row.

    Detects interrupt vs terminal by inspecting state.result and state.eligibility_result.
    """
    if not isinstance(result_state, dict):
        session.status = ApplicationSession.Status.ERROR
        session.error_message = f"Graph returned non-dict state: {type(result_state)}"
        session.save(update_fields=["status", "error_message", "updated_at"])
        return

    # Always copy progress indicators
    session.current_step = result_state.get("current_step") or session.current_step
    if "step_history" in result_state:
        session.step_history = result_state["step_history"]
    if "eligibility_result" in result_state:
        session.eligibility_result = result_state["eligibility_result"]
    if "asset_mapping" in result_state:
        session.asset_mapping = result_state["asset_mapping"]
    if "tailored_documents" in result_state:
        session.tailored_documents = result_state["tailored_documents"]
    if "application_package" in result_state:
        session.application_package = result_state["application_package"]
    if "application_record" in result_state:
        session.application_record = result_state["application_record"]
    session.discovered_apply_url = result_state.get("discovered_apply_url") or ""
    session.discovery_method = result_state.get("discovery_method") or ""
    session.discovery_confidence = result_state.get("discovery_confidence")

    result = result_state.get("result") or {}

    # Error path
    if result.get("status") == "error":
        if result.get("error_code") == "CANCELLED":
            session.status = ApplicationSession.Status.CANCELLED
        elif result.get("error_code") == "PROFILE_INCOMPLETE_AFTER_RETRIES":
            session.status = ApplicationSession.Status.INELIGIBLE
        else:
            session.status = ApplicationSession.Status.ERROR
        session.error_message = result.get("user_message", "")
        session.save()
        return

    # Eligibility verdicts
    elig = result_state.get("eligibility_result") or {}
    if elig.get("decision") == "ineligible":
        session.status = ApplicationSession.Status.INELIGIBLE
        session.error_message = "; ".join(elig.get("reasons", []))[:2000]
        session.save()
        return

    # Currently held at an interrupt? Use current_step to disambiguate.
    current = result_state.get("current_step", "")
    if current == "human_gate_0":
        session.status = ApplicationSession.Status.AWAITING_PROFILE_COMPLETION
    elif current == "human_gate_1":
        session.status = ApplicationSession.Status.AWAITING_DOCUMENT_MAPPING
    elif current == "human_gate_2":
        session.status = ApplicationSession.Status.AWAITING_FINAL_APPROVAL
    else:
        # Reached a terminal — distinguish submitted vs handoff by submission_type
        package = result_state.get("application_package") or {}
        if package.get("submission_type") == "internal":
            session.status = ApplicationSession.Status.FINALIZING
            session.submission_type = "internal"
        else:
            session.status = ApplicationSession.Status.COMPLETED_HANDOFF
            session.submission_type = "handoff"
            session.completed_at = timezone.now()

    session.save()
```

- [ ] **Step 4: Run tests**

Run: `python manage.py test ai_services.tests.test_auto_apply_adapter.StartSessionTests -v 2`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add ai_services/auto_apply_adapter.py ai_services/tests/test_auto_apply_adapter.py
git commit -m "feat(ai_services): add start_session and initial-phase runner for auto_apply"
```

---

## Task B6: Adapter — three gate-resume handlers

**Files:**
- Modify: `backend/ai_services/auto_apply_adapter.py` (append)
- Modify: `backend/ai_services/tests/test_auto_apply_adapter.py` (append)

- [ ] **Step 1: Write failing test**

Append to `backend/ai_services/tests/test_auto_apply_adapter.py`:

```python
class ResumeHandlerSignaturesTests(TestCase):
    def test_all_three_resume_handlers_exist(self):
        from ai_services.auto_apply_adapter import (
            resume_session_gate_0, resume_session_gate_1, resume_session_gate_2,
        )
        # Just verify they're importable callables
        assert callable(resume_session_gate_0)
        assert callable(resume_session_gate_1)
        assert callable(resume_session_gate_2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python manage.py test ai_services.tests.test_auto_apply_adapter.ResumeHandlerSignaturesTests -v 2`
Expected: FAIL — handlers not defined.

- [ ] **Step 3: Append handlers to adapter**

Append to `backend/ai_services/auto_apply_adapter.py`:

```python
# ─── Gate Resume Handlers ────────────────────────────────────────────────────

def resume_session_gate_0(session, payload: Dict[str, Any]) -> None:
    """Resume after gate 0 (profile completion).

    payload shape:
      {
        "profile_updates": {...},  // optional fields to write to Student
        "uploaded_cv": <File>      // optional; written to StudentCV
      }
    Writes profile updates to the DB *before* resuming the graph, so the
    eligibility re-check sees the updated state.
    """
    student = session.student

    profile_updates = payload.get("profile_updates") or {}
    if profile_updates:
        for field in ("nationality", "location"):
            if field in profile_updates:
                setattr(student, field, profile_updates[field])
        student.save()

    uploaded_cv = payload.get("uploaded_cv")
    if uploaded_cv is not None:
        cv = StudentCV.objects.filter(student=student).first() or StudentCV(student=student)
        cv.cv_file = uploaded_cv
        cv.save()

    session.gate_0_response = {"profile_completed": True}
    session.status = ApplicationSession.Status.PROCESSING
    session.save(update_fields=["gate_0_response", "status", "updated_at"])

    threading.Thread(
        target=_run_graph_resume_phase,
        args=(session.id, {"profile_completed": True}),
        daemon=True,
    ).start()


def resume_session_gate_1(session, payload: Dict[str, Any]) -> None:
    """Resume after gate 1 (document mapping).

    payload shape:
      {
        "confirm": True,
        "overrides": [{"document_type": "...", "use_default": True | "uploaded_file_ref": "..."}],
        "additional_uploads": [...]   // multipart files already saved by view
      }
    """
    session.gate_1_response = payload
    session.status = ApplicationSession.Status.PROCESSING
    session.save(update_fields=["gate_1_response", "status", "updated_at"])

    resume_value = {
        "confirm": payload.get("confirm", True),
        "overrides": payload.get("overrides", []),
    }
    threading.Thread(
        target=_run_graph_resume_phase,
        args=(session.id, resume_value),
        daemon=True,
    ).start()


def resume_session_gate_2(session, payload: Dict[str, Any]) -> None:
    """Resume after gate 2 (final approval).

    payload shape:
      {
        "approved": True | False,
        "feedback": [{"document_type": "...", "comment": "..."}]   // optional
      }
    If approved=False, terminates the session as CANCELLED (Spec Q2 v1 cap).
    Otherwise resumes; freshness re-check before resume (Spec §6.4).
    """
    if not payload.get("approved", False):
        session.gate_2_response = payload
        session.status = ApplicationSession.Status.CANCELLED
        session.error_message = "User rejected the tailored package at final review."
        session.save(update_fields=["gate_2_response", "status", "error_message", "updated_at"])
        # Inject cancel into the graph so the thread terminates cleanly
        _gate_2_cancel_graph(session)
        return

    # Freshness re-check (Spec §6.4)
    fresh = build_auto_apply_opportunity_snapshot(session.opportunity_type, session.opportunity_id)
    if fresh is None:
        session.status = ApplicationSession.Status.INELIGIBLE
        session.error_message = "This opportunity has just closed or been removed."
        session.save(update_fields=["status", "error_message", "updated_at"])
        _gate_2_cancel_graph(session)
        return

    session.gate_2_response = payload
    session.status = ApplicationSession.Status.FINALIZING
    session.save(update_fields=["gate_2_response", "status", "updated_at"])

    threading.Thread(
        target=_run_graph_resume_phase,
        args=(session.id, {"approved": True}),
        daemon=True,
    ).start()


def _gate_2_cancel_graph(session) -> None:
    """Inject CANCELLED marker so the suspended graph thread can drain."""
    setup_conn = psycopg.connect(_get_db_url(), autocommit=True)
    try:
        checkpointer = PostgresSaver(setup_conn)
        agentic_cancel_session(str(session.thread_id), checkpointer)
    finally:
        setup_conn.close()


def _run_graph_resume_phase(session_id: int, resume_value: Dict[str, Any]) -> None:
    """Resume the graph from its current interrupt with `resume_value`."""
    conn = None
    try:
        django_db.close_old_connections()
        session = ApplicationSession.objects.get(id=session_id)
        graph, conn = _build_checkpointed_auto_apply_graph()
        config = {"configurable": {"thread_id": str(session.thread_id)}}

        result_state = graph.invoke(Command(resume=resume_value), config)
        _persist_state_after_phase(session, result_state)

        # If finalizing on internal-submit path, write the Application row now.
        if session.status == ApplicationSession.Status.FINALIZING:
            finalize_internal_submission(session)

    except Exception as exc:
        logger.exception("auto_apply: resume phase failed for session %s", session_id)
        try:
            session = ApplicationSession.objects.get(id=session_id)
            session.status = ApplicationSession.Status.ERROR
            session.error_message = str(exc)[:2000]
            session.save(update_fields=["status", "error_message", "updated_at"])
        except Exception:
            pass
    finally:
        if conn:
            try: conn.close()
            except Exception: pass
        django_db.close_old_connections()
```

- [ ] **Step 4: Run tests**

Run: `python manage.py test ai_services.tests.test_auto_apply_adapter.ResumeHandlerSignaturesTests -v 2`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add ai_services/auto_apply_adapter.py ai_services/tests/test_auto_apply_adapter.py
git commit -m "feat(ai_services): add three gate-resume handlers and resume runner"
```

---

## Task B7: Adapter — `finalize_internal_submission` + `cancel_session`

**Files:**
- Modify: `backend/ai_services/auto_apply_adapter.py` (append)
- Modify: `backend/ai_services/tests/test_auto_apply_adapter.py` (append)

- [ ] **Step 1: Write failing test**

Append to `backend/ai_services/tests/test_auto_apply_adapter.py`:

```python
class FinalizeInternalSubmissionTests(TestCase):
    def setUp(self):
        from django.contrib.auth.models import User
        user = User.objects.create_user(username="u3", email="u3@x.com", password="x")
        self.student = Student.objects.create(user=user)
        self.job = JobOpportunity.objects.create(
            title="Internal SWE", company="UppGrad", site="manual", employer_id=1,
        )

    def test_writes_application_row_with_pdf(self):
        from ai_services.auto_apply_adapter import finalize_internal_submission
        from jobs.models import Application
        import uuid

        session = ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=self.job.id,
            opportunity_snapshot={"id": self.job.id, "employer_id": 1},
            status=ApplicationSession.Status.FINALIZING,
            submission_type="internal",
            tailored_documents={
                "CV": {"content": "Tailored CV body."},
                "Cover Letter": {"content": "Tailored cover letter body."},
            },
            application_package={"submission_type": "internal"},
        )
        finalize_internal_submission(session)

        session.refresh_from_db()
        assert session.status == ApplicationSession.Status.COMPLETED_SUBMITTED
        assert session.application is not None
        app = session.application
        assert app.status == Application.Status.SUBMITTED
        assert app.cover_letter == "Tailored cover letter body."
        assert app.resume_file.name   # FileField populated with rendered PDF
```

- [ ] **Step 2: Run to verify it fails**

Run: `python manage.py test ai_services.tests.test_auto_apply_adapter.FinalizeInternalSubmissionTests -v 2`
Expected: FAIL — `finalize_internal_submission` not defined.

- [ ] **Step 3: Append to adapter**

Append to `backend/ai_services/auto_apply_adapter.py`:

```python
from django.core.files.base import ContentFile

# ─── Internal Submission Finalizer ──────────────────────────────────────────

def finalize_internal_submission(session) -> None:
    """Write a real Application row using Django ORM (Spec A3 — no HTTP loopback).

    Called after the graph terminates with submission_type='internal'.
    """
    from jobs.models import Application, JobOpportunity
    from .document_renderer import render_text_to_pdf

    job = JobOpportunity.objects.filter(id=session.opportunity_id).first()
    if job is None:
        session.status = ApplicationSession.Status.ERROR
        session.error_message = "Opportunity disappeared during finalization."
        session.save(update_fields=["status", "error_message", "updated_at"])
        return

    tailored = session.tailored_documents or {}
    cv_content = (tailored.get("CV") or {}).get("content", "")
    cl_content = (tailored.get("Cover Letter") or {}).get("content", "")

    pdf_bytes = render_text_to_pdf(cv_content, title=f"CV — {session.student.user.email}")
    pdf_filename = f"applications/resumes/{session.thread_id}.pdf"

    application = Application(
        student=session.student,
        job=job,
        cover_letter=cl_content,
        status=Application.Status.SUBMITTED,
    )
    application.resume_file.save(pdf_filename, ContentFile(pdf_bytes), save=False)
    application.save()

    session.application = application
    session.status = ApplicationSession.Status.COMPLETED_SUBMITTED
    session.completed_at = timezone.now()
    session.save(update_fields=[
        "application", "status", "completed_at", "updated_at",
    ])

    logger.info("auto_apply: created Application id=%s for session %s", application.id, session.id)


# ─── Cancel ──────────────────────────────────────────────────────────────────

def cancel_session(session) -> None:
    """Cancel a running auto_apply session (Spec A7).

    Best-effort: the currently-executing node finishes; subsequent nodes
    short-circuit. Status flips to CANCELLED immediately.
    """
    setup_conn = psycopg.connect(_get_db_url(), autocommit=True)
    try:
        checkpointer = PostgresSaver(setup_conn)
        agentic_cancel_session(str(session.thread_id), checkpointer)
    finally:
        setup_conn.close()

    session.status = ApplicationSession.Status.CANCELLED
    session.error_message = "Session cancelled by user."
    session.save(update_fields=["status", "error_message", "updated_at"])
```

- [ ] **Step 4: Run tests**

Run: `python manage.py test ai_services.tests.test_auto_apply_adapter.FinalizeInternalSubmissionTests -v 2`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add ai_services/auto_apply_adapter.py ai_services/tests/test_auto_apply_adapter.py
git commit -m "feat(ai_services): add finalize_internal_submission (ORM write) and cancel_session"
```

---

## Task B8: Serializers

**Files:**
- Modify: `backend/ai_services/serializers.py`
- Test: `backend/ai_services/tests/test_application_session_serializers.py`

- [ ] **Step 1: Write failing test**

```python
# backend/ai_services/tests/test_application_session_serializers.py
from django.test import SimpleTestCase
from rest_framework.test import APIRequestFactory


class ApplicationSessionStartSerializerTests(SimpleTestCase):
    def test_valid_input(self):
        from ai_services.serializers import ApplicationSessionStartSerializer
        s = ApplicationSessionStartSerializer(data={
            "opportunity_type": "job", "opportunity_id": 42,
        })
        assert s.is_valid(), s.errors

    def test_invalid_opportunity_type(self):
        from ai_services.serializers import ApplicationSessionStartSerializer
        s = ApplicationSessionStartSerializer(data={
            "opportunity_type": "invalid", "opportunity_id": 42,
        })
        assert not s.is_valid()

    def test_missing_id(self):
        from ai_services.serializers import ApplicationSessionStartSerializer
        s = ApplicationSessionStartSerializer(data={"opportunity_type": "job"})
        assert not s.is_valid()


class GateResumeSerializerTests(SimpleTestCase):
    def test_gate_0_accepts_profile_updates_or_file(self):
        from ai_services.serializers import Gate0ResumeSerializer
        s = Gate0ResumeSerializer(data={"profile_updates": {"location": "London"}})
        assert s.is_valid(), s.errors

    def test_gate_1_requires_confirm(self):
        from ai_services.serializers import Gate1ResumeSerializer
        s = Gate1ResumeSerializer(data={})
        assert not s.is_valid()
        s = Gate1ResumeSerializer(data={"confirm": True})
        assert s.is_valid(), s.errors

    def test_gate_2_requires_approved_bool(self):
        from ai_services.serializers import Gate2ResumeSerializer
        assert Gate2ResumeSerializer(data={"approved": True}).is_valid()
        assert Gate2ResumeSerializer(data={"approved": False}).is_valid()
        assert not Gate2ResumeSerializer(data={}).is_valid()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python manage.py test ai_services.tests.test_application_session_serializers -v 2`
Expected: FAIL — serializers not defined.

- [ ] **Step 3: Add serializers**

Append to `backend/ai_services/serializers.py`:

```python
from .models import ApplicationSession


class ApplicationSessionStartSerializer(serializers.Serializer):
    opportunity_type = serializers.ChoiceField(
        choices=[("job", "Job"), ("masters", "Masters"), ("phd", "PhD"), ("scholarship", "Scholarship")],
    )
    opportunity_id = serializers.IntegerField(min_value=1)


class ApplicationSessionSerializer(serializers.ModelSerializer):
    application_id = serializers.IntegerField(source="application_id", read_only=True)

    class Meta:
        model = ApplicationSession
        fields = [
            "id", "thread_id", "status",
            "opportunity_type", "opportunity_id", "opportunity_snapshot",
            "current_step", "step_history",
            "eligibility_result", "asset_mapping", "tailored_documents",
            "application_package", "submission_type",
            "discovered_apply_url", "discovery_method", "discovery_confidence",
            "error_message", "application_id",
            "created_at", "updated_at", "completed_at",
        ]
        read_only_fields = fields


class Gate0ResumeSerializer(serializers.Serializer):
    profile_updates = serializers.DictField(required=False, default=dict)
    uploaded_cv = serializers.FileField(required=False, allow_null=True)


class Gate1ResumeSerializer(serializers.Serializer):
    confirm = serializers.BooleanField(required=True)
    overrides = serializers.ListField(
        child=serializers.DictField(), required=False, default=list,
    )
    additional_uploads = serializers.ListField(
        child=serializers.FileField(), required=False, default=list,
    )


class Gate2ResumeSerializer(serializers.Serializer):
    approved = serializers.BooleanField(required=True)
    feedback = serializers.ListField(
        child=serializers.DictField(), required=False, default=list,
    )
```

- [ ] **Step 4: Run tests**

Run: `python manage.py test ai_services.tests.test_application_session_serializers -v 2`
Expected: all 6 pass.

- [ ] **Step 5: Commit**

```bash
git add ai_services/serializers.py ai_services/tests/test_application_session_serializers.py
git commit -m "feat(ai_services): add ApplicationSession serializers"
```

---

## Task B9: Views — start, list, detail, cancel

**Files:**
- Modify: `backend/ai_services/views.py`
- Modify: `backend/ai_services/urls.py`
- Test: `backend/ai_services/tests/test_application_session_views.py`

- [ ] **Step 1: Write failing test**

```python
# backend/ai_services/tests/test_application_session_views.py
import uuid
from unittest.mock import patch
from django.contrib.auth.models import User
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient, APITestCase

from accounts.models import Student
from jobs.models import JobOpportunity
from ai_services.models import ApplicationSession


class ApplicationSessionViewsTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="s1", email="s1@x.com", first_name="Stu", last_name="Dent", password="x",
        )
        self.student = Student.objects.create(user=self.user)
        self.client.force_authenticate(self.user)
        self.job = JobOpportunity.objects.create(
            title="SWE", company="Acme", site="linkedin", employer_id=None,
            url="https://linkedin.com/jobs/view/1",
        )

    @patch("ai_services.auto_apply_adapter.start_session")
    def test_start_creates_session_and_launches(self, fake_start):
        url = reverse("application-session-list-create")
        resp = self.client.post(url, {"opportunity_type": "job", "opportunity_id": self.job.id}, format="json")
        assert resp.status_code == status.HTTP_201_CREATED, resp.data
        assert ApplicationSession.objects.filter(student=self.student).count() == 1
        fake_start.assert_called_once()

    def test_start_rejects_closed_job(self):
        self.job.is_closed = True
        self.job.save()
        url = reverse("application-session-list-create")
        resp = self.client.post(url, {"opportunity_type": "job", "opportunity_id": self.job.id}, format="json")
        assert resp.status_code == status.HTTP_404_NOT_FOUND

    @patch("ai_services.auto_apply_adapter.start_session")
    def test_start_rejects_active_duplicate(self, _):
        url = reverse("application-session-list-create")
        self.client.post(url, {"opportunity_type": "job", "opportunity_id": self.job.id}, format="json")
        resp2 = self.client.post(url, {"opportunity_type": "job", "opportunity_id": self.job.id}, format="json")
        assert resp2.status_code == status.HTTP_409_CONFLICT

    def test_detail_returns_session(self):
        s = ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=self.job.id,
            opportunity_snapshot={"id": self.job.id},
        )
        resp = self.client.get(reverse("application-session-detail", args=[s.id]))
        assert resp.status_code == 200
        assert resp.data["id"] == s.id

    @patch("ai_services.auto_apply_adapter.cancel_session")
    def test_cancel(self, fake_cancel):
        s = ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=self.job.id,
            opportunity_snapshot={"id": self.job.id},
        )
        resp = self.client.post(reverse("application-session-cancel", args=[s.id]))
        assert resp.status_code == status.HTTP_200_OK
        fake_cancel.assert_called_once()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python manage.py test ai_services.tests.test_application_session_views -v 2`
Expected: FAIL — views and routes not defined.

- [ ] **Step 3: Add views**

Append to `backend/ai_services/views.py`:

```python
import uuid
import logging
from .models import ApplicationSession
from .serializers import (
    ApplicationSessionStartSerializer, ApplicationSessionSerializer,
    Gate0ResumeSerializer, Gate1ResumeSerializer, Gate2ResumeSerializer,
)
from .auto_apply_adapter import (
    start_session as auto_apply_start,
    resume_session_gate_0, resume_session_gate_1, resume_session_gate_2,
    cancel_session as auto_apply_cancel,
    build_auto_apply_opportunity_snapshot,
)

_logger = logging.getLogger(__name__)


class ApplicationSessionListCreateView(APIView):
    permission_classes = [IsStudent]

    def get(self, request):
        student = request.user.student
        sessions = ApplicationSession.objects.filter(student=student)
        return Response(ApplicationSessionSerializer(sessions, many=True).data)

    def post(self, request):
        ser = ApplicationSessionStartSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)

        student = request.user.student
        opp_type = ser.validated_data["opportunity_type"]
        opp_id = ser.validated_data["opportunity_id"]

        snapshot = build_auto_apply_opportunity_snapshot(opp_type, opp_id)
        if snapshot is None:
            return Response(
                {"error": "Opportunity not found or has been closed."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Check for active duplicate
        active = ApplicationSession.objects.filter(
            student=student,
            opportunity_type=opp_type, opportunity_id=opp_id,
            status__in=ApplicationSession.ACTIVE_STATUSES,
        ).first()
        if active:
            return Response(
                {"error": "An active session already exists for this opportunity.",
                 "active_session_id": active.id},
                status=status.HTTP_409_CONFLICT,
            )

        session = ApplicationSession.objects.create(
            student=student, thread_id=uuid.uuid4(),
            opportunity_type=opp_type, opportunity_id=opp_id,
            opportunity_snapshot=snapshot,
        )

        try:
            auto_apply_start(session)
        except Exception as exc:
            _logger.exception("Failed to launch auto_apply session %s", session.id)
            session.status = ApplicationSession.Status.ERROR
            session.error_message = f"Failed to start: {exc}"
            session.save(update_fields=["status", "error_message", "updated_at"])
            return Response(
                {"error": session.error_message, "session_id": session.id},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            ApplicationSessionSerializer(session).data,
            status=status.HTTP_201_CREATED,
        )


class ApplicationSessionDetailView(APIView):
    permission_classes = [IsStudent]

    def get(self, request, session_id):
        try:
            session = ApplicationSession.objects.get(id=session_id, student=request.user.student)
        except ApplicationSession.DoesNotExist:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(ApplicationSessionSerializer(session).data)


class ApplicationSessionCancelView(APIView):
    permission_classes = [IsStudent]

    def post(self, request, session_id):
        try:
            session = ApplicationSession.objects.get(id=session_id, student=request.user.student)
        except ApplicationSession.DoesNotExist:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        if session.status not in ApplicationSession.ACTIVE_STATUSES:
            return Response(
                {"error": f"Session is '{session.status}', cannot cancel."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        auto_apply_cancel(session)
        return Response({"message": "Session cancelled."})
```

- [ ] **Step 4: Add URL routes**

Modify `backend/ai_services/urls.py` — add to imports and `urlpatterns`:

```python
from .views import (
    # ... existing imports ...
    ApplicationSessionListCreateView,
    ApplicationSessionDetailView,
    ApplicationSessionCancelView,
)

urlpatterns += [
    path("ai/application-sessions/", ApplicationSessionListCreateView.as_view(), name="application-session-list-create"),
    path("ai/application-sessions/<int:session_id>/", ApplicationSessionDetailView.as_view(), name="application-session-detail"),
    path("ai/application-sessions/<int:session_id>/cancel/", ApplicationSessionCancelView.as_view(), name="application-session-cancel"),
]
```

- [ ] **Step 5: Run tests**

Run: `python manage.py test ai_services.tests.test_application_session_views -v 2`
Expected: all 5 pass.

- [ ] **Step 6: Commit**

```bash
git add ai_services/views.py ai_services/urls.py ai_services/tests/test_application_session_views.py
git commit -m "feat(ai_services): add start/detail/cancel views for ApplicationSession"
```

---

## Task B10: Views — three resume endpoints

**Files:**
- Modify: `backend/ai_services/views.py`
- Modify: `backend/ai_services/urls.py`
- Modify: `backend/ai_services/tests/test_application_session_views.py`

- [ ] **Step 1: Write failing test**

Append to `backend/ai_services/tests/test_application_session_views.py`:

```python
class GateResumeViewsTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="s2", email="s2@x.com", password="x")
        self.student = Student.objects.create(user=self.user)
        self.client.force_authenticate(self.user)
        self.job = JobOpportunity.objects.create(title="J", company="C", site="linkedin", employer_id=None)
        self.session = ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=self.job.id,
            opportunity_snapshot={"id": self.job.id, "is_closed": False},
        )

    @patch("ai_services.auto_apply_adapter.resume_session_gate_0")
    def test_gate_0_resume_only_when_in_correct_status(self, fake_resume):
        # Wrong status
        resp = self.client.post(
            reverse("application-session-resume-gate-0", args=[self.session.id]),
            {"profile_updates": {"location": "Istanbul"}}, format="json",
        )
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

        # Correct status
        self.session.status = ApplicationSession.Status.AWAITING_PROFILE_COMPLETION
        self.session.save()
        resp = self.client.post(
            reverse("application-session-resume-gate-0", args=[self.session.id]),
            {"profile_updates": {"location": "Istanbul"}}, format="json",
        )
        assert resp.status_code == status.HTTP_202_ACCEPTED, resp.data
        fake_resume.assert_called_once()

    @patch("ai_services.auto_apply_adapter.resume_session_gate_1")
    def test_gate_1_resume(self, fake_resume):
        self.session.status = ApplicationSession.Status.AWAITING_DOCUMENT_MAPPING
        self.session.save()
        resp = self.client.post(
            reverse("application-session-resume-gate-1", args=[self.session.id]),
            {"confirm": True, "overrides": []}, format="json",
        )
        assert resp.status_code == status.HTTP_202_ACCEPTED
        fake_resume.assert_called_once()

    @patch("ai_services.auto_apply_adapter.resume_session_gate_2")
    def test_gate_2_resume(self, fake_resume):
        self.session.status = ApplicationSession.Status.AWAITING_FINAL_APPROVAL
        self.session.save()
        resp = self.client.post(
            reverse("application-session-resume-gate-2", args=[self.session.id]),
            {"approved": True}, format="json",
        )
        assert resp.status_code == status.HTTP_202_ACCEPTED
        fake_resume.assert_called_once()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python manage.py test ai_services.tests.test_application_session_views.GateResumeViewsTests -v 2`
Expected: FAIL — routes not defined.

- [ ] **Step 3: Add views**

Append to `backend/ai_services/views.py`:

```python
class _GateResumeBaseView(APIView):
    permission_classes = [IsStudent]
    expected_status: str = ""
    serializer_cls = None
    handler = None

    def post(self, request, session_id):
        try:
            session = ApplicationSession.objects.get(id=session_id, student=request.user.student)
        except ApplicationSession.DoesNotExist:
            return Response({"error": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        if session.status != self.expected_status:
            return Response(
                {"error": f"Session is '{session.status}', expected '{self.expected_status}'."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        ser = self.serializer_cls(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=status.HTTP_400_BAD_REQUEST)
        type(self).handler(session, dict(ser.validated_data))
        return Response({"message": "Resume submitted."}, status=status.HTTP_202_ACCEPTED)


class ApplicationSessionResumeGate0View(_GateResumeBaseView):
    expected_status = ApplicationSession.Status.AWAITING_PROFILE_COMPLETION
    serializer_cls = Gate0ResumeSerializer
    handler = staticmethod(resume_session_gate_0)


class ApplicationSessionResumeGate1View(_GateResumeBaseView):
    expected_status = ApplicationSession.Status.AWAITING_DOCUMENT_MAPPING
    serializer_cls = Gate1ResumeSerializer
    handler = staticmethod(resume_session_gate_1)


class ApplicationSessionResumeGate2View(_GateResumeBaseView):
    expected_status = ApplicationSession.Status.AWAITING_FINAL_APPROVAL
    serializer_cls = Gate2ResumeSerializer
    handler = staticmethod(resume_session_gate_2)
```

- [ ] **Step 4: Add URL routes**

Append to `backend/ai_services/urls.py`:

```python
from .views import (
    ApplicationSessionResumeGate0View,
    ApplicationSessionResumeGate1View,
    ApplicationSessionResumeGate2View,
)

urlpatterns += [
    path("ai/application-sessions/<int:session_id>/resume-gate-0/",
         ApplicationSessionResumeGate0View.as_view(), name="application-session-resume-gate-0"),
    path("ai/application-sessions/<int:session_id>/resume-gate-1/",
         ApplicationSessionResumeGate1View.as_view(), name="application-session-resume-gate-1"),
    path("ai/application-sessions/<int:session_id>/resume-gate-2/",
         ApplicationSessionResumeGate2View.as_view(), name="application-session-resume-gate-2"),
]
```

- [ ] **Step 5: Run tests**

Run: `python manage.py test ai_services.tests.test_application_session_views -v 2`
Expected: all 8 (5 from B9 + 3 here) pass.

- [ ] **Step 6: Commit**

```bash
git add ai_services/views.py ai_services/urls.py ai_services/tests/test_application_session_views.py
git commit -m "feat(ai_services): add three gate-resume endpoints for ApplicationSession"
```

---

## Task B11: Per-status stale-session janitor

**Files:**
- Create: `backend/ai_services/janitor.py`
- Test: `backend/ai_services/tests/test_janitor.py`

- [ ] **Step 1: Write failing test**

```python
# backend/ai_services/tests/test_janitor.py
import uuid
from datetime import timedelta
from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

from accounts.models import Student
from ai_services.models import ApplicationSession
from jobs.models import JobOpportunity


class JanitorTests(TestCase):
    def setUp(self):
        user = User.objects.create_user(username="j1", email="j1@x.com", password="x")
        self.student = Student.objects.create(user=user)
        self.job = JobOpportunity.objects.create(title="J", company="C", site="linkedin", employer_id=None)

    def _mk(self, status, age_minutes):
        s = ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=self.job.id,
            opportunity_snapshot={}, status=status,
        )
        # Backdate updated_at
        ApplicationSession.objects.filter(id=s.id).update(
            updated_at=timezone.now() - timedelta(minutes=age_minutes)
        )
        return s

    def test_processing_stale_at_15_min(self):
        from ai_services.janitor import cleanup_stale_application_sessions
        old = self._mk(ApplicationSession.Status.PROCESSING, 20)
        fresh = self._mk(ApplicationSession.Status.PROCESSING, 5)
        # Need different opportunity_id to avoid the unique constraint
        ApplicationSession.objects.filter(id=fresh.id).update(opportunity_id=self.job.id + 1)
        cleanup_stale_application_sessions()
        old.refresh_from_db(); fresh.refresh_from_db()
        assert old.status == ApplicationSession.Status.ERROR
        assert fresh.status == ApplicationSession.Status.PROCESSING

    def test_finalizing_stale_at_30_min(self):
        from ai_services.janitor import cleanup_stale_application_sessions
        old = self._mk(ApplicationSession.Status.FINALIZING, 35)
        cleanup_stale_application_sessions()
        old.refresh_from_db()
        assert old.status == ApplicationSession.Status.ERROR

    def test_awaiting_states_stale_at_7_days(self):
        from ai_services.janitor import cleanup_stale_application_sessions
        s_recent = self._mk(ApplicationSession.Status.AWAITING_DOCUMENT_MAPPING, 60 * 24 * 3)  # 3 days
        s_recent_2 = self._mk(ApplicationSession.Status.AWAITING_PROFILE_COMPLETION, 60 * 24 * 3)
        ApplicationSession.objects.filter(id=s_recent_2.id).update(opportunity_id=self.job.id + 1)
        s_old = self._mk(ApplicationSession.Status.AWAITING_FINAL_APPROVAL, 60 * 24 * 8)  # 8 days
        ApplicationSession.objects.filter(id=s_old.id).update(opportunity_id=self.job.id + 2)

        cleanup_stale_application_sessions()
        s_recent.refresh_from_db(); s_old.refresh_from_db()
        assert s_recent.status == ApplicationSession.Status.AWAITING_DOCUMENT_MAPPING
        assert s_old.status == ApplicationSession.Status.ERROR
```

- [ ] **Step 2: Run to verify it fails**

Run: `python manage.py test ai_services.tests.test_janitor -v 2`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the janitor**

```python
# backend/ai_services/janitor.py
"""Per-status stale-session cleanup (Spec T2).

Run from a management command, AppConfig.ready, or a scheduled job.
Flips sessions older than their per-status TTL to ERROR with a timeout message.
"""
from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict

from django.utils import timezone

from .models import ApplicationSession

logger = logging.getLogger(__name__)


_TTL_MINUTES: Dict[str, int] = {
    ApplicationSession.Status.PROCESSING:                  15,
    ApplicationSession.Status.FINALIZING:                  30,
    ApplicationSession.Status.AWAITING_PROFILE_COMPLETION: 60 * 24 * 7,   # 7 days
    ApplicationSession.Status.AWAITING_DOCUMENT_MAPPING:   60 * 24 * 7,
    ApplicationSession.Status.AWAITING_FINAL_APPROVAL:     60 * 24 * 7,
}


def cleanup_stale_application_sessions() -> int:
    """Flip stale sessions to ERROR. Returns total count cleaned."""
    total = 0
    now = timezone.now()
    for status, minutes in _TTL_MINUTES.items():
        cutoff = now - timedelta(minutes=minutes)
        count = ApplicationSession.objects.filter(
            status=status, updated_at__lt=cutoff,
        ).update(
            status=ApplicationSession.Status.ERROR,
            error_message="Session timed out — please start fresh.",
        )
        if count:
            logger.info("janitor: cleaned %d stale %s sessions", count, status)
            total += count
    return total
```

- [ ] **Step 4: Run tests**

Run: `python manage.py test ai_services.tests.test_janitor -v 2`
Expected: all 3 pass.

- [ ] **Step 5: Commit**

```bash
git add ai_services/janitor.py ai_services/tests/test_janitor.py
git commit -m "feat(ai_services): add per-status TTL janitor for ApplicationSession"
```

---

## Task B12: Run full backend test suite + manual smoke

**Files:** none (verification)

- [ ] **Step 1: Run all ai_services tests**

Run: `python manage.py test ai_services -v 2`
Expected: every test from B2–B11 passes; existing ai_services tests untouched.

- [ ] **Step 2: Run the broader backend suite**

Run: `python manage.py test -v 1`
Expected: no regressions in other apps.

- [ ] **Step 3: Manual smoke against staging Neon (one of each opportunity type)**

Set up:
```bash
export DATABASE_URL="<staging-neon-url>"
export UPPGRAD_LLM_PROVIDER=openai
export OPENAI_API_KEY="..."
python manage.py migrate
python manage.py runserver
```

For each opportunity type (job-internal, job-external, masters, phd, scholarship), in a separate terminal:

```bash
# Replace TOKEN with the test student's token (see TEST_USERS.md)
curl -X POST http://localhost:8000/api/ai/application-sessions/ \
  -H "Authorization: Token $TOKEN" -H "Content-Type: application/json" \
  -d '{"opportunity_type": "job", "opportunity_id": <REAL_ID>}'
# Returns 201 with session id

curl http://localhost:8000/api/ai/application-sessions/<id>/ \
  -H "Authorization: Token $TOKEN"
# Poll until status == 'awaiting_document_mapping'

curl -X POST http://localhost:8000/api/ai/application-sessions/<id>/resume-gate-1/ \
  -H "Authorization: Token $TOKEN" -H "Content-Type: application/json" \
  -d '{"confirm": true, "overrides": []}'
# Returns 202

# Poll until awaiting_final_approval, then approve
curl -X POST http://localhost:8000/api/ai/application-sessions/<id>/resume-gate-2/ \
  -H "Authorization: Token $TOKEN" -H "Content-Type: application/json" \
  -d '{"approved": true}'
```

Expected outcomes:
- `job-internal`: session ends in `completed_submitted`, `application_id` populated, `Application` row exists in DB.
- `job-external`, `masters`, `phd`, `scholarship`: session ends in `completed_handoff`, `application_package` populated.

- [ ] **Step 4: Document smoke results in commit**

```bash
git commit --allow-empty -m "test: smoke-tested auto_apply integration against staging Neon"
```

(Empty commit acts as a marker for the verification pass.)

---

## Self-review

**Spec coverage:**

- ✅ A1 (backend pre-loads opportunity) — Tasks B4 + A4
- ✅ A2 (`ApplicationSession` model, one active per student+opportunity) — Task B2
- ✅ A3 (Django ORM `Application` writes, no HTTP) — Task B7
- ✅ A4 (in-app handoff terminal) — handled by existing `package_and_handoff` node, surfaced via Task B5's `_persist_state_after_phase`
- ✅ A5 (two distinct success terminals) — Task B2 status enum + B5 transitions
- ✅ A6 (discovery state fields added now) — Task A1 + Task B2
- ✅ A7 (cancel via `graph.update_state`) — Tasks A9 + B7
- ✅ A8 (mirror document_feedback pattern) — Tasks B5 + B6 + B7
- ✅ §3.1 model schema — Task B2
- ✅ §4.1–§4.5 API surface — Tasks B8 + B9 + B10
- ✅ §6.1 profile-from-state helper — Tasks A2 + A3
- ✅ §6.2 human_gate_0 real interrupt + iteration cap — Tasks A5 + A6
- ✅ §6.3 submit_internal records intent only — Task A7
- ✅ §6.4 load_opportunity short-circuit + freshness re-check at gate 2 — Tasks A4 + B6
- ✅ §6.5 PDF rendering for internal submit — Task B3
- ✅ §6.6 cancel_session helper — Task A9
- ✅ §10.Q1 session uploads scoped to session-uploads/ dir — handled in B6 via uploaded_cv saving to StudentCV (Q1.a applies only to gate-1 additional uploads, deferred)
- ✅ §10.Q2 gate 2 reject = terminate v1 — Task B6 (gate 2 handler short-circuits on approved=False)
- ✅ §10.Q3 per-status TTLs — Task B11
- ✅ §10.Q5 per-doc-type tailoring caps — Task A8
- ✅ §11 out-of-scope items — explicitly not implemented (no discovery node logic, no external auto-submit, no polymorphic doc store)

**Placeholder scan:** every step has either complete code or an exact runnable command. No "TBD" / "implement later" / "similar to task N" patterns.

**Type consistency:**
- `AutoApplyState` keys (`profile_snapshot`, `discovered_apply_url`, `discovery_method`, `discovery_confidence`, `gate_0_iteration_count`) — defined in A1, consumed in A2, A4, A5.
- `resolve_profile(state)` signature — defined A2, consumed A3.
- `_truncate_to_cap(content, doc_type)` — defined A8, used in same file.
- `cancel_session(thread_id, checkpointer)` — defined A9 (agentic), consumed B7 (backend) where backend wraps as `cancel_session(session)` and constructs the checkpointer.
- `build_auto_apply_opportunity_snapshot(opp_type, opp_id) -> Optional[Dict]` — defined B4, consumed B6 + B9.
- `build_auto_apply_profile_snapshot(student) -> Dict` — defined B4, consumed B5.
- `_build_checkpointed_auto_apply_graph()` — defined B5, consumed B5/B6.
- `_persist_state_after_phase(session, result_state)` — defined B5, consumed B6.
- `finalize_internal_submission(session)` — defined B7, consumed B6.
- `ApplicationSession.Status.*` — defined B2, consumed B5/B6/B7/B9/B10/B11.
- `ApplicationSession.ACTIVE_STATUSES` — defined B2, consumed B9 (cancel + duplicate check).
- Three resume handlers `resume_session_gate_{0,1,2}(session, payload)` — defined B6, wired to view classes via `staticmethod` in B10.
- `Gate{0,1,2}ResumeSerializer` — defined B8, consumed B10.

No drift detected.
