# Apply-URL Discovery v2 + Eligibility Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current LinkedIn-fetch-then-fail scrape path with a search-based discovery pipeline (Brave) that resolves a job's real apply URL, then fetches it via httpx-first / Playwright-fallback to extract employer-specific document requirements. Plus a small eligibility-node cleanup that separates compatibility *warnings* from material-readiness *gating*.

**Architecture:** Phase 0 splits compatibility checks (location/age/degree/discipline/nationality) out of the eligibility decision path — they become warnings carried through state into the handoff package, surfaced on the UI, but no longer hard-block the workflow. Phase 1-5 introduce a uniform discovery pipeline: search → verify → fetch (httpx, Playwright fallback under env var) → evaluate. No ATS-specific shortcuts (Greenhouse-API style) — uniform pipeline. Phase 6 adds a Postgres-backed cross-user cache on the backend side; agentic repo stays DB-free.

**Tech Stack:** Python 3.11+, LangGraph, httpx (new dep), rapidfuzz (new dep), Brave Search API (no SDK — single HTTP client), Crawl4AI / Playwright (optional, env-gated), Postgres (Neon, backend side only), Django 4.2, DRF, pytest, pytest-asyncio, respx (already present).

**Spec / earlier plan refs:**
- Spec: `docs/superpowers/specs/2026-04-26-auto-apply-backend-integration.md` (settled architecture; eligibility cleanup follows resolved discussion)
- Original discovery plan (paused): `docs/superpowers/plans/2026-04-26-apply-url-discovery.md` (this plan supersedes it)

---

## Architectural decisions locked

| ID | Decision |
|---|---|
| **D1** | Eligibility node hard-blocks ONLY for deadline-passed and missing user-supplied (non-generatable) docs. Compatibility issues (location, age, degree, discipline, nationality) become warnings carried through to handoff. |
| **D2** | No ATS-specific shortcuts (no Greenhouse `?questions=true` path in v1). Single uniform pipeline: search → fetch → evaluate. |
| **D3** | Two-tier fetcher: **httpx-first** (always available), **Playwright/Crawl4AI fallback** when (a) `UPPGRAD_BROWSER_SCRAPE_ENABLED=true` AND (b) httpx output looks thin (anti-bot wall, JS shell, thin keyword count). |
| **D4** | Brave Search opt-in via `UPPGRAD_SEARCH_PROVIDER=brave` + `BRAVE_SEARCH_API_KEY`. Without these → discovery is no-op pass-through (matches current behavior, falls through to assumed defaults). |
| **D5** | Cache (`JobApplyUrlDiscovery`) lives on **backend** side as a Django model. Agentic repo stays DB-free. Adapter handles cache lookup pre-invoke and write post-invoke. |
| **D6** | Drop the closed-postings cleanup hook. `last_verified_at` + 14-day re-verify gate handles staleness. |
| **D7** | Discovery node sits between `load_opportunity` and `scrape_application_page` for jobs only. Internal jobs (`employer_id == 1`) skip it entirely. |
| **D8** | ToS gray area (programmatic scraping of public-facing ATS endpoints) accepted. Industry norm. Documented as open risk. |

---

## File Structure

### Phase 0 — Eligibility cleanup

| Path | Responsibility | Status |
|---|---|---|
| `src/uppgrad_agentic/workflows/auto_apply/state.py` | Add `compatibility_warnings: List[str]` | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/nodes/eligibility_and_readiness.py` | Move compatibility issues out of decision path; only deadline + missing user-supplied docs gate | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/nodes/package_and_handoff.py` | Include `warnings` field in `application_package` | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/nodes/submit_internal.py` | Include `warnings` field in `application_package` | Modify |
| `backend/ai_services/models.py` | Add `compatibility_warnings = models.JSONField(default=list, blank=True)` | Modify |
| `backend/ai_services/migrations/0008_compatibility_warnings.py` | Auto-generated | Create |
| `backend/ai_services/auto_apply_adapter.py` | `_persist_state_after_phase` reads `compatibility_warnings` from result_state | Modify |
| `backend/ai_services/serializers.py` | Expose `compatibility_warnings` field on `ApplicationSessionSerializer` | Modify |

### Phase 1-5 — Discovery (agentic repo)

| Path | Responsibility | Status |
|---|---|---|
| `pyproject.toml` | Add `httpx>=0.27`, `rapidfuzz>=3.10`, dev: `respx>=0.21`, `crawl4ai>=0.4.0` (optional) | Modify |
| `src/uppgrad_agentic/common/llm.py` | Add `get_search_provider()` factory mirroring `get_llm()` | Modify |
| `src/uppgrad_agentic/tools/search.py` | `SearchProvider` ABC + `BraveSearchProvider` + `SearchResult` model | Create |
| `src/uppgrad_agentic/tools/web_fetcher.py` | Tiered fetcher: httpx → Playwright (env-gated, lazy-imported) | Create |
| `src/uppgrad_agentic/tools/url_discovery.py` | Verification scoring + 3-tier orchestration | Create |
| `src/uppgrad_agentic/workflows/auto_apply/nodes/discover_apply_url.py` | Graph node calling `tools.url_discovery` | Create |
| `src/uppgrad_agentic/workflows/auto_apply/nodes/scrape_application_page.py` | Use `state['discovered_apply_url']` + `web_fetcher` | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/graph.py` | Insert `discover_apply_url` between `load_opportunity` and `scrape_application_page` for jobs | Modify |

### Phase 6 — Backend cache

| Path | Responsibility | Status |
|---|---|---|
| `backend/ai_services/models.py` | Add `JobApplyUrlDiscovery` model | Modify |
| `backend/ai_services/migrations/0009_job_apply_url_discovery.py` | Auto-generated | Create |
| `backend/ai_services/auto_apply_adapter.py` | `_lookup_discovery_cache()` + `_persist_discovery_to_cache()` helpers; integrate into `_run_graph_initial_phase` and `_persist_state_after_phase` | Modify |

### Tests

All new files get tests. Existing test files extended where modifying behaviour. Specific paths called out per task.

---

# PHASE 0 — Eligibility Cleanup

All Phase 0 tasks span both repos.

## Task 0.1: Add `compatibility_warnings` to AutoApplyState

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/state.py`
- Test: `tests/workflows/auto_apply/test_state_compatibility_warnings.py`

- [ ] **Step 1: Write failing test**

```python
# tests/workflows/auto_apply/test_state_compatibility_warnings.py
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState


def test_state_declares_compatibility_warnings():
    keys = AutoApplyState.__annotations__
    assert "compatibility_warnings" in keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd AgenticWorkflows_LangChain-LangGraph && uv run pytest tests/workflows/auto_apply/test_state_compatibility_warnings.py -v`
Expected: FAIL — key not present.

- [ ] **Step 3: Add field**

In `src/uppgrad_agentic/workflows/auto_apply/state.py`, add after `gate_0_iteration_count: int`:

```python
    # compatibility warnings (Spec follow-up — deadline-passed + missing
    # user-supplied docs are the only hard-block reasons; everything else
    # like location mismatch / age cap / degree level becomes a non-blocking
    # warning the UI surfaces on the apply screen and the handoff package).
    compatibility_warnings: List[str]
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/workflows/auto_apply/test_state_compatibility_warnings.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/state.py tests/workflows/auto_apply/test_state_compatibility_warnings.py
git commit -m "feat(auto_apply): add compatibility_warnings to AutoApplyState"
```

---

## Task 0.2: Refactor eligibility node — compatibility issues become warnings, not blocks

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/nodes/eligibility_and_readiness.py`
- Test: `tests/workflows/auto_apply/test_eligibility_compatibility_warnings.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/workflows/auto_apply/test_eligibility_compatibility_warnings.py
from uppgrad_agentic.workflows.auto_apply.nodes.eligibility_and_readiness import (
    eligibility_and_readiness,
)


def _job_state(is_remote=False, job_location="Ankara, TR", user_location="Istanbul, TR",
               deadline=None, has_cv=True):
    return {
        "opportunity_type": "job",
        "opportunity_data": {
            "is_closed": False, "is_remote": is_remote,
            "title": "X", "company": "Y", "location": job_location,
            "deadline": deadline,
        },
        "normalized_requirements": [
            {"requirement_type": "document", "document_type": "CV",
             "is_assumed": False, "confidence": 0.9},
        ],
        "profile_snapshot": {
            "name": "U", "email": "u@x.com",
            "location": user_location,
            "uploaded_documents": {"CV": has_cv},
            "document_texts": {"CV": "..."} if has_cv else {},
        },
    }


def test_location_mismatch_emits_warning_not_block():
    out = eligibility_and_readiness(_job_state(
        is_remote=False, job_location="Ankara, TR", user_location="Istanbul, TR",
    ))
    assert out["eligibility_result"]["decision"] == "ready"
    assert any("Ankara" in w for w in out["compatibility_warnings"])


def test_remote_job_no_warning():
    out = eligibility_and_readiness(_job_state(
        is_remote=True, user_location="Istanbul, TR",
    ))
    assert out["eligibility_result"]["decision"] == "ready"
    assert out["compatibility_warnings"] == []


def test_deadline_passed_still_hard_blocks():
    out = eligibility_and_readiness(_job_state(deadline="2020-01-01"))
    assert out["eligibility_result"]["decision"] == "ineligible"
    # deadline reason goes into eligibility_result.reasons, not compatibility_warnings
    assert "deadline" in out["eligibility_result"]["reasons"][0].lower()


def test_scholarship_age_cap_emits_warning_not_block():
    state = {
        "opportunity_type": "scholarship",
        "opportunity_data": {
            "title": "S", "req_age": "Under 35",
            "data": {},
        },
        "normalized_requirements": [
            {"requirement_type": "document", "document_type": "CV",
             "is_assumed": False, "confidence": 0.9},
        ],
        "profile_snapshot": {
            "name": "U", "email": "u@x.com", "age": 36,
            "uploaded_documents": {"CV": True},
            "document_texts": {"CV": "..."},
        },
    }
    out = eligibility_and_readiness(state)
    assert out["eligibility_result"]["decision"] == "ready"
    assert any("under 35" in w.lower() for w in out["compatibility_warnings"])


def test_phd_degree_mismatch_emits_warning_not_block():
    state = {
        "opportunity_type": "phd",
        "opportunity_data": {
            "title": "PhD CS", "degree_type": "PhD",
            "data": {"requirements": {"academic": "MSc required"}},
        },
        "normalized_requirements": [
            {"requirement_type": "document", "document_type": "CV",
             "is_assumed": False, "confidence": 0.9},
        ],
        "profile_snapshot": {
            "name": "U", "email": "u@x.com", "degree_level": "BSc",
            "uploaded_documents": {"CV": True},
            "document_texts": {"CV": "..."},
        },
    }
    out = eligibility_and_readiness(state)
    assert out["eligibility_result"]["decision"] == "ready"
    assert any("masters" in w.lower() for w in out["compatibility_warnings"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/workflows/auto_apply/test_eligibility_compatibility_warnings.py -v`
Expected: FAIL — tests assert `decision="ready"` but current code returns `"ineligible"` for these cases.

- [ ] **Step 3: Refactor `eligibility_and_readiness`**

In `src/uppgrad_agentic/workflows/auto_apply/nodes/eligibility_and_readiness.py`, replace the body of `eligibility_and_readiness` (the function starting at the bottom of the file) with:

```python
def eligibility_and_readiness(state: AutoApplyState) -> dict:
    updates = {"current_step": "eligibility_and_readiness", "step_history": ["eligibility_and_readiness"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    opportunity_type = state.get("opportunity_type", "")
    opportunity_data = state.get("opportunity_data") or {}
    normalized_requirements = state.get("normalized_requirements") or []
    from uppgrad_agentic.workflows.auto_apply._profile import resolve_profile
    profile = resolve_profile(state)

    # ------------------------------------------------------------------
    # 1. Hard block: deadline passed (objective, non-arguable)
    # ------------------------------------------------------------------
    deadline_passed, deadline_reason = _check_deadline(opportunity_data)
    if deadline_passed:
        result = EligibilityResult(
            decision="ineligible",
            reasons=[deadline_reason],
            missing_fields=[],
        )
        return {**updates, "eligibility_result": result.model_dump(),
                "compatibility_warnings": []}

    # ------------------------------------------------------------------
    # 2. Compatibility warnings (NOT blocks). Surface in handoff package.
    # ------------------------------------------------------------------
    if opportunity_type == "job":
        warnings = _check_job_eligibility(opportunity_data, profile)
    elif opportunity_type in ("masters", "phd"):
        warnings = _check_program_eligibility(opportunity_data, profile)
    elif opportunity_type == "scholarship":
        warnings = _check_scholarship_eligibility(opportunity_data, profile)
    else:
        warnings = []

    # ------------------------------------------------------------------
    # 3. Profile / required-doc completeness (gate 0 trigger)
    # ------------------------------------------------------------------
    missing_fields = _check_profile_completeness(profile, normalized_requirements)

    if missing_fields:
        doc_missing = [f for f in missing_fields if f.startswith("document:")]
        profile_missing = [f for f in missing_fields if not f.startswith("document:")]
        pending_reasons: List[str] = []
        if profile_missing:
            pending_reasons.append(
                f"Your profile is missing required fields: {', '.join(profile_missing)}."
            )
        if doc_missing:
            doc_names = [f.removeprefix("document:") for f in doc_missing]
            pending_reasons.append(
                f"The following documents have not been uploaded yet: {', '.join(doc_names)}."
            )
        result = EligibilityResult(
            decision="pending",
            reasons=pending_reasons,
            missing_fields=missing_fields,
        )
        return {**updates, "eligibility_result": result.model_dump(),
                "compatibility_warnings": warnings}

    # ------------------------------------------------------------------
    # 4. Ready
    # ------------------------------------------------------------------
    result = EligibilityResult(
        decision="ready",
        reasons=["All hard checks passed; required documents are present or generatable."],
        missing_fields=[],
    )
    return {**updates, "eligibility_result": result.model_dump(),
            "compatibility_warnings": warnings}
```

The existing `_check_job_eligibility`, `_check_program_eligibility`, `_check_scholarship_eligibility`, `_check_deadline`, `_check_profile_completeness` functions stay as-is — only their *consumption* in the main function changes. They already return `List[str]` of issue strings; we just route them to `compatibility_warnings` instead of using them to drive the decision.

Also remove the `is_closed` issue from `_check_job_eligibility` (line ~117) since closed jobs are already filtered by the adapter at session-start time:

```python
def _check_job_eligibility(opportunity_data: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    # NOTE: is_closed is filtered by the backend adapter at session start; no need to re-check here.
    if not opportunity_data.get("is_remote", False):
        job_location = (opportunity_data.get("location") or "").lower()
        user_location = (profile.get("location") or "").lower()
        if job_location and user_location and job_location not in user_location and user_location not in job_location:
            issues.append(
                f"Job is on-site in '{opportunity_data.get('location')}' but your location is '{profile.get('location')}'. "
                "Remote work is not offered."
            )
    return issues
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_eligibility_compatibility_warnings.py tests/workflows/auto_apply/test_eligibility_generatable_docs.py -v`
Expected: all pass. The pre-existing generatable-docs tests still work because we didn't change `_check_profile_completeness`.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/nodes/eligibility_and_readiness.py tests/workflows/auto_apply/test_eligibility_compatibility_warnings.py
git commit -m "feat(auto_apply): split compatibility warnings from material-readiness gating

Eligibility now hard-blocks ONLY for deadline-passed and missing user-supplied
docs. Location mismatch / age caps / degree-level / discipline / nationality
checks become warnings carried in state.compatibility_warnings — surfaced in
the UI and handoff package, but no longer a workflow terminator."
```

---

## Task 0.3: Thread warnings into `application_package`

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/nodes/package_and_handoff.py`
- Modify: `src/uppgrad_agentic/workflows/auto_apply/nodes/submit_internal.py`
- Test: `tests/workflows/auto_apply/test_package_includes_warnings.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/workflows/auto_apply/test_package_includes_warnings.py
from uppgrad_agentic.workflows.auto_apply.nodes.package_and_handoff import package_and_handoff
from uppgrad_agentic.workflows.auto_apply.nodes.submit_internal import submit_internal


def _state(warnings):
    return {
        "opportunity_type": "job",
        "opportunity_id": "1",
        "opportunity_data": {"id": 1, "title": "X", "company": "Y"},
        "compatibility_warnings": warnings,
        "tailored_documents": {"CV": {"content": "...", "tailoring_depth": "light"}},
        "scraped_requirements": {"status": "failed", "confidence": 0.0, "source": ""},
    }


def test_handoff_package_carries_warnings():
    out = package_and_handoff(_state(["Job is on-site in Ankara, you're in Istanbul"]))
    pkg = out["application_package"]
    assert "warnings" in pkg
    assert pkg["warnings"] == ["Job is on-site in Ankara, you're in Istanbul"]


def test_handoff_package_warnings_empty_list_when_no_issues():
    out = package_and_handoff(_state([]))
    assert out["application_package"]["warnings"] == []


def test_internal_submit_package_carries_warnings():
    state = _state(["Test warning"])
    state["tailored_documents"] = {
        "CV": {"content": "cv content"},
        "Cover Letter": {"content": "cl content"},
    }
    out = submit_internal(state)
    pkg = out["application_package"]
    assert pkg["warnings"] == ["Test warning"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/workflows/auto_apply/test_package_includes_warnings.py -v`
Expected: FAIL — `warnings` key not yet in package.

- [ ] **Step 3: Add `warnings` key to handoff package**

In `src/uppgrad_agentic/workflows/auto_apply/nodes/package_and_handoff.py`, find the `package: Dict[str, Any] = {` block and add a `"warnings"` key:

```python
    package: Dict[str, Any] = {
        "documents": {
            doc_type: {
                "content": info.get("content", ""),
                "tailoring_depth": info.get("tailoring_depth", ""),
                "char_count": len(info.get("content") or ""),
            }
            for doc_type, info in tailored_documents.items()
            if not info.get("skip") and info.get("tailoring_depth") != "none"
        },
        "opportunity": {
            "type": opportunity_type,
            "id": state.get("opportunity_id", ""),
            "title": title,
            "organisation": org,
            "application_url": url,
        },
        "submission_type": "handoff",
        "warnings": list(state.get("compatibility_warnings") or []),
    }
```

- [ ] **Step 4: Add `warnings` key to internal-submit package**

In `src/uppgrad_agentic/workflows/auto_apply/nodes/submit_internal.py`, find the `package: Dict[str, Any] = {` block and add the warnings key:

```python
    package: Dict[str, Any] = {
        "CV": cv_content,
        "Cover Letter": cl_content,
        "submission_type": "internal",
        "warnings": list(state.get("compatibility_warnings") or []),
    }
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_package_includes_warnings.py tests/workflows/auto_apply/ -q`
Expected: all green (3 new tests + all prior tests).

- [ ] **Step 6: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/nodes/package_and_handoff.py src/uppgrad_agentic/workflows/auto_apply/nodes/submit_internal.py tests/workflows/auto_apply/test_package_includes_warnings.py
git commit -m "feat(auto_apply): carry compatibility_warnings into application_package"
```

---

## Task 0.4: Backend — `ApplicationSession.compatibility_warnings` + persistence + serializer

**Files:**
- Modify: `backend/ai_services/models.py`
- Create: `backend/ai_services/migrations/0008_compatibility_warnings.py` (auto-generated)
- Modify: `backend/ai_services/auto_apply_adapter.py`
- Modify: `backend/ai_services/serializers.py`
- Test: `backend/ai_services/tests/test_application_session_compatibility_warnings.py`

- [ ] **Step 1: Write failing test**

```python
# backend/ai_services/tests/test_application_session_compatibility_warnings.py
import uuid
from django.contrib.auth.models import User
from django.test import TestCase

from accounts.models import Student
from ai_services.models import ApplicationSession
from ai_services.serializers import ApplicationSessionSerializer


class CompatibilityWarningsTests(TestCase):
    def setUp(self):
        u = User.objects.create_user(username="cw", email="cw@x.com", password="x")
        self.student = Student.objects.create(user=u)

    def test_field_defaults_to_empty_list(self):
        s = ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=1, opportunity_snapshot={},
        )
        self.assertEqual(s.compatibility_warnings, [])

    def test_field_accepts_warning_list(self):
        s = ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=1, opportunity_snapshot={},
            compatibility_warnings=["W1", "W2"],
        )
        s.refresh_from_db()
        self.assertEqual(s.compatibility_warnings, ["W1", "W2"])

    def test_serializer_exposes_field(self):
        s = ApplicationSession(
            id=99, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=1, opportunity_snapshot={},
            compatibility_warnings=["X"],
        )
        data = ApplicationSessionSerializer(s).data
        self.assertEqual(data["compatibility_warnings"], ["X"])

    def test_persist_state_after_phase_writes_warnings(self):
        from ai_services.auto_apply_adapter import _persist_state_after_phase
        s = ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=1, opportunity_snapshot={},
        )
        _persist_state_after_phase(s, {
            "current_step": "asset_mapping",
            "step_history": ["x"],
            "compatibility_warnings": ["Job is on-site in Ankara"],
            "result": {"status": "ok"},
            "eligibility_result": {"decision": "ready"},
        }, pending_node="human_gate_1")
        s.refresh_from_db()
        self.assertEqual(s.compatibility_warnings, ["Job is on-site in Ankara"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python manage.py test ai_services.tests.test_application_session_compatibility_warnings -v 2`
Expected: FAIL — field not defined.

- [ ] **Step 3: Add field to model**

In `backend/ai_services/models.py`, inside `ApplicationSession`, add after `discovery_confidence`:

```python
    compatibility_warnings = models.JSONField(default=list, blank=True)
```

- [ ] **Step 4: Generate migration**

Run: `cd backend && python manage.py makemigrations ai_services -n compatibility_warnings`
Expected: creates `0008_compatibility_warnings.py`.

- [ ] **Step 5: Update `_persist_state_after_phase`**

In `backend/ai_services/auto_apply_adapter.py`, in `_persist_state_after_phase`, add after the existing `discovery_*` field copies:

```python
    if "compatibility_warnings" in result_state:
        session.compatibility_warnings = list(result_state.get("compatibility_warnings") or [])
```

- [ ] **Step 6: Update serializer**

In `backend/ai_services/serializers.py`, in `ApplicationSessionSerializer.Meta.fields`, add `"compatibility_warnings"` to the list (alongside `discovery_method` etc.).

- [ ] **Step 7: Apply migration locally**

Run: `cd backend && python manage.py migrate ai_services`
Expected: applied successfully.

- [ ] **Step 8: Run tests**

Run: `python manage.py test ai_services.tests.test_application_session_compatibility_warnings -v 2`
Expected: all 4 pass.

- [ ] **Step 9: Commit**

```bash
git add ai_services/models.py ai_services/migrations/0008_compatibility_warnings.py ai_services/auto_apply_adapter.py ai_services/serializers.py ai_services/tests/test_application_session_compatibility_warnings.py
git commit -m "feat(ai_services): persist and surface compatibility_warnings on ApplicationSession"
```

---

# PHASE 1 — Search Provider

## Task 1.1: `SearchProvider` ABC + `BraveSearchProvider` + factory

**Files:**
- Modify: `src/uppgrad_agentic/pyproject.toml`
- Modify: `src/uppgrad_agentic/common/llm.py`
- Create: `src/uppgrad_agentic/tools/search.py`
- Test: `tests/tools/test_search.py`
- Test: `tests/common/test_search_provider_factory.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_search.py
import httpx
import pytest
import respx

from uppgrad_agentic.tools.search import (
    BraveSearchProvider, SearchResult, SearchProvider,
)


def test_search_result_model():
    r = SearchResult(url="https://x.com/job/1", title="Engineer", snippet="...")
    assert r.url == "https://x.com/job/1"


def test_search_provider_is_abstract():
    with pytest.raises(TypeError):
        SearchProvider()


@respx.mock
def test_brave_provider_returns_results():
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {"url": "https://boards.greenhouse.io/acme/jobs/1",
                         "title": "SWE @ Acme", "description": "Apply now"},
                        {"url": "https://example.com/2", "title": "Other", "description": "..."},
                    ]
                }
            },
        )
    )
    provider = BraveSearchProvider(api_key="test")
    results = provider.search('"SWE" "Acme"', count=3)
    assert len(results) == 2
    assert results[0].url == "https://boards.greenhouse.io/acme/jobs/1"


@respx.mock
def test_brave_provider_returns_empty_on_429():
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        return_value=httpx.Response(429, json={"error": "rate limited"})
    )
    assert BraveSearchProvider(api_key="test").search('"x"', count=3) == []


@respx.mock
def test_brave_provider_returns_empty_on_network_error():
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        side_effect=httpx.ConnectError("boom")
    )
    assert BraveSearchProvider(api_key="test").search('"x"', count=3) == []


@respx.mock
def test_brave_provider_truncates_to_count():
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        return_value=httpx.Response(200, json={
            "web": {"results": [{"url": f"https://x.com/{i}", "title": str(i), "description": ""}
                                for i in range(10)]}
        })
    )
    assert len(BraveSearchProvider(api_key="test").search('"x"', count=3)) == 3
```

```python
# tests/common/test_search_provider_factory.py
from uppgrad_agentic.common.llm import get_search_provider


def test_returns_none_when_no_provider_configured(monkeypatch):
    monkeypatch.delenv("UPPGRAD_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    assert get_search_provider() is None


def test_returns_brave_provider_when_configured(monkeypatch):
    monkeypatch.setenv("UPPGRAD_SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    p = get_search_provider()
    assert p is not None
    assert p.__class__.__name__ == "BraveSearchProvider"


def test_returns_none_when_brave_key_missing(monkeypatch):
    monkeypatch.setenv("UPPGRAD_SEARCH_PROVIDER", "brave")
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    assert get_search_provider() is None


def test_unknown_provider_returns_none(monkeypatch):
    monkeypatch.setenv("UPPGRAD_SEARCH_PROVIDER", "google")
    assert get_search_provider() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_search.py tests/common/test_search_provider_factory.py -v`
Expected: FAIL — `tools.search` module not found.

- [ ] **Step 3: Add deps to `pyproject.toml`**

Append `httpx>=0.27.0` and `rapidfuzz>=3.10.0` to the `[project].dependencies` block; append `respx>=0.21.0` to the `[dependency-groups].dev` block.

- [ ] **Step 4: Run `uv sync --dev`**

Run: `uv sync --dev`
Expected: lockfile updates, packages installed.

- [ ] **Step 5: Implement `tools/search.py`**

```python
# src/uppgrad_agentic/tools/search.py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    url: str = Field(...)
    title: str = Field(default="")
    snippet: str = Field(default="")


class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, count: int = 3) -> List[SearchResult]:
        """Run a web search; return at most `count` results. Never raises."""
        raise NotImplementedError


class BraveSearchProvider(SearchProvider):
    _ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
    _TIMEOUT = 10.0

    def __init__(self, api_key: str):
        self._api_key = api_key

    def search(self, query: str, count: int = 3) -> List[SearchResult]:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        params = {"q": query, "count": min(count, 20)}
        try:
            resp = httpx.get(self._ENDPOINT, headers=headers, params=params, timeout=self._TIMEOUT)
        except httpx.HTTPError as exc:
            logger.warning("brave search: network error — %s", exc)
            return []
        if resp.status_code != 200:
            logger.warning("brave search: HTTP %s — %s", resp.status_code, resp.text[:200])
            return []
        try:
            payload = resp.json()
        except ValueError:
            logger.warning("brave search: non-JSON response")
            return []
        web_results = (payload.get("web") or {}).get("results") or []
        out: List[SearchResult] = []
        for item in web_results[:count]:
            url = item.get("url") or ""
            if not url:
                continue
            out.append(SearchResult(
                url=url,
                title=item.get("title") or "",
                snippet=item.get("description") or "",
            ))
        return out
```

- [ ] **Step 6: Add `get_search_provider()` to `common/llm.py`**

Append to `src/uppgrad_agentic/common/llm.py`:

```python
def get_search_provider():
    """Return a SearchProvider instance or None if not configured.

    Mirrors get_llm() opt-in pattern. Callers MUST handle None by returning
    a degraded result, never by raising.
    """
    import os
    provider_name = os.getenv("UPPGRAD_SEARCH_PROVIDER", "").lower()
    if provider_name != "brave":
        return None
    api_key = os.getenv("BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        return None
    from uppgrad_agentic.tools.search import BraveSearchProvider
    return BraveSearchProvider(api_key=api_key)
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/tools/test_search.py tests/common/test_search_provider_factory.py -v`
Expected: all 9 pass.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/uppgrad_agentic/tools/search.py src/uppgrad_agentic/common/llm.py tests/tools/test_search.py tests/common/test_search_provider_factory.py
git commit -m "feat(tools): add SearchProvider ABC, BraveSearchProvider, get_search_provider factory"
```

---

# PHASE 2 — Web Fetcher

## Task 2.1: httpx-only fetcher with thin-content detection

**Files:**
- Create: `src/uppgrad_agentic/tools/web_fetcher.py`
- Test: `tests/tools/test_web_fetcher_httpx.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_web_fetcher_httpx.py
import httpx
import pytest
import respx

from uppgrad_agentic.tools.web_fetcher import fetch_url, FetchResult


@respx.mock
def test_returns_success_for_substantial_html():
    body = "<html><body>" + ("Real apply page content. " * 200) + "<form><input type='file' name='resume'></form></body></html>"
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url("https://acme.com/jobs/1")
    assert isinstance(result, FetchResult)
    assert result.success is True
    assert result.thin is False
    assert result.http_status == 200
    assert "apply page content" in result.text


@respx.mock
def test_returns_thin_for_404():
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(404, text="Page not found"))
    result = fetch_url("https://acme.com/jobs/1")
    assert result.success is False
    assert result.thin is True
    assert result.http_status == 404


@respx.mock
def test_returns_thin_for_anti_bot_keywords():
    body = "<html><body>Cloudflare. Please complete the captcha to continue. JavaScript required.</body></html>"
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url("https://acme.com/jobs/1")
    assert result.success is True
    assert result.thin is True
    assert "cloudflare" in result.thin_signals[0].lower() or "captcha" in result.thin_signals[0].lower()


@respx.mock
def test_returns_thin_for_short_body():
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(200, text="short"))
    result = fetch_url("https://acme.com/jobs/1")
    assert result.thin is True


@respx.mock
def test_returns_failure_on_network_error():
    respx.get("https://acme.com/jobs/1").mock(side_effect=httpx.ConnectError("boom"))
    result = fetch_url("https://acme.com/jobs/1")
    assert result.success is False
    assert result.http_status == 0
    assert "boom" in result.error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_web_fetcher_httpx.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `tools/web_fetcher.py`**

```python
# src/uppgrad_agentic/tools/web_fetcher.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 15.0
_MAX_BYTES = 500_000

# Heuristics for "this httpx response is not a real apply page"
_THIN_KEYWORDS = [
    "404", "page not found", "access denied",
    "javascript required", "enable javascript",
    "cloudflare", "robot", "captcha", "challenge-platform",
    "please verify you are human",
]
_MIN_BODY_BYTES = 500


@dataclass
class FetchResult:
    success: bool                    # HTTP fetch returned 2xx
    thin: bool                       # Content looks like anti-bot, JS shell, short, or 4xx
    text: str                        # Body (HTML), truncated to _MAX_BYTES
    http_status: int
    error: str = ""
    thin_signals: List[str] = field(default_factory=list)
    used_browser: bool = False       # True when we escalated to Playwright/Crawl4AI


def _detect_thin(text: str, status: int) -> tuple[bool, List[str]]:
    if status >= 400:
        return True, [f"http_status={status}"]
    if len(text.strip()) < _MIN_BODY_BYTES:
        return True, [f"body_len={len(text)}"]
    lowered = text.lower()
    hits = [kw for kw in _THIN_KEYWORDS if kw in lowered]
    if len(hits) >= 2:
        return True, hits
    return False, []


def fetch_url(url: str) -> FetchResult:
    """Fetch a URL using httpx. Always returns a FetchResult (never raises).

    Caller can inspect `.thin` to decide whether to escalate to a browser.
    The browser escalation lives in `fetch_url_with_fallback` (Task 2.2).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; UppGrad-Bot/1.0; +https://uppgrad.com)"
        ),
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=_TIMEOUT, follow_redirects=True)
    except httpx.HTTPError as exc:
        logger.warning("fetch_url: network error for %s — %s", url, exc)
        return FetchResult(
            success=False, thin=True, text="",
            http_status=0, error=str(exc),
            thin_signals=["network_error"],
        )

    text = resp.text[:_MAX_BYTES]
    thin, signals = _detect_thin(text, resp.status_code)

    return FetchResult(
        success=resp.status_code < 400,
        thin=thin,
        text=text,
        http_status=resp.status_code,
        thin_signals=signals,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/tools/test_web_fetcher_httpx.py -v`
Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/tools/web_fetcher.py tests/tools/test_web_fetcher_httpx.py
git commit -m "feat(tools): add httpx-based web_fetcher with thin-content detection"
```

---

## Task 2.2: Playwright fallback (env-gated)

**Files:**
- Modify: `src/uppgrad_agentic/tools/web_fetcher.py`
- Test: `tests/tools/test_web_fetcher_browser_fallback.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_web_fetcher_browser_fallback.py
from unittest.mock import patch, AsyncMock, MagicMock

import httpx
import pytest
import respx

from uppgrad_agentic.tools.web_fetcher import fetch_url_with_fallback, FetchResult


@respx.mock
def test_returns_httpx_result_when_fallback_disabled(monkeypatch):
    monkeypatch.delenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", raising=False)
    body = "<html><body>" + ("real content " * 200) + "</body></html>"
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url_with_fallback("https://acme.com/jobs/1")
    assert result.used_browser is False
    assert result.thin is False


@respx.mock
def test_no_fallback_when_httpx_succeeds_with_substantial_content(monkeypatch):
    monkeypatch.setenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "true")
    body = "<html><body>" + ("real content " * 200) + "</body></html>"
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(200, text=body))
    result = fetch_url_with_fallback("https://acme.com/jobs/1")
    assert result.used_browser is False


@respx.mock
def test_fallback_fires_when_httpx_thin_and_browser_enabled(monkeypatch):
    monkeypatch.setenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "true")
    body = "Cloudflare. JavaScript required."
    respx.get("https://acme.com/jobs/1").mock(return_value=httpx.Response(200, text=body))

    fake_crawl = AsyncMock(return_value=MagicMock(
        success=True, markdown="Real apply page content from Playwright. " * 50,
        html="", status_code=200, metadata={"title": "Apply"},
    ))
    fake_crawler = AsyncMock()
    fake_crawler.__aenter__.return_value = fake_crawler
    fake_crawler.__aexit__.return_value = False
    fake_crawler.arun = fake_crawl

    with patch("uppgrad_agentic.tools.web_fetcher._build_async_crawler", return_value=fake_crawler):
        result = fetch_url_with_fallback("https://acme.com/jobs/1")
    assert result.used_browser is True
    assert result.success is True
    assert "Real apply page content" in result.text


def test_fallback_silently_skipped_when_crawl4ai_not_installed(monkeypatch):
    monkeypatch.setenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "true")
    with patch("uppgrad_agentic.tools.web_fetcher._build_async_crawler",
               side_effect=ImportError("crawl4ai not installed")):
        with respx.mock:
            respx.get("https://acme.com/jobs/1").mock(
                return_value=httpx.Response(200, text="Cloudflare. JavaScript required."))
            result = fetch_url_with_fallback("https://acme.com/jobs/1")
    # Returns the httpx (thin) result, browser flag stays False
    assert result.used_browser is False
    assert result.thin is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_web_fetcher_browser_fallback.py -v`
Expected: FAIL — `fetch_url_with_fallback` not defined.

- [ ] **Step 3: Add fallback to `tools/web_fetcher.py`**

Append to `src/uppgrad_agentic/tools/web_fetcher.py`:

```python
import asyncio


def _browser_fallback_enabled() -> bool:
    return os.getenv("UPPGRAD_BROWSER_SCRAPE_ENABLED", "").lower() in ("true", "1", "yes")


def _build_async_crawler():
    """Construct a Crawl4AI crawler. Raises ImportError if crawl4ai missing.

    Patched in tests to inject a fake crawler.
    """
    from crawl4ai import AsyncWebCrawler  # noqa: lazy import — heavy
    return AsyncWebCrawler(verbose=False)


async def _crawl_with_browser(url: str, timeout_seconds: float = 25.0) -> FetchResult:
    """Use Crawl4AI / Playwright to fetch a URL when httpx returned thin content."""
    try:
        crawler = _build_async_crawler()
    except ImportError as exc:
        logger.warning("web_fetcher: crawl4ai not installed — skipping browser fallback (%s)", exc)
        raise

    async with crawler:
        try:
            result = await crawler.arun(url=url, page_timeout=int(timeout_seconds * 1000))
        except Exception as exc:
            logger.warning("web_fetcher: crawl4ai error for %s — %s", url, exc)
            return FetchResult(
                success=False, thin=True, text="",
                http_status=0, error=str(exc),
                thin_signals=["browser_error"], used_browser=True,
            )

    if not getattr(result, "success", False):
        return FetchResult(
            success=False, thin=True, text="",
            http_status=getattr(result, "status_code", 0) or 0,
            error=getattr(result, "error_message", "") or "crawl unsuccessful",
            thin_signals=["crawl_unsuccessful"], used_browser=True,
        )

    md = (getattr(result, "markdown", "") or "")[:_MAX_BYTES]
    thin, signals = _detect_thin(md, getattr(result, "status_code", 200))
    return FetchResult(
        success=True, thin=thin, text=md,
        http_status=getattr(result, "status_code", 200) or 200,
        thin_signals=signals, used_browser=True,
    )


def fetch_url_with_fallback(url: str) -> FetchResult:
    """Fetch with httpx; escalate to Playwright/Crawl4AI when configured AND httpx is thin."""
    httpx_result = fetch_url(url)
    if not httpx_result.thin:
        return httpx_result
    if not _browser_fallback_enabled():
        return httpx_result
    try:
        return asyncio.run(_crawl_with_browser(url))
    except ImportError:
        # Browser fallback enabled but crawl4ai not installed — return httpx result silently
        return httpx_result
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/tools/test_web_fetcher_browser_fallback.py tests/tools/test_web_fetcher_httpx.py -v`
Expected: all 9 pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/tools/web_fetcher.py tests/tools/test_web_fetcher_browser_fallback.py
git commit -m "feat(tools): add env-gated Playwright/Crawl4AI fallback to web_fetcher"
```

---

# PHASE 3 — URL Discovery Orchestration

## Task 3.1: Verification scoring (pure function)

**Files:**
- Create: `src/uppgrad_agentic/tools/url_discovery.py` (verification half)
- Test: `tests/tools/test_url_discovery_verify.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_url_discovery_verify.py
from datetime import datetime, timedelta, timezone

from uppgrad_agentic.tools.url_discovery import score_candidate, VerifyInputs


def _job(title="Senior Backend Engineer", company="Acme Corp",
         posted_iso=None, location="London, UK"):
    return {
        "id": 1,
        "title": title,
        "company": company,
        "posted_time": posted_iso or datetime.now(timezone.utc).isoformat(),
        "location": location,
    }


def test_perfect_match_passes():
    inputs = VerifyInputs(
        candidate_url="https://boards.greenhouse.io/acme/jobs/1",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text="Acme Corp hiring Senior Backend Engineer in London. Apply now.",
        candidate_posted_at=datetime.now(timezone.utc),
        job=_job(),
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is True
    assert score.confidence >= 0.7


def test_title_mismatch_fails():
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Marketing Coordinator",
        candidate_text="Marketing role at Acme.",
        candidate_posted_at=None,
        job=_job(),
        tier="ats",
    )
    assert score_candidate(inputs).passed is False


def test_company_missing_fails_for_tier1():
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Senior Backend Engineer",
        candidate_text="A great backend role somewhere.",
        candidate_posted_at=None,
        job=_job(),
        tier="ats",
    )
    assert score_candidate(inputs).passed is False


def test_company_missing_ok_for_careers_tier():
    inputs = VerifyInputs(
        candidate_url="https://acmecorp.com/careers/role",
        candidate_title="Senior Backend Engineer",
        candidate_text="Backend engineer position. Apply via this form.",
        candidate_posted_at=None,
        job=_job(),
        tier="careers",
    )
    assert score_candidate(inputs).passed is True


def test_old_posting_lowers_confidence():
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text="Acme Corp is hiring",
        candidate_posted_at=datetime.now(timezone.utc) - timedelta(days=400),
        job=_job(posted_iso=datetime.now(timezone.utc).isoformat()),
        tier="ats",
    )
    assert score_candidate(inputs).confidence < 0.85


def test_tier3_uses_stricter_threshold():
    inputs = VerifyInputs(
        candidate_url="https://random-board.com/1",
        candidate_title="Senior Backend Engineer at Acme",
        candidate_text="Acme is hiring backend engineer",
        candidate_posted_at=None,
        job=_job(title="Senior Backend Engineer (Platform)", company="Acme Corp"),
        tier="generic",
    )
    assert score_candidate(inputs).passed is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_url_discovery_verify.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement verification half of `tools/url_discovery.py`**

```python
# src/uppgrad_agentic/tools/url_discovery.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Literal, Optional

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

Tier = Literal["ats", "careers", "generic"]

_TIER_THRESHOLDS = {
    "ats": 0.70,
    "careers": 0.65,
    "generic": 0.80,
}
_TITLE_FUZZY_MIN = 85


@dataclass
class VerifyInputs:
    candidate_url: str
    candidate_title: str
    candidate_text: str
    candidate_posted_at: Optional[datetime]
    job: dict
    tier: Tier


@dataclass
class VerificationScore:
    passed: bool
    confidence: float
    reasons: List[str]


def _parse_iso_or_none(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def score_candidate(inputs: VerifyInputs) -> VerificationScore:
    reasons: List[str] = []
    job = inputs.job
    job_title = (job.get("title") or "").strip()
    job_company = (job.get("company") or "").strip()

    haystack = f"{inputs.candidate_title}\n{inputs.candidate_text[:2000]}"
    title_score = fuzz.partial_ratio(job_title.lower(), haystack.lower()) if job_title else 0
    if title_score < _TITLE_FUZZY_MIN:
        return VerificationScore(passed=False, confidence=0.0,
                                 reasons=[f"title fuzzy {title_score} < {_TITLE_FUZZY_MIN}"])

    if inputs.tier != "careers":
        company_in_url = bool(job_company) and job_company.lower().replace(" ", "") in inputs.candidate_url.lower()
        company_in_text = bool(job_company) and re.search(re.escape(job_company), inputs.candidate_text, re.IGNORECASE) is not None
        if not (company_in_url or company_in_text):
            return VerificationScore(passed=False, confidence=0.0,
                                     reasons=[f"company '{job_company}' not present"])

    confidence = 0.85
    reasons.append(f"title fuzzy {title_score}")

    job_posted = _parse_iso_or_none(job.get("posted_time"))
    if inputs.candidate_posted_at and job_posted:
        delta_days = abs((inputs.candidate_posted_at - job_posted).days)
        if delta_days > 180:
            confidence -= 0.20
            reasons.append(f"freshness off by {delta_days}d")

    job_loc_tokens = {tok.strip().lower() for tok in (job.get("location") or "").split(",") if tok.strip()}
    if job_loc_tokens:
        loc_hit = any(tok in inputs.candidate_text.lower() for tok in job_loc_tokens)
        if not loc_hit:
            confidence -= 0.10
            reasons.append("location not on page")

    confidence = max(0.0, min(1.0, confidence))
    threshold = _TIER_THRESHOLDS[inputs.tier]
    return VerificationScore(passed=confidence >= threshold, confidence=confidence, reasons=reasons)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/tools/test_url_discovery_verify.py -v`
Expected: all 6 pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/tools/url_discovery.py tests/tools/test_url_discovery_verify.py
git commit -m "feat(tools): add candidate-URL verification scoring"
```

---

## Task 3.2: 3-tier discovery orchestration

**Files:**
- Modify: `src/uppgrad_agentic/tools/url_discovery.py`
- Test: `tests/tools/test_url_discovery_orchestration.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_url_discovery_orchestration.py
from unittest.mock import MagicMock

import pytest

from uppgrad_agentic.tools.url_discovery import (
    discover_apply_url, _build_ats_query, _build_careers_query, _build_generic_query,
    DiscoveryResult,
)
from uppgrad_agentic.tools.search import SearchResult
from uppgrad_agentic.tools.web_fetcher import FetchResult


def _job(title="Senior Backend Engineer", company="Acme Corp",
         url_direct=None, company_url=None):
    return {
        "id": 42, "title": title, "company": company,
        "url": "https://www.linkedin.com/jobs/view/42",
        "url_direct": url_direct,
        "company_url": company_url,
        "posted_time": "2026-04-20T00:00:00Z",
        "location": "London, UK",
    }


def test_short_circuit_when_url_direct_present():
    job = _job(url_direct="https://acme.com/apply/1")
    result = discover_apply_url(job, search_provider=None)
    assert result.method == "url_direct"
    assert result.url == "https://acme.com/apply/1"
    assert result.confidence == 1.0


def test_failed_when_no_search_provider_and_no_url_direct():
    result = discover_apply_url(_job(), search_provider=None)
    assert result.method == "failed"
    assert result.url == ""


def test_ats_query_format():
    q = _build_ats_query("Senior Backend Engineer", "Acme Corp")
    assert '"Senior Backend Engineer"' in q
    assert '"Acme Corp"' in q
    assert "site:greenhouse.io" in q
    assert "site:lever.co" in q
    assert "site:myworkdayjobs.com" in q


def test_careers_query_format():
    q = _build_careers_query("Senior Backend Engineer", "https://acme.com/about")
    assert '"Senior Backend Engineer"' in q
    assert "site:acme.com" in q


def test_careers_query_returns_none_without_company_url():
    q = _build_careers_query("Senior Backend Engineer", None)
    assert q is None


def test_ats_tier_returns_first_verified(monkeypatch):
    job = _job()
    fake_search = MagicMock()
    fake_search.search.return_value = [
        SearchResult(url="https://boards.greenhouse.io/acme/jobs/1",
                     title="Senior Backend Engineer at Acme Corp", snippet="Apply now"),
    ]
    fake_fetch = MagicMock(return_value=FetchResult(
        success=True, thin=False,
        text="Acme Corp is hiring Senior Backend Engineer in London",
        http_status=200,
    ))
    monkeypatch.setattr("uppgrad_agentic.tools.url_discovery.fetch_url_with_fallback", fake_fetch)

    result = discover_apply_url(job, search_provider=fake_search)
    assert result.method == "ats"
    assert result.url.startswith("https://boards.greenhouse.io/")


def test_falls_through_to_careers_when_ats_fails(monkeypatch):
    job = _job(company_url="https://acmecorp.com")
    fake_search = MagicMock()
    fake_search.search.side_effect = [
        [SearchResult(url="https://boards.greenhouse.io/acme/jobs/1",
                      title="Marketing Manager", snippet="")],
        [SearchResult(url="https://acmecorp.com/careers/role",
                      title="Senior Backend Engineer", snippet="")],
    ]

    def fake_fetch(url):
        if "greenhouse" in url:
            return FetchResult(success=True, thin=False, text="Marketing role.", http_status=200)
        return FetchResult(success=True, thin=False,
                           text="Senior Backend Engineer position. Apply.",
                           http_status=200)

    monkeypatch.setattr("uppgrad_agentic.tools.url_discovery.fetch_url_with_fallback", fake_fetch)

    result = discover_apply_url(job, search_provider=fake_search)
    assert result.method == "careers"
    assert result.url == "https://acmecorp.com/careers/role"


def test_returns_failed_when_all_tiers_miss(monkeypatch):
    job = _job(company_url="https://acmecorp.com")
    fake_search = MagicMock()
    fake_search.search.return_value = []
    monkeypatch.setattr("uppgrad_agentic.tools.url_discovery.fetch_url_with_fallback", MagicMock())

    result = discover_apply_url(job, search_provider=fake_search)
    assert result.method == "failed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tools/test_url_discovery_orchestration.py -v`
Expected: FAIL — `discover_apply_url` not defined yet.

- [ ] **Step 3: Append orchestration to `tools/url_discovery.py`**

```python
# Append to src/uppgrad_agentic/tools/url_discovery.py

from urllib.parse import urlparse

from uppgrad_agentic.tools.search import SearchProvider, SearchResult
from uppgrad_agentic.tools.web_fetcher import fetch_url_with_fallback


_ATS_DOMAINS = [
    "greenhouse.io", "lever.co", "ashbyhq.com", "workable.com",
    "smartrecruiters.com", "myworkdayjobs.com", "bamboohr.com",
    "jobvite.com", "recruitee.com",
]


@dataclass
class DiscoveryResult:
    url: str
    method: str            # 'url_direct' | 'ats' | 'careers' | 'generic' | 'failed'
    confidence: float


def _build_ats_query(title: str, company: str) -> str:
    sites = " OR ".join(f"site:{d}" for d in _ATS_DOMAINS)
    return f'"{title}" "{company}" ({sites})'


def _extract_company_domain(company_url: Optional[str]) -> Optional[str]:
    if not company_url:
        return None
    try:
        parsed = urlparse(company_url if "://" in company_url else f"https://{company_url}")
    except ValueError:
        return None
    host = (parsed.netloc or parsed.path).lower().lstrip("www.")
    return host or None


def _build_careers_query(title: str, company_url: Optional[str]) -> Optional[str]:
    domain = _extract_company_domain(company_url)
    if not domain:
        return None
    return f'"{title}" site:{domain}'


def _build_generic_query(title: str, company: str) -> str:
    return f'"{title}" "{company}" apply'


def _verify_one(candidate: SearchResult, job: dict, tier: Tier) -> Optional[VerificationScore]:
    fetch = fetch_url_with_fallback(candidate.url)
    if not fetch.success:
        return None
    candidate_title = candidate.title  # title-from-fetch parsing deferred
    inputs = VerifyInputs(
        candidate_url=candidate.url,
        candidate_title=candidate_title,
        candidate_text=fetch.text,
        candidate_posted_at=None,
        job=job,
        tier=tier,
    )
    score = score_candidate(inputs)
    return score if score.passed else None


def _try_tier(candidates: List[SearchResult], job: dict, tier: Tier) -> Optional[tuple[SearchResult, VerificationScore]]:
    for cand in candidates:
        verified = _verify_one(cand, job, tier)
        if verified is not None:
            return cand, verified
    return None


def discover_apply_url(
    job: dict,
    search_provider: Optional[SearchProvider],
) -> DiscoveryResult:
    """Synchronous discovery orchestrator.

    Caching (Phase 6) lives in the backend adapter — agentic stays DB-free.
    Callers that want cached results should consult the cache *before* calling
    this function and skip if a hit was found.
    """
    url_direct = (job.get("url_direct") or "").strip()
    if url_direct:
        return DiscoveryResult(url=url_direct, method="url_direct", confidence=1.0)

    if search_provider is None:
        return DiscoveryResult(url="", method="failed", confidence=0.0)

    title = (job.get("title") or "").strip()
    company = (job.get("company") or "").strip()
    if not title or not company:
        return DiscoveryResult(url="", method="failed", confidence=0.0)

    # Tier 1: ATS
    ats_results = search_provider.search(_build_ats_query(title, company), count=3)
    hit = _try_tier(ats_results, job, "ats")
    if hit:
        cand, score = hit
        return DiscoveryResult(url=cand.url, method="ats", confidence=score.confidence)

    # Tier 2: Careers
    careers_q = _build_careers_query(title, job.get("company_url"))
    if careers_q:
        careers_results = search_provider.search(careers_q, count=3)
        hit = _try_tier(careers_results, job, "careers")
        if hit:
            cand, score = hit
            return DiscoveryResult(url=cand.url, method="careers", confidence=score.confidence)

    # Tier 3: Generic
    generic_results = search_provider.search(_build_generic_query(title, company), count=3)
    hit = _try_tier(generic_results, job, "generic")
    if hit:
        cand, score = hit
        return DiscoveryResult(url=cand.url, method="generic", confidence=score.confidence)

    return DiscoveryResult(url="", method="failed", confidence=0.0)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/tools/test_url_discovery_orchestration.py tests/tools/test_url_discovery_verify.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/tools/url_discovery.py tests/tools/test_url_discovery_orchestration.py
git commit -m "feat(tools): add 3-tier search orchestration with verification"
```

---

# PHASE 4 — Discovery Graph Node

## Task 4.1: `discover_apply_url` graph node

**Files:**
- Create: `src/uppgrad_agentic/workflows/auto_apply/nodes/discover_apply_url.py`
- Test: `tests/workflows/auto_apply/test_discover_apply_url.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/workflows/auto_apply/test_discover_apply_url.py
from unittest.mock import patch

from uppgrad_agentic.workflows.auto_apply.nodes.discover_apply_url import (
    discover_apply_url_node,
)
from uppgrad_agentic.tools.url_discovery import DiscoveryResult


def _state(opp_type="job", url_direct=None, employer_id=None,
           preset_url="", preset_method=""):
    return {
        "opportunity_type": opp_type,
        "opportunity_id": "1",
        "opportunity_data": {
            "id": 42, "title": "SWE", "company": "Acme",
            "url": "https://linkedin.com/jobs/view/42",
            "url_direct": url_direct,
            "employer_id": employer_id,
            "company_url": None, "posted_time": None, "location": "",
        },
        "discovered_apply_url": preset_url or None,
        "discovery_method": preset_method or None,
    }


def test_skips_for_non_jobs():
    out = discover_apply_url_node(_state(opp_type="masters"))
    assert out["current_step"] == "discover_apply_url"
    assert "discovery_method" not in out


def test_skips_for_internal_jobs():
    out = discover_apply_url_node(_state(employer_id=1))
    assert out["discovery_method"] == "skipped_internal"
    assert out["discovered_apply_url"] is None


def test_uses_cached_when_already_in_state():
    state = _state(preset_url="https://cached.com/job/1", preset_method="ats")
    out = discover_apply_url_node(state)
    # Cache hit — no work done; we propagate the cached values
    assert out["discovered_apply_url"] == "https://cached.com/job/1"
    assert out["discovery_method"] == "ats"


def test_url_direct_short_circuit_via_orchestrator():
    state = _state(url_direct="https://acme.com/apply/1")
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.discover_apply_url.discover_apply_url",
        return_value=DiscoveryResult(url="https://acme.com/apply/1",
                                     method="url_direct", confidence=1.0),
    ):
        out = discover_apply_url_node(state)
    assert out["discovered_apply_url"] == "https://acme.com/apply/1"
    assert out["discovery_method"] == "url_direct"


def test_failed_path_does_not_set_error():
    state = _state(url_direct=None)
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.discover_apply_url.discover_apply_url",
        return_value=DiscoveryResult(url="", method="failed", confidence=0.0),
    ):
        out = discover_apply_url_node(state)
    assert out["discovered_apply_url"] is None
    assert out["discovery_method"] == "failed"
    assert "result" not in out


def test_short_circuits_on_upstream_error():
    state = _state()
    state["result"] = {"status": "error", "error_code": "X"}
    out = discover_apply_url_node(state)
    assert out == {"current_step": "discover_apply_url", "step_history": ["discover_apply_url"]}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/workflows/auto_apply/test_discover_apply_url.py -v`
Expected: FAIL — node module not found.

- [ ] **Step 3: Implement the node**

```python
# src/uppgrad_agentic/workflows/auto_apply/nodes/discover_apply_url.py
from __future__ import annotations

import logging

from uppgrad_agentic.common.llm import get_search_provider
from uppgrad_agentic.tools.url_discovery import discover_apply_url
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)


def discover_apply_url_node(state: AutoApplyState) -> dict:
    updates = {"current_step": "discover_apply_url", "step_history": ["discover_apply_url"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    if state.get("opportunity_type") != "job":
        return updates

    opportunity_data = state.get("opportunity_data") or {}

    # Internal jobs (employer_id == 1) submit through the platform — discovery N/A.
    if opportunity_data.get("employer_id") == 1:
        return {
            **updates,
            "discovered_apply_url": None,
            "discovery_method": "skipped_internal",
            "discovery_confidence": 0.0,
        }

    # Cache hit — adapter pre-loaded a known-good URL.
    cached_url = state.get("discovered_apply_url")
    cached_method = state.get("discovery_method")
    if cached_url and cached_method and cached_method != "failed":
        return {
            **updates,
            "discovered_apply_url": cached_url,
            "discovery_method": cached_method,
            "discovery_confidence": state.get("discovery_confidence") or 0.0,
        }

    search_provider = get_search_provider()
    result = discover_apply_url(opportunity_data, search_provider=search_provider)

    logger.info(
        "discover_apply_url: method=%s confidence=%.2f url=%s",
        result.method, result.confidence, result.url,
    )

    return {
        **updates,
        "discovered_apply_url": result.url or None,
        "discovery_method": result.method,
        "discovery_confidence": result.confidence,
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_discover_apply_url.py -v`
Expected: all 6 pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/nodes/discover_apply_url.py tests/workflows/auto_apply/test_discover_apply_url.py
git commit -m "feat(auto_apply): add discover_apply_url graph node"
```

---

## Task 4.2: Wire `discover_apply_url` into the graph

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/graph.py`
- Test: `tests/workflows/auto_apply/test_graph_discovery_routing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/workflows/auto_apply/test_graph_discovery_routing.py
from uppgrad_agentic.workflows.auto_apply.graph import build_graph


def test_graph_includes_discover_apply_url_node():
    g = build_graph().get_graph()
    assert "discover_apply_url" in g.nodes


def test_load_opportunity_routes_to_discover_for_jobs():
    g = build_graph().get_graph()
    edges = [e for e in g.edges if e.source == "load_opportunity"]
    targets = {e.target for e in edges}
    assert "discover_apply_url" in targets


def test_discover_routes_to_scrape():
    g = build_graph().get_graph()
    edges = [e for e in g.edges if e.source == "discover_apply_url"]
    targets = {e.target for e in edges}
    assert "scrape_application_page" in targets
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/workflows/auto_apply/test_graph_discovery_routing.py -v`
Expected: FAIL.

- [ ] **Step 3: Modify `graph.py`**

Add the import after the other node imports in `src/uppgrad_agentic/workflows/auto_apply/graph.py`:

```python
from uppgrad_agentic.workflows.auto_apply.nodes.discover_apply_url import discover_apply_url_node
```

Replace the existing `_route_after_load` function:

```python
def _route_after_load(state: AutoApplyState) -> str:
    if state.get("result", {}).get("status") == "error":
        return "end_with_error"
    if state.get("opportunity_type") == "job":
        return "discover_apply_url"
    return "determine_requirements"
```

Add a new router:

```python
def _route_after_discovery(state: AutoApplyState) -> str:
    if state.get("result", {}).get("status") == "error":
        return "end_with_error"
    return "scrape_application_page"
```

In `build_graph`, register the new node and rewire the conditional edges. Find the existing `g.add_conditional_edges("load_opportunity", _route_after_load, ...)` block and replace with:

```python
    g.add_node("discover_apply_url", discover_apply_url_node)

    g.add_conditional_edges(
        "load_opportunity",
        _route_after_load,
        {
            "discover_apply_url": "discover_apply_url",
            "determine_requirements": "determine_requirements",
            "end_with_error": "end_with_error",
        },
    )

    g.add_conditional_edges(
        "discover_apply_url",
        _route_after_discovery,
        {
            "scrape_application_page": "scrape_application_page",
            "end_with_error": "end_with_error",
        },
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_graph_discovery_routing.py tests/workflows/auto_apply/ -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/graph.py tests/workflows/auto_apply/test_graph_discovery_routing.py
git commit -m "feat(auto_apply): wire discover_apply_url between load_opportunity and scrape"
```

---

# PHASE 5 — Replace `scrape_application_page`

## Task 5.1: Switch scrape to use `web_fetcher` + `discovered_apply_url`

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/nodes/scrape_application_page.py`
- Test: `tests/workflows/auto_apply/test_scrape_application_page_fetcher.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/workflows/auto_apply/test_scrape_application_page_fetcher.py
from unittest.mock import patch

from uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page import (
    scrape_application_page,
)
from uppgrad_agentic.tools.web_fetcher import FetchResult


def _state(discovered=None, method="ats"):
    return {
        "opportunity_type": "job",
        "opportunity_id": "1",
        "opportunity_data": {"id": 42, "title": "SWE", "company": "Acme",
                             "url": "https://linkedin.com/x", "url_direct": None},
        "discovered_apply_url": discovered,
        "discovery_method": method,
    }


def test_no_discovered_url_records_failed():
    out = scrape_application_page(_state(discovered=None, method="failed"))
    sr = out["scraped_requirements"]
    assert sr["status"] == "failed"
    assert sr["raw_content"] == ""


def test_skips_for_non_jobs():
    state = _state(discovered="https://acme.com/job/1")
    state["opportunity_type"] = "masters"
    out = scrape_application_page(state)
    assert "scraped_requirements" not in out


def test_uses_discovered_url_records_content():
    state = _state(discovered="https://boards.greenhouse.io/acme/jobs/1", method="ats")
    fake_fetch = FetchResult(
        success=True, thin=False,
        text="Apply now. Upload CV and Cover Letter.",
        http_status=200,
    )
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page.fetch_url_with_fallback",
        return_value=fake_fetch,
    ):
        out = scrape_application_page(state)
    sr = out["scraped_requirements"]
    assert sr["status"] == "partial"
    assert sr["source"] == "https://boards.greenhouse.io/acme/jobs/1"
    assert "Upload CV" in sr["raw_content"]


def test_thin_response_records_failed():
    state = _state(discovered="https://x.com/job/1", method="ats")
    fake_fetch = FetchResult(
        success=True, thin=True,
        text="Cloudflare. Captcha.",
        http_status=200,
        thin_signals=["cloudflare", "captcha"],
    )
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page.fetch_url_with_fallback",
        return_value=fake_fetch,
    ):
        out = scrape_application_page(state)
    sr = out["scraped_requirements"]
    assert sr["status"] == "failed"


def test_short_circuits_on_upstream_error():
    state = _state(discovered="https://x.com/1")
    state["result"] = {"status": "error"}
    out = scrape_application_page(state)
    assert "scraped_requirements" not in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/workflows/auto_apply/test_scrape_application_page_fetcher.py -v`
Expected: FAIL — node still uses `requests.get`.

- [ ] **Step 3: Replace the node implementation**

Overwrite `src/uppgrad_agentic/workflows/auto_apply/nodes/scrape_application_page.py`:

```python
from __future__ import annotations

import logging

from uppgrad_agentic.tools.web_fetcher import fetch_url_with_fallback
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)


def scrape_application_page(state: AutoApplyState) -> dict:
    updates = {"current_step": "scrape_application_page", "step_history": ["scrape_application_page"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    if state.get("opportunity_type") != "job":
        return updates

    target_url = (state.get("discovered_apply_url") or "").strip()

    if not target_url:
        logger.info("scrape_application_page: no discovered URL — recording failed scrape")
        return {
            **updates,
            "scraped_requirements": {
                "status": "failed",
                "requirements": [],
                "confidence": 0.0,
                "source": "",
                "raw_content": "",
                "http_status": 0,
                "error": "no apply URL discovered",
            },
        }

    fetch = fetch_url_with_fallback(target_url)

    if not fetch.success or fetch.thin:
        logger.warning(
            "scrape_application_page: fetch thin/failed for %s (status=%s, signals=%s)",
            target_url, fetch.http_status, fetch.thin_signals,
        )
        return {
            **updates,
            "scraped_requirements": {
                "status": "failed",
                "requirements": [],
                "confidence": 0.0,
                "source": target_url,
                "raw_content": "",
                "http_status": fetch.http_status,
                "error": fetch.error or f"thin: {','.join(fetch.thin_signals)}",
            },
        }

    logger.info("scrape_application_page: fetched %d chars from %s (browser=%s)",
                len(fetch.text), target_url, fetch.used_browser)
    return {
        **updates,
        "scraped_requirements": {
            "status": "partial",   # evaluate_scrape sets the final status
            "requirements": [],
            "confidence": 0.0,
            "source": target_url,
            "raw_content": fetch.text,
            "http_status": fetch.http_status,
        },
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_scrape_application_page_fetcher.py tests/workflows/auto_apply/ -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/nodes/scrape_application_page.py tests/workflows/auto_apply/test_scrape_application_page_fetcher.py
git commit -m "refactor(auto_apply): scrape via web_fetcher against discovered URL"
```

---

# PHASE 6 — Backend Cache

## Task 6.1: `JobApplyUrlDiscovery` model + migration

**Files:**
- Modify: `backend/ai_services/models.py`
- Create: `backend/ai_services/migrations/0009_job_apply_url_discovery.py` (auto-generated)
- Test: `backend/ai_services/tests/test_job_apply_url_discovery_model.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/ai_services/tests/test_job_apply_url_discovery_model.py
from datetime import timedelta
from django.test import TestCase
from django.utils import timezone

from ai_services.models import JobApplyUrlDiscovery


class JobApplyUrlDiscoveryModelTests(TestCase):
    def test_can_create(self):
        d = JobApplyUrlDiscovery.objects.create(
            job_id=42,
            discovered_url="https://boards.greenhouse.io/acme/jobs/1",
            discovery_method="ats",
            discovery_confidence=0.9,
        )
        self.assertEqual(d.job_id, 42)
        self.assertIsNotNone(d.discovered_at)
        self.assertIsNotNone(d.last_verified_at)

    def test_job_id_is_pk(self):
        from django.db.utils import IntegrityError
        JobApplyUrlDiscovery.objects.create(
            job_id=42, discovered_url="x", discovery_method="ats", discovery_confidence=0.9,
        )
        with self.assertRaises(IntegrityError):
            JobApplyUrlDiscovery.objects.create(
                job_id=42, discovered_url="y", discovery_method="careers", discovery_confidence=0.7,
            )

    def test_invalid_method_rejected(self):
        from django.core.exceptions import ValidationError
        d = JobApplyUrlDiscovery(
            job_id=42, discovered_url="x", discovery_method="bogus",
            discovery_confidence=0.5,
        )
        with self.assertRaises(ValidationError):
            d.full_clean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python manage.py test ai_services.tests.test_job_apply_url_discovery_model -v 2`
Expected: FAIL — model not defined.

- [ ] **Step 3: Add the model**

Append to `backend/ai_services/models.py`:

```python
class JobApplyUrlDiscovery(models.Model):
    """Cross-user cache of resolved apply URLs per linkedin_jobs row.

    A row exists when discovery succeeded (method != 'failed'). Cache lookups
    that find a row older than 14 days re-discover and update last_verified_at.

    No closed-postings cleanup hook — the staleness gate is sufficient because
    closed postings are filtered out of user-facing surfaces and never trigger
    a cache lookup in practice.
    """
    METHOD_CHOICES = [
        ("url_direct", "URL Direct"),
        ("ats", "ATS"),
        ("careers", "Careers"),
        ("generic", "Generic"),
    ]

    job_id = models.BigIntegerField(primary_key=True)
    discovered_url = models.TextField()
    discovery_method = models.CharField(max_length=20, choices=METHOD_CHOICES)
    discovery_confidence = models.FloatField()
    discovered_at = models.DateTimeField(auto_now_add=True)
    last_verified_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "job_apply_url_discovery"
        indexes = [
            models.Index(fields=["discovery_method"]),
            models.Index(fields=["last_verified_at"]),
        ]

    def __str__(self):
        return f"JobApplyUrlDiscovery(job={self.job_id}, method={self.discovery_method})"
```

- [ ] **Step 4: Generate migration**

Run: `python manage.py makemigrations ai_services -n job_apply_url_discovery`
Expected: creates `0009_job_apply_url_discovery.py`.

- [ ] **Step 5: Apply migration**

Run: `python manage.py migrate ai_services`

- [ ] **Step 6: Run tests**

Run: `python manage.py test ai_services.tests.test_job_apply_url_discovery_model -v 2`
Expected: 3 pass.

- [ ] **Step 7: Commit**

```bash
git add ai_services/models.py ai_services/migrations/0009_job_apply_url_discovery.py ai_services/tests/test_job_apply_url_discovery_model.py
git commit -m "feat(ai_services): add JobApplyUrlDiscovery cache table"
```

---

## Task 6.2: Adapter cache lookup helper + integration into `_run_graph_initial_phase`

**Files:**
- Modify: `backend/ai_services/auto_apply_adapter.py`
- Test: `backend/ai_services/tests/test_auto_apply_adapter_discovery_cache.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/ai_services/tests/test_auto_apply_adapter_discovery_cache.py
from datetime import timedelta
from django.test import TestCase
from django.utils import timezone

from ai_services.auto_apply_adapter import (
    _lookup_discovery_cache, _MAX_CACHE_AGE_DAYS,
)
from ai_services.models import JobApplyUrlDiscovery


class DiscoveryCacheLookupTests(TestCase):
    def test_returns_none_for_missing(self):
        result = _lookup_discovery_cache(99999)
        self.assertIsNone(result)

    def test_returns_fresh_cached_row(self):
        JobApplyUrlDiscovery.objects.create(
            job_id=42, discovered_url="https://x.com/1",
            discovery_method="ats", discovery_confidence=0.9,
        )
        result = _lookup_discovery_cache(42)
        self.assertIsNotNone(result)
        self.assertEqual(result["url"], "https://x.com/1")
        self.assertEqual(result["method"], "ats")

    def test_returns_none_for_stale_row(self):
        d = JobApplyUrlDiscovery.objects.create(
            job_id=42, discovered_url="https://x.com/1",
            discovery_method="ats", discovery_confidence=0.9,
        )
        # backdate
        stale = timezone.now() - timedelta(days=_MAX_CACHE_AGE_DAYS + 1)
        JobApplyUrlDiscovery.objects.filter(job_id=42).update(last_verified_at=stale)
        result = _lookup_discovery_cache(42)
        self.assertIsNone(result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python manage.py test ai_services.tests.test_auto_apply_adapter_discovery_cache -v 2`
Expected: FAIL — `_lookup_discovery_cache` not defined.

- [ ] **Step 3: Add the helper + wire into initial phase**

In `backend/ai_services/auto_apply_adapter.py`, add near the top (after the model imports):

```python
from datetime import timedelta
from .models import JobApplyUrlDiscovery

_MAX_CACHE_AGE_DAYS = 14


def _lookup_discovery_cache(job_id: int) -> Optional[Dict[str, Any]]:
    """Return cached discovery result or None when missing/stale.

    Stale = last_verified_at older than _MAX_CACHE_AGE_DAYS. Stale rows are
    NOT deleted here; the next successful discovery overwrites them via
    _persist_discovery_to_cache.
    """
    row = JobApplyUrlDiscovery.objects.filter(job_id=job_id).first()
    if row is None:
        return None
    cutoff = timezone.now() - timedelta(days=_MAX_CACHE_AGE_DAYS)
    if row.last_verified_at < cutoff:
        return None
    return {
        "url": row.discovered_url,
        "method": row.discovery_method,
        "confidence": row.discovery_confidence,
    }
```

In `_run_graph_initial_phase`, after building `profile_snapshot` and before `graph.invoke`, inject the cached discovery into initial state:

```python
        initial_state = {
            "opportunity_type": session.opportunity_type,
            "opportunity_id": str(session.opportunity_id),
            "opportunity_data": opportunity_snapshot,
            "profile_snapshot": profile_snapshot,
            "gate_0_iteration_count": 0,
        }

        # Cache lookup — only meaningful for jobs (matches discover_apply_url node logic)
        if session.opportunity_type == "job":
            cached = _lookup_discovery_cache(int(session.opportunity_id))
            if cached:
                initial_state["discovered_apply_url"] = cached["url"]
                initial_state["discovery_method"] = cached["method"]
                initial_state["discovery_confidence"] = cached["confidence"]
                logger.info(
                    "auto_apply: cache hit for job %s — method=%s",
                    session.opportunity_id, cached["method"],
                )
```

- [ ] **Step 4: Run tests**

Run: `python manage.py test ai_services.tests.test_auto_apply_adapter_discovery_cache -v 2`
Expected: 3 pass.

- [ ] **Step 5: Commit**

```bash
git add ai_services/auto_apply_adapter.py ai_services/tests/test_auto_apply_adapter_discovery_cache.py
git commit -m "feat(ai_services): adapter looks up discovery cache before graph invoke"
```

---

## Task 6.3: Adapter cache write in `_persist_state_after_phase`

**Files:**
- Modify: `backend/ai_services/auto_apply_adapter.py`
- Test: `backend/ai_services/tests/test_auto_apply_adapter_discovery_cache.py` (extend)

- [ ] **Step 1: Add failing test**

Append to `backend/ai_services/tests/test_auto_apply_adapter_discovery_cache.py`:

```python
import uuid
from django.contrib.auth.models import User

from accounts.models import Student
from ai_services.models import ApplicationSession


class DiscoveryCachePersistTests(TestCase):
    def setUp(self):
        u = User.objects.create_user(username="dc", email="dc@x.com", password="x")
        self.student = Student.objects.create(user=u)

    def test_persist_writes_cache_on_successful_discovery(self):
        from ai_services.auto_apply_adapter import _persist_state_after_phase
        s = ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=42,
            opportunity_snapshot={"id": 42},
        )
        _persist_state_after_phase(s, {
            "current_step": "scrape_application_page",
            "step_history": ["x"],
            "discovered_apply_url": "https://boards.greenhouse.io/acme/jobs/1",
            "discovery_method": "ats",
            "discovery_confidence": 0.85,
            "result": {},
        }, pending_node="human_gate_1")

        row = JobApplyUrlDiscovery.objects.filter(job_id=42).first()
        self.assertIsNotNone(row)
        self.assertEqual(row.discovered_url, "https://boards.greenhouse.io/acme/jobs/1")
        self.assertEqual(row.discovery_method, "ats")

    def test_persist_does_not_write_for_failed(self):
        from ai_services.auto_apply_adapter import _persist_state_after_phase
        s = ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=43,
            opportunity_snapshot={"id": 43},
        )
        _persist_state_after_phase(s, {
            "current_step": "scrape_application_page",
            "step_history": ["x"],
            "discovered_apply_url": None,
            "discovery_method": "failed",
            "discovery_confidence": 0.0,
            "result": {},
        }, pending_node="")
        self.assertFalse(JobApplyUrlDiscovery.objects.filter(job_id=43).exists())

    def test_persist_does_not_write_for_skipped_internal(self):
        from ai_services.auto_apply_adapter import _persist_state_after_phase
        s = ApplicationSession.objects.create(
            student=self.student, thread_id=uuid.uuid4(),
            opportunity_type="job", opportunity_id=44,
            opportunity_snapshot={"id": 44, "employer_id": 1},
        )
        _persist_state_after_phase(s, {
            "current_step": "discover_apply_url",
            "step_history": ["x"],
            "discovered_apply_url": None,
            "discovery_method": "skipped_internal",
            "discovery_confidence": 0.0,
            "result": {},
        }, pending_node="determine_requirements")
        self.assertFalse(JobApplyUrlDiscovery.objects.filter(job_id=44).exists())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python manage.py test ai_services.tests.test_auto_apply_adapter_discovery_cache.DiscoveryCachePersistTests -v 2`
Expected: FAIL — adapter doesn't write to cache yet.

- [ ] **Step 3: Add `_persist_discovery_to_cache` + call from `_persist_state_after_phase`**

In `backend/ai_services/auto_apply_adapter.py`, add:

```python
_CACHEABLE_METHODS = {"url_direct", "ats", "careers", "generic"}


def _persist_discovery_to_cache(
    job_id: int, url: str, method: str, confidence: float,
) -> None:
    """Upsert a discovery row into the cache. Skips for non-cacheable methods."""
    if method not in _CACHEABLE_METHODS or not url:
        return
    JobApplyUrlDiscovery.objects.update_or_create(
        job_id=job_id,
        defaults={
            "discovered_url": url,
            "discovery_method": method,
            "discovery_confidence": confidence,
        },
    )
```

In `_persist_state_after_phase`, after the existing `discovery_*` field copies, add:

```python
    # Persist successful discovery to the cross-user cache.
    if session.opportunity_type == "job" and session.discovery_method:
        _persist_discovery_to_cache(
            int(session.opportunity_id),
            session.discovered_apply_url,
            session.discovery_method,
            session.discovery_confidence or 0.0,
        )
```

- [ ] **Step 4: Run tests**

Run: `python manage.py test ai_services.tests.test_auto_apply_adapter_discovery_cache -v 2`
Expected: all 6 pass.

- [ ] **Step 5: Commit**

```bash
git add ai_services/auto_apply_adapter.py ai_services/tests/test_auto_apply_adapter_discovery_cache.py
git commit -m "feat(ai_services): adapter writes successful discovery to cache after each phase"
```

---

# PHASE 7 — Smoke Verification

## Task 7.1: Live test against Neon dev with Brave Search enabled

**Files:** none (verification only)

- [ ] **Step 1: Run full agentic suite**

Run: `cd AgenticWorkflows_LangChain-LangGraph && uv run pytest tests/ -q`
Expected: all green.

- [ ] **Step 2: Run full backend ai_services suite**

Run: `cd backend && python manage.py test ai_services -v 1`
Expected: all new tests pass; pre-existing 1 unrelated failure stays.

- [ ] **Step 3: Live integration drive**

Set environment for the next process:
```bash
export UPPGRAD_LLM_PROVIDER=openai
export UPPGRAD_OPENAI_MODEL=gpt-4o-mini
export UPPGRAD_SEARCH_PROVIDER=brave
export BRAVE_SEARCH_API_KEY=<your key>
# UPPGRAD_BROWSER_SCRAPE_ENABLED stays unset (httpx-only fallback)
```

Drive a live external job through the workflow against Neon dev (use the same script pattern as the original integration smoke test). Confirm:
- `discovery_method` populated to one of `url_direct | ats | careers | generic | failed` (not blank)
- `JobApplyUrlDiscovery.objects.filter(job_id=<id>).exists()` is True when method != failed
- A second session for the same job hits the cache (log line: `cache hit for job ...`)
- `compatibility_warnings` propagates from eligibility into `application_package.warnings`

- [ ] **Step 4: Document smoke results**

```bash
git commit --allow-empty -m "test: smoke-verified discovery v2 against Neon dev with Brave Search"
```

---

## Self-review

**Spec coverage:**
- ✅ Eligibility cleanup (D1) — Tasks 0.1-0.4
- ✅ Uniform pipeline, no Greenhouse special case (D2) — Tasks 3.x, 5.1
- ✅ httpx-first + Playwright fallback (D3) — Tasks 2.1, 2.2
- ✅ Brave opt-in via env var (D4) — Tasks 1.1, search-provider factory
- ✅ Backend-owned cache (D5) — Tasks 6.1-6.3
- ✅ No closed-postings cleanup (D6) — explicitly skipped
- ✅ Discovery before scrape, jobs only (D7) — Tasks 4.1, 4.2
- ✅ ToS acceptance (D8) — documented in plan header

**Placeholder scan:** every step has runnable code or exact commands; no TBD/TODO/etc.

**Type consistency:**
- `FetchResult` (Task 2.1) — consumed by Tasks 2.2, 3.2, 5.1 with same shape.
- `DiscoveryResult` (Task 3.2) — consumed by Tasks 4.1.
- `VerifyInputs`, `VerificationScore` (Task 3.1) — consumed by Task 3.2.
- `SearchProvider`, `SearchResult` (Task 1.1) — consumed by Tasks 3.2, 4.1.
- `_lookup_discovery_cache(job_id)` returns `Optional[Dict[str, Any]]` with keys `url`, `method`, `confidence` — consistent across Tasks 6.2 (define) and 6.3 (use).
- `_persist_discovery_to_cache(job_id, url, method, confidence)` signature consistent.
- State key names (`discovered_apply_url`, `discovery_method`, `discovery_confidence`, `compatibility_warnings`) used consistently across agentic and backend.
- `discover_apply_url_node` (Task 4.1) consumes `state["opportunity_data"]["employer_id"]` and `state["discovered_apply_url"]`/`discovery_method` for cache short-circuit.

No drift detected.
