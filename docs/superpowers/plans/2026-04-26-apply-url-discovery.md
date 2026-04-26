# Apply-URL Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find an externally-applicable URL for LinkedIn jobs that lack `url_direct`, by discovering the posting on a public ATS (Greenhouse, Lever, Ashby, Workable, SmartRecruiters, Workday) or the company's careers site, then scrape requirements from there with Crawl4AI instead of from LinkedIn.

**Architecture:** Insert a `discover_apply_url` node between `load_opportunity` and `scrape_application_page`. It returns a discovered URL (`url_direct` if already in DB, ATS hit, careers-site hit, or `None`). When discovery succeeds, `scrape_application_page` uses Crawl4AI against the discovered URL; when it fails, the existing graceful fallback to assumed defaults runs unchanged. Discovery results are cached in a new `job_apply_url_discovery` table keyed on `linkedin_jobs.id`, invalidated when the closed-postings scraper marks the posting closed.

**Tech Stack:** Python 3.11+, LangGraph, Crawl4AI (new dep), Brave Search API (new dep — single HTTP client, no SDK), rapidfuzz (new dep), Pydantic, pytest, pytest-asyncio (new dep). No backend changes in this plan beyond the new table migration; FastAPI integration remains out of scope per CLAUDE.md.

---

> **Status note (2026-04-26):** This plan is paused pending backend integration. Reasons: (1) the cross-user discovery cache cannot exist meaningfully until the agentic workflow shares a Postgres connection with the rest of the platform — an in-process cache would be thrown away on integration; (2) discovery thresholds (0.7/0.65/0.8) and query templates can only be tuned against real `linkedin_jobs` rows, which `_fetch_opportunity` does not yet return; (3) the closed-postings cache-invalidation hook (originally Task 12, `bitirme/db_utils.py` patch) has been **dropped** from this plan — closed jobs are filtered out of user-facing surfaces so a stale row is never read, and the existing `last_verified_at` + 14-day re-verification gate already covers the rare close-then-repost edge case without cross-repo coupling. When this plan is resumed: replace `InMemoryDiscoveryCache` with a Postgres-backed implementation, drop the `mark_job_closed` change from Task 12 (keep only the SQL migration), and re-tune verification thresholds against real data before committing them.

---

## Scope

**In scope:**
1. New `tools/search.py` with a `SearchProvider` ABC + `BraveSearchProvider` impl.
2. New `tools/url_discovery.py` orchestrating three discovery tiers + verification.
3. New `tools/url_discovery_cache.py` for read-through cache (in-process for now; DB integration spec'd but stubbed pending backend wiring per existing pattern).
4. New `nodes/discover_apply_url.py` graph node.
5. Replace `scrape_application_page._fetch_url` with a Crawl4AI-backed scraper that targets the discovered URL.
6. New `AutoApplyState` fields: `discovered_apply_url`, `discovery_method`, `discovery_confidence`.
7. Migration script for `job_apply_url_discovery` (SQL-only — no Django model needed since this lives next to the closed-postings cleanup hook in `bitirme/`).
8. One-line cleanup hook in `bitirme/db_utils.mark_job_closed` to delete the cache row.

**Out of scope (deferred):**
- SearXNG fallback (revisit if Brave recall disappoints).
- Playwright tier for Workday-class JS portals (Crawl4AI's headless mode covers most cases; escalate later if evaluation says otherwise).
- External form auto-submission (already deferred per CLAUDE.md).
- Real DB lookups for `_fetch_opportunity` (separate integration task).

---

## File Structure

| Path | Responsibility | Status |
|---|---|---|
| `src/uppgrad_agentic/tools/search.py` | `SearchProvider` ABC + `BraveSearchProvider` + `SearchResult` model | Create |
| `src/uppgrad_agentic/tools/url_discovery.py` | `discover_apply_url(job)` orchestration: ATS tier, careers tier, generic tier, verification | Create |
| `src/uppgrad_agentic/tools/url_discovery_cache.py` | `get_cached(job_id)`, `set_cached(...)`, `invalidate(job_id)` — in-process LRU now, DB-backed swap later | Create |
| `src/uppgrad_agentic/tools/web_scraper.py` | Thin async wrapper over Crawl4AI returning markdown + extracted page metadata | Create |
| `src/uppgrad_agentic/workflows/auto_apply/nodes/discover_apply_url.py` | Graph node calling `tools.url_discovery.discover_apply_url` | Create |
| `src/uppgrad_agentic/workflows/auto_apply/nodes/scrape_application_page.py` | Switch from `requests.get` → `tools.web_scraper.scrape_url`; consume `state["discovered_apply_url"]` | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/state.py` | Add `discovered_apply_url`, `discovery_method`, `discovery_confidence` | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/schemas.py` | Add `DiscoveryResult` Pydantic model | Modify |
| `src/uppgrad_agentic/workflows/auto_apply/graph.py` | Insert `discover_apply_url` between `load_opportunity` and `scrape_application_page` for jobs only | Modify |
| `src/uppgrad_agentic/common/llm.py` | Add `get_search_provider()` factory mirroring `get_llm()` opt-in pattern | Modify |
| `pyproject.toml` | Add `crawl4ai`, `rapidfuzz`, `httpx`, `pytest-asyncio` | Modify |
| `tests/tools/test_search.py` | Unit tests for `BraveSearchProvider` (mocked HTTP) | Create |
| `tests/tools/test_url_discovery.py` | Unit tests for tier orchestration + verification scoring | Create |
| `tests/tools/test_url_discovery_cache.py` | Unit tests for cache get/set/invalidate | Create |
| `tests/tools/test_web_scraper.py` | Unit tests for the Crawl4AI wrapper (mocked) | Create |
| `tests/workflows/auto_apply/test_discover_apply_url.py` | Node-level test | Create |
| `tests/workflows/auto_apply/test_graph_discovery_routing.py` | End-to-end graph test for the new branch | Create |
| `migrations/2026-04-26_job_apply_url_discovery.sql` | SQL migration for the cache table (lives in agentic repo since cache is owned here) | Create |

---

## State Schema Additions

```python
# Added to AutoApplyState in state.py
discovered_apply_url: Optional[str]   # final URL handed to scrape_application_page
discovery_method: Optional[str]       # 'url_direct' | 'ats' | 'careers' | 'generic' | 'failed'
discovery_confidence: Optional[float] # 0.0–1.0
```

`scraped_requirements.source` continues to hold whichever URL was actually scraped — that may equal `discovered_apply_url` or be empty if discovery failed. They're not duplicates: discovery emits a URL; scraping records what it actually fetched.

---

## DB Schema (cache table)

```sql
CREATE TABLE IF NOT EXISTS job_apply_url_discovery (
    job_id              BIGINT PRIMARY KEY REFERENCES linkedin_jobs(id) ON DELETE CASCADE,
    discovered_url      TEXT NOT NULL,
    discovery_method    TEXT NOT NULL CHECK (discovery_method IN ('url_direct','ats','careers','generic')),
    discovery_confidence DOUBLE PRECISION NOT NULL,
    discovered_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_verified_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS job_apply_url_discovery_method_idx
  ON job_apply_url_discovery (discovery_method);
```

Cleanup contract: `mark_job_closed(conn, table, job_id)` in `bitirme/db_utils.py` issues a `DELETE FROM job_apply_url_discovery WHERE job_id = %s` in the same transaction that flips `is_closed = True`. Re-verification: rows with `last_verified_at` older than 14 days are eligible for re-discovery (a re-run of `discover_apply_url` updates `last_verified_at`).

---

## Discovery algorithm

**Tier 1 — ATS-targeted.** Single search query:
```
"<title>" "<company>" (site:greenhouse.io OR site:lever.co OR site:ashbyhq.com 
  OR site:workable.com OR site:smartrecruiters.com OR site:myworkdayjobs.com 
  OR site:bamboohr.com OR site:jobvite.com OR site:recruitee.com)
```
Take top 3 results → verify each → return first verified hit.

**Tier 2 — Company careers.** Skip if no `company_url` and inference fails. Otherwise:
```
"<title>" site:<company_domain>
```
Top 3 → verify → return first verified hit.

**Tier 3 — Generic.** Last resort:
```
"<title>" "<company>" apply
```
Top 3 → verify with a stricter threshold → return first verified hit.

**Verification scoring** for a candidate page (after fetching its `<title>` + first 2KB):

| Check | Type | Weight |
|---|---|---|
| Fuzzy title match (rapidfuzz `partial_ratio`) ≥ 85 | Hard (Tier 1/3); Soft (Tier 2 — domain already constrains) | required |
| Company name in candidate page text or domain | Hard for Tier 1/3; free for Tier 2 | required |
| Posting freshness (if extractable) within 180d of `linkedin_jobs.posted_time` | Soft | -0.2 if violated |
| Location overlap | Soft | -0.1 if violated |

Accept threshold: Tier 1 ≥ 0.7, Tier 2 ≥ 0.65, Tier 3 ≥ 0.8 (stricter because no `site:` filter constrains noise).

---

## Tasks

### Task 1: Add dependencies and the search-provider env-var contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/uppgrad_agentic/common/llm.py`
- Test: `tests/common/test_search_provider_factory.py`

- [ ] **Step 1: Write failing test for `get_search_provider`**

```python
# tests/common/test_search_provider_factory.py
import os
import pytest
from uppgrad_agentic.common.llm import get_search_provider


def test_returns_none_when_no_provider_configured(monkeypatch):
    monkeypatch.delenv("UPPGRAD_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    assert get_search_provider() is None


def test_returns_brave_provider_when_configured(monkeypatch):
    monkeypatch.setenv("UPPGRAD_SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    provider = get_search_provider()
    assert provider is not None
    assert provider.__class__.__name__ == "BraveSearchProvider"


def test_returns_none_when_brave_key_missing(monkeypatch):
    monkeypatch.setenv("UPPGRAD_SEARCH_PROVIDER", "brave")
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    assert get_search_provider() is None


def test_unknown_provider_returns_none(monkeypatch):
    monkeypatch.setenv("UPPGRAD_SEARCH_PROVIDER", "google")
    assert get_search_provider() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/common/test_search_provider_factory.py -v`
Expected: FAIL — `get_search_provider` not defined / `BraveSearchProvider` import missing.

- [ ] **Step 3: Add dependencies to `pyproject.toml`**

```toml
# pyproject.toml — replace [project].dependencies block with:
dependencies = [
    "langchain-core>=1.2.4",
    "langchain-openai>=1.1.6",
    "langgraph>=1.0.5",
    "pydantic>=2.12.5",
    "pypdf>=6.4.2",
    "python-docx>=1.2.0",
    "setuptools>=80.9.0",
    "crawl4ai>=0.4.0",
    "rapidfuzz>=3.10.0",
    "httpx>=0.27.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "respx>=0.21.0",
]
```

- [ ] **Step 4: Add `get_search_provider` factory to `common/llm.py`**

Append at the bottom of `src/uppgrad_agentic/common/llm.py`:

```python
def get_search_provider():
    """Return a SearchProvider instance or None if not configured.

    Mirrors get_llm() opt-in pattern. Nodes that need search MUST handle None
    by returning a degraded result, never by raising.
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

- [ ] **Step 5: Install deps**

Run: `uv sync`
Expected: lockfile updates, no errors.

- [ ] **Step 6: Run test — still failing (BraveSearchProvider not yet defined)**

Run: `uv run pytest tests/common/test_search_provider_factory.py -v`
Expected: 3 of 4 tests pass; one fails on import. We finish wiring the import in Task 2 and re-run then.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/uppgrad_agentic/common/llm.py tests/common/test_search_provider_factory.py
git commit -m "chore(deps): add crawl4ai, rapidfuzz, httpx; add search provider factory stub"
```

---

### Task 2: `SearchProvider` ABC + `BraveSearchProvider`

**Files:**
- Create: `src/uppgrad_agentic/tools/search.py`
- Test: `tests/tools/test_search.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_search.py
import httpx
import pytest
import respx

from uppgrad_agentic.tools.search import (
    BraveSearchProvider,
    SearchResult,
    SearchProvider,
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
                        {"url": "https://boards.greenhouse.io/acme/jobs/1", "title": "SWE @ Acme", "description": "Apply now"},
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
    assert results[0].title == "SWE @ Acme"
    assert results[0].snippet == "Apply now"


@respx.mock
def test_brave_provider_returns_empty_on_429():
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        return_value=httpx.Response(429, json={"error": "rate limited"})
    )
    provider = BraveSearchProvider(api_key="test")
    results = provider.search('"SWE"', count=3)
    assert results == []


@respx.mock
def test_brave_provider_returns_empty_on_network_error():
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        side_effect=httpx.ConnectError("boom")
    )
    provider = BraveSearchProvider(api_key="test")
    results = provider.search('"SWE"', count=3)
    assert results == []


@respx.mock
def test_brave_provider_truncates_to_count():
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        return_value=httpx.Response(
            200,
            json={"web": {"results": [{"url": f"https://x.com/{i}", "title": str(i), "description": ""} for i in range(10)]}},
        )
    )
    provider = BraveSearchProvider(api_key="test")
    results = provider.search('"SWE"', count=3)
    assert len(results) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tools/test_search.py -v`
Expected: FAIL — `tools.search` module not found.

- [ ] **Step 3: Implement the module**

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

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/tools/test_search.py tests/common/test_search_provider_factory.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/tools/search.py tests/tools/test_search.py
git commit -m "feat(tools): add SearchProvider ABC and BraveSearchProvider"
```

---

### Task 3: Crawl4AI scraper wrapper

**Files:**
- Create: `src/uppgrad_agentic/tools/web_scraper.py`
- Test: `tests/tools/test_web_scraper.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_web_scraper.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from uppgrad_agentic.tools.web_scraper import scrape_url, ScrapeOutcome


@pytest.mark.asyncio
async def test_scrape_returns_markdown_on_success():
    fake_result = MagicMock(success=True, markdown="# Apply now\nUpload CV", html="<h1>Apply</h1>", status_code=200, metadata={"title": "Engineer"})
    fake_crawler = AsyncMock()
    fake_crawler.__aenter__.return_value = fake_crawler
    fake_crawler.__aexit__.return_value = False
    fake_crawler.arun.return_value = fake_result

    with patch("uppgrad_agentic.tools.web_scraper.AsyncWebCrawler", return_value=fake_crawler):
        outcome = await scrape_url("https://acme.com/jobs/1")

    assert isinstance(outcome, ScrapeOutcome)
    assert outcome.success is True
    assert "Apply now" in outcome.markdown
    assert outcome.page_title == "Engineer"
    assert outcome.http_status == 200


@pytest.mark.asyncio
async def test_scrape_returns_failure_on_crawl_error():
    fake_crawler = AsyncMock()
    fake_crawler.__aenter__.return_value = fake_crawler
    fake_crawler.__aexit__.return_value = False
    fake_crawler.arun.side_effect = RuntimeError("boom")

    with patch("uppgrad_agentic.tools.web_scraper.AsyncWebCrawler", return_value=fake_crawler):
        outcome = await scrape_url("https://acme.com/jobs/1")

    assert outcome.success is False
    assert outcome.markdown == ""
    assert outcome.error == "boom"


@pytest.mark.asyncio
async def test_scrape_returns_failure_when_crawl_unsuccessful():
    fake_result = MagicMock(success=False, markdown="", html="", status_code=403, metadata={}, error_message="forbidden")
    fake_crawler = AsyncMock()
    fake_crawler.__aenter__.return_value = fake_crawler
    fake_crawler.__aexit__.return_value = False
    fake_crawler.arun.return_value = fake_result

    with patch("uppgrad_agentic.tools.web_scraper.AsyncWebCrawler", return_value=fake_crawler):
        outcome = await scrape_url("https://acme.com/jobs/1")

    assert outcome.success is False
    assert outcome.http_status == 403
    assert "forbidden" in outcome.error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tools/test_web_scraper.py -v`
Expected: FAIL — `tools.web_scraper` not found.

- [ ] **Step 3: Implement the wrapper**

```python
# src/uppgrad_agentic/tools/web_scraper.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ScrapeOutcome:
    success: bool
    markdown: str
    html: str
    page_title: str
    http_status: int
    error: str = ""


async def scrape_url(url: str, timeout_seconds: float = 20.0) -> ScrapeOutcome:
    """Scrape a URL using Crawl4AI. Never raises; returns ScrapeOutcome with success=False on failure."""
    from crawl4ai import AsyncWebCrawler  # lazy import — Crawl4AI is heavy

    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url, page_timeout=int(timeout_seconds * 1000))
    except Exception as exc:
        logger.warning("scrape_url: crawler raised — %s", exc)
        return ScrapeOutcome(success=False, markdown="", html="", page_title="", http_status=0, error=str(exc))

    if not getattr(result, "success", False):
        return ScrapeOutcome(
            success=False,
            markdown="",
            html="",
            page_title="",
            http_status=getattr(result, "status_code", 0) or 0,
            error=getattr(result, "error_message", "") or "crawl unsuccessful",
        )

    metadata = getattr(result, "metadata", {}) or {}
    return ScrapeOutcome(
        success=True,
        markdown=getattr(result, "markdown", "") or "",
        html=getattr(result, "html", "") or "",
        page_title=metadata.get("title", "") or "",
        http_status=getattr(result, "status_code", 200) or 200,
    )
```

- [ ] **Step 4: Configure pytest-asyncio**

Append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/tools/test_web_scraper.py -v`
Expected: all 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/uppgrad_agentic/tools/web_scraper.py tests/tools/test_web_scraper.py pyproject.toml
git commit -m "feat(tools): add Crawl4AI-backed web_scraper wrapper"
```

---

### Task 4: `DiscoveryResult` schema + state additions

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/schemas.py`
- Modify: `src/uppgrad_agentic/workflows/auto_apply/state.py`
- Test: `tests/workflows/auto_apply/test_schemas_discovery.py`

- [ ] **Step 1: Write failing test**

```python
# tests/workflows/auto_apply/test_schemas_discovery.py
import pytest
from pydantic import ValidationError
from uppgrad_agentic.workflows.auto_apply.schemas import DiscoveryResult


def test_discovery_result_valid():
    r = DiscoveryResult(
        url="https://boards.greenhouse.io/acme/jobs/1",
        method="ats",
        confidence=0.9,
    )
    assert r.method == "ats"
    assert r.confidence == 0.9


def test_discovery_result_failed_method_allows_empty_url():
    r = DiscoveryResult(url="", method="failed", confidence=0.0)
    assert r.url == ""


def test_discovery_result_rejects_unknown_method():
    with pytest.raises(ValidationError):
        DiscoveryResult(url="https://x.com", method="bogus", confidence=0.5)


def test_discovery_result_clamps_confidence():
    with pytest.raises(ValidationError):
        DiscoveryResult(url="https://x.com", method="ats", confidence=1.5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/workflows/auto_apply/test_schemas_discovery.py -v`
Expected: FAIL — `DiscoveryResult` not found.

- [ ] **Step 3: Add `DiscoveryResult` to schemas.py**

Append to `src/uppgrad_agentic/workflows/auto_apply/schemas.py`:

```python
DiscoveryMethod = Literal["url_direct", "ats", "careers", "generic", "failed"]


class DiscoveryResult(BaseModel):
    url: str = Field(default="", description="Discovered application URL, empty when method='failed'")
    method: DiscoveryMethod = Field(..., description="Which tier produced this result")
    confidence: float = Field(..., ge=0.0, le=1.0)
    cached: bool = Field(default=False, description="True if the result came from the cache")
```

- [ ] **Step 4: Add state fields**

Modify `src/uppgrad_agentic/workflows/auto_apply/state.py` — add three keys inside `AutoApplyState` after `opportunity_data`:

```python
    # apply-URL discovery
    discovered_apply_url: Optional[str]
    discovery_method: Optional[str]   # 'url_direct' | 'ats' | 'careers' | 'generic' | 'failed'
    discovery_confidence: Optional[float]
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_schemas_discovery.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/schemas.py src/uppgrad_agentic/workflows/auto_apply/state.py tests/workflows/auto_apply/test_schemas_discovery.py
git commit -m "feat(auto_apply): add DiscoveryResult schema and state fields"
```

---

### Task 5: Discovery cache (in-process)

**Files:**
- Create: `src/uppgrad_agentic/tools/url_discovery_cache.py`
- Test: `tests/tools/test_url_discovery_cache.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_url_discovery_cache.py
import pytest
from datetime import datetime, timedelta, timezone

from uppgrad_agentic.tools.url_discovery_cache import (
    InMemoryDiscoveryCache,
    CachedDiscovery,
)


def test_set_then_get_returns_entry():
    cache = InMemoryDiscoveryCache()
    cache.set_cached(123, "https://x.com/1", "ats", 0.9)
    entry = cache.get_cached(123)
    assert entry is not None
    assert entry.discovered_url == "https://x.com/1"
    assert entry.method == "ats"
    assert entry.confidence == 0.9


def test_get_missing_returns_none():
    cache = InMemoryDiscoveryCache()
    assert cache.get_cached(999) is None


def test_invalidate_removes_entry():
    cache = InMemoryDiscoveryCache()
    cache.set_cached(123, "https://x.com/1", "ats", 0.9)
    cache.invalidate(123)
    assert cache.get_cached(123) is None


def test_invalidate_missing_is_noop():
    cache = InMemoryDiscoveryCache()
    cache.invalidate(999)  # must not raise


def test_stale_after_14_days():
    cache = InMemoryDiscoveryCache()
    cache.set_cached(123, "https://x.com/1", "ats", 0.9)
    entry = cache.get_cached(123)
    entry.last_verified_at = datetime.now(timezone.utc) - timedelta(days=15)
    cache._store[123] = entry
    fresh = cache.get_cached(123, max_age_days=14)
    assert fresh is None  # stale entries return None when max_age enforced


def test_set_overwrites_previous_entry():
    cache = InMemoryDiscoveryCache()
    cache.set_cached(123, "https://x.com/1", "ats", 0.9)
    cache.set_cached(123, "https://y.com/2", "careers", 0.7)
    entry = cache.get_cached(123)
    assert entry.discovered_url == "https://y.com/2"
    assert entry.method == "careers"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/tools/test_url_discovery_cache.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the cache**

```python
# src/uppgrad_agentic/tools/url_discovery_cache.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class CachedDiscovery:
    job_id: int
    discovered_url: str
    method: str
    confidence: float
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class InMemoryDiscoveryCache:
    """In-process discovery cache — placeholder until Postgres-backed cache is wired.

    Replace with a DB-backed implementation that talks to job_apply_url_discovery
    once the API/backend integration phase begins. Method signatures must stay
    identical so callers don't change.
    """

    def __init__(self):
        self._store: Dict[int, CachedDiscovery] = {}

    def get_cached(self, job_id: int, max_age_days: Optional[int] = None) -> Optional[CachedDiscovery]:
        entry = self._store.get(job_id)
        if entry is None:
            return None
        if max_age_days is not None:
            age = datetime.now(timezone.utc) - entry.last_verified_at
            if age > timedelta(days=max_age_days):
                return None
        return entry

    def set_cached(self, job_id: int, url: str, method: str, confidence: float) -> None:
        self._store[job_id] = CachedDiscovery(
            job_id=job_id, discovered_url=url, method=method, confidence=confidence,
        )

    def invalidate(self, job_id: int) -> None:
        self._store.pop(job_id, None)


# Module-level singleton — tests can construct their own instance for isolation.
_DEFAULT_CACHE = InMemoryDiscoveryCache()


def get_default_cache() -> InMemoryDiscoveryCache:
    return _DEFAULT_CACHE
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/tools/test_url_discovery_cache.py -v`
Expected: all 6 pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/tools/url_discovery_cache.py tests/tools/test_url_discovery_cache.py
git commit -m "feat(tools): add in-memory discovery cache (DB-backed swap deferred)"
```

---

### Task 6: Verification scoring (pure function, no I/O)

**Files:**
- Create: `src/uppgrad_agentic/tools/url_discovery.py` (verification half — orchestration added in Task 7)
- Test: `tests/tools/test_url_discovery_verify.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_url_discovery_verify.py
import pytest
from datetime import datetime, timezone, timedelta

from uppgrad_agentic.tools.url_discovery import score_candidate, VerifyInputs


def _job(title="Senior Backend Engineer", company="Acme Corp", posted_iso=None, location="London, UK"):
    return {
        "id": 1,
        "title": title,
        "company": company,
        "posted_time": posted_iso or datetime.now(timezone.utc).isoformat(),
        "location": location,
    }


def test_perfect_match_scores_above_threshold():
    inputs = VerifyInputs(
        candidate_url="https://boards.greenhouse.io/acme/jobs/1",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text="Acme Corp is hiring a Senior Backend Engineer in London. Apply now.",
        candidate_posted_at=datetime.now(timezone.utc),
        job=_job(),
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is True
    assert score.confidence >= 0.7


def test_title_mismatch_fails_hard_check():
    inputs = VerifyInputs(
        candidate_url="https://boards.greenhouse.io/acme/jobs/1",
        candidate_title="Marketing Coordinator",
        candidate_text="Acme Corp marketing role.",
        candidate_posted_at=None,
        job=_job(),
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is False


def test_company_missing_fails_for_tier1():
    inputs = VerifyInputs(
        candidate_url="https://boards.greenhouse.io/other/jobs/1",
        candidate_title="Senior Backend Engineer",
        candidate_text="A great backend role somewhere.",
        candidate_posted_at=None,
        job=_job(),
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is False


def test_company_missing_ok_for_careers_tier():
    # Tier 2 (`site:<company_domain>`) — domain already constrains, company text not required.
    inputs = VerifyInputs(
        candidate_url="https://acmecorp.com/careers/role-1",
        candidate_title="Senior Backend Engineer",
        candidate_text="Backend engineer position. Apply via this form.",
        candidate_posted_at=None,
        job=_job(),
        tier="careers",
    )
    score = score_candidate(inputs)
    assert score.passed is True


def test_old_posting_lowers_confidence():
    inputs = VerifyInputs(
        candidate_url="https://boards.greenhouse.io/acme/jobs/1",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text="Acme Corp is hiring",
        candidate_posted_at=datetime.now(timezone.utc) - timedelta(days=400),
        job=_job(posted_iso=datetime.now(timezone.utc).isoformat()),
        tier="ats",
    )
    score = score_candidate(inputs)
    # Still passes hard checks but confidence is reduced
    assert score.confidence < 0.85


def test_tier3_uses_stricter_threshold():
    # Borderline case that would pass tier 1 but should fail tier 3
    inputs = VerifyInputs(
        candidate_url="https://random-board.com/jobs/1",
        candidate_title="Senior Backend Engineer at Acme",  # close but not exact
        candidate_text="Acme is hiring backend engineer",
        candidate_posted_at=None,
        job=_job(title="Senior Backend Engineer (Platform)", company="Acme Corp"),
        tier="generic",
    )
    score = score_candidate(inputs)
    # Generic tier requires confidence >= 0.8 — the rough match shouldn't clear that bar
    assert score.passed is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/tools/test_url_discovery_verify.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement verification**

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

# Hard-required title fuzzy threshold (rapidfuzz partial_ratio, 0–100)
_TITLE_FUZZY_MIN = 85


@dataclass
class VerifyInputs:
    candidate_url: str
    candidate_title: str
    candidate_text: str
    candidate_posted_at: Optional[datetime]
    job: dict             # subset of linkedin_jobs row: id, title, company, posted_time, location
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

    # ---------- Hard check: title fuzzy match
    haystack = f"{inputs.candidate_title}\n{inputs.candidate_text[:2000]}"
    title_score = fuzz.partial_ratio(job_title.lower(), haystack.lower()) if job_title else 0
    if title_score < _TITLE_FUZZY_MIN:
        return VerificationScore(passed=False, confidence=0.0, reasons=[f"title fuzzy {title_score} < {_TITLE_FUZZY_MIN}"])

    # ---------- Hard check: company match (waived for careers tier — domain enforces it)
    if inputs.tier != "careers":
        company_in_url = job_company and job_company.lower().replace(" ", "") in inputs.candidate_url.lower()
        company_in_text = job_company and re.search(re.escape(job_company), inputs.candidate_text, re.IGNORECASE) is not None
        if not (company_in_url or company_in_text):
            return VerificationScore(
                passed=False, confidence=0.0,
                reasons=[f"company '{job_company}' not present in URL or page"],
            )

    # ---------- Soft signals (start at 0.85, deduct for misses)
    confidence = 0.85
    reasons.append(f"title fuzzy {title_score}")

    # Freshness — only applied if we have both timestamps
    job_posted = _parse_iso_or_none(job.get("posted_time"))
    if inputs.candidate_posted_at and job_posted:
        delta_days = abs((inputs.candidate_posted_at - job_posted).days)
        if delta_days > 180:
            confidence -= 0.20
            reasons.append(f"posting freshness off by {delta_days}d")

    # Location overlap — soft only
    job_loc_tokens = {tok.strip().lower() for tok in (job.get("location") or "").split(",") if tok.strip()}
    if job_loc_tokens:
        loc_hit = any(tok in inputs.candidate_text.lower() for tok in job_loc_tokens)
        if not loc_hit:
            confidence -= 0.10
            reasons.append("location not found on page")

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

### Task 7: Discovery orchestration (three tiers)

**Files:**
- Modify: `src/uppgrad_agentic/tools/url_discovery.py`
- Test: `tests/tools/test_url_discovery_orchestration.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_url_discovery_orchestration.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from uppgrad_agentic.tools.url_discovery import discover_apply_url, _build_ats_query, _build_careers_query
from uppgrad_agentic.tools.search import SearchResult
from uppgrad_agentic.tools.web_scraper import ScrapeOutcome


def _job(title="Senior Backend Engineer", company="Acme Corp", url_direct=None, company_url=None):
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
    result = discover_apply_url(job, search_provider=None, cache=None)
    assert result.method == "url_direct"
    assert result.url == "https://acme.com/apply/1"
    assert result.confidence == 1.0


def test_returns_failed_when_no_search_provider_and_no_url_direct():
    job = _job(url_direct=None)
    result = discover_apply_url(job, search_provider=None, cache=None)
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


@pytest.mark.asyncio
async def test_ats_tier_returns_first_verified():
    job = _job()
    fake_search = MagicMock()
    fake_search.search.return_value = [
        SearchResult(url="https://boards.greenhouse.io/acme/jobs/1", title="Senior Backend Engineer at Acme Corp", snippet="Apply now"),
    ]
    fake_scrape = AsyncMock(return_value=ScrapeOutcome(
        success=True,
        markdown="Acme Corp is hiring Senior Backend Engineer in London",
        html="",
        page_title="Senior Backend Engineer at Acme Corp",
        http_status=200,
    ))
    with patch("uppgrad_agentic.tools.url_discovery.scrape_url", fake_scrape):
        result = await _async_discover(job, fake_search)
    assert result.method == "ats"
    assert result.url.startswith("https://boards.greenhouse.io/")


@pytest.mark.asyncio
async def test_falls_through_to_careers_when_ats_fails_verification():
    job = _job(company_url="https://acmecorp.com")
    fake_search = MagicMock()
    fake_search.search.side_effect = [
        # Tier 1 — verification will fail (wrong title)
        [SearchResult(url="https://boards.greenhouse.io/acme/jobs/1", title="Marketing Manager", snippet="")],
        # Tier 2 — verification will pass
        [SearchResult(url="https://acmecorp.com/careers/role-1", title="Senior Backend Engineer", snippet="")],
    ]

    async def fake_scrape(url, timeout_seconds=20.0):
        if "greenhouse" in url:
            return ScrapeOutcome(success=True, markdown="Marketing role", html="", page_title="Marketing Manager", http_status=200)
        return ScrapeOutcome(success=True, markdown="Senior Backend Engineer position. Apply.", html="", page_title="Senior Backend Engineer", http_status=200)

    with patch("uppgrad_agentic.tools.url_discovery.scrape_url", fake_scrape):
        result = await _async_discover(job, fake_search)
    assert result.method == "careers"
    assert result.url == "https://acmecorp.com/careers/role-1"


@pytest.mark.asyncio
async def test_returns_failed_when_all_tiers_miss():
    job = _job(company_url="https://acmecorp.com")
    fake_search = MagicMock()
    fake_search.search.return_value = []  # every tier returns no results
    fake_scrape = AsyncMock()
    with patch("uppgrad_agentic.tools.url_discovery.scrape_url", fake_scrape):
        result = await _async_discover(job, fake_search)
    assert result.method == "failed"
    assert result.url == ""


@pytest.mark.asyncio
async def test_cache_hit_short_circuits():
    job = _job()
    cache = MagicMock()
    cache.get_cached.return_value = MagicMock(
        discovered_url="https://cached.example.com/job/42",
        method="ats",
        confidence=0.9,
    )
    fake_search = MagicMock()
    result = discover_apply_url(job, search_provider=fake_search, cache=cache)
    assert result.url == "https://cached.example.com/job/42"
    assert result.method == "ats"
    assert result.cached is True
    fake_search.search.assert_not_called()


# Helper: orchestrator is sync from caller's POV (it runs an asyncio event loop internally),
# but for these tests we exercise the async core directly.
async def _async_discover(job, fake_search):
    from uppgrad_agentic.tools.url_discovery import _discover_async
    return await _discover_async(job, fake_search)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/tools/test_url_discovery_orchestration.py -v`
Expected: FAIL — `discover_apply_url` and `_discover_async` not defined.

- [ ] **Step 3: Add orchestration to `url_discovery.py`**

Append to `src/uppgrad_agentic/tools/url_discovery.py`:

```python
import asyncio
from urllib.parse import urlparse

from uppgrad_agentic.tools.search import SearchProvider, SearchResult
from uppgrad_agentic.tools.web_scraper import scrape_url
from uppgrad_agentic.tools.url_discovery_cache import InMemoryDiscoveryCache
from uppgrad_agentic.workflows.auto_apply.schemas import DiscoveryResult


_ATS_DOMAINS = [
    "greenhouse.io", "lever.co", "ashbyhq.com", "workable.com",
    "smartrecruiters.com", "myworkdayjobs.com", "bamboohr.com",
    "jobvite.com", "recruitee.com",
]


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


async def _verify_one(candidate: SearchResult, job: dict, tier: Tier) -> Optional[VerificationScore]:
    outcome = await scrape_url(candidate.url)
    if not outcome.success:
        return None
    candidate_title = outcome.page_title or candidate.title
    inputs = VerifyInputs(
        candidate_url=candidate.url,
        candidate_title=candidate_title,
        candidate_text=outcome.markdown,
        candidate_posted_at=None,  # extraction TBD; freshness is soft anyway
        job=job,
        tier=tier,
    )
    score = score_candidate(inputs)
    if not score.passed:
        return None
    return score


async def _try_tier(candidates: List[SearchResult], job: dict, tier: Tier) -> Optional[tuple[SearchResult, VerificationScore]]:
    for cand in candidates:
        verified = await _verify_one(cand, job, tier)
        if verified is not None:
            return cand, verified
    return None


async def _discover_async(job: dict, search_provider: SearchProvider) -> DiscoveryResult:
    title = (job.get("title") or "").strip()
    company = (job.get("company") or "").strip()
    if not title or not company:
        return DiscoveryResult(url="", method="failed", confidence=0.0)

    # Tier 1 — ATS
    ats_results = search_provider.search(_build_ats_query(title, company), count=3)
    hit = await _try_tier(ats_results, job, "ats")
    if hit:
        cand, score = hit
        return DiscoveryResult(url=cand.url, method="ats", confidence=score.confidence)

    # Tier 2 — Careers
    careers_query = _build_careers_query(title, job.get("company_url"))
    if careers_query:
        careers_results = search_provider.search(careers_query, count=3)
        hit = await _try_tier(careers_results, job, "careers")
        if hit:
            cand, score = hit
            return DiscoveryResult(url=cand.url, method="careers", confidence=score.confidence)

    # Tier 3 — Generic
    generic_results = search_provider.search(_build_generic_query(title, company), count=3)
    hit = await _try_tier(generic_results, job, "generic")
    if hit:
        cand, score = hit
        return DiscoveryResult(url=cand.url, method="generic", confidence=score.confidence)

    return DiscoveryResult(url="", method="failed", confidence=0.0)


def discover_apply_url(
    job: dict,
    search_provider: Optional[SearchProvider],
    cache: Optional[InMemoryDiscoveryCache] = None,
) -> DiscoveryResult:
    """Synchronous entry point. Caller does not need to manage an event loop."""
    # Short-circuit 1: url_direct already in DB
    url_direct = (job.get("url_direct") or "").strip()
    if url_direct:
        return DiscoveryResult(url=url_direct, method="url_direct", confidence=1.0)

    # Short-circuit 2: cache hit (only if caller supplied a cache + job has an id)
    job_id = job.get("id")
    if cache is not None and job_id is not None:
        cached = cache.get_cached(int(job_id), max_age_days=14)
        if cached is not None:
            return DiscoveryResult(
                url=cached.discovered_url, method=cached.method,
                confidence=cached.confidence, cached=True,
            )

    if search_provider is None:
        return DiscoveryResult(url="", method="failed", confidence=0.0)

    result = asyncio.run(_discover_async(job, search_provider))

    # Write through to cache on success
    if cache is not None and job_id is not None and result.method != "failed":
        cache.set_cached(int(job_id), result.url, result.method, result.confidence)
    return result
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/tools/test_url_discovery_orchestration.py tests/tools/test_url_discovery_verify.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/tools/url_discovery.py tests/tools/test_url_discovery_orchestration.py
git commit -m "feat(tools): orchestrate three-tier apply-URL discovery with verification"
```

---

### Task 8: `discover_apply_url` graph node

**Files:**
- Create: `src/uppgrad_agentic/workflows/auto_apply/nodes/discover_apply_url.py`
- Test: `tests/workflows/auto_apply/test_discover_apply_url.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/workflows/auto_apply/test_discover_apply_url.py
from unittest.mock import patch, MagicMock

from uppgrad_agentic.workflows.auto_apply.nodes.discover_apply_url import discover_apply_url_node
from uppgrad_agentic.workflows.auto_apply.schemas import DiscoveryResult


def _state(opportunity_type="job", url_direct=None):
    return {
        "opportunity_type": opportunity_type,
        "opportunity_id": "job-001",
        "opportunity_data": {
            "id": 42, "title": "SWE", "company": "Acme",
            "url": "https://www.linkedin.com/jobs/view/42",
            "url_direct": url_direct,
            "company_url": None, "posted_time": None, "location": "",
        },
    }


def test_skips_for_non_job_opportunities():
    out = discover_apply_url_node(_state(opportunity_type="masters"))
    assert out["current_step"] == "discover_apply_url"
    assert "discovered_apply_url" not in out  # node doesn't set when skipping


def test_skips_for_internal_jobs():
    state = _state()
    state["opportunity_data"]["employer_id"] = 1
    out = discover_apply_url_node(state)
    assert out.get("discovery_method") in (None, "skipped_internal")


def test_records_url_direct_path():
    state = _state(url_direct="https://acme.com/apply/1")
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.discover_apply_url.discover_apply_url",
        return_value=DiscoveryResult(url="https://acme.com/apply/1", method="url_direct", confidence=1.0),
    ):
        out = discover_apply_url_node(state)
    assert out["discovered_apply_url"] == "https://acme.com/apply/1"
    assert out["discovery_method"] == "url_direct"
    assert out["discovery_confidence"] == 1.0


def test_records_failed_path_without_setting_error():
    state = _state(url_direct=None)
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.discover_apply_url.discover_apply_url",
        return_value=DiscoveryResult(url="", method="failed", confidence=0.0),
    ):
        out = discover_apply_url_node(state)
    assert out["discovered_apply_url"] is None
    assert out["discovery_method"] == "failed"
    # Critical: must NOT set result.status=error — graceful degradation only
    assert "result" not in out


def test_short_circuits_on_upstream_error():
    state = _state()
    state["result"] = {"status": "error", "error_code": "X", "user_message": "boom"}
    out = discover_apply_url_node(state)
    assert out == {"current_step": "discover_apply_url", "step_history": ["discover_apply_url"]}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/workflows/auto_apply/test_discover_apply_url.py -v`
Expected: FAIL — node module not found.

- [ ] **Step 3: Implement the node**

```python
# src/uppgrad_agentic/workflows/auto_apply/nodes/discover_apply_url.py
from __future__ import annotations

import logging

from uppgrad_agentic.common.llm import get_search_provider
from uppgrad_agentic.tools.url_discovery import discover_apply_url
from uppgrad_agentic.tools.url_discovery_cache import get_default_cache
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)


def discover_apply_url_node(state: AutoApplyState) -> dict:
    updates = {"current_step": "discover_apply_url", "step_history": ["discover_apply_url"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    if state.get("opportunity_type") != "job":
        return updates  # discovery only meaningful for jobs

    opportunity_data = state.get("opportunity_data") or {}

    # Internal jobs (employer_id == 1) submit through the platform — discovery is N/A.
    if opportunity_data.get("employer_id") == 1:
        return {
            **updates,
            "discovered_apply_url": None,
            "discovery_method": "skipped_internal",
            "discovery_confidence": 0.0,
        }

    search_provider = get_search_provider()
    result = discover_apply_url(
        opportunity_data,
        search_provider=search_provider,
        cache=get_default_cache(),
    )

    logger.info(
        "discover_apply_url: method=%s confidence=%.2f cached=%s url=%s",
        result.method, result.confidence, result.cached, result.url,
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
Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/nodes/discover_apply_url.py tests/workflows/auto_apply/test_discover_apply_url.py
git commit -m "feat(auto_apply): add discover_apply_url graph node"
```

---

### Task 9: Replace `scrape_application_page` with Crawl4AI + discovered URL

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/nodes/scrape_application_page.py`
- Test: `tests/workflows/auto_apply/test_scrape_application_page.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/workflows/auto_apply/test_scrape_application_page.py
import pytest
from unittest.mock import patch, AsyncMock

from uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page import scrape_application_page
from uppgrad_agentic.tools.web_scraper import ScrapeOutcome


def _state(discovered=None, method="ats"):
    return {
        "opportunity_type": "job",
        "opportunity_id": "job-001",
        "opportunity_data": {"id": 42, "title": "SWE", "company": "Acme", "url": "https://linkedin.com/x", "url_direct": None},
        "discovered_apply_url": discovered,
        "discovery_method": method,
    }


def test_no_discovered_url_records_failed_status():
    out = scrape_application_page(_state(discovered=None, method="failed"))
    sr = out.get("scraped_requirements")
    assert sr is not None
    assert sr["status"] == "failed"
    assert sr["raw_content"] == ""


def test_skips_for_non_jobs():
    state = _state(discovered="https://acme.com/job/1")
    state["opportunity_type"] = "masters"
    out = scrape_application_page(state)
    assert "scraped_requirements" not in out


def test_uses_discovered_url_and_records_content():
    state = _state(discovered="https://boards.greenhouse.io/acme/jobs/1", method="ats")
    fake_outcome = ScrapeOutcome(
        success=True,
        markdown="Apply now. Upload CV and Cover Letter.",
        html="<html>...</html>",
        page_title="Software Engineer at Acme",
        http_status=200,
    )
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page.scrape_url",
        AsyncMock(return_value=fake_outcome),
    ):
        out = scrape_application_page(state)
    sr = out["scraped_requirements"]
    assert sr["status"] == "partial"
    assert sr["source"] == "https://boards.greenhouse.io/acme/jobs/1"
    assert sr["raw_content"] == "Apply now. Upload CV and Cover Letter."
    assert sr["http_status"] == 200


def test_records_failed_when_crawl_fails():
    state = _state(discovered="https://x.com/job/1", method="ats")
    fake_outcome = ScrapeOutcome(success=False, markdown="", html="", page_title="", http_status=403, error="forbidden")
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page.scrape_url",
        AsyncMock(return_value=fake_outcome),
    ):
        out = scrape_application_page(state)
    sr = out["scraped_requirements"]
    assert sr["status"] == "failed"
    assert sr["http_status"] == 403


def test_short_circuits_on_upstream_error():
    state = _state(discovered="https://x.com/1")
    state["result"] = {"status": "error", "error_code": "X", "user_message": "boom"}
    out = scrape_application_page(state)
    assert "scraped_requirements" not in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/workflows/auto_apply/test_scrape_application_page.py -v`
Expected: most tests fail because the node currently uses `requests.get` and ignores `discovered_apply_url`.

- [ ] **Step 3: Replace the node implementation**

Overwrite `src/uppgrad_agentic/workflows/auto_apply/nodes/scrape_application_page.py`:

```python
from __future__ import annotations

import asyncio
import logging

from uppgrad_agentic.tools.web_scraper import scrape_url
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)

_SCRAPE_TIMEOUT = 20.0  # seconds


def scrape_application_page(state: AutoApplyState) -> dict:
    updates = {"current_step": "scrape_application_page", "step_history": ["scrape_application_page"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    if state.get("opportunity_type") != "job":
        return updates

    target_url = (state.get("discovered_apply_url") or "").strip()

    if not target_url:
        # Discovery failed or skipped — emit a clean failed scrape; determine_requirements
        # will fall back to assumed defaults.
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

    outcome = asyncio.run(scrape_url(target_url, timeout_seconds=_SCRAPE_TIMEOUT))

    if not outcome.success:
        logger.warning("scrape_application_page: crawl failed for %s — %s", target_url, outcome.error)
        return {
            **updates,
            "scraped_requirements": {
                "status": "failed",
                "requirements": [],
                "confidence": 0.0,
                "source": target_url,
                "raw_content": "",
                "http_status": outcome.http_status,
                "error": outcome.error,
            },
        }

    logger.info("scrape_application_page: fetched %d chars from %s", len(outcome.markdown), target_url)
    return {
        **updates,
        "scraped_requirements": {
            "status": "partial",   # evaluate_scrape will set the final status
            "requirements": [],
            "confidence": 0.0,
            "source": target_url,
            "raw_content": outcome.markdown,
            "http_status": outcome.http_status,
        },
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_scrape_application_page.py -v`
Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/nodes/scrape_application_page.py tests/workflows/auto_apply/test_scrape_application_page.py
git commit -m "refactor(auto_apply): scrape via Crawl4AI against discovered URL"
```

---

### Task 10: Wire `discover_apply_url` into the graph

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/graph.py`
- Test: `tests/workflows/auto_apply/test_graph_discovery_routing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/workflows/auto_apply/test_graph_discovery_routing.py
from unittest.mock import patch, AsyncMock

from uppgrad_agentic.workflows.auto_apply.graph import build_graph
from uppgrad_agentic.workflows.auto_apply.schemas import DiscoveryResult
from uppgrad_agentic.tools.web_scraper import ScrapeOutcome


def test_graph_includes_discover_apply_url_node():
    graph = build_graph()
    nodes = list(graph.get_graph().nodes)
    assert "discover_apply_url" in nodes


def test_job_path_runs_discover_before_scrape(monkeypatch):
    """End-to-end: a job opportunity should pass through discover → scrape with the discovered URL."""
    monkeypatch.delenv("UPPGRAD_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("UPPGRAD_SEARCH_PROVIDER", raising=False)

    graph = build_graph()
    seen_step_history = []

    discovery = DiscoveryResult(url="https://boards.greenhouse.io/acme/jobs/1", method="ats", confidence=0.9)
    fake_scrape = AsyncMock(return_value=ScrapeOutcome(
        success=True, markdown="Apply now. Upload CV.", html="", page_title="SWE @ Acme", http_status=200,
    ))

    with patch("uppgrad_agentic.workflows.auto_apply.nodes.discover_apply_url.discover_apply_url",
               return_value=discovery), \
         patch("uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page.scrape_url", fake_scrape):
        # Use the existing CLI-style invocation that stops at human_gate_1 (interrupts).
        # We only need to verify ordering — abort by raising at human_gate_1.
        thread_config = {"configurable": {"thread_id": "test-1"}}
        try:
            for ev in graph.stream(
                {"opportunity_type": "job", "opportunity_id": "job-001"},
                config=thread_config,
                stream_mode="values",
            ):
                seen_step_history = ev.get("step_history") or seen_step_history
                if "human_gate_0" in (ev.get("step_history") or []) or "human_gate_1" in (ev.get("step_history") or []):
                    break
        except Exception:
            pass

    # discover_apply_url must appear and must come before scrape_application_page
    assert "discover_apply_url" in seen_step_history
    assert "scrape_application_page" in seen_step_history
    assert seen_step_history.index("discover_apply_url") < seen_step_history.index("scrape_application_page")


def test_non_job_path_skips_discovery():
    graph = build_graph()
    seen_step_history = []
    thread_config = {"configurable": {"thread_id": "test-2"}}
    try:
        for ev in graph.stream(
            {"opportunity_type": "masters", "opportunity_id": "prog-001"},
            config=thread_config,
            stream_mode="values",
        ):
            seen_step_history = ev.get("step_history") or seen_step_history
            if "human_gate_0" in (ev.get("step_history") or []) or "human_gate_1" in (ev.get("step_history") or []):
                break
    except Exception:
        pass

    # discover_apply_url node should NOT appear for non-job opportunities
    # (We route past it with a conditional edge.)
    assert "discover_apply_url" not in seen_step_history
    assert "scrape_application_page" not in seen_step_history
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/workflows/auto_apply/test_graph_discovery_routing.py -v`
Expected: FAIL — `discover_apply_url` not yet in graph.

- [ ] **Step 3: Modify graph.py**

In `src/uppgrad_agentic/workflows/auto_apply/graph.py`:

a) Add the import after the other node imports:

```python
from uppgrad_agentic.workflows.auto_apply.nodes.discover_apply_url import discover_apply_url_node
```

b) Replace the existing `_route_after_load` function with:

```python
def _route_after_load(state: AutoApplyState) -> str:
    if state.get("result", {}).get("status") == "error":
        return "end_with_error"
    if state.get("opportunity_type") == "job":
        return "discover_apply_url"
    return "determine_requirements"
```

c) Add a new router after discovery:

```python
def _route_after_discovery(state: AutoApplyState) -> str:
    if state.get("result", {}).get("status") == "error":
        return "end_with_error"
    return "scrape_application_page"
```

d) In `build_graph`, register the new node (after the `load_opportunity` line):

```python
    g.add_node("discover_apply_url", discover_apply_url_node)
```

e) Update the conditional edges from `load_opportunity` (replace the existing `add_conditional_edges` for `load_opportunity`):

```python
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

Run: `uv run pytest tests/workflows/auto_apply/test_graph_discovery_routing.py -v`
Expected: all 3 pass.

- [ ] **Step 5: Run the full auto_apply test suite**

Run: `uv run pytest tests/workflows/auto_apply/ tests/tools/ tests/common/ -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/graph.py tests/workflows/auto_apply/test_graph_discovery_routing.py
git commit -m "feat(auto_apply): wire discover_apply_url before scrape_application_page"
```

---

### Task 11: Surface discovery metadata in `record_application`

**Files:**
- Modify: `src/uppgrad_agentic/workflows/auto_apply/nodes/record_application.py`
- Test: `tests/workflows/auto_apply/test_record_application_discovery.py`

- [ ] **Step 1: Read current record_application to understand fields**

Run: `cat src/uppgrad_agentic/workflows/auto_apply/nodes/record_application.py`

(Used to confirm it currently records `scrape_status` + `scrape_confidence`; we extend with discovery_method and discovery_confidence so application records carry full provenance.)

- [ ] **Step 2: Write failing test**

```python
# tests/workflows/auto_apply/test_record_application_discovery.py
from uppgrad_agentic.workflows.auto_apply.nodes.record_application import record_application


def test_records_discovery_metadata_for_jobs():
    state = {
        "opportunity_type": "job",
        "opportunity_id": "job-001",
        "opportunity_data": {"id": 42, "title": "SWE", "company": "Acme"},
        "discovery_method": "ats",
        "discovery_confidence": 0.9,
        "discovered_apply_url": "https://boards.greenhouse.io/acme/jobs/1",
        "tailored_documents": {"CV": {"content": "..."}},
        "scraped_requirements": {"status": "full", "confidence": 0.85},
        "application_package": {"submission_type": "external"},
    }
    out = record_application(state)
    rec = out["application_record"]
    assert rec["discovery_method"] == "ats"
    assert rec["discovery_confidence"] == 0.9
    assert rec["discovered_apply_url"] == "https://boards.greenhouse.io/acme/jobs/1"


def test_omits_discovery_metadata_for_non_jobs():
    state = {
        "opportunity_type": "masters",
        "opportunity_id": "prog-001",
        "opportunity_data": {"id": 1, "title": "MSc CS"},
        "tailored_documents": {"CV": {"content": "..."}},
        "application_package": {"submission_type": "external"},
    }
    out = record_application(state)
    rec = out["application_record"]
    assert "discovery_method" not in rec
    assert "discovered_apply_url" not in rec
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/workflows/auto_apply/test_record_application_discovery.py -v`
Expected: FAIL — current record_application does not pass through discovery metadata.

- [ ] **Step 4: Patch `record_application.py`**

Inside `record_application` (existing function), locate where the `application_record` dict is built for jobs. Add three keys (only when `opportunity_type == 'job'`):

```python
    if state.get("opportunity_type") == "job":
        record["discovery_method"] = state.get("discovery_method", "")
        record["discovery_confidence"] = state.get("discovery_confidence", 0.0)
        record["discovered_apply_url"] = state.get("discovered_apply_url", "")
```

Place this immediately before the existing `scrape_status` / `scrape_confidence` writes so all jobs-only metadata sits together. (If those writes don't currently exist in record_application, add them in the same block:)

```python
        record["scrape_status"] = (state.get("scraped_requirements") or {}).get("status", "")
        record["scrape_confidence"] = (state.get("scraped_requirements") or {}).get("confidence", 0.0)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/workflows/auto_apply/test_record_application_discovery.py -v`
Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add src/uppgrad_agentic/workflows/auto_apply/nodes/record_application.py tests/workflows/auto_apply/test_record_application_discovery.py
git commit -m "feat(auto_apply): persist discovery provenance in application_record"
```

---

### Task 12: SQL migration for `job_apply_url_discovery`

**Files:**
- Create: `migrations/2026-04-26_job_apply_url_discovery.sql`
- Modify: `bitirme/db_utils.py` (closed-postings cleanup hook)
- Test: manual review (no Python code path runs against this in the agentic repo yet — DB integration is deferred per CLAUDE.md)

- [ ] **Step 1: Create the migration**

Create `migrations/2026-04-26_job_apply_url_discovery.sql`:

```sql
-- Up migration
CREATE TABLE IF NOT EXISTS job_apply_url_discovery (
    job_id              BIGINT PRIMARY KEY REFERENCES linkedin_jobs(id) ON DELETE CASCADE,
    discovered_url      TEXT NOT NULL,
    discovery_method    TEXT NOT NULL CHECK (discovery_method IN ('url_direct','ats','careers','generic')),
    discovery_confidence DOUBLE PRECISION NOT NULL,
    discovered_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_verified_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS job_apply_url_discovery_method_idx
  ON job_apply_url_discovery (discovery_method);

CREATE INDEX IF NOT EXISTS job_apply_url_discovery_last_verified_idx
  ON job_apply_url_discovery (last_verified_at);

-- Down migration (commented — apply manually if reverting)
-- DROP INDEX IF EXISTS job_apply_url_discovery_last_verified_idx;
-- DROP INDEX IF EXISTS job_apply_url_discovery_method_idx;
-- DROP TABLE IF EXISTS job_apply_url_discovery;
```

- [ ] **Step 2: Patch `mark_job_closed` in `bitirme/db_utils.py`**

Locate `mark_job_closed` (around line 285 per earlier inspection). Inside the same transaction, add a delete against the cache table BEFORE the existing UPDATE:

```python
def mark_job_closed(conn, table: str, job_id: str):
    with conn.cursor() as cur:
        # Invalidate the apply-URL discovery cache for this job
        cur.execute("DELETE FROM job_apply_url_discovery WHERE job_id = %s", (job_id,))
        # Existing close logic (preserved)
        cur.execute(
            f"""
            UPDATE {table}
            SET is_closed = TRUE
            WHERE job_id = %s
            """,
            (job_id,),
        )
```

(Adapt the closed-update SQL to match the existing version — only the new `DELETE` line is being added.)

- [ ] **Step 3: Verify the SQL parses**

Run: `psql --no-psqlrc -1 -c "EXPLAIN $(cat migrations/2026-04-26_job_apply_url_discovery.sql | grep -v '^--' | tr '\n' ' ')"` against any Postgres with a `linkedin_jobs(id)` table — or simpler, copy the SQL into pgAdmin/Neon console and run.

Expected: table created without error.

- [ ] **Step 4: Commit**

```bash
git add migrations/2026-04-26_job_apply_url_discovery.sql
cd ../bitirme && git add db_utils.py && cd ../AgenticWorkflows_LangChain-LangGraph
# Note: bitirme is a separate repo; commit there independently
git commit -m "feat(db): add job_apply_url_discovery cache table migration"
```

(Per `feedback_git_hygiene` memory: never branch/commit in the parent `cs491-2` repo. The agentic-repo and bitirme-repo commits stay separate.)

---

### Task 13: Smoke test the full discovery path with real Brave API (optional integration check)

**Files:**
- Create: `tests/integration/test_discovery_smoke.py`

- [ ] **Step 1: Add the smoke test (skipped without API key)**

```python
# tests/integration/test_discovery_smoke.py
import os
import pytest

from uppgrad_agentic.tools.url_discovery import discover_apply_url
from uppgrad_agentic.common.llm import get_search_provider


@pytest.mark.skipif(not os.getenv("BRAVE_SEARCH_API_KEY"), reason="needs BRAVE_SEARCH_API_KEY")
def test_real_discovery_for_known_external_apply_job():
    """Smoke test against a known-good public job. Update inputs to a current real job before running."""
    job = {
        "id": 99999, "title": "Software Engineer", "company": "Stripe",
        "url": "https://www.linkedin.com/jobs/view/99999",
        "url_direct": None, "company_url": "https://stripe.com",
        "posted_time": "2026-04-01T00:00:00Z", "location": "Remote",
    }
    os.environ["UPPGRAD_SEARCH_PROVIDER"] = "brave"
    provider = get_search_provider()
    assert provider is not None
    result = discover_apply_url(job, search_provider=provider, cache=None)
    # Loose assertion — just expect *some* discovery success or a clean failed result
    assert result.method in ("ats", "careers", "generic", "failed")
    if result.method != "failed":
        assert result.url.startswith("http")
        assert result.confidence >= 0.65
```

- [ ] **Step 2: Run with API key set**

Run: `BRAVE_SEARCH_API_KEY=<your-key> UPPGRAD_SEARCH_PROVIDER=brave uv run pytest tests/integration/test_discovery_smoke.py -v -s`
Expected: passes if discovery works against a real job, or returns `failed` cleanly. Either is acceptable for this smoke test (it's about not exploding).

- [ ] **Step 3: Document in CLAUDE.md**

Open `CLAUDE.md`, in the "Auto-Apply Workflow → Scraping status" section, replace the "Planned improvement" block with:

```markdown
### Scraping status

The auto-apply scraping path now follows two stages:

1. **`discover_apply_url`** — for jobs with `url_direct` populated, that URL is used
   directly. For jobs without `url_direct`, a three-tier discovery runs against an
   external search provider (Brave by default; pluggable via
   `UPPGRAD_SEARCH_PROVIDER`):
   - Tier 1 (ATS): `site:greenhouse.io OR site:lever.co OR …` targeted query
   - Tier 2 (Careers): `site:<company_domain>` query when `company_url` is known
   - Tier 3 (Generic): `"<title>" "<company>" apply` fallback
   Each candidate URL is scraped with Crawl4AI and verified for title/company
   match before being accepted. Discovery results are cached in
   `job_apply_url_discovery` and invalidated when the closed-postings scraper
   marks the job closed.

2. **`scrape_application_page`** — fetches the discovered URL via Crawl4AI,
   producing markdown for `evaluate_scrape` to parse.

When discovery fails or the provider is not configured, the workflow degrades
gracefully to assumed default requirements (CV + Cover Letter for jobs).
LinkedIn pages themselves are never scraped — `url_direct` paths only.

Future improvements (deferred):
- SearXNG fallback when Brave recall is insufficient.
- Playwright tier for Workday-class JS portals that Crawl4AI cannot render.
- External form auto-submission (handled in a separate plan).
```

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_discovery_smoke.py CLAUDE.md
git commit -m "docs: document apply-URL discovery in CLAUDE.md; add smoke test"
```

---

## Self-review

**Spec coverage:**
- ✅ url_direct short-circuit (Task 7, `discover_apply_url` first lines)
- ✅ Three-tier discovery: ATS, careers, generic (Task 7)
- ✅ Verification with title/company/freshness/location (Task 6)
- ✅ Crawl4AI for the actual scrape (Task 3, used in Task 7's `_verify_one` and Task 9's `scrape_application_page`)
- ✅ Brave Search provider behind a swappable interface (Task 2)
- ✅ Cache table + cleanup hook on closure (Task 12)
- ✅ Discovery provenance in `application_record` (Task 11)
- ✅ Internal job (`employer_id == 1`) early skip (Task 8)
- ✅ Graceful degradation when discovery or provider absent (Tasks 8, 9)
- ✅ State additions (Task 4)
- ✅ Graph wiring with conditional routing for jobs vs others (Task 10)

**Placeholders scanned:** none — every step has concrete code or an exact command.

**Type consistency:**
- `DiscoveryResult` defined in Task 4, consumed identically in Tasks 7, 8, 10, 11.
- `ScrapeOutcome` (Task 3) consumed identically in Tasks 7, 9.
- `SearchProvider.search(query, count) -> List[SearchResult]` consistent across Tasks 2, 7.
- `discover_apply_url(job, search_provider, cache)` signature consistent in Tasks 7, 8.
- `score_candidate(VerifyInputs) -> VerificationScore` consistent in Tasks 6, 7.
- Cache method names `get_cached`, `set_cached`, `invalidate` match across Tasks 5, 7, 8, 12.
- Tier names `"ats" | "careers" | "generic"` and method names `"url_direct" | "ats" | "careers" | "generic" | "failed" | "skipped_internal"` are consistent across Tasks 4, 6, 7, 8, 11.

No drift detected.
