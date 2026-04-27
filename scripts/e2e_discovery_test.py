"""Live end-to-end discovery test against prod linkedin_jobs rows.

Loads BRAVE_SEARCH_API_KEY + UPPGRAD_SEARCH_PROVIDER from backend/.env.
Honors UPPGRAD_BROWSER_SCRAPE_ENABLED for the browser-fallback path.

Calls discover_apply_url ONCE per job (single-fetch architecture preserved).
Per-tier visibility comes from logging hooks installed on SearchProvider.search
and fetch_url_with_fallback — they record what the orchestrator does, they do
not run a parallel walkthrough.

Usage:
    uv run python scripts/e2e_discovery_test.py <job_id> [<job_id> ...]
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

_BACKEND_ENV = Path(__file__).resolve().parents[2] / "backend" / ".env"
load_dotenv(_BACKEND_ENV)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("uppgrad_agentic.tools.url_discovery").setLevel(logging.INFO)
logging.getLogger("uppgrad_agentic.tools.web_fetcher").setLevel(logging.INFO)
logging.getLogger("uppgrad_agentic.tools.search").setLevel(logging.INFO)

from uppgrad_agentic.common.llm import get_search_provider  # noqa: E402
from uppgrad_agentic.tools import url_discovery as ud  # noqa: E402


def fetch_job(conn, job_id: int) -> dict | None:
    cur = conn.execute(
        """
        SELECT id, title, company, location, posted_time, description,
               url, url_direct, company_url, employer_id,
               COALESCE(is_closed, false) AS is_closed
        FROM linkedin_jobs WHERE id = %s
        """,
        (job_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    cols = [c.name for c in cur.description]
    job = dict(zip(cols, row))
    pt = job.get("posted_time")
    if pt and not isinstance(pt, str):
        job["posted_time"] = pt.isoformat()
    return job


def cross_check(job: dict, result_text: str) -> dict:
    text_lower = (result_text or "").lower()
    title_l = (job.get("title") or "").lower()
    company_l = (job.get("company") or "").lower()
    out = {
        "title_substring_match": title_l in text_lower,
        "company_substring_match": company_l in text_lower,
        "location_tokens_in_text": [],
    }
    loc_tokens = [t.strip().lower() for t in (job.get("location") or "").split(",") if len(t.strip()) >= 3]
    out["location_tokens_in_text"] = [t for t in loc_tokens if t in text_lower]
    out["passes_human_check"] = (
        out["title_substring_match"] and (out["company_substring_match"] or "url" in title_l)
    )
    return out


class _RecordingProvider:
    """Pass-through wrapper around a real SearchProvider that prints each query+result."""
    def __init__(self, inner):
        self._inner = inner

    def search(self, query, count=3):
        print(f"\n  brave query: {query}")
        results = self._inner.search(query, count=count)
        print(f"  brave returned {len(results)} candidates:")
        for i, r in enumerate(results):
            print(f"    [{i}] {r.url}")
            print(f"        title={r.title!r}")
            print(f"        snippet={(r.snippet or '')[:120]!r}")
        return results


def _wrap_fetch(orig):
    def wrapped(url):
        fr = orig(url)
        print(f"    fetch {url}")
        print(f"      → status={fr.http_status} thin={fr.thin} used_browser={fr.used_browser} "
              f"signals={fr.thin_signals} text_len={len(fr.text)}")
        return fr
    return wrapped


def _wrap_score(orig):
    def wrapped(inputs):
        score = orig(inputs)
        print(f"    score [{inputs.tier}] {inputs.candidate_url}: "
              f"passed={score.passed} confidence={score.confidence:.2f} reasons={score.reasons}")
        return score
    return wrapped


def run(job_id: int, conn) -> None:
    job = fetch_job(conn, job_id)
    if not job:
        print(f"[{job_id}] NOT FOUND in linkedin_jobs")
        return

    print("=" * 80)
    print(f"job_id={job['id']}  title={job['title']!r}")
    print(f"  company={job['company']!r}  location={job['location']!r}")
    print(f"  url_direct={job['url_direct']!r}")
    print(f"  company_url={job['company_url']!r}")
    print(f"  employer_id={job['employer_id']}  is_closed={job['is_closed']}")
    print(f"  posted_time={job['posted_time']}")
    print(f"  browser_fallback_enabled={os.getenv('UPPGRAD_BROWSER_SCRAPE_ENABLED', '<unset>')}")

    provider = get_search_provider()
    print(f"  search_provider={type(provider).__name__ if provider else None}")
    if provider is None:
        print("  ERROR: no search provider configured. Set UPPGRAD_SEARCH_PROVIDER + BRAVE_SEARCH_API_KEY.")
        return

    # Install pass-through hooks so we can see per-tier behavior WITHOUT
    # duplicating any of the orchestrator's work.
    orig_fetch = ud.fetch_url_with_fallback
    orig_score = ud.score_candidate
    ud.fetch_url_with_fallback = _wrap_fetch(orig_fetch)
    ud.score_candidate = _wrap_score(orig_score)
    try:
        result = ud.discover_apply_url(job, _RecordingProvider(provider))
    finally:
        ud.fetch_url_with_fallback = orig_fetch
        ud.score_candidate = orig_score

    print()
    print(f"DISCOVERY RESULT: method={result.method}  confidence={result.confidence:.2f}")
    print(f"  url={result.url!r}")
    print(f"  http_status={result.http_status}  text_len={len(result.text)}")
    if result.method != "failed" and result.text:
        check = cross_check(job, result.text)
        print(f"  cross_check: {json.dumps(check, indent=2)}")


def main():
    if len(sys.argv) < 2:
        print("usage: e2e_discovery_test.py <job_id> [<job_id> ...]")
        sys.exit(2)
    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url) as conn:
        for arg in sys.argv[1:]:
            try:
                run(int(arg), conn)
            except Exception as exc:
                print(f"[{arg}] ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
