"""Live end-to-end test of the Opportunity Intelligence pipeline.

Drives the real chain (discover_apply_url → scrape_application_page →
evaluate_scrape → determine_requirements) against prod linkedin_jobs rows
with real Brave Search + real OpenAI LLM. Prints the extracted
requirements and a raw-content snippet so a human reviewer can cross-check
whether the LLM extraction matches what the page actually says.

Usage:
    UPPGRAD_BROWSER_SCRAPE_ENABLED=true \
      uv run python scripts/e2e_scrape_test.py <job_id> [<job_id> ...]
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
# override=True so an empty OPENAI_API_KEY/BRAVE key in the calling shell
# doesn't shadow real values from backend/.env (otherwise get_llm returns None).
load_dotenv(_BACKEND_ENV, override=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("uppgrad_agentic.tools.url_discovery").setLevel(logging.INFO)
logging.getLogger("uppgrad_agentic.tools.web_fetcher").setLevel(logging.INFO)
logging.getLogger("uppgrad_agentic.workflows.auto_apply.nodes").setLevel(logging.INFO)
logging.getLogger("uppgrad_agentic.common.llm").setLevel(logging.WARNING)

from uppgrad_agentic.workflows.auto_apply.nodes.discover_apply_url import (  # noqa: E402
    discover_apply_url_node,
)
from uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page import (  # noqa: E402
    scrape_application_page,
)
from uppgrad_agentic.workflows.auto_apply.nodes.evaluate_scrape import evaluate_scrape  # noqa: E402
from uppgrad_agentic.workflows.auto_apply.nodes.extract_form_fields import (  # noqa: E402
    extract_form_fields,
)
from uppgrad_agentic.workflows.auto_apply.nodes.determine_requirements import (  # noqa: E402
    determine_requirements,
)


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


def merge(state: dict, updates: dict) -> dict:
    """Mimic LangGraph's reducer: merge node return into state, with list-add semantics for step_history."""
    out = dict(state)
    for k, v in updates.items():
        if k == "step_history":
            out[k] = (out.get(k) or []) + (v or [])
        else:
            out[k] = v
    return out


def run(job_id: int, conn) -> None:
    job = fetch_job(conn, job_id)
    if not job:
        print(f"[{job_id}] NOT FOUND")
        return

    print("=" * 100)
    print(f"job_id={job['id']}  title={job['title']!r}")
    print(f"  company={job['company']!r}  location={job['location']!r}")
    print(f"  url_direct={job['url_direct']!r}")
    print(f"  browser_fallback={os.getenv('UPPGRAD_BROWSER_SCRAPE_ENABLED', '<unset>')}  "
          f"llm={'on' if os.getenv('OPENAI_API_KEY') else 'off'}")

    state = {
        "opportunity_type": "job",
        "opportunity_id": str(job["id"]),
        "opportunity_data": job,
        "result": {},
    }

    # 1. Discover URL
    print("\n── discover_apply_url ──")
    state = merge(state, discover_apply_url_node(state))
    print(f"  discovery_method={state.get('discovery_method')}  "
          f"confidence={state.get('discovery_confidence'):.2f}  "
          f"posting_closed={state.get('posting_closed', False)}")
    print(f"  url={state.get('discovered_apply_url')!r}")
    print(f"  page_content_len={len(state.get('discovered_page_content') or '')}")

    if state.get("discovery_method") in ("failed", None) or not state.get("discovered_apply_url"):
        print("  → discovery did not produce a URL; skipping scrape pipeline")
        return

    # 2. Scrape application page
    print("\n── scrape_application_page ──")
    state = merge(state, scrape_application_page(state))
    sr = state.get("scraped_requirements") or {}
    raw = sr.get("raw_content") or ""
    print(f"  scrape_status={sr.get('status')}  http_status={sr.get('http_status')}  "
          f"raw_content_len={len(raw)}")

    # 3. Evaluate scrape (LLM extraction)
    print("\n── evaluate_scrape (LLM) ──")
    state = merge(state, evaluate_scrape(state))
    sr = state.get("scraped_requirements") or {}
    print(f"  llm_status={sr.get('status')}  confidence={sr.get('confidence', 0.0):.2f}")
    print(f"  extracted requirements ({len(sr.get('requirements', []))}):")
    for r in sr.get("requirements", []):
        is_assumed = "(assumed)" if r.get("is_assumed") else "(scraped)"
        print(f"    - {r.get('requirement_type')}/{r.get('document_type', '')!r:<25} "
              f"conf={r.get('confidence', 0):.2f} {is_assumed}")

    # 4. Extract form fields (NEW — Phase 2)
    print("\n── extract_form_fields (LLM, form HTML only) ──")
    print(f"  overview_url={state.get('discovered_apply_url')!r}")
    print(f"  form_url    ={state.get('discovered_form_url')!r}")
    print(f"  state.discovered_raw_html len = {len(state.get('discovered_raw_html') or '')}")
    print(f"  state.scraped_requirements.raw_html len = {len((state.get('scraped_requirements') or {}).get('raw_html') or '')}")
    state = merge(state, extract_form_fields(state))
    fields = state.get("form_fields") or []
    print(f"  extracted {len(fields)} form field(s):")
    for f in fields:
        opts = f"  options={f.get('options')[:5]!r}" if f.get("options") else ""
        req = " *" if f.get("required") else ""
        print(f"    - [{f.get('field_type'):>9}] {f.get('label')!r:<45} "
              f"name={f.get('name')!r:<25} src={f.get('expected_source')}{req}{opts}")

    # 5. Final determine_requirements
    print("\n── determine_requirements ──")
    state = merge(state, determine_requirements(state))
    nr = state.get("normalized_requirements") or []
    print(f"  final normalized requirements ({len(nr)}):")
    for r in nr:
        is_assumed = "(assumed)" if r.get("is_assumed") else "(scraped)"
        print(f"    - {r.get('requirement_type')}/{r.get('document_type', '')!r:<25} "
              f"conf={r.get('confidence', 0):.2f} {is_assumed}")

    # 5. Cross-check: print a snippet of the actual scraped page so the human
    # reviewer can compare extracted requirements against page reality.
    print("\n── raw content snippet (for human cross-check) ──")
    snippet = raw[:1500] if raw else ""
    print(snippet)
    if len(raw) > 1500:
        print(f"\n  ... [truncated; total {len(raw)} chars]")


def main():
    if len(sys.argv) < 2:
        print("usage: e2e_scrape_test.py <job_id> [<job_id> ...]")
        sys.exit(2)
    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url) as conn:
        for arg in sys.argv[1:]:
            try:
                run(int(arg), conn)
            except Exception as exc:
                import traceback
                print(f"[{arg}] ERROR: {type(exc).__name__}: {exc}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
