"""Live integration sanity-check for the productionized form-filler stack.

Drives the existing pipeline (discovery → scrape → extract_form_fields) to
get FormField list for a real prod job, then runs the productionized
compute_form_values + fill_form_async (NOT the PoC scripts) end-to-end.

Mirrors PoC v4 but uses the new modules, so a green run here means PR A's
production stack works against real ATSes.

Usage:
    UPPGRAD_BROWSER_SCRAPE_ENABLED=true \
      uv run python scripts/live_verify_form_filler.py [<job_id>] [--headed] [--no-llm]
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

_BACKEND_ENV = Path(__file__).resolve().parents[2] / "backend" / ".env"
load_dotenv(_BACKEND_ENV, override=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from uppgrad_agentic.tools.playwright_filler import fill_form_async  # noqa: E402
from uppgrad_agentic.tools.value_planner import compute_form_values  # noqa: E402
from uppgrad_agentic.workflows.auto_apply.nodes.discover_apply_url import (  # noqa: E402
    discover_apply_url_node,
)
from uppgrad_agentic.workflows.auto_apply.nodes.extract_form_fields import (  # noqa: E402
    extract_form_fields,
)
from uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page import (  # noqa: E402
    scrape_application_page,
)


_MOCK_PROFILE = {
    "first_name": "Koray", "last_name": "Sevil", "full_name": "Koray Sevil",
    "email": "koraysevil@gmail.com", "phone": "+90 555 123 4567",
    "country": "Turkey", "city": "Istanbul", "location": "Istanbul, Turkey",
    "linkedin": "https://www.linkedin.com/in/koraysevil",
    "github": "https://github.com/koraysevil",
    "website": "https://koraysevil.com",
}
_MOCK_RESUME_PATH = "/Users/koraysevil/Desktop/Senior/cs491-2/test_resume.pdf"


def fetch_job(conn, job_id: int) -> dict | None:
    cur = conn.execute(
        """
        SELECT id, title, company, location, posted_time, description,
               url, url_direct, company_url, employer_id,
               COALESCE(is_closed, false) AS is_closed
        FROM linkedin_jobs WHERE id = %s
        """, (job_id,))
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
    out = dict(state)
    for k, v in updates.items():
        if k == "step_history":
            out[k] = (out.get(k) or []) + (v or [])
        else:
            out[k] = v
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    headed = "--headed" in sys.argv
    use_llm = "--no-llm" not in sys.argv
    job_id = int(args[0]) if args else 228527

    db_url = os.environ["DATABASE_URL"]
    with psycopg.connect(db_url) as conn:
        job = fetch_job(conn, job_id)
    if not job:
        print(f"job {job_id} not found"); sys.exit(1)
    print(f"=== job {job_id}: {job['title']!r} @ {job['company']!r} ({job['location']}) ===")

    # 1. Existing pipeline: discovery → scrape → extract_form_fields
    state = {"opportunity_type": "job", "opportunity_id": str(job_id),
             "opportunity_data": job, "result": {}}
    state = merge(state, discover_apply_url_node(state))
    print(f"  discovery: method={state.get('discovery_method')} "
          f"form_url={state.get('discovered_form_url')!r}")
    if not state.get("discovered_form_url"):
        print("  no form URL"); sys.exit(2)
    state = merge(state, scrape_application_page(state))
    state = merge(state, extract_form_fields(state))
    fields = state.get("form_fields") or []
    print(f"  extracted {len(fields)} field(s)")
    if not fields:
        print("  no fields"); sys.exit(2)

    # 2. Productionized planner — note: tailored_documents includes a CV path
    #    so file uploads work; mock answer for free-text.
    docs = {"CV": {"file_path": _MOCK_RESUME_PATH, "content": "<elided>"}}
    plans = compute_form_values(fields, _MOCK_PROFILE, docs, job)
    print(f"  planned {sum(1 for p in plans if p.status == 'filled')} "
          f"of {len(plans)} field(s) (rest skipped at plan time)")

    # 3. Productionized filler
    llm = None
    if use_llm:
        from uppgrad_agentic.common.llm import get_llm
        llm = get_llm()
        print(f"  llm tier-4 enabled: {bool(llm)}")
    print(f"=== filling form (headed={headed}) ===")
    result = asyncio.run(
        fill_form_async(
            state["discovered_form_url"], plans,
            llm=llm, headless=not headed,
        )
    )

    print("\n=== fill results ===")
    for r in result.reports:
        marker = {"ok": "✅", "ok_llm": "🤖", "plan_skip": "⏭ "}.get(r.outcome, "❌")
        print(f"  {marker} [{r.field_type:>9}] {r.label!r:<55} {r.outcome:<20} {r.detail[:90]}")

    print(f"\n=== summary ===")
    print(f"  success: {result.success}  submit_clicked: {result.submit_clicked}  "
          f"captcha: {result.captcha_detected}")
    print(f"  total: {result.fields_total}  "
          f"filled_native: {result.fields_filled_native}  "
          f"filled_llm: {result.fields_filled_llm}  "
          f"skipped: {result.fields_skipped}  "
          f"failed: {result.fields_failed}")
    print(f"  llm_picker_calls: {result.llm_picker_calls}")
    if result.error:
        print(f"  ERROR: {result.error}")


if __name__ == "__main__":
    main()
