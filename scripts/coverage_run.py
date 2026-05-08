"""Coverage harness — drive auto_apply graph up to gate 1 across many opportunities.

Mirrors `e2e_auto_apply_harness.py cmd_start` but:
  - Runs N opportunities in one process (no separate `start` invocations).
  - Supports all four opportunity types (job / masters / phd / scholarship).
  - Writes a per-row JSON for each session and a combined markdown table.
  - Uses MemorySaver — gate-1/gate-2 resume isn't needed for ATS coverage signal.

Usage:
  uv run python scripts/coverage_run.py \
    --student-id 16 \
    --cv-path /Users/koraysevil/Desktop/Senior/cs491-2/test_resume.pdf \
    --output-dir docs/coverage_run_2026_05_03_post_fix
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import psycopg
from psycopg.rows import dict_row

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from uppgrad_agentic.tools.documents import extract_text_from_file
from uppgrad_agentic.workflows.auto_apply.graph import build_graph
from uppgrad_agentic.workflows.auto_apply.nodes.asset_mapping import _USER_SUPPLIED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("coverage")
# Quiet down crawl4ai
logging.getLogger("crawl4ai").setLevel(logging.WARNING)


# ─── Targets ─────────────────────────────────────────────────────────────────

@dataclass
class Target:
    label: str
    opp_type: str  # "job" | "masters" | "phd" | "scholarship"
    opp_id: int
    expected_ats: str = ""  # informational — for the report column


TARGETS: List[Target] = [
    # Ashby (3)
    Target("ashby-monumental",     "job", 129010, "ashby"),
    Target("ashby-robinradar",     "job", 127874, "ashby"),
    Target("ashby-filigran",       "job", 127628, "ashby"),
    # Greenhouse (3)
    Target("gh-sofico",            "job", 130136, "greenhouse"),
    Target("gh-dept",              "job", 129958, "greenhouse"),
    Target("gh-adyen",             "job", 128803, "greenhouse"),
    # Lever (3) — Mendix and Wypoon were broken on prev run
    Target("lever-tsmg",           "job", 130560, "lever"),
    Target("lever-wypoon",         "job", 129801, "lever"),
    Target("lever-mendix",         "job", 127838, "lever"),
    # SmartRecruiters (3) — DYKA was broken on prev run
    Target("sr-wasco",             "job", 129862, "smartrecruiters"),
    Target("sr-dyka",              "job", 128136, "smartrecruiters"),
    Target("sr-abercrombie",       "job", 119388, "smartrecruiters"),
    # Workable (2) — Phoenix was broken on prev run
    Target("wkbl-phoenix",         "job", 107732, "workable"),
    Target("wkbl-debenhams",       "job", 101902, "workable"),
    # Workday (2) — auth-walled, expected to gracefully fall back to defaults
    Target("wday-prysmian",        "job", 129969, "workday"),
    Target("wday-safeguard",       "job", 129018, "workday"),
    # Programs (2)
    Target("masters-mba-turan",    "masters", 71103),
    Target("masters-boise-pop",    "masters", 71091),
    # Scholarships (2)
    Target("schol-goostree",       "scholarship", 8361),
    Target("schol-griffithlaw",    "scholarship", 8357),
]


# ─── Snapshot builders (mirror backend/ai_services/auto_apply_adapter.py) ─────

def fetch_job(conn: psycopg.Connection, job_id: int) -> Dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, title, company, company_url, location, description, url, url_direct,
                   site, employer_id, posted_time, is_closed, is_remote, salary,
                   job_type, job_level
            FROM linkedin_jobs WHERE id = %s
            """,
            (job_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"job id={job_id} not found")
    if isinstance(row.get("posted_time"), (datetime, date)):
        row["posted_time"] = row["posted_time"].isoformat()
    return row


def fetch_program(conn: psycopg.Connection, prog_id: int) -> Dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT id, title, university, url, location, duration, degree_type,
                      study_mode, program_type, tuition_fee, venue, data
               FROM programs WHERE id = %s""",
            (prog_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"program id={prog_id} not found")
    row["data"] = row["data"] or {}
    return row


def fetch_scholarship(conn: psycopg.Connection, sch_id: int) -> Dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT id, title, url, provider_name, disciplines, location, deadline,
                      scholarship_type, coverage, description, benefits, eligibility_text,
                      req_disciplines, req_locations, req_nationality, req_age,
                      req_study_experience, application_info, data
               FROM scholarships WHERE id = %s""",
            (sch_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"scholarship id={sch_id} not found")

    # CSV → list[str] for agentic eligibility checks (mirrors _split_csv in adapter)
    def split_csv(value: Optional[str]) -> List[str]:
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    for k in ("req_disciplines", "req_locations", "req_nationality"):
        row[k] = split_csv(row.get(k))
    row["data"] = row["data"] or {}
    return row


def fetch_profile(conn: psycopg.Connection, student_id: int, cv_path: Optional[str]) -> Dict[str, Any]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT s.id AS student_id, s.user_id, s.location, s.bio,
                   s.linkedin_url, s.github_url, s.phone_number, s.languages,
                   u.first_name, u.last_name, u.email, u.username
            FROM accounts_student s JOIN auth_user u ON s.user_id = u.id
            WHERE s.id = %s
            """,
            (student_id,),
        )
        s = cur.fetchone()
    if s is None:
        raise ValueError(f"student id={student_id} not found")

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT university, title_obtained, end_year, gpa, major
               FROM accounts_studenteducation WHERE student_id = %s ORDER BY end_year DESC NULLS LAST""",
            (student_id,),
        )
        education = [
            {"degree": r["title_obtained"], "institution": r["university"],
             "year": r["end_year"], "gpa": float(r["gpa"]) if r["gpa"] else None,
             "major": r.get("major") or ""}
            for r in cur.fetchall()
        ]

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """SELECT cs.name FROM accounts_student_skills sks
               JOIN common_skill cs ON sks.skill_id = cs.id WHERE sks.student_id = %s""",
            (student_id,),
        )
        skills = [r["name"] for r in cur.fetchall()]

    cv_text = ""
    if cv_path:
        try:
            cv_text = extract_text_from_file(cv_path).text
        except Exception as exc:
            logger.warning("CV extraction failed: %s", exc)

    name = f"{s['first_name'] or ''} {s['last_name'] or ''}".strip() or s["username"]
    return {
        "name": name,
        "email": s["email"],
        "age": None,
        "nationality": "",
        "location": s["location"] or "",
        "degree_level": education[0]["degree"] if education else "",
        "disciplines": skills,
        "gpa": education[0]["gpa"] if education and education[0].get("gpa") else None,
        "uploaded_documents": {
            "CV": bool(cv_text),
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


# ─── Per-target run ───────────────────────────────────────────────────────────

@dataclass
class RowResult:
    label: str
    opp_type: str
    opp_id: int
    expected_ats: str
    title: str
    company_or_provider: str
    discovery_method: str
    discovered_form_url: str
    posting_closed: bool
    form_field_count: int
    form_field_breakdown: Dict[str, int]  # by field_type
    requirement_breakdown: Dict[str, int]  # by category
    requirement_labels: List[str]
    eligibility_status: str
    compatibility_warnings: List[str]
    # Full-flow fields (populated when graph reaches gate 2 / END)
    tailored_documents: Dict[str, int]    # doc_type -> char count
    tailored_answers_count: int
    evaluation_warnings: List[str]
    final_status: str                     # result.status
    submission_type: str                  # 'handoff' | 'internal' | ''
    duration_seconds: float
    error: str = ""


def _build_gate1_payload(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Auto-fill defaults: documents auto_generate (skip USER_SUPPLIED that
    can't be generated), texts auto_generate, misc auto_fill. Mirrors
    e2e harness's `build_gate1_payload` with no upload overrides — the
    coverage run doesn't need user-supplied uploads to test the flow."""
    requirements: Dict[str, Dict[str, Any]] = {}
    for item in items:
        rid = str(item["id"])
        cat = item.get("category")
        if cat == "document":
            doc_type = (item.get("document_type") or "").strip()
            if doc_type in _USER_SUPPLIED:
                requirements[rid] = {
                    "choice": "ignore_for_now" if not item.get("required") else "skip",
                    "uploaded_text": None, "user_prompt": None,
                }
            else:
                requirements[rid] = {"choice": "auto_generate", "uploaded_text": None, "user_prompt": None}
        elif cat in ("text", "misc"):
            requirements[rid] = {"choice": "auto_generate", "uploaded_text": None, "user_prompt": None}
        else:
            requirements[rid] = {"choice": "ignore_for_now", "uploaded_text": None, "user_prompt": None}
    return {"requirements": requirements, "misc_strategy": "auto_fill"}


def _find_interrupt(state: Dict[str, Any]) -> Optional[Any]:
    """Locate the most recent interrupt payload in graph state, if any."""
    for tasks in (state.get("__interrupt__") or []):
        return tasks
    interrupts = state.get("__interrupt__")
    if isinstance(interrupts, list) and interrupts:
        first = interrupts[0]
        if hasattr(first, "value"):
            return first.value
        return first
    return None


def _empty_row(target: Target, title: str, who: str, dur: float, err: str) -> RowResult:
    return RowResult(
        label=target.label, opp_type=target.opp_type, opp_id=target.opp_id,
        expected_ats=target.expected_ats, title=title, company_or_provider=who,
        discovery_method="", discovered_form_url="", posting_closed=False,
        form_field_count=0, form_field_breakdown={}, requirement_breakdown={},
        requirement_labels=[], eligibility_status="error", compatibility_warnings=[],
        tailored_documents={}, tailored_answers_count=0, evaluation_warnings=[],
        final_status="", submission_type="", duration_seconds=dur, error=err,
    )


def run_one(conn: psycopg.Connection, target: Target, profile: Dict[str, Any]) -> RowResult:
    t0 = time.time()
    try:
        if target.opp_type == "job":
            opp = fetch_job(conn, target.opp_id)
            title, who = opp.get("title") or "", opp.get("company") or ""
        elif target.opp_type in ("masters", "phd"):
            opp = fetch_program(conn, target.opp_id)
            title, who = opp.get("title") or "", opp.get("university") or ""
        elif target.opp_type == "scholarship":
            opp = fetch_scholarship(conn, target.opp_id)
            title, who = opp.get("title") or "", opp.get("provider_name") or ""
        else:
            raise ValueError(f"unsupported type {target.opp_type}")
    except Exception as exc:
        return _empty_row(target, "", "", time.time() - t0, f"fetch: {exc}")

    initial_state = {
        "opportunity_type": target.opp_type,
        "opportunity_id": str(target.opp_id),
        "opportunity_data": opp,
        "profile_snapshot": profile,
    }

    cp = MemorySaver()
    graph = build_graph(checkpointer=cp)
    config = {"configurable": {"thread_id": target.label}}

    # ── Phase 1: invoke up to gate 1 ─────────────────────────────────────────
    try:
        result = graph.invoke(initial_state, config=config)
    except Exception as exc:
        return _empty_row(target, title, who, time.time() - t0, f"graph(start): {exc}")

    # Capture pre-gate-1 state immediately so we have form_fields even if
    # tailoring blows up downstream.
    form_fields = result.get("form_fields") or []
    breakdown_field: Dict[str, int] = {}
    for f in form_fields:
        ft = f.get("field_type") or "unknown"
        breakdown_field[ft] = breakdown_field.get(ft, 0) + 1
    items = result.get("requirement_items") or []
    breakdown_cat: Dict[str, int] = {}
    for it in items:
        c = it.get("category") or "unknown"
        breakdown_cat[c] = breakdown_cat.get(c, 0) + 1
    el = result.get("eligibility_result") or {}
    discovery_method = result.get("discovery_method") or ""
    discovered_form_url = result.get("discovered_form_url") or ""
    posting_closed = bool(result.get("posting_closed") or False)
    compatibility = result.get("compatibility_warnings") or []
    eligibility_status = el.get("status") or ""

    # If the graph terminated before gate 1 (ineligible / past deadline / error path),
    # return now — no tailoring to do.
    if not items or _find_interrupt(result) is None:
        final_result = result.get("result") or {}
        return RowResult(
            label=target.label, opp_type=target.opp_type, opp_id=target.opp_id,
            expected_ats=target.expected_ats, title=title, company_or_provider=who,
            discovery_method=discovery_method, discovered_form_url=discovered_form_url,
            posting_closed=posting_closed, form_field_count=len(form_fields),
            form_field_breakdown=breakdown_field, requirement_breakdown=breakdown_cat,
            requirement_labels=[(it.get("label") or "")[:60] for it in items],
            eligibility_status=eligibility_status, compatibility_warnings=compatibility,
            tailored_documents={}, tailored_answers_count=0, evaluation_warnings=[],
            final_status=str(final_result.get("status") or ""),
            submission_type="", duration_seconds=round(time.time() - t0, 2),
        )

    # ── Phase 2: resume gate 1 with auto-defaults → run through tailoring ───
    gate1_payload = _build_gate1_payload(items)
    try:
        result = graph.invoke(Command(resume=gate1_payload), config=config)
    except Exception as exc:
        return RowResult(
            label=target.label, opp_type=target.opp_type, opp_id=target.opp_id,
            expected_ats=target.expected_ats, title=title, company_or_provider=who,
            discovery_method=discovery_method, discovered_form_url=discovered_form_url,
            posting_closed=posting_closed, form_field_count=len(form_fields),
            form_field_breakdown=breakdown_field, requirement_breakdown=breakdown_cat,
            requirement_labels=[(it.get("label") or "")[:60] for it in items],
            eligibility_status=eligibility_status, compatibility_warnings=compatibility,
            tailored_documents={}, tailored_answers_count=0, evaluation_warnings=[],
            final_status="", submission_type="", duration_seconds=round(time.time() - t0, 2),
            error=f"graph(gate1): {exc}",
        )

    tailored_docs = {
        dt: len(info.get("content") or "")
        for dt, info in (result.get("tailored_documents") or {}).items()
    }
    tailored_answers_count = len(result.get("tailored_answers") or {})
    eval_warnings = (result.get("evaluation_result") or {}).get("warnings") or []

    # ── Phase 3: gate 2 — approve, no auto-submit ────────────────────────────
    if _find_interrupt(result) is not None:
        gate2_payload = {"approved": True, "attempt_auto_submit": False, "feedback": ""}
        try:
            result = graph.invoke(Command(resume=gate2_payload), config=config)
        except Exception as exc:
            return RowResult(
                label=target.label, opp_type=target.opp_type, opp_id=target.opp_id,
                expected_ats=target.expected_ats, title=title, company_or_provider=who,
                discovery_method=discovery_method, discovered_form_url=discovered_form_url,
                posting_closed=posting_closed, form_field_count=len(form_fields),
                form_field_breakdown=breakdown_field, requirement_breakdown=breakdown_cat,
                requirement_labels=[(it.get("label") or "")[:60] for it in items],
                eligibility_status=eligibility_status, compatibility_warnings=compatibility,
                tailored_documents=tailored_docs, tailored_answers_count=tailored_answers_count,
                evaluation_warnings=eval_warnings, final_status="", submission_type="",
                duration_seconds=round(time.time() - t0, 2),
                error=f"graph(gate2): {exc}",
            )

    final_result = result.get("result") or {}
    pkg = result.get("application_package") or {}
    submission_type = pkg.get("submission_type") or ""

    return RowResult(
        label=target.label, opp_type=target.opp_type, opp_id=target.opp_id,
        expected_ats=target.expected_ats, title=title, company_or_provider=who,
        discovery_method=discovery_method, discovered_form_url=discovered_form_url,
        posting_closed=posting_closed, form_field_count=len(form_fields),
        form_field_breakdown=breakdown_field, requirement_breakdown=breakdown_cat,
        requirement_labels=[(it.get("label") or "")[:60] for it in items],
        eligibility_status=eligibility_status, compatibility_warnings=compatibility,
        tailored_documents=tailored_docs, tailored_answers_count=tailored_answers_count,
        evaluation_warnings=eval_warnings,
        final_status=str(final_result.get("status") or ""),
        submission_type=submission_type,
        duration_seconds=round(time.time() - t0, 2),
    )


# ─── Report rendering ─────────────────────────────────────────────────────────

def render_md(rows: List[RowResult]) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    out: List[str] = []
    out.append(f"# Coverage run — {today} (post URL hygiene + raw_html cap fix)\n")
    out.append(
        "End-to-end coverage on 20 opportunities (16 jobs + 2 masters + 2 scholarships) "
        "after PR #21 (`fix(ats_form_urls)` URL parsing + 2MB raw_html cap). Same student "
        "across all sessions: koray.sevil.b@gmail.com (student_id=16) with `test_resume.pdf`. "
        "Driven via `scripts/coverage_run.py`.\n"
    )

    # Per-session table
    out.append("## Per-session results\n")
    out.append(
        "| # | Type | ATS | Posting | Form fields | Reqs (doc/text/misc) | Tailored docs | "
        "Answers | Eval warnings | Final | Outcome |"
    )
    out.append(
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for i, r in enumerate(rows, 1):
        ff_count = r.form_field_count
        ff_count_str = f"**{ff_count}**" if ff_count == 0 else str(ff_count)
        rb = r.requirement_breakdown
        reqs = f"{rb.get('document', 0)} / {rb.get('text', 0)} / {rb.get('misc', 0)}"
        outcome = _outcome(r)
        title_short = (r.title[:48] + "…") if len(r.title) > 48 else r.title
        ats_or_type = r.expected_ats or r.opp_type
        tdocs = ", ".join(f"{k}({v}c)" for k, v in r.tailored_documents.items()) or "—"
        out.append(
            f"| {i} | {r.opp_type} | {ats_or_type} | {r.company_or_provider} — {title_short} | "
            f"{ff_count_str} | {reqs} | {tdocs} | "
            f"{r.tailored_answers_count} | {len(r.evaluation_warnings)} | "
            f"{r.final_status or '—'} | {outcome} |"
        )

    # Tally
    n = len(rows)
    job_rows = [r for r in rows if r.opp_type == "job"]
    extracted_real = [r for r in job_rows if r.form_field_count >= 3]
    fell_back = [r for r in job_rows if r.form_field_count == 0 and "myworkdayjobs" not in r.discovered_form_url]
    workday_graceful = [r for r in job_rows if r.form_field_count == 0 and ("workday" in r.expected_ats)]

    out.append("\n## Tally\n")
    out.append(f"- Total opportunities: **{n}**")
    out.append(f"- Jobs: **{len(job_rows)}**")
    out.append(f"  - Form fields extracted (≥3 visible inputs): **{len(extracted_real)}**")
    out.append(f"  - Workday graceful no-fields (auth wall expected): **{len(workday_graceful)}**")
    out.append(f"  - Other zero-field fallbacks: **{len(fell_back) - len(workday_graceful)}**")
    out.append(f"- Programs: **{len([r for r in rows if r.opp_type in ('masters','phd')])}**")
    out.append(f"- Scholarships: **{len([r for r in rows if r.opp_type == 'scholarship'])}**\n")

    # Errors
    errs = [r for r in rows if r.error]
    if errs:
        out.append("## Errors\n")
        for r in errs:
            out.append(f"- `{r.label}`: {r.error}")
        out.append("")

    return "\n".join(out)


def _outcome(r: RowResult) -> str:
    if r.error:
        return "❌ error"
    if r.opp_type == "job":
        if r.form_field_count == 0:
            if r.expected_ats == "workday":
                return "✓ correctly handled (auth wall)"
            return "❌ extraction failed → defaults"
        if r.form_field_count >= 3:
            return "✅ extracted"
        return f"⚠️ partial ({r.form_field_count} fields)"
    # programs / scholarships
    if r.requirement_breakdown:
        return "✅ requirements parsed"
    return "⚠️ no requirements"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Coverage run across 20 opportunities")
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL", ""), help="Postgres DSN")
    ap.add_argument("--student-id", type=int, required=True)
    ap.add_argument("--cv-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--only-label", default="", help="Filter to a single target label (debug)")
    args = ap.parse_args()

    if not args.dsn:
        sys.exit("--dsn or DATABASE_URL required")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pg = psycopg.connect(args.dsn)
    try:
        profile = fetch_profile(pg, args.student_id, args.cv_path)
        targets = [t for t in TARGETS if not args.only_label or t.label == args.only_label]
        print(f"Profile: name={profile['name']!r} skills={len(profile['disciplines'])} cv_chars={len(profile.get('document_texts', {}).get('CV', ''))}")
        print(f"Running {len(targets)} target(s)\n")

        rows: List[RowResult] = []
        for i, t in enumerate(targets, 1):
            print(f"[{i}/{len(targets)}] {t.label} (type={t.opp_type} id={t.opp_id})...")
            r = run_one(pg, t, profile)
            rows.append(r)
            (out_dir / f"{t.label}.json").write_text(
                json.dumps(asdict(r), indent=2, default=str), encoding="utf-8"
            )
            print(f"   → form_fields={r.form_field_count} reqs={r.requirement_breakdown} "
                  f"discovery={r.discovery_method or '—'} t={r.duration_seconds}s "
                  f"{('ERR=' + r.error) if r.error else ''}")
    finally:
        pg.close()

    md = render_md(rows)
    md_path = out_dir / "coverage_report.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"\nReport: {md_path}")


if __name__ == "__main__":
    main()
