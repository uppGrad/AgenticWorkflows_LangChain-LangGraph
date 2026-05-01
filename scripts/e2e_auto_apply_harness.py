"""End-to-end harness for the auto_apply graph against real Neon data.

Mirrors backend/ai_services/auto_apply_adapter.build_auto_apply_*_snapshot
without depending on Django. Drives the graph with a SqliteSaver checkpointer
so gate-1 / gate-2 resumes can happen across separate process invocations.

Usage:

  # Phase 1: start fresh, run discover/scrape/extract/asset_mapping, stop at gate 1
  uv run python scripts/e2e_auto_apply_harness.py start \\
    --run-label planet-127928 \\
    --opportunity-id 127928 \\
    --student-id 20 \\
    --cv-path "C:/Users/alioz/Desktop/Ali Özhavala CV.pdf"

  # Phase 2: resume gate 1 with auto-fill defaults (auto_generate every doc/text,
  # auto_fill misc) — rerun with --upload <doc_type>=<path> to upload specific docs
  uv run python scripts/e2e_auto_apply_harness.py resume-gate-1 \\
    --run-label planet-127928

  uv run python scripts/e2e_auto_apply_harness.py resume-gate-1 \\
    --run-label planet-127928 \\
    --upload "CV=C:/Users/alioz/Desktop/Ali Özhavala CV.pdf"

  # Phase 3: approve gate 2 (default) -> package_and_handoff -> END
  uv run python scripts/e2e_auto_apply_harness.py resume-gate-2 \\
    --run-label planet-127928

Outputs land under C:/Users/alioz/Desktop/auto_apply_e2e_run/<run-label>/.

Env vars (set before running):
  OPENAI_API_KEY, BRAVE_SEARCH_API_KEY, UPPGRAD_SEARCH_PROVIDER=brave,
  UPPGRAD_LLM_PROVIDER=openai, UPPGRAD_BROWSER_SCRAPE_ENABLED=true,
  NEON_DSN  (the Neon postgres connection string)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from datetime import date, datetime
from typing import Any, Dict, List, Optional

# Windows console defaults to cp1252; force utf-8 so Turkish chars in names /
# CV paths and any unicode in scraped page content print without crashing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import psycopg
from psycopg.rows import dict_row

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from uppgrad_agentic.tools.documents import extract_text_from_file
from uppgrad_agentic.workflows.auto_apply.graph import build_graph
from uppgrad_agentic.workflows.auto_apply.nodes.asset_mapping import (
    _GENERATABLE,
    _USER_SUPPLIED,
)


OUTPUT_ROOT = pathlib.Path(r"C:\Users\alioz\Desktop\auto_apply_e2e_run")
SQLITE_DB = OUTPUT_ROOT / "checkpoints.sqlite"

logger = logging.getLogger("e2e_harness")


# ── Neon snapshot builders ───────────────────────────────────────────────────

def fetch_opportunity(conn: psycopg.Connection, job_id: int) -> Dict[str, Any]:
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
        sys.exit(f"Job id={job_id} not found in linkedin_jobs")
    if row["is_closed"]:
        sys.exit(f"Job id={job_id} is_closed=true — pick another job")
    if isinstance(row["posted_time"], (datetime, date)):
        row["posted_time"] = row["posted_time"].isoformat()
    return row


def fetch_profile_snapshot(
    conn: psycopg.Connection,
    student_id: int,
    cv_path: Optional[str],
) -> Dict[str, Any]:
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
        sys.exit(f"Student id={student_id} not found")

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT university, title_obtained, end_year, gpa
            FROM accounts_studenteducation WHERE student_id = %s
            ORDER BY end_year DESC NULLS LAST
            """,
            (student_id,),
        )
        education = [
            {
                "degree": r["title_obtained"],
                "institution": r["university"],
                "year": r["end_year"],
                "gpa": float(r["gpa"]) if r["gpa"] else None,
            }
            for r in cur.fetchall()
        ]

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT cs.name FROM accounts_student_skills sks
            JOIN common_skill cs ON sks.skill_id = cs.id WHERE sks.student_id = %s
            """,
            (student_id,),
        )
        skills = [r["name"] for r in cur.fetchall()]

    cv_text = ""
    if cv_path:
        try:
            cv_text = extract_text_from_file(cv_path).text
        except Exception as exc:
            logger.warning("CV text extraction failed: %s", exc)

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


# ── Output helpers ───────────────────────────────────────────────────────────

def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def find_interrupt_value(state: Dict[str, Any]) -> Optional[Any]:
    interrupts = state.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", first)


def slug(s: str, max_len: int = 60) -> str:
    out = "".join(c if c.isalnum() else "_" for c in s).strip("_")
    return out[:max_len] or "item"


def write_tailored_outputs(out_dir: pathlib.Path, state: Dict[str, Any]) -> None:
    docs = state.get("tailored_documents") or {}
    answers = state.get("tailored_answers") or {}

    docs_dir = out_dir / "tailored_documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for doc_type, info in docs.items():
        content = info.get("content") or "(empty)"
        meta = (
            f"# {doc_type}\n\n"
            f"- source: {info.get('source')}\n"
            f"- tailoring_depth: {info.get('tailoring_depth')}\n"
            f"- llm_used: {info.get('llm_used')}\n"
            f"- passes: {info.get('passes')}\n"
            f"- char_count: {len(info.get('content') or '')}\n\n"
            f"---\n\n"
        )
        (docs_dir / f"{slug(doc_type)}.md").write_text(meta + content, encoding="utf-8")

    answers_dir = out_dir / "tailored_answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    for key, info in answers.items():
        question = info.get("question") or ""
        meta = (
            f"# {question}\n\n"
            f"- form_field_index: {info.get('form_field_index')}\n"
            f"- llm_used: {info.get('llm_used')}\n"
            f"- char_count: {len(info.get('content') or '')}\n\n"
            f"---\n\n"
        )
        fname = f"{key}_{slug(question, 50)}.md"
        (answers_dir / fname).write_text(meta + (info.get("content") or "(empty)"), encoding="utf-8")


# ── Gate 1 payload construction ──────────────────────────────────────────────

def build_gate1_payload(
    requirement_items: List[Dict[str, Any]],
    upload_overrides: Dict[str, str],
    ignore_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """Auto-fill the gate-1 resume payload.

    upload_overrides maps document_type -> file path. For each matching
    document item, the file is read with extract_text_from_file and the
    choice becomes 'upload' with that text. Everything else: documents
    and texts auto_generate, USER_SUPPLIED docs without an upload are
    skipped (or kept required -> invalid; we still emit choice=skip and
    let validation catch it). Misc -> auto_fill.
    """
    ignore_ids = ignore_ids or set()
    requirements: Dict[str, Dict[str, Any]] = {}
    for item in requirement_items:
        rid = str(item["id"])
        cat = item.get("category")
        if rid in ignore_ids or item["id"] in ignore_ids:
            requirements[rid] = {
                "choice": "ignore_for_now",
                "uploaded_text": None,
                "user_prompt": None,
            }
            continue
        if cat == "document":
            doc_type = (item.get("document_type") or "").strip()
            override = upload_overrides.get(doc_type) or upload_overrides.get(item.get("label") or "")
            if override:
                try:
                    text = extract_text_from_file(override).text
                except Exception as exc:
                    logger.warning("upload extraction failed for %s (%s): %s", doc_type, override, exc)
                    text = ""
                requirements[rid] = {
                    "choice": "upload",
                    "uploaded_text": text,
                    "user_prompt": None,
                }
            elif doc_type in _USER_SUPPLIED:
                # No upload provided for a user-supplied doc; can't auto_generate.
                # Use ignore_for_now if not required; skip otherwise (invalid;
                # gate validation will surface the error).
                requirements[rid] = {
                    "choice": "ignore_for_now" if not item.get("required") else "skip",
                    "uploaded_text": None,
                    "user_prompt": None,
                }
            else:
                requirements[rid] = {
                    "choice": "auto_generate",
                    "uploaded_text": None,
                    "user_prompt": None,
                }
        elif cat == "text":
            requirements[rid] = {
                "choice": "auto_generate",
                "uploaded_text": None,
                "user_prompt": None,
            }
        elif cat == "misc":
            requirements[rid] = {
                "choice": "auto_generate",
                "uploaded_text": None,
                "user_prompt": None,
            }
        else:
            requirements[rid] = {
                "choice": "ignore_for_now",
                "uploaded_text": None,
                "user_prompt": None,
            }
    return {"requirements": requirements, "misc_strategy": "auto_fill"}


# ── Print helpers ────────────────────────────────────────────────────────────

def print_discovery_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "discovered_apply_url": state.get("discovered_apply_url"),
        "discovery_method": state.get("discovery_method"),
        "discovery_confidence": state.get("discovery_confidence"),
        "discovered_form_url": state.get("discovered_form_url"),
        "scrape_status": (state.get("scraped_requirements") or {}).get("status"),
        "scrape_confidence": (state.get("scraped_requirements") or {}).get("confidence"),
        "form_fields_count": len(state.get("form_fields") or []),
        "posting_closed": state.get("posting_closed"),
        "compatibility_warnings": state.get("compatibility_warnings"),
        "eligibility": (state.get("eligibility_result") or {}).get("decision"),
    }
    print("\n=== DISCOVERY / SCRAPE SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


def print_requirement_items(items: List[Dict[str, Any]]) -> None:
    by_cat: Dict[str, int] = {}
    for it in items:
        by_cat[it["category"]] = by_cat.get(it["category"], 0) + 1
    print(f"\n=== REQUIREMENT ITEMS ({len(items)}: {by_cat}) ===")
    for it in items:
        flag = "REQ" if it.get("required") else "opt"
        doc = it.get("document_type") or ""
        print(f"  #{it['id']} [{it['category']:8s} {flag}] {doc:18s} | {(it.get('label') or '')[:80]}")


# ── Subcommands ──────────────────────────────────────────────────────────────

def cmd_start(args: argparse.Namespace) -> None:
    out_dir = OUTPUT_ROOT / args.run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    pg = psycopg.connect(args.dsn)
    try:
        opportunity_data = fetch_opportunity(pg, args.opportunity_id)
        profile_snapshot = fetch_profile_snapshot(pg, args.student_id, args.cv_path)
    finally:
        pg.close()

    write_json(out_dir / "opportunity_data.json", opportunity_data)
    cv_chars = len(profile_snapshot.get("document_texts", {}).get("CV", ""))
    write_json(
        out_dir / "profile_snapshot.json",
        {**profile_snapshot, "document_texts": {"CV": f"<{cv_chars} chars>"} if cv_chars else {}},
    )
    print(f"Opportunity: id={opportunity_data['id']} title={opportunity_data['title']!r} company={opportunity_data['company']!r}")
    print(f"Profile: name={profile_snapshot['name']!r} skills={len(profile_snapshot['disciplines'])} cv_chars={cv_chars}")

    initial_state = {
        "opportunity_type": "job",
        "opportunity_id": str(opportunity_data["id"]),
        "opportunity_data": opportunity_data,
        "profile_snapshot": profile_snapshot,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(SQLITE_DB)) as cp:
        graph = build_graph(checkpointer=cp)
        config = {"configurable": {"thread_id": args.run_label}}
        result = graph.invoke(initial_state, config=config)

    summary = print_discovery_summary(result)
    write_json(out_dir / "discovery_summary.json", summary)
    write_json(out_dir / "form_fields.json", result.get("form_fields") or [])
    items = result.get("requirement_items") or []
    write_json(out_dir / "requirement_items.json", items)
    print_requirement_items(items)

    interrupt_val = find_interrupt_value(result)
    if interrupt_val is None:
        write_json(out_dir / "final_state_no_interrupt.json", _safe_state(result))
        print("\n[NO INTERRUPT] Workflow finished without hitting a gate.")
        print(f"  result: {result.get('result')}")
        print(f"  current_step: {result.get('current_step')}")
        return

    write_json(out_dir / "gate1_interrupt_payload.json", interrupt_val)
    print(f"\n[STOPPED AT GATE 1] Outputs in {out_dir}")
    print("Next: review requirement_items.json, then run `resume-gate-1` "
          "(optionally with --upload doc_type=path).")


def cmd_resume_gate1(args: argparse.Namespace) -> None:
    out_dir = OUTPUT_ROOT / args.run_label
    if not out_dir.exists():
        sys.exit(f"Run dir {out_dir} not found — did you run `start` first?")

    items_file = out_dir / "requirement_items.json"
    if not items_file.exists():
        sys.exit(f"requirement_items.json missing in {out_dir}")
    items = json.loads(items_file.read_text(encoding="utf-8"))

    upload_overrides: Dict[str, str] = {}
    for spec in args.upload or []:
        if "=" not in spec:
            sys.exit(f"--upload must be doc_type=path, got {spec!r}")
        k, v = spec.split("=", 1)
        upload_overrides[k.strip()] = v.strip()

    ignore_ids = set()
    for raw in args.ignore or []:
        try:
            ignore_ids.add(int(raw))
        except ValueError:
            ignore_ids.add(raw)

    payload = build_gate1_payload(items, upload_overrides, ignore_ids)
    write_json(out_dir / "gate1_resume_payload.json", payload)
    print("Resuming gate 1 with payload:")
    for rid, entry in payload["requirements"].items():
        upl = "(uploaded_text=Y)" if entry.get("uploaded_text") else ""
        print(f"  #{rid}: {entry['choice']} {upl}")
    print(f"  misc_strategy: {payload['misc_strategy']}")

    with SqliteSaver.from_conn_string(str(SQLITE_DB)) as cp:
        graph = build_graph(checkpointer=cp)
        config = {"configurable": {"thread_id": args.run_label}}
        result = graph.invoke(Command(resume=payload), config=config)

    write_json(out_dir / "tailoring_state.json", _safe_state(result))
    write_json(out_dir / "human_review_1.json", result.get("human_review_1") or {})
    write_json(out_dir / "evaluation_result.json", result.get("evaluation_result") or {})
    write_tailored_outputs(out_dir, result)

    print("\n=== TAILORED DOCUMENTS ===")
    for doc_type, info in (result.get("tailored_documents") or {}).items():
        c = info.get("content") or ""
        print(f"  {doc_type}: chars={len(c)} depth={info.get('tailoring_depth')} "
              f"source={info.get('source')} llm_used={info.get('llm_used')} passes={info.get('passes')}")
    print("\n=== TAILORED ANSWERS ===")
    for key, info in (result.get("tailored_answers") or {}).items():
        c = info.get("content") or ""
        q = (info.get("question") or "")[:80]
        print(f"  ffi={key}: chars={len(c)} llm_used={info.get('llm_used')} | {q}")

    interrupt_val = find_interrupt_value(result)
    if interrupt_val is None:
        print("\n[NO INTERRUPT] Workflow finished without hitting gate 2.")
        print(f"  result: {result.get('result')}")
        return

    write_json(out_dir / "gate2_interrupt_payload.json", interrupt_val)
    print(f"\n[STOPPED AT GATE 2] Outputs in {out_dir}")
    print("Next: run `resume-gate-2` (default: approve=True, attempt_auto_submit=False).")


def cmd_resume_gate2(args: argparse.Namespace) -> None:
    out_dir = OUTPUT_ROOT / args.run_label
    if not out_dir.exists():
        sys.exit(f"Run dir {out_dir} not found")

    payload = {
        "approved": not args.reject,
        "attempt_auto_submit": False,
        "feedback": {},
    }
    write_json(out_dir / "gate2_resume_payload.json", payload)
    print(f"Resuming gate 2 with payload: {payload}")

    with SqliteSaver.from_conn_string(str(SQLITE_DB)) as cp:
        graph = build_graph(checkpointer=cp)
        config = {"configurable": {"thread_id": args.run_label}}
        result = graph.invoke(Command(resume=payload), config=config)

    write_json(out_dir / "final_state.json", _safe_state(result))
    write_json(out_dir / "application_package.json", result.get("application_package") or {})
    write_json(out_dir / "application_record.json", result.get("application_record") or {})

    print("\n=== FINAL ===")
    print(f"  current_step: {result.get('current_step')}")
    print(f"  step_history (last 8): {(result.get('step_history') or [])[-8:]}")
    pkg = result.get("application_package") or {}
    print(f"  application_package keys: {list(pkg.keys())}")
    print(f"  application_record: {result.get('application_record')}")
    print(f"  result: {result.get('result')}")
    print(f"\nWritten to {out_dir}.")


def _safe_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the largest fields so the dumped JSON is readable."""
    out = dict(state)
    out.pop("profile_snapshot", None)
    if "discovered_page_content" in out:
        out["discovered_page_content"] = f"<{len(out['discovered_page_content'] or '')} chars>"
    if "discovered_raw_html" in out:
        out["discovered_raw_html"] = f"<{len(out['discovered_raw_html'] or '')} chars>"
    sr = out.get("scraped_requirements") or {}
    if sr.get("raw_content"):
        sr["raw_content"] = f"<{len(sr['raw_content'])} chars>"
    if sr.get("raw_html"):
        sr["raw_html"] = f"<{len(sr['raw_html'])} chars>"
    out["scraped_requirements"] = sr
    out.pop("__interrupt__", None)
    return out


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-apply E2E harness")
    ap.add_argument("--dsn", default=os.environ.get("NEON_DSN"),
                    help="Neon postgres DSN (or NEON_DSN env var)")
    ap.add_argument("--run-label", required=True,
                    help="Used as both thread_id and output subdir name")

    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("start", help="Run from start to gate 1")
    p1.add_argument("--opportunity-id", type=int, required=True)
    p1.add_argument("--student-id", type=int, required=True)
    p1.add_argument("--cv-path", required=True, help="Path to source CV PDF")
    p1.set_defaults(func=cmd_start)

    p2 = sub.add_parser("resume-gate-1", help="Resume gate 1 -> gate 2")
    p2.add_argument("--upload", action="append",
                    help="Repeatable: doc_type=path (e.g. --upload 'CV=C:/x.pdf')")
    p2.add_argument("--ignore", action="append",
                    help="Repeatable: requirement id to mark ignore_for_now (e.g. --ignore 3)")
    p2.set_defaults(func=cmd_resume_gate1)

    p3 = sub.add_parser("resume-gate-2", help="Resume gate 2 -> END")
    p3.add_argument("--reject", action="store_true",
                    help="Reject at gate 2 instead of approving")
    p3.set_defaults(func=cmd_resume_gate2)

    args = ap.parse_args()
    if not args.dsn:
        sys.exit("--dsn or NEON_DSN env var required")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
