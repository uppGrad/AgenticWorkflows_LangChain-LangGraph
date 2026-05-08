"""Coverage harness: run walker + verifier on a randomised sample of
prod job postings and aggregate signal.

Usage:
    PYTHONPATH=src UPPGRAD_FORM_DISCOVERY_VERIFY=1 \
        OPENAI_API_KEY=... \
        uv run --no-sync python scripts/coverage_harness.py

Reads URL candidates from `scripts/coverage_urls.json` (an array of
{id, title, company, url_direct, ats} records). Cleans `&urlHash=...`
artefacts left by the scraper. Runs each through:
  1. discover_form_fields_with_screenshot_async
  2. verify_fields_with_vision (if env flag set)
  3. records per-URL outcome
Writes incremental progress to `/tmp/coverage_results.json` so a crash
mid-run doesn't lose data.

Stops once 50 URLs have produced a non-empty walker output. Replaces
failures (no fields, navigation timeout) with the next candidate.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.WARNING,  # quiet by default; bump to INFO for chatter
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("coverage_harness")
logger.setLevel(logging.INFO)


SCRIPT_DIR = Path(__file__).parent
URLS_FILE = SCRIPT_DIR / "coverage_urls.json"
RESULTS_FILE = Path("/tmp/coverage_results.json")
TARGET_SUCCESS = 50
PER_URL_TIMEOUT_S = 90  # generous — walker + verifier can run ~30s


def _clean_url(url: str) -> str:
    """Strip the scraper's bad `&urlHash=...` artefact (often appended
    without a `?` separator). Without this, the URL is malformed and
    hits a 4xx."""
    # The artefact starts at literal "&urlHash=" or "?urlHash=" or
    # mid-path "&urlHash=". Cut the entire suffix.
    for marker in ("&urlHash=", "?urlHash=", "/&urlHash="):
        if marker in url:
            url = url.split(marker)[0]
    # Some URLs ended `/jobs/12345&urlHash=...` — after split the
    # trailing slash is fine. Some had `?gh_src=...&urlHash=...` —
    # the gh_src was stripped too. That's fine; gh_src is just
    # tracking and not needed for the form.
    return url


async def _run_one_with_timeout(url: str) -> Dict[str, Any]:
    """Run the walker + verifier on one URL with a hard timeout.
    Returns a result dict (never raises)."""
    from uppgrad_agentic.tools.form_discoverer import (
        discover_form_fields_with_screenshot_async,
    )
    from uppgrad_agentic.tools.form_verifier import verify_fields_with_vision

    t0 = time.time()
    try:
        fields, screenshot = await asyncio.wait_for(
            discover_form_fields_with_screenshot_async(url),
            timeout=PER_URL_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return {
            "url": url, "outcome": "timeout",
            "duration_s": round(time.time() - t0, 1),
        }
    except Exception as exc:
        return {
            "url": url, "outcome": "walker_error",
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            "duration_s": round(time.time() - t0, 1),
        }

    walker_count = len(fields)
    if walker_count == 0:
        return {
            "url": url, "outcome": "no_fields",
            "duration_s": round(time.time() - t0, 1),
            "screenshot_bytes": len(screenshot or b""),
        }

    # Run verifier (sync — wraps an LLM call internally).
    t_verify_start = time.time()
    try:
        verified = verify_fields_with_vision(fields, screenshot)
    except Exception as exc:
        verified = fields
        verify_err = f"{type(exc).__name__}: {str(exc)[:200]}"
    else:
        verify_err = None
    verify_duration = round(time.time() - t_verify_start, 1)

    return {
        "url": url,
        "outcome": "ok",
        "duration_s": round(time.time() - t0, 1),
        "verify_duration_s": verify_duration,
        "verify_error": verify_err,
        "screenshot_bytes": len(screenshot or b""),
        "walker_count": walker_count,
        "verified_count": len(verified),
        "walker_fields": [
            {
                "label": f.get("label", "")[:120],
                "field_type": f.get("field_type", ""),
                "name": f.get("name", "")[:60],
                "required": bool(f.get("required")),
                "options_count": len(f.get("options") or []),
                "options_preview": list((f.get("options") or []))[:5],
                "role": f.get("role", ""),
            } for f in fields
        ],
        "verified_fields": [
            {
                "label": f.get("label", "")[:120],
                "field_type": f.get("field_type", ""),
                "name": f.get("name", "")[:60],
                "required": bool(f.get("required")),
                "options_count": len(f.get("options") or []),
                "options_preview": list((f.get("options") or []))[:5],
                "role": f.get("role", ""),
            } for f in verified
        ],
    }


def _summarise(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate signal for the human readable summary."""
    total = len(results)
    by_outcome: Dict[str, int] = {}
    for r in results:
        by_outcome[r.get("outcome", "?")] = by_outcome.get(r.get("outcome", "?"), 0) + 1

    ok = [r for r in results if r.get("outcome") == "ok"]

    walker_total = sum(r["walker_count"] for r in ok)
    verified_total = sum(r["verified_count"] for r in ok)

    # Generic ATS section headings the verifier should have replaced.
    GENERIC_LABELS = {
        "application", "personal information", "eeoc", "demographics",
        "voluntary disclosures", "voluntary self-identification",
        "additional information section",
    }

    walker_generic_labels = 0
    verified_generic_labels = 0
    walker_radio_with_one_or_zero_opts = 0
    verified_radio_with_proper_opts = 0
    walker_combobox_no_opts = 0
    verified_combobox_no_opts = 0

    for r in ok:
        for f in r["walker_fields"]:
            if f["label"].lower().strip() in GENERIC_LABELS:
                walker_generic_labels += 1
            if f["field_type"] == "radio" and f["options_count"] <= 1:
                walker_radio_with_one_or_zero_opts += 1
            if (f["role"] == "combobox" or f["field_type"] == "select") and f["options_count"] == 0:
                walker_combobox_no_opts += 1
        for f in r["verified_fields"]:
            if f["label"].lower().strip() in GENERIC_LABELS:
                verified_generic_labels += 1
            if f["field_type"] == "radio" and f["options_count"] >= 2:
                verified_radio_with_proper_opts += 1
            if (f["role"] == "combobox" or f["field_type"] == "select") and f["options_count"] == 0:
                verified_combobox_no_opts += 1

    durations = [r["duration_s"] for r in ok if "duration_s" in r]
    avg_duration = sum(durations) / len(durations) if durations else 0
    verify_durations = [r["verify_duration_s"] for r in ok if "verify_duration_s" in r and r.get("verify_duration_s") is not None]
    avg_verify_duration = sum(verify_durations) / len(verify_durations) if verify_durations else 0

    return {
        "total_attempts": total,
        "by_outcome": by_outcome,
        "successful": len(ok),
        "walker_field_count_total": walker_total,
        "verified_field_count_total": verified_total,
        "avg_fields_per_form_walker": round(walker_total / len(ok), 1) if ok else 0,
        "avg_fields_per_form_verified": round(verified_total / len(ok), 1) if ok else 0,
        "labels_generic_walker": walker_generic_labels,
        "labels_generic_verified": verified_generic_labels,
        "labels_generic_reduction": walker_generic_labels - verified_generic_labels,
        "radios_no_real_options_walker": walker_radio_with_one_or_zero_opts,
        "radios_with_options_after_verify": verified_radio_with_proper_opts,
        "comboboxes_no_options_walker": walker_combobox_no_opts,
        "comboboxes_no_options_verified": verified_combobox_no_opts,
        "avg_session_seconds": round(avg_duration, 1),
        "avg_verify_seconds": round(avg_verify_duration, 1),
    }


async def main() -> int:
    if not URLS_FILE.exists():
        print(f"missing {URLS_FILE}; populate it with candidate records first")
        return 1
    candidates = json.loads(URLS_FILE.read_text())
    print(f"loaded {len(candidates)} candidates")

    results: List[Dict[str, Any]] = []
    successful = 0
    for i, cand in enumerate(candidates):
        if successful >= TARGET_SUCCESS:
            break
        raw_url = cand.get("url_direct") or ""
        url = _clean_url(raw_url)
        record_id = cand.get("id")
        ats = cand.get("ats", "?")
        title = cand.get("title", "")[:80]
        company = cand.get("company", "")[:40]

        print(
            f"[{i + 1}/{len(candidates)}] (success={successful}/{TARGET_SUCCESS}) "
            f"id={record_id} ats={ats} {company} — {title}",
            flush=True,
        )
        if not url:
            results.append({"id": record_id, "url": None, "outcome": "no_url"})
            RESULTS_FILE.write_text(json.dumps(results, indent=2))
            continue

        result = await _run_one_with_timeout(url)
        result["id"] = record_id
        result["ats"] = ats
        result["title"] = title
        result["company"] = company
        results.append(result)
        if result.get("outcome") == "ok":
            successful += 1
            print(
                f"    ✓ ok — walker={result['walker_count']} verified={result['verified_count']} "
                f"({result['duration_s']}s, verify {result.get('verify_duration_s', '?')}s)",
                flush=True,
            )
        else:
            print(f"    ✗ {result.get('outcome')} ({result.get('duration_s', '?')}s)", flush=True)

        # Persist incrementally — survive a Ctrl-C / OOM mid-run.
        RESULTS_FILE.write_text(json.dumps(results, indent=2))

    summary = _summarise(results)
    summary_path = Path("/tmp/coverage_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nfull results: {RESULTS_FILE}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
