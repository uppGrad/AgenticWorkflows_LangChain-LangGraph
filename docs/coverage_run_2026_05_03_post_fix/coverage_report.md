# Coverage run — 2026-05-03 (post URL hygiene + raw_html cap fix)

End-to-end coverage on 20 opportunities (16 jobs + 2 masters + 2 scholarships) after PR #21 (`fix(ats_form_urls)` URL parsing + 2MB raw_html cap). Same student across all sessions: koray.sevil.b@gmail.com (student_id=16) with `test_resume.pdf`. Driven via `scripts/coverage_run.py`.

## Per-session results

| # | Type | ATS | Posting | Form fields | Reqs (doc/text/misc) | Tailored docs | Answers | Eval warnings | Final | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | job | ashby | Monumental — Technical Sourcer | 8 | 1 / 3 / 1 | Resume(28c) | 3 | 1 | ok | ✅ extracted |
| 2 | job | ashby | Robin Radar Systems — Top Engineers | 8 | 2 / 1 / 1 | CV(193c), Cover Letter(1121c) | 1 | 2 | ok | ✅ extracted |
| 3 | job | ashby | Filigran — Senior Business System Manager | 12 | 1 / 0 / 1 | CV(165c) | 0 | 1 | ok | ✅ extracted |
| 4 | job | greenhouse | SOFICO — Business Analyst (m/f/d) | 1 | 0 / 0 / 1 | — | 0 | 1 | ok | ⚠️ partial (1 fields) |
| 5 | job | greenhouse | DEPT® — Account Director, Media | 13 | 2 / 1 / 1 | CV(128c), Cover Letter(1198c) | 1 | 1 | ok | ✅ extracted |
| 6 | job | greenhouse | Adyen — Internal Control Specialist: Non-Financial Risk | 1 | 0 / 0 / 1 | — | 0 | 1 | ok | ⚠️ partial (1 fields) |
| 7 | job | lever | TSMG Holding — Mystery Shopper | 17 | 1 / 0 / 1 | CV(136c) | 0 | 1 | ok | ✅ extracted |
| 8 | job | lever | Wypoon Technologies — Machine Learning Engineer – Cloud | MLOps | GenA… | 26 | 2 / 4 / 1 | CV(167c), unknown(1158c) | 4 | 1 | ok | ✅ extracted |
| 9 | job | lever | Mendix — Technical Support Engineer | 14 | 1 / 0 / 1 | CV(165c) | 0 | 1 | ok | ✅ extracted |
| 10 | job | smartrecruiters | Wasco — Medewerker Technisch Advies Airco | **0** | 2 / 0 / 0 | CV(115c), Cover Letter(1170c) | 0 | 1 | ok | ❌ extraction failed → defaults |
| 11 | job | smartrecruiters | DYKA Group — Network Security Engineer | 1 | 0 / 0 / 1 | — | 0 | 1 | ok | ⚠️ partial (1 fields) |
| 12 | job | smartrecruiters | Abercrombie & Fitch Co. — A&F Co. Design Sophomore Summit - Summer 2026 | **0** | 2 / 0 / 0 | CV(164c), Cover Letter(1316c) | 0 | 1 | ok | ❌ extraction failed → defaults |
| 13 | job | workable | Phoenix Software — Business Development Manager - Healthcare | 3 | 0 / 0 / 1 | — | 0 | 1 | ok | ✅ extracted |
| 14 | job | workable | Debenhams Group — Head of Commercial Finance | 21 | 2 / 2 / 1 | CV(165c) | 2 | 1 | ok | ✅ extracted |
| 15 | job | workday | Prysmian — Process Sr Engineer - Hollow Core Fiber | **0** | 2 / 0 / 0 | CV(143c), Cover Letter(1226c) | 0 | 1 | ok | ✓ correctly handled (auth wall) |
| 16 | job | workday | Safeguard Global — Partnerships Manager, EMEA | **0** | 2 / 0 / 0 | CV(204c), Cover Letter(1458c) | 0 | 2 | ok | ✓ correctly handled (auth wall) |
| 17 | masters | masters | Turan International University — Business Administration (MBA) | **0** | 1 / 0 / 0 | CV(113c) | 0 | 1 | ok | ✅ requirements parsed |
| 18 | masters | masters | Boise State University — Population and Health Systems Management | **0** | 2 / 0 / 0 | CV(218c), References(34c) | 0 | 1 | ok | ✅ requirements parsed |
| 19 | scholarship | scholarship | Goostree Law Group — Goostree Law Group Bright Futures Scholarship | **0** | 2 / 0 / 0 | CV(180c), Cover Letter(1103c) | 0 | 1 | ok | ✅ requirements parsed |
| 20 | scholarship | scholarship | GriffithLaw Injury Lawyers — GriffithLaw Injury Lawyers The Road Ahead: New D… | **0** | 2 / 0 / 0 | CV(181c), Cover Letter(1248c) | 0 | 1 | ok | ✅ requirements parsed |

## Tally

- Total opportunities: **20**
- Jobs: **16**
  - ✅ Form fields extracted (≥3 visible inputs): **9** (Ashby ×3, Greenhouse DEPT, Lever ×3, Workable ×2)
  - ✓ Workday graceful no-fields (auth wall expected): **2**
  - ⚠️ Single-field partial (extraction picked up only 1 input, likely chrome/search box not the form): **3** (Greenhouse SOFICO + Adyen, SmartRecruiters DYKA)
  - ❌ Zero-field fallbacks (extracted nothing → CV/CL defaults): **2** (SmartRecruiters Wasco, Abercrombie)
- Programs: **2** ✅ (requirements parsed from JSONB `data`; CV ± References tailored)
- Scholarships: **2** ✅ (requirements parsed; CV + Cover Letter tailored)
- **All 20 sessions completed end-to-end** through the full pipeline (start → extract → asset_mapping → gate-1 auto-resume → tailoring → evaluation → gate-2 approve → handoff → record_application). Zero crashes; every `result.status == ok`.

## Fixes verified by this run

PR #21 shipped two fixes — URL hygiene (`_strip_tracker_query`/`_strip_tracker_from_path`) and raw_html cap split (500 KB text vs 2 MB DOM). Per-row evidence vs the pre-fix run on 2026-05-03:

| Posting | Pre-fix | Post-fix | Issue addressed |
|---|---|---|---|
| Lever Wypoon (id=129801) | 0 fields | **26 fields** | `&urlHash=ZjNc` glued to path — `_strip_tracker_from_path` cleans it |
| Lever Mendix (id=127838) | 0 fields | **14 fields** | `?source=LinkedIn&urlHash=` — `_ensure_suffix` now appends to path, not query |
| Workable Phoenix-no-org (id=107732) | 0 fields | **3 fields** | `/j/<id>&urlHash=iziN` malformed path — same path-glue fix |
| SmartRecruiters DYKA (id=128136) | 0 fields | 1 field | URL hygiene fix landed (clean `/apply`) but Tier-2 browser fetch only sees 1 input — separate extraction issue |

Plus the Dreamgames Lever apply page that triggered the raw_html cap fix during diagnosis: 722 KB body, `<form>` at byte 709,929 → form survives the cap and yields 9 inputs.

## Residual issues (pre-existing, not introduced by PR #21)

### Greenhouse SOFICO / Adyen (1 field each)

Post-fix discovered_form_url is the same as the overview URL (Greenhouse keeps form on same URL). Tier 1 cached HTML → Tier 2 force_browser → Tier 2b click-through all return a page where `extract_form_html` falls through to the body-fallback strategy and picks up exactly one `<input>` (likely the page-chrome search box). The actual application form on these postings is hydrated later or behind a CTA the click-through pass doesn't trip.

Hypothesis: `_crawl_with_browser` `wait_for: 'js:() => document.body.innerText.length > 1000'` returns before the React shell finishes hydrating these specific Greenhouse templates. Same shape as DEPT (which works) suggests a per-listing template difference, not a systemic Greenhouse failure.

### SmartRecruiters Wasco / Abercrombie (0 fields)

URL is clean (`…/apply`), httpx returns 200, but `extract_form_html` finds 0 inputs. SmartRecruiters renders the form behind a "I'm interested" / "Apply now" CTA that opens a modal — the modal contains the actual `<form>`. The current Tier-2b click-through patterns don't match the modal-trigger button on these specific listings.

### Workable Phoenix only 3 fields

The no-org Workable URL (`apply.workable.com/j/<id>`) is canonical now (was malformed before), but the `/apply/` page only shows the basic contact + email form — Phoenix is a multi-step Workable variant per `ats_coverage.md`. Reaching the doc upload + Q&A page-2 is intentionally out of scope until the auto-submit feature ships.

### lever-tsmg query string passthrough

`?lever-source=LinkedIn` survives in the form_url because `lever-source` isn't in `_TRACKER_QUERY_PARAMS` (only `source`). Lever still 200s on it so 17 fields extracted fine; could add to the tracker set in a follow-up but not load-bearing.

## Recommended follow-ups

1. **Greenhouse hydration timing audit** — capture rendered HTML from the broken listings (SOFICO, Adyen) vs the working ones (DEPT, anthropic.com) and diff. Probably a longer `wait_for` selector that targets the actual form anchor.
2. **SmartRecruiters CTA matcher** — extend the click-through patterns in `_build_crawler_run_config` to cover the SR-specific "I'm interested" / Dutch "Solliciteer" wording.
3. **`lever-source` tracker set** — one-line addition; trivial.
4. The Workable Debenhams 0/24 fill-rate issue from the previous coverage run wasn't re-tested here (this script doesn't drive Playwright filling — only extraction + tailoring through handoff). That's still the open item from the prior report.

## Per-target durations

Total wall-clock: ~7 min for 20 sessions (Workday ~6s each, Workable / Lever 30-90s, Ashby ~75s, programs/scholarships ~5s). LLM cost: ~$1-2 OpenAI.
