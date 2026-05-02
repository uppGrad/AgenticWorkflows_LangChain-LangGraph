# Coverage run — 2026-05-03

End-to-end auto-apply against 12 recent open job postings (one per ATS,
two of each, all posted 2026-03-13 to 2026-03-16). Same student
(`browsertest_legacy@uppgrad.dev`), same browser-fallback configuration,
same agent prompts. Driven post-merge of the `/apply/` URL rewrite +
LLM cap bump (PR #20).

## Per-session results

| # | ATS | Posting | Form fields | Reqs (doc/text/misc) | CV engine | CL engine | Auto-fill | Outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | Ashby | Airwallex — Corporate Secretary | 16 | 1 / 0 / 1 | tectonic | — | 7/16 (44%) | ✅ handoff |
| 2 | Ashby | Filigran — Senior Business System Manager | 12 | 1 / 0 / 1 | tectonic | — | 7/12 (58%) | ✅ handoff |
| 3 | Greenhouse | Catawiki — Security Engineer | 21 | 2 / 0 / 1 | tectonic | tectonic | 16/21 (76%) | ✅ handoff |
| 4 | Greenhouse | Planet — Senior SWE Geometry | 17 | 2 / 1 / 1 | tectonic | tectonic | 9/17 (53%) | ✅ handoff |
| 5 | Lever | Mendix — Technical Support Engineer | **0** | 2 (defaults) / 0 / 0 | — | — | 0/0 — extraction never reached | ❌ form-extraction failed |
| 6 | Lever | Wypoon — ML Engineer | **0** | 2 (defaults) / 0 / 0 | — | — | 0/0 — extraction never reached | ❌ form-extraction failed |
| 7 | SmartRecruiters | DYKA — Network Security Engineer | **0** | 2 (defaults) / 0 / 0 | — | — | 0/0 — extraction never reached | ❌ form-extraction failed |
| 8 | SmartRecruiters | Wasco — Medewerker Technisch Advies | **1** | 0 / 0 / 1 (misc) | — | — | 0/1 (0%) | ⚠️ partial — only 1 input found |
| 9 | Workable | Phoenix Software (no-org URL) | **0** | 2 (defaults) / 0 / 0 | — | — | 0/0 — extraction never reached | ❌ form-extraction failed |
| 10 | Workable | Debenhams Group | **24** | 3 / 2 / 1 | (no tailored docs?) | (no tailored docs?) | 0/24 (0%) | ⚠️ extraction good, fill broken |
| 11 | Workday | Lamb Weston — Manager Plant Tech Services | 0 | 2 (defaults) / 0 / 0 | — | — | 0/0 | ✓ correctly handled (auth wall) |
| 12 | Workday | Enza Zaden — Project Lead Facilities | 0 | 2 (defaults) / 0 / 0 | — | — | 0/0 | ✓ correctly handled (auth wall) |

## Tally

- **Full success** (form fields extracted + CV/CL tailored + auto-fill landed): **4/12** (33%)
  - Both Greenhouse (4 of 4 documents Tectonic-rendered, 53-76% fill rate)
  - Both Ashby (single-doc form, 44-58% fill rate)
- **Partial success** (extraction worked, fill or tailoring broken): **2/12** (17%)
  - Workable Debenhams: 24 fields extracted but 0 filled
  - SmartRecruiters Wasco: 1 field extracted (likely an LLM under-extraction; HTML had ~24KB)
- **Form-extraction failed** (silent fallback to `[CV, Cover Letter]` defaults): **4/12** (33%)
  - Both Lever, one SmartRecruiters, one Workable-no-org
- **Correctly degraded** (auth wall, defaults shown): **2/12** (17%)
  - Both Workday — graph short-circuits at `discovered_form_url=None`, no crash, defaults surface at gate-1.

## Root cause #1: URL hygiene — `&urlHash=...` query-string poisoning

LinkedIn's job scraper appends a tracking parameter (`&urlHash=<hash>`)
to every `url_direct` it stores. For ATSes whose form URL is a path
suffix (`/apply` for Lever/SmartRecruiters, `/apply/` for Workable),
our `_ensure_suffix` does naive string-level appending without
parsing the URL — which produces malformed URLs whenever the input
already carries a query string:

| Source URL (from `linkedin_jobs.url_direct`) | After `_ensure_suffix` | What we actually fetch |
|---|---|---|
| `…/wypoon/.../apply&urlHash=ZjNc` | `…/apply&urlHash=ZjNc/apply` | malformed path |
| `…/mendix/.../apply?source=LinkedIn&urlHash=zlzS` | `…/apply?source=LinkedIn&urlHash=zlzS/apply` | malformed path |
| `…/REXEL1/744000114960857-...?trid=...&urlHash=yQ7q` | `…?trid=...&urlHash=yQ7q/apply` | `/apply` lands inside the query string |
| `…/debenhamsgroup/j/53D0ECE60E?utm_source=linkedin.com&urlHash=nDer` | `…?utm_source=linkedin.com&urlHash=nDer/apply/` | tolerated by Workable's path matcher (24 fields extracted) |
| `…/j/41ED129B39&urlHash=iziN` | `…/j/41ED129B39&urlHash=iziN/apply/` | malformed path; Workable returns blank shell |

Impact: Lever, SmartRecruiters, and Workable-no-org variants (~33% of
the recent-postings sample) silently fall back to `[CV, Cover Letter]`
defaults and produce no form fields at all. The user sees a session
that "works" (handoff completes) but auto-fill does literally nothing.

**Fix**: in `tools/ats_form_urls`, parse the URL, work on the `path`
component only, then reconstruct with the original query / fragment
preserved (or drop the LinkedIn-specific `urlHash` param at the same
time — it's pure tracker noise). One-screen change.

## Root cause #2: SmartRecruiters under-extraction

SmartRecruiters/Wasco extracted only 1 input from a 24KB form_html
(LLM dropped 23 fields). May be a prompt-quality issue with the
extract_form_fields system prompt, OR the HTML cleaner is too
aggressive. Same shape as the recent Workable single-page bug pre-PR
#20 (95KB → truncated to 80K → only 4 fields surfaced). Diagnostic
log line `raw_inputs=N` will help triage live.

## Root cause #3: Workable Debenhams — extraction good, fill = 0

Form-extraction surfaced 24 fields including 3 documents, 2 text Q&A,
1 misc. Auto-fill ran but landed `0/24` — every field locator missed.
Likely because Workable's `/apply/` route fully hydrates within the
extraction-time browser fetch but at fill time the playwright_filler
runs a different browser session that hits a different DOM state.
Worth a separate dive.

## Already-known limits surfaced cleanly

- **Workday** (sessions 11+12): correctly returned `discovered_form_url=None`,
  graph short-circuits at extract_form_fields, gate-1 shows `[CV, Cover Letter]`
  defaults. User clicks handoff, applies manually. ✓ Behaves as
  documented in `ats_coverage.md`.
- **Workable multi-step**: not exercised in this run (Phoenix Software
  failed for URL-hygiene reasons before multi-step would have mattered;
  Debenhams happened to be single-page). Multi-step remains intentionally
  out of scope per the auto-submit-feature gate.

## Recommended next actions

1. **Ship URL-hygiene fix in `_ensure_suffix`**. Single-PR, ~10 lines, unblocks
   ~33% of postings. Lever / SmartRecruiters / Workable-no-org all
   recover.
2. **Investigate the Workable Debenhams 0/24 fill failure** — same
   Tier-1 locator path that worked on Vertigo (8/10) and AISquared (8/10),
   so there's something different about Debenhams' DOM. Probably a
   short fix once we have the page in front of us.
3. **Re-extract LLM prompt audit** — SmartRecruiters Wasco extracted 1/?
   inputs from 24KB, suggests the extract prompt has an under-fitting
   bias on certain markup shapes. Worth a separate triage.

## Appendix — opportunity IDs

| ID | ATS | Posting |
|---|---|---|
| 126815 | ashby | Airwallex |
| 127628 | ashby | Filigran |
| 128629 | greenhouse | Catawiki |
| 127928 | greenhouse | Planet |
| 127838 | lever | Mendix |
| 129801 | lever | Wypoon |
| 128136 | smartrecruiters | DYKA Group |
| 129862 | smartrecruiters | Wasco |
| 107732 | workable | Phoenix Software |
| 101902 | workable | Debenhams Group |
| 127989 | workday | Lamb Weston EMEA |
| 127018 | workday | Enza Zaden |
