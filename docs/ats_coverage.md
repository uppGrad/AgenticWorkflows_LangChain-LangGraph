# ATS coverage — what auto-apply can and cannot reach today

A living tracker of which Applicant Tracking Systems / careers-page setups
the auto-apply graph handles end-to-end, partially, or not at all. Update
when something is verified or breaks.

The two pipeline stages this document is about:

1. **Discovery** — `discover_apply_url`: takes a `linkedin_jobs` row and
   produces a `discovered_apply_url` + `discovery_method`. Confidence
   gated on title fuzzy match ≥85 + corroborators (company-in-text,
   location, posted-time, description-keyword overlap).
2. **Form extraction** — `extract_form_fields`: takes the apply URL and
   produces structured `FormField` records the rest of the graph
   (asset_mapping, value_planner, playwright_filler) consumes. Failure
   here silently falls back to per-type defaults at `asset_mapping`
   (`[CV, Cover Letter]` for jobs).

A "✅ full" row means **both stages** complete and produce real form fields
that map back to the actual ATS form (not the per-type defaults).

## Coverage matrix

| ATS / host | Discovery | Form extraction | Notes | Last verified |
|---|---|---|---|---|
| Greenhouse (`*.greenhouse.io`, `boards.greenhouse.io`, `job-boards.greenhouse.io`) | ✅ Tier 1 ATS | ✅ full | Form on same URL after JS hydration. Tier 2 (`force_browser_fetch`) handles the React shell. | 2026-04 (Anthropic) |
| Ashby (`jobs.ashbyhq.com`) | ✅ Tier 1 ATS | ✅ full | Form on `/application` suffix. `ats_form_urls` rewrites the URL. | 2026-04 |
| Lever (`jobs.lever.co`) | ✅ Tier 1 ATS | ✅ full | Form on `/apply` suffix. | 2026-04 |
| SmartRecruiters (`*.smartrecruiters.com`) | ✅ Tier 1 ATS | ⚠️ partial | Form on `/apply` suffix. Single-page-gated forms now extract via Tier 2b click-through (PR #13). Multi-step variants reach page 1 only. | 2026-05 |
| Workable (`apply.workable.com/<org>/j/<id>/`) | ✅ Tier 1 ATS | ⚠️ partial | Listing page 1 (contact info) extracts via Tier 2b click-through. **Page 2+ (CV upload + cover letter + custom questions) NOT reached** — needs recursive click-through across "Continue" buttons. See "Known limitations" below. | 2026-05 |
| Workable iframe-embed | ✅ Tier 2 careers | ✅ full | Same as the listing case once iframe is followed. | 2026-04 |
| Workday (`*.myworkdayjobs.com`) | ✅ Tier 1 ATS | ❌ blocked | Auth wall — no public form URL. `ats_form_urls.resolve_application_form_url` returns `None`; the graph correctly produces no form fields and the user gets a handoff package. | 2026-04 |
| MongoDB careers (cross-origin Greenhouse iframe) | ✅ Tier 2 careers | ✅ full | `extract_ats_iframe_src` follows the iframe to the Greenhouse form. | 2026-04 |
| Anthropic careers (`anthropic.com/careers`, then Greenhouse `job-boards.greenhouse.io`) | ✅ Tier 1 ATS | ✅ full | `boards.greenhouse.io` 301 → `job-boards.greenhouse.io` resolved by httpx. Tier 2 handles SPA hydration. | 2026-04 |
| Direct careers pages (company-domain, no ATS) | ✅ Tier 2 careers | varies | Per-page judgement — depends on whether the form is rendered server-side or behind a CTA. Tier 2b click-through covers most of the latter. | 2026-04 |
| LinkedIn-only ("ghost") postings (no `url_direct`) | ❌ Tier 3 generic returns `failed` | n/a | Bare LinkedIn listings with no external apply path. Discovery correctly flags as `failed`; we surface the package without an apply URL. Real fix is upstream in the LinkedIn scraper. | 2026-04 |
| Indeed / Glassdoor (when `url_direct` points there) | ❌ blocklisted at Tier 2 | n/a | Treated as aggregators; we don't try to apply through them. | 2026-04 |

## Known limitations

### Workable multi-step apply flow

Workable's `/j/<slug>/` URL is a hydration shell with the company logo, JD,
and an "Apply for this job" CTA. Our Tier 2b click-through (PR #13) clicks
that CTA and reaches the contact-info page, but Workable's apply flow
continues over multiple pages:

```
page 0: listing            (CTA: "Apply for this job")
page 1: contact info       (CTA: "Continue")        ← Tier 2b stops here
page 2: CV upload          (CTA: "Continue")
page 3: cover letter / Q&A (CTA: "Submit")
```

End-to-end coverage requires recursive click-through across page transitions.
The current Tier 2b is single-click only. Until that lands, Workable
postings will surface profile-fillable RequirementItems but **no
CV/Cover Letter requirement** — even though both will be required at
submit time. Practical workaround for the user: use the "package and
bounce" path (gate-1 `ignore_for_now` for required items) and finish in
the browser.

Tracked: TODO open dedicated issue for `extract_form_fields` recursive
click-through.

### `expected_source` classification quality

Live tests showed noise: Anthropic Country resolved to `unknown`, Notion
free-text Location resolved to `user_profile`. The `extract_form_fields`
system prompt needs few-shot examples before auto-fill is reliable
across ATSes. See top-level CLAUDE.md "Open / out-of-scope".

### LinkedIn ghost-postings burn Brave budget

Many `linkedin_jobs` rows are tracker-only (apply-button bookmarks
pointing back to LinkedIn). Discovery correctly returns `failed`, but
spends 1-3 Brave calls finding that out. A negative cache (~7 day TTL)
on `failed` would help; the real fix is at scraper-ingestion time.

## How to add an entry

When you verify a new ATS, append a row to the matrix above. Format:

- **Discovery** column: `✅ Tier N <method>` / `❌ <reason>`. The method
  matches `DiscoveryResult.method` (`url_direct` / `ats` / `careers` /
  `generic` / `failed` / `closed` / `skipped_internal`).
- **Form extraction** column: `✅ full` / `⚠️ partial — <one-liner>` /
  `❌ blocked — <reason>`.
- **Notes**: anything special — URL rule, hydration quirk, click-through
  requirement, multi-step caveat.
- **Last verified**: YYYY-MM (the smoke test or live session that
  produced the entry).

Partial / blocked rows should also have a "Known limitations" subsection
spelling out what doesn't work and what the fix shape would look like.
