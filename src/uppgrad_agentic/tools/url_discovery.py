from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Literal, Optional, Tuple
from urllib.parse import urlparse

from rapidfuzz import fuzz

from uppgrad_agentic.tools.ats_form_urls import resolve_application_form_url
from uppgrad_agentic.tools.search import SearchProvider, SearchResult
from uppgrad_agentic.tools.web_fetcher import FetchResult, fetch_url_with_fallback

logger = logging.getLogger(__name__)

Tier = Literal["ats", "careers", "generic"]

# Title fuzzy match is a hard prerequisite; below this we don't even score corroborators.
_TITLE_FUZZY_MIN = 85

# Multi-factor verification: number of corroborating signals required per tier.
# Title fuzzy match is always required as a prerequisite. Then at least N of:
#   {company-in-text, location-match, posted-time-match, description-keyword-overlap}
# Careers tier needs only 1 because the search-time `site:<company-domain>`
# constraint already proves the company. ATS / generic need 2.
#
# Why 2 for ATS: matching only company-in-URL is trivially satisfied by any
# Greenhouse page hosted by that company — including pages for completely
# different locations or roles. Live test confirmed: a Schwyz, Switzerland
# linkedin_jobs row matched a Greenhouse URL for an Ohio role purely on
# title-fuzz + same-company. That's the false positive we want to prevent.
_CORROBORATORS_REQUIRED = {
    "ats": 2,
    "careers": 1,
    "generic": 2,
}

_DESCRIPTION_KEYWORD_HIT_THRESHOLD = 3   # ≥3 of extracted distinctive terms must appear
_DESCRIPTION_KEYWORD_LIMIT = 10          # extract at most 10 distinctive tokens

# Minimum fuzzy-match score for company-vs-slug normalization. Below this we
# treat the slug as a different company and reject the candidate outright.
# Example normalizations: 'github' vs 'github' = 100, 'notion' vs 'notionhq' = ~80
# (substring match), 'github' vs 'formaaiinc' = ~10 → reject.
_SLUG_FUZZY_MIN = 70

# Stopwords for keyword extraction. Kept small and focused on common
# job-description boilerplate that would otherwise dominate frequency counts.
_DESCRIPTION_STOPWORDS = {
    "company", "experience", "candidate", "position", "looking",
    "responsibilities", "requirements", "preferred", "skills", "ability",
    "support", "across", "various", "include", "working", "primary",
    "duties", "applicant", "minimum", "qualifications", "successful",
    "qualified", "develop", "communicate", "developing", "providing",
    "professional", "growth", "opportunities", "additional", "applications",
    "applying", "applies", "ensures", "ensure", "ensuring",
}


@dataclass
class VerifyInputs:
    candidate_url: str
    candidate_title: str
    candidate_text: str
    candidate_posted_at: Optional[datetime]
    job: dict
    tier: Tier


@dataclass
class VerificationScore:
    passed: bool
    confidence: float
    reasons: List[str]


@dataclass
class DiscoveryResult:
    url: str
    method: str            # 'url_direct' | 'ats' | 'careers' | 'generic' | 'closed' | 'failed'
    confidence: float
    text: str = ""         # verified page content; populated when verification fetched
    raw_html: str = ""     # raw rendered HTML when available (for downstream form-field extraction)
    http_status: int = 0
    posting_closed: bool = False  # True when the listing exists but is no longer accepting applications
    form_url: Optional[str] = None  # Apply-form URL from per-ATS rules; None when not reachable (Workday auth wall)


# Phrases that definitively indicate a posting is closed/stale. A page that
# verifies on title+company+location but contains any of these is NOT
# actionable for auto-apply — we record it (with method='closed') so the
# workflow can tell the user, and skip it during tier matching in case a
# later tier finds the same role open elsewhere.
_CLOSED_POSTING_PHRASES = [
    "no longer accepting applications",
    "this position has been filled",
    "this job has been closed",
    "applications closed",
    "applications have closed",
    "we are no longer accepting applications",
]


def _detect_closed_posting(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in _CLOSED_POSTING_PHRASES)


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


def _extract_distinctive_keywords(description: str) -> List[str]:
    """Pull lowercase tokens (≥6 chars) from a job description for use as a
    corroborating signal during verification. We pick the most frequent
    non-stopword tokens, which empirically correlate with role-specific terms
    (technologies, methodologies, named tools, location-suffixes, etc.).

    Returns at most _DESCRIPTION_KEYWORD_LIMIT tokens.
    """
    if not description:
        return []
    tokens = re.findall(r"[a-z0-9]{6,}", description.lower())
    counts = Counter(tok for tok in tokens if tok not in _DESCRIPTION_STOPWORDS)
    return [tok for tok, _ in counts.most_common(_DESCRIPTION_KEYWORD_LIMIT)]


def _normalize_company(name: str) -> str:
    """Lowercase + strip non-alphanumeric. 'GitHub Inc.' → 'githubinc'."""
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _extract_ats_company_slug(url: str) -> Optional[str]:
    """Pull the company identifier from a known ATS URL pattern. Returns None
    if the host doesn't match a recognized ATS or the slug can't be extracted.
    Callers MUST treat None as 'unknown ATS, skip slug check' (do not reject).

    Patterns supported (host → where the slug lives):
      *.greenhouse.io path: /<slug>/jobs/<id>           (boards / job-boards)
      jobs.lever.co path:   /<slug>/<id>
      jobs.ashbyhq.com path:/<slug>/<id>
      apply.workable.com:   /<slug>/j/<id>
      jobs.jobvite.com:     /<slug>/...
      *.recruitee.com:      subdomain
      *.bamboohr.com:       subdomain (e.g., <slug>.bamboohr.com)
      *.workable.com:       subdomain (when not 'apply')
      *.myworkdayjobs.com:  subdomain (Workday tenant — may not match company,
                            e.g., 'github.wd1.myworkdayjobs.com'. Returned as-is;
                            slug check is fuzzy enough to handle this case.)
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    parts = [p for p in path.split("/") if p]

    if host.endswith(".greenhouse.io") or host == "greenhouse.io":
        if parts:
            return parts[0]
    if host == "jobs.lever.co" or host.endswith(".lever.co"):
        if host == "jobs.lever.co" and parts:
            return parts[0]
    if host == "jobs.ashbyhq.com":
        if parts:
            return parts[0]
    if host == "apply.workable.com":
        if parts:
            return parts[0]
    if host == "jobs.jobvite.com":
        if parts:
            return parts[0]
    if host.endswith(".recruitee.com"):
        return host.split(".")[0]
    if host.endswith(".bamboohr.com"):
        return host.split(".")[0]
    if host.endswith(".workable.com") and host != "apply.workable.com":
        return host.split(".")[0]
    if host.endswith(".myworkdayjobs.com"):
        return host.split(".")[0]
    if host.endswith(".smartrecruiters.com"):
        # smartrecruiters: split between subdomain and path; prefer first path segment
        if parts:
            return parts[0]
        return host.split(".")[0]
    return None


def _slug_matches_company(slug: str, company: str) -> bool:
    """True when the URL slug plausibly identifies the company. Tolerant to
    common normalization differences: 'NotionHQ' slug vs 'Notion' company,
    'GitHub-inc' slug vs 'GitHub Inc' company.
    """
    s = _normalize_company(slug)
    c = _normalize_company(company)
    if not s or not c:
        return False
    if s == c or s in c or c in s:
        return True
    return fuzz.partial_ratio(s, c) >= _SLUG_FUZZY_MIN


def score_candidate(inputs: VerifyInputs) -> VerificationScore:
    """Multi-factor verification: title fuzz (gate) + N corroborators (gate).

    Title fuzz < 85 → reject.
    Title fuzz passes but corroborators < tier minimum → reject.
    Otherwise pass with confidence scaled by extra corroborators.
    """
    reasons: List[str] = []
    job = inputs.job
    job_title = (job.get("title") or "").strip()
    job_company = (job.get("company") or "").strip()

    # ── ATS-tier hard guard: URL slug must identify the queried company ──
    # The URL host+path of a known ATS unambiguously names the employer.
    # Without this guard, a same-titled role at a different company whose
    # description mentions the queried company name as a *tool* (e.g. GitHub,
    # Stripe, Notion) trivially satisfies the textual company-match
    # corroborator and false-passes verification. Live failure: GitHub query
    # matched a Forma.ai Greenhouse posting via the Databricks/S3/GitHub
    # tooling list. For unrecognized hosts we skip the guard (return None
    # from the extractor) — text-based corroborators carry the load there.
    if inputs.tier == "ats" and job_company:
        slug = _extract_ats_company_slug(inputs.candidate_url)
        if slug is not None and not _slug_matches_company(slug, job_company):
            return VerificationScore(
                passed=False, confidence=0.0,
                reasons=[f"ATS slug '{slug}' does not match company '{job_company}'"],
            )

    # ── Hard prerequisite: title fuzzy match against candidate body ──
    haystack = f"{inputs.candidate_title}\n{inputs.candidate_text[:2000]}"
    title_score = fuzz.partial_ratio(job_title.lower(), haystack.lower()) if job_title else 0
    if title_score < _TITLE_FUZZY_MIN:
        return VerificationScore(passed=False, confidence=0.0,
                                 reasons=[f"title fuzzy {title_score} < {_TITLE_FUZZY_MIN}"])
    reasons.append(f"title fuzzy {title_score}")

    text_lower = inputs.candidate_text.lower()
    candidate_url_lower = inputs.candidate_url.lower()
    corroborators = 0

    # ── Signal 1: company match (URL or text) ──
    if job_company:
        company_in_url = job_company.lower().replace(" ", "") in candidate_url_lower
        company_in_text = re.search(re.escape(job_company), inputs.candidate_text, re.IGNORECASE) is not None
        if company_in_url or company_in_text:
            corroborators += 1
            reasons.append("company match")

    # ── Signal 2: location match ──
    job_loc_tokens = {
        tok.strip().lower()
        for tok in (job.get("location") or "").split(",")
        if len(tok.strip()) >= 3
    }
    if job_loc_tokens:
        loc_hits = sum(1 for tok in job_loc_tokens if tok in text_lower)
        if loc_hits >= 1:
            corroborators += 1
            reasons.append(f"location {loc_hits}/{len(job_loc_tokens)}")

    # ── Signal 3: posted-time match (within 180 days) ──
    job_posted = _parse_iso_or_none(job.get("posted_time"))
    if inputs.candidate_posted_at and job_posted:
        delta_days = abs((inputs.candidate_posted_at - job_posted).days)
        if delta_days <= 180:
            corroborators += 1
            reasons.append(f"freshness {delta_days}d")

    # ── Signal 4: description keyword overlap ──
    job_description = job.get("description") or ""
    if job_description:
        distinctive = _extract_distinctive_keywords(job_description)
        if distinctive:
            kw_hits = sum(1 for kw in distinctive if kw in text_lower)
            if kw_hits >= _DESCRIPTION_KEYWORD_HIT_THRESHOLD:
                corroborators += 1
                reasons.append(f"keywords {kw_hits}/{len(distinctive)}")

    # ── Decision ──
    required = _CORROBORATORS_REQUIRED[inputs.tier]
    if corroborators < required:
        return VerificationScore(
            passed=False, confidence=0.0,
            reasons=[*reasons, f"corroborators {corroborators}/{required}"],
        )

    # Confidence: 0.70 base + 0.05 per extra corroborator + 0.10 if title is near-perfect
    extra = corroborators - required
    bump = 0.10 if title_score >= 95 else 0.0
    confidence = min(1.0, 0.70 + 0.05 * extra + bump)
    reasons.append(f"corroborators {corroborators}/{required}")
    return VerificationScore(passed=True, confidence=confidence, reasons=reasons)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

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


# Domains that show up in linkedin_jobs.company_url but aren't real company
# career sites. Searching `site:<these>` for an apply page is wasted Brave budget.
_CAREERS_DOMAIN_BLOCKLIST = {
    "linkedin.com", "indeed.com", "glassdoor.com", "monster.com", "ziprecruiter.com",
    "facebook.com", "twitter.com", "x.com", "instagram.com", "youtube.com",
    "github.com", "medium.com", "wikipedia.org", "crunchbase.com",
}


def _build_careers_query(title: str, company_url: Optional[str]) -> Optional[str]:
    domain = _extract_company_domain(company_url)
    if not domain:
        return None
    # Strip leading subdomain when checking the blocklist (e.g. www.linkedin.com → linkedin.com)
    base = ".".join(domain.split(".")[-2:]) if domain.count(".") >= 1 else domain
    if base in _CAREERS_DOMAIN_BLOCKLIST:
        logger.info("careers tier skipped: company_url resolves to non-careers domain %s", domain)
        return None
    return f'"{title}" site:{domain}'


def _build_generic_query(title: str, company: str) -> str:
    return f'"{title}" "{company}" apply'


@dataclass
class _VerifiedHit:
    candidate: SearchResult
    score: VerificationScore
    fetch: FetchResult
    posting_closed: bool = False


def _verify_one(
    candidate: SearchResult, job: dict, tier: Tier,
) -> Optional[_VerifiedHit]:
    """Fetch + verify a candidate. Reject thin pages outright before scoring.

    Returns a `_VerifiedHit` on accept; None on any reject. Pages that pass
    verification but contain closed-posting phrases are returned with
    `posting_closed=True` — the orchestrator decides whether to surface them
    (no other tier found an open match) or skip them (later tier hit open).
    The FetchResult is propagated so the orchestrator can hand its `text`
    forward instead of re-fetching the same URL during scrape.
    """
    fetch = fetch_url_with_fallback(candidate.url)
    if not fetch.success:
        return None
    if fetch.thin:
        # Thin pages can't be verified — captcha walls, 404s, JS shells.
        # Skip without scoring.
        return None
    inputs = VerifyInputs(
        candidate_url=candidate.url,
        candidate_title=candidate.title,
        candidate_text=fetch.text,
        candidate_posted_at=None,
        job=job,
        tier=tier,
    )
    score = score_candidate(inputs)
    if not score.passed:
        return None
    return _VerifiedHit(
        candidate=candidate, score=score, fetch=fetch,
        posting_closed=_detect_closed_posting(fetch.text),
    )


def _try_tier(
    candidates: List[SearchResult], job: dict, tier: Tier,
    closed_hits: List[_VerifiedHit],
) -> Optional[_VerifiedHit]:
    """Iterate candidates; return the first OPEN verified hit. Closed hits are
    appended to `closed_hits` so the orchestrator can surface one as a
    `method='closed'` result if no tier produces an open match."""
    for cand in candidates:
        verified = _verify_one(cand, job, tier)
        if verified is None:
            continue
        if verified.posting_closed:
            closed_hits.append(verified)
            continue
        return verified
    return None


def discover_apply_url(
    job: dict,
    search_provider: Optional[SearchProvider],
) -> DiscoveryResult:
    """Synchronous discovery orchestrator.

    Caching (Phase 6) lives in the backend adapter — agentic stays DB-free.
    Callers that want cached results should consult the cache *before* calling
    this function and skip if a hit was found.

    On a successful search-driven verification, the returned DiscoveryResult
    carries the verified page text + http_status so the downstream
    `scrape_application_page` doesn't have to re-fetch the same URL.
    """
    url_direct = (job.get("url_direct") or "").strip()
    if url_direct:
        # url_direct path doesn't fetch during discovery — scrape will do it.
        return DiscoveryResult(
            url=url_direct, method="url_direct", confidence=1.0,
            form_url=resolve_application_form_url(url_direct),
        )

    if search_provider is None:
        return DiscoveryResult(url="", method="failed", confidence=0.0)

    title = (job.get("title") or "").strip()
    company = (job.get("company") or "").strip()
    if not title or not company:
        return DiscoveryResult(url="", method="failed", confidence=0.0)

    closed_hits: List[_VerifiedHit] = []

    # Tier 1: ATS
    ats_results = search_provider.search(_build_ats_query(title, company), count=3)
    hit = _try_tier(ats_results, job, "ats", closed_hits)
    if hit:
        return DiscoveryResult(
            url=hit.candidate.url, method="ats", confidence=hit.score.confidence,
            text=hit.fetch.text, raw_html=hit.fetch.raw_html, http_status=hit.fetch.http_status,
            form_url=resolve_application_form_url(hit.candidate.url),
        )

    # Tier 2: Careers
    careers_q = _build_careers_query(title, job.get("company_url"))
    if careers_q:
        careers_results = search_provider.search(careers_q, count=3)
        hit = _try_tier(careers_results, job, "careers", closed_hits)
        if hit:
            return DiscoveryResult(
                url=hit.candidate.url, method="careers", confidence=hit.score.confidence,
                text=hit.fetch.text, raw_html=hit.fetch.raw_html, http_status=hit.fetch.http_status,
                form_url=resolve_application_form_url(hit.candidate.url),
            )

    # Tier 3: Generic
    generic_results = search_provider.search(_build_generic_query(title, company), count=3)
    hit = _try_tier(generic_results, job, "generic", closed_hits)
    if hit:
        return DiscoveryResult(
            url=hit.candidate.url, method="generic", confidence=hit.score.confidence,
            text=hit.fetch.text, raw_html=hit.fetch.raw_html, http_status=hit.fetch.http_status,
            form_url=resolve_application_form_url(hit.candidate.url),
        )

    # No open match anywhere. If we found ANY page that verified-but-closed,
    # surface the first one as a 'closed' result so the workflow can tell the
    # user the listing is closed alongside the default-package handoff.
    if closed_hits:
        first = closed_hits[0]
        return DiscoveryResult(
            url=first.candidate.url, method="closed",
            confidence=first.score.confidence,
            text=first.fetch.text, raw_html=first.fetch.raw_html,
            http_status=first.fetch.http_status,
            posting_closed=True,
            form_url=resolve_application_form_url(first.candidate.url),
        )

    return DiscoveryResult(url="", method="failed", confidence=0.0)
