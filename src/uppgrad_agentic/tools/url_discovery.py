from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Literal, Optional, Tuple
from urllib.parse import urlparse

from rapidfuzz import fuzz

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
    method: str            # 'url_direct' | 'ats' | 'careers' | 'generic' | 'failed'
    confidence: float
    text: str = ""         # verified page content; populated when verification fetched
    http_status: int = 0


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


def _verify_one(
    candidate: SearchResult, job: dict, tier: Tier,
) -> Optional[Tuple[VerificationScore, FetchResult]]:
    """Fetch + verify a candidate. Reject thin pages outright before scoring.

    Returns (score, fetch) on a verified accept; None on any reject. The
    FetchResult is propagated so the orchestrator can hand its `text` forward
    instead of refetching the same URL during scrape.
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
    return score, fetch


def _try_tier(
    candidates: List[SearchResult], job: dict, tier: Tier,
) -> Optional[Tuple[SearchResult, VerificationScore, FetchResult]]:
    for cand in candidates:
        verified = _verify_one(cand, job, tier)
        if verified is not None:
            score, fetch = verified
            return cand, score, fetch
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
        return DiscoveryResult(url=url_direct, method="url_direct", confidence=1.0)

    if search_provider is None:
        return DiscoveryResult(url="", method="failed", confidence=0.0)

    title = (job.get("title") or "").strip()
    company = (job.get("company") or "").strip()
    if not title or not company:
        return DiscoveryResult(url="", method="failed", confidence=0.0)

    # Tier 1: ATS
    ats_results = search_provider.search(_build_ats_query(title, company), count=3)
    hit = _try_tier(ats_results, job, "ats")
    if hit:
        cand, score, fetch = hit
        return DiscoveryResult(
            url=cand.url, method="ats", confidence=score.confidence,
            text=fetch.text, http_status=fetch.http_status,
        )

    # Tier 2: Careers
    careers_q = _build_careers_query(title, job.get("company_url"))
    if careers_q:
        careers_results = search_provider.search(careers_q, count=3)
        hit = _try_tier(careers_results, job, "careers")
        if hit:
            cand, score, fetch = hit
            return DiscoveryResult(
                url=cand.url, method="careers", confidence=score.confidence,
                text=fetch.text, http_status=fetch.http_status,
            )

    # Tier 3: Generic
    generic_results = search_provider.search(_build_generic_query(title, company), count=3)
    hit = _try_tier(generic_results, job, "generic")
    if hit:
        cand, score, fetch = hit
        return DiscoveryResult(
            url=cand.url, method="generic", confidence=score.confidence,
            text=fetch.text, http_status=fetch.http_status,
        )

    return DiscoveryResult(url="", method="failed", confidence=0.0)
