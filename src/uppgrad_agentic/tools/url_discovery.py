from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Literal, Optional

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

Tier = Literal["ats", "careers", "generic"]

_TIER_THRESHOLDS = {
    "ats": 0.70,
    "careers": 0.65,
    "generic": 0.80,
}
_TITLE_FUZZY_MIN = 85


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


def score_candidate(inputs: VerifyInputs) -> VerificationScore:
    reasons: List[str] = []
    job = inputs.job
    job_title = (job.get("title") or "").strip()
    job_company = (job.get("company") or "").strip()

    haystack = f"{inputs.candidate_title}\n{inputs.candidate_text[:2000]}"
    title_score = fuzz.partial_ratio(job_title.lower(), haystack.lower()) if job_title else 0
    if title_score < _TITLE_FUZZY_MIN:
        return VerificationScore(passed=False, confidence=0.0,
                                 reasons=[f"title fuzzy {title_score} < {_TITLE_FUZZY_MIN}"])

    if inputs.tier != "careers":
        company_in_url = bool(job_company) and job_company.lower().replace(" ", "") in inputs.candidate_url.lower()
        company_in_text = bool(job_company) and re.search(re.escape(job_company), inputs.candidate_text, re.IGNORECASE) is not None
        if not (company_in_url or company_in_text):
            return VerificationScore(passed=False, confidence=0.0,
                                     reasons=[f"company '{job_company}' not present"])

    confidence = 0.85
    reasons.append(f"title fuzzy {title_score}")

    job_posted = _parse_iso_or_none(job.get("posted_time"))
    if inputs.candidate_posted_at and job_posted:
        delta_days = abs((inputs.candidate_posted_at - job_posted).days)
        if delta_days > 180:
            confidence -= 0.20
            reasons.append(f"freshness off by {delta_days}d")

    job_loc_tokens = {tok.strip().lower() for tok in (job.get("location") or "").split(",") if tok.strip()}
    if job_loc_tokens:
        loc_hit = any(tok in inputs.candidate_text.lower() for tok in job_loc_tokens)
        if not loc_hit:
            confidence -= 0.10
            reasons.append("location not on page")

    confidence = max(0.0, min(1.0, confidence))
    threshold = _TIER_THRESHOLDS[inputs.tier]
    return VerificationScore(passed=confidence >= threshold, confidence=confidence, reasons=reasons)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

from urllib.parse import urlparse

from uppgrad_agentic.tools.search import SearchProvider, SearchResult
from uppgrad_agentic.tools.web_fetcher import fetch_url_with_fallback


_ATS_DOMAINS = [
    "greenhouse.io", "lever.co", "ashbyhq.com", "workable.com",
    "smartrecruiters.com", "myworkdayjobs.com", "bamboohr.com",
    "jobvite.com", "recruitee.com",
]


@dataclass
class DiscoveryResult:
    url: str
    method: str            # 'url_direct' | 'ats' | 'careers' | 'generic' | 'failed'
    confidence: float


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


def _build_careers_query(title: str, company_url: Optional[str]) -> Optional[str]:
    domain = _extract_company_domain(company_url)
    if not domain:
        return None
    return f'"{title}" site:{domain}'


def _build_generic_query(title: str, company: str) -> str:
    return f'"{title}" "{company}" apply'


def _verify_one(candidate: SearchResult, job: dict, tier: Tier) -> Optional[VerificationScore]:
    fetch = fetch_url_with_fallback(candidate.url)
    if not fetch.success:
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
    return score if score.passed else None


def _try_tier(candidates: List[SearchResult], job: dict, tier: Tier):
    for cand in candidates:
        verified = _verify_one(cand, job, tier)
        if verified is not None:
            return cand, verified
    return None


def discover_apply_url(
    job: dict,
    search_provider: Optional[SearchProvider],
) -> DiscoveryResult:
    """Synchronous discovery orchestrator.

    Caching (Phase 6) lives in the backend adapter — agentic stays DB-free.
    Callers that want cached results should consult the cache *before* calling
    this function and skip if a hit was found.
    """
    url_direct = (job.get("url_direct") or "").strip()
    if url_direct:
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
        cand, score = hit
        return DiscoveryResult(url=cand.url, method="ats", confidence=score.confidence)

    # Tier 2: Careers
    careers_q = _build_careers_query(title, job.get("company_url"))
    if careers_q:
        careers_results = search_provider.search(careers_q, count=3)
        hit = _try_tier(careers_results, job, "careers")
        if hit:
            cand, score = hit
            return DiscoveryResult(url=cand.url, method="careers", confidence=score.confidence)

    # Tier 3: Generic
    generic_results = search_provider.search(_build_generic_query(title, company), count=3)
    hit = _try_tier(generic_results, job, "generic")
    if hit:
        cand, score = hit
        return DiscoveryResult(url=cand.url, method="generic", confidence=score.confidence)

    return DiscoveryResult(url="", method="failed", confidence=0.0)
