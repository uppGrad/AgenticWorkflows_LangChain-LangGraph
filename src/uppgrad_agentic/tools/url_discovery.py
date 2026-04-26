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
