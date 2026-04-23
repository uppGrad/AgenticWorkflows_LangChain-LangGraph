from __future__ import annotations

import logging
import re
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.workflows.auto_apply.schemas import NormalizedRequirement, ScrapeResult
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)

_MAX_CONTENT_FOR_LLM = 8_000

# Signals that suggest the page has real application content
_RICH_SIGNALS = [
    "apply now", "apply online", "submit application", "upload", "cv", "resume",
    "cover letter", "requirements", "qualifications", "responsibilities",
    "how to apply", "application form", "attach", "attach your",
]

# Signals that suggest the page is mostly navigation/marketing, not an application page
_THIN_SIGNALS = [
    "404", "page not found", "access denied", "javascript required",
    "enable javascript", "cloudflare", "robot", "captcha",
]

_DEFAULT_JOB_REQUIREMENTS = [
    NormalizedRequirement(
        requirement_type="document",
        document_type="CV",
        is_assumed=True,
        confidence=0.9,
    ),
    NormalizedRequirement(
        requirement_type="document",
        document_type="Cover Letter",
        is_assumed=True,
        confidence=0.8,
    ),
]


SYSTEM = """You are assessing the quality of a scraped job application page.

Given the raw page text, return a JSON object with:
- status: "full" | "partial" | "failed"
  - full: the page clearly shows the application form or explicit required documents list
  - partial: some application info present but incomplete or behind a login wall
  - failed: no useful application content (404, Cloudflare block, login-only wall, marketing page)
- confidence: 0.0–1.0 (how confident you are in the status assessment)
- requirements: list of objects, each with:
    - requirement_type: "document" | "eligibility" | "language" | "other"
    - document_type: e.g. "CV", "Cover Letter", "Portfolio", "References"
    - is_assumed: false (these are scraped, not assumed)
    - confidence: 0.0–1.0

Only include requirements you can clearly identify. If status is failed, return an empty list.
Return raw JSON only, no markdown fences."""


def _heuristic_evaluate(raw_content: str) -> ScrapeResult:
    """Assess scrape quality without LLM."""
    text_lower = raw_content.lower()

    thin_hits = sum(1 for s in _THIN_SIGNALS if s in text_lower)
    rich_hits = sum(1 for s in _RICH_SIGNALS if s in text_lower)

    if thin_hits >= 2 or len(raw_content.strip()) < 500:
        return ScrapeResult(
            status="failed",
            requirements=[],
            confidence=0.85,
            source="",
        )

    requirements: List[NormalizedRequirement] = []

    cv_patterns = [r"\bcv\b", r"\bresume\b", r"\bcurriculum vitae\b"]
    if any(re.search(p, text_lower) for p in cv_patterns):
        requirements.append(
            NormalizedRequirement(
                requirement_type="document",
                document_type="CV",
                is_assumed=False,
                confidence=0.8,
            )
        )

    cover_patterns = [r"\bcover letter\b", r"\bcovering letter\b", r"\bmotivation letter\b"]
    if any(re.search(p, text_lower) for p in cover_patterns):
        requirements.append(
            NormalizedRequirement(
                requirement_type="document",
                document_type="Cover Letter",
                is_assumed=False,
                confidence=0.8,
            )
        )

    portfolio_patterns = [r"\bportfolio\b", r"\bwork samples?\b"]
    if any(re.search(p, text_lower) for p in portfolio_patterns):
        requirements.append(
            NormalizedRequirement(
                requirement_type="document",
                document_type="Portfolio",
                is_assumed=False,
                confidence=0.7,
            )
        )

    if rich_hits >= 3 and requirements:
        status = "full"
        confidence = min(0.65 + 0.05 * rich_hits, 0.85)
    elif rich_hits >= 1 or requirements:
        status = "partial"
        confidence = min(0.45 + 0.05 * rich_hits, 0.70)
    else:
        status = "failed"
        confidence = 0.60

    return ScrapeResult(
        status=status,
        requirements=requirements,
        confidence=confidence,
        source="",
    )


def _llm_evaluate(raw_content: str, llm) -> ScrapeResult | None:
    """Use LLM to assess scrape quality and extract requirements."""
    import json

    snippet = raw_content[:_MAX_CONTENT_FOR_LLM]
    structured = llm.with_structured_output(ScrapeResult)
    try:
        result: ScrapeResult = structured.invoke([
            SystemMessage(content=SYSTEM),
            HumanMessage(content=f"Page content:\n\n{snippet}"),
        ])
        return result
    except Exception as exc:
        logger.warning("evaluate_scrape: LLM evaluation failed — %s", exc)
        return None


def evaluate_scrape(state: AutoApplyState) -> dict:
    updates = {"current_step": "evaluate_scrape", "step_history": ["evaluate_scrape"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    # Only runs for job opportunities
    if state.get("opportunity_type") != "job":
        return updates

    scraped = state.get("scraped_requirements") or {}
    source = scraped.get("source", "")

    # If scraping itself already failed (no raw content), surface that result directly
    if scraped.get("status") == "failed" and not scraped.get("raw_content"):
        return {
            **updates,
            "scraped_requirements": {
                **scraped,
                "requirements": [r.model_dump() for r in _DEFAULT_JOB_REQUIREMENTS],
            },
        }

    raw_content = scraped.get("raw_content", "")

    llm = get_llm()
    if llm is not None:
        result = _llm_evaluate(raw_content, llm)
        if result is None:
            result = _heuristic_evaluate(raw_content)
    else:
        result = _heuristic_evaluate(raw_content)

    # If we got nothing useful, fall back to assumed defaults
    if result.status == "failed" or not result.requirements:
        requirements = _DEFAULT_JOB_REQUIREMENTS
        logger.info(
            "evaluate_scrape: status=%s — falling back to assumed default requirements",
            result.status,
        )
    else:
        requirements = result.requirements

    return {
        **updates,
        "scraped_requirements": {
            "status": result.status,
            "requirements": [r.model_dump() for r in requirements],
            "confidence": result.confidence,
            "source": source,
        },
    }
