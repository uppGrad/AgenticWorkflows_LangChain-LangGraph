"""Evaluate the quality of the page returned by the apply-URL discovery step.

This node is a STATUS-ONLY second-pass on the discovered page contents:
confirms the page actually serves a real, open job application and not a
broken intermediate (404, closed posting, anti-bot wall slipped past
discovery's gates, marketing landing page, etc.).

It does NOT extract requirements from the page. Form requirements come from
`extract_form_fields` (real DOM <input>/<select>/<textarea>) — JD prose on
a posting page is recruiter copy and is not authoritative for what the
application form actually collects.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.workflows.auto_apply.schemas import ScrapeStatus
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)

_MAX_CONTENT_FOR_LLM = 8_000

_RICH_SIGNALS = [
    "apply now", "apply online", "submit application", "upload", "cv", "resume",
    "cover letter", "requirements", "qualifications", "responsibilities",
    "how to apply", "application form", "attach", "attach your",
]

_THIN_SIGNALS = [
    "404", "page not found", "access denied", "javascript required",
    "enable javascript", "cloudflare", "robot", "captcha",
]


class _ScrapeStatusOnly(BaseModel):
    """Status-only LLM structured output. Requirement extraction was removed
    on purpose — see module docstring."""
    status: ScrapeStatus = Field(..., description="full | partial | failed")
    confidence: float = Field(..., ge=0.0, le=1.0)


SYSTEM = """You are verifying the quality of a discovered job application page.

Given the raw page text, classify whether the page is a real, fetchable
application/listing page or a broken intermediate.

Return a JSON object with:
- status: "full" | "partial" | "failed"
  - full:    page clearly shows a real job listing with apply affordances
  - partial: real job content present but limited (login wall, partial render)
  - failed:  no useful content (404, anti-bot block, marketing landing,
             closed posting page)
- confidence: 0.0–1.0

DO NOT enumerate or extract document requirements from the page text. Form
requirements are read from the DOM, not the marketing copy.

Return raw JSON only, no markdown fences."""


def _heuristic_status(raw_content: str) -> _ScrapeStatusOnly:
    """Status-only fallback when no LLM is configured."""
    text_lower = raw_content.lower()
    thin_hits = sum(1 for s in _THIN_SIGNALS if s in text_lower)
    rich_hits = sum(1 for s in _RICH_SIGNALS if s in text_lower)

    if thin_hits >= 2 or len(raw_content.strip()) < 500:
        return _ScrapeStatusOnly(status="failed", confidence=0.85)
    if rich_hits >= 3:
        return _ScrapeStatusOnly(status="full", confidence=min(0.65 + 0.05 * rich_hits, 0.85))
    if rich_hits >= 1:
        return _ScrapeStatusOnly(status="partial", confidence=min(0.45 + 0.05 * rich_hits, 0.70))
    return _ScrapeStatusOnly(status="failed", confidence=0.60)


def _llm_status(raw_content: str, llm) -> _ScrapeStatusOnly | None:
    snippet = raw_content[:_MAX_CONTENT_FOR_LLM]
    structured = llm.with_structured_output(_ScrapeStatusOnly)
    try:
        return structured.invoke([
            SystemMessage(content=SYSTEM),
            HumanMessage(content=f"Page content:\n\n{snippet}"),
        ])
    except Exception as exc:
        logger.warning("evaluate_scrape: LLM status check failed — %s", exc)
        return None


def evaluate_scrape(state: AutoApplyState) -> dict:
    updates = {"current_step": "evaluate_scrape", "step_history": ["evaluate_scrape"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    if state.get("opportunity_type") != "job":
        return updates

    scraped = dict(state.get("scraped_requirements") or {})
    source = scraped.get("source", "")
    raw_content = scraped.get("raw_content", "")

    # Already failed upstream — don't burn an LLM call to confirm.
    if scraped.get("status") == "failed" and not raw_content:
        return {**updates, "scraped_requirements": scraped}

    llm = get_llm()
    if llm is not None:
        result = _llm_status(raw_content, llm) or _heuristic_status(raw_content)
    else:
        result = _heuristic_status(raw_content)

    logger.info(
        "evaluate_scrape: source=%s status=%s confidence=%.2f",
        source, result.status, result.confidence,
    )

    # Preserve raw_content / raw_html / http_status / source on the dict so
    # downstream nodes (extract_form_fields fallback path, telemetry) can
    # still see them. Only the status verdict is replaced.
    return {
        **updates,
        "scraped_requirements": {
            **scraped,
            "status": result.status,
            "confidence": result.confidence,
            "source": source,
        },
    }
