from __future__ import annotations

import logging
from typing import List

from uppgrad_agentic.workflows.auto_apply.schemas import NormalizedRequirement
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default assumed requirements per opportunity type
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, List[NormalizedRequirement]] = {
    "job": [
        NormalizedRequirement(requirement_type="document", document_type="CV", is_assumed=True, confidence=0.9),
        NormalizedRequirement(requirement_type="document", document_type="Cover Letter", is_assumed=True, confidence=0.8),
    ],
    "masters": [
        NormalizedRequirement(requirement_type="document", document_type="CV", is_assumed=True, confidence=0.9),
        NormalizedRequirement(requirement_type="document", document_type="SOP", is_assumed=True, confidence=0.85),
    ],
    "phd": [
        NormalizedRequirement(requirement_type="document", document_type="CV", is_assumed=True, confidence=0.9),
        NormalizedRequirement(requirement_type="document", document_type="SOP", is_assumed=True, confidence=0.85),
    ],
    "scholarship": [
        NormalizedRequirement(requirement_type="document", document_type="CV", is_assumed=True, confidence=0.9),
        NormalizedRequirement(requirement_type="document", document_type="Cover Letter", is_assumed=True, confidence=0.8),
    ],
}


def _parse_program_requirements(data: dict) -> List[NormalizedRequirement]:
    """Parse requirements from the data json field of a program record."""
    requirements: List[NormalizedRequirement] = []
    reqs = data.get("requirements") or {}

    other = (reqs.get("other") or "").lower()

    doc_keywords = {
        "cv": ("document", "CV"),
        "curriculum vitae": ("document", "CV"),
        "resume": ("document", "CV"),
        "sop": ("document", "SOP"),
        "statement of purpose": ("document", "SOP"),
        "personal statement": ("document", "Personal Statement"),
        "cover letter": ("document", "Cover Letter"),
        "transcript": ("document", "Transcript"),
        "reference": ("document", "References"),
        "recommendation": ("document", "References"),
        "research proposal": ("document", "Research Proposal"),
        "writing sample": ("document", "Writing Sample"),
        "portfolio": ("document", "Portfolio"),
    }

    seen: set[str] = set()
    for keyword, (req_type, doc_type) in doc_keywords.items():
        if keyword in other and doc_type not in seen:
            requirements.append(
                NormalizedRequirement(
                    requirement_type=req_type,
                    document_type=doc_type,
                    is_assumed=False,
                    confidence=0.85,
                )
            )
            seen.add(doc_type)

    if reqs.get("english"):
        requirements.append(
            NormalizedRequirement(
                requirement_type="language",
                document_type="English Proficiency Test",
                is_assumed=False,
                confidence=0.9,
            )
        )

    # Baseline: always include CV if not already present
    if "CV" not in seen:
        requirements.insert(
            0,
            NormalizedRequirement(
                requirement_type="document",
                document_type="CV",
                is_assumed=True,
                confidence=0.9,
            ),
        )

    return requirements


def _parse_scholarship_requirements(data: dict) -> List[NormalizedRequirement]:
    """Parse requirements from the data json field of a scholarship record."""
    requirements: List[NormalizedRequirement] = []
    required_docs = data.get("required_documents") or []

    doc_map = {
        "cv": "CV",
        "curriculum vitae": "CV",
        "resume": "CV",
        "personal statement": "Personal Statement",
        "cover letter": "Cover Letter",
        "motivation letter": "Cover Letter",
        "transcript": "Transcript",
        "reference": "References",
        "recommendation": "References",
        "portfolio": "Portfolio",
    }

    seen: set[str] = set()
    for raw_doc in required_docs:
        lower = raw_doc.lower()
        matched = False
        for keyword, normalized in doc_map.items():
            if keyword in lower and normalized not in seen:
                requirements.append(
                    NormalizedRequirement(
                        requirement_type="document",
                        document_type=normalized,
                        is_assumed=False,
                        confidence=0.9,
                    )
                )
                seen.add(normalized)
                matched = True
                break
        if not matched:
            # Preserve unrecognized docs verbatim
            requirements.append(
                NormalizedRequirement(
                    requirement_type="document",
                    document_type=raw_doc,
                    is_assumed=False,
                    confidence=0.75,
                )
            )
            seen.add(raw_doc)

    # Baseline fallback if nothing was parsed
    if not requirements:
        return [r.model_dump() for r in _DEFAULTS["scholarship"]]  # type: ignore[return-value]

    return requirements


def determine_requirements(state: AutoApplyState) -> dict:
    updates = {"current_step": "determine_requirements", "step_history": ["determine_requirements"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    opportunity_type = state.get("opportunity_type", "")
    opportunity_data = state.get("opportunity_data") or {}

    # ------------------------------------------------------------------
    # Job: use scraped requirements if available, else fall back to defaults
    # ------------------------------------------------------------------
    if opportunity_type == "job":
        scraped = state.get("scraped_requirements") or {}
        scraped_reqs = scraped.get("requirements") or []
        scrape_status = scraped.get("status", "failed")

        if scrape_status == "full" and scraped_reqs:
            logger.info("determine_requirements: using fully scraped requirements (%d items)", len(scraped_reqs))
            return {**updates, "normalized_requirements": scraped_reqs}

        if scrape_status == "partial" and scraped_reqs:
            # Merge scraped with defaults, deduplicating by document_type
            seen_types: set[str] = {r.get("document_type", "") for r in scraped_reqs}
            merged = list(scraped_reqs)
            for default in _DEFAULTS["job"]:
                if default.document_type not in seen_types:
                    merged.append(default.model_dump())
            logger.info(
                "determine_requirements: partial scrape — merged %d scraped + %d defaults",
                len(scraped_reqs),
                len(merged) - len(scraped_reqs),
            )
            return {**updates, "normalized_requirements": merged}

        logger.info("determine_requirements: scrape failed/empty — using assumed defaults for job")
        return {**updates, "normalized_requirements": [r.model_dump() for r in _DEFAULTS["job"]]}

    # ------------------------------------------------------------------
    # Masters / PhD: parse from data json
    # ------------------------------------------------------------------
    if opportunity_type in ("masters", "phd"):
        data = opportunity_data.get("data") or {}
        if data:
            parsed = _parse_program_requirements(data)
            if parsed:
                logger.info("determine_requirements: parsed %d requirements from program data json", len(parsed))
                return {**updates, "normalized_requirements": [r.model_dump() if hasattr(r, "model_dump") else r for r in parsed]}

        logger.info("determine_requirements: no parseable data json — using assumed defaults for %s", opportunity_type)
        return {**updates, "normalized_requirements": [r.model_dump() for r in _DEFAULTS[opportunity_type]]}

    # ------------------------------------------------------------------
    # Scholarship: parse from data json
    # ------------------------------------------------------------------
    if opportunity_type == "scholarship":
        data = opportunity_data.get("data") or {}
        if data:
            parsed = _parse_scholarship_requirements(data)
            if parsed:
                logger.info("determine_requirements: parsed %d requirements from scholarship data json", len(parsed))
                return {**updates, "normalized_requirements": [r.model_dump() if hasattr(r, "model_dump") else r for r in parsed]}

        logger.info("determine_requirements: no parseable scholarship data — using assumed defaults")
        return {**updates, "normalized_requirements": [r.model_dump() for r in _DEFAULTS["scholarship"]]}

    # Unknown type — already validated upstream, but guard anyway
    return {
        **updates,
        "result": {
            "status": "error",
            "error_code": "INVALID_OPPORTUNITY_TYPE",
            "user_message": f"Cannot determine requirements for unknown opportunity type: {opportunity_type}",
        },
    }
