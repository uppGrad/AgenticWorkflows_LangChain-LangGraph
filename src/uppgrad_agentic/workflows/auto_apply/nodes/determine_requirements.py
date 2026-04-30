from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.tools.canonical_doc_types import (
    ALL_TYPES as _CANONICAL_TYPES,
    classify_label,
)
from uppgrad_agentic.workflows.auto_apply.schemas import NormalizedRequirement
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical document-type tagging on file fields (jobs path)
# ---------------------------------------------------------------------------

class _CanonicalDocClassification(BaseModel):
    """LLM structured output: per-label canonical type."""
    label: str = Field(..., description="The exact label text passed in")
    canonical_document_type: str = Field(
        default="",
        description="One of the allowed canonical types, or empty string if no type fits",
    )


class _CanonicalDocBatch(BaseModel):
    classifications: List[_CanonicalDocClassification] = Field(default_factory=list)


def _llm_classify_labels(labels: List[str]) -> Dict[str, str]:
    """Single-batch LLM classification of file-field labels into canonical
    doc types. Returns a dict keyed by label. Empty/missing on LLM failure.
    """
    if not labels:
        return {}
    llm = get_llm()
    if llm is None:
        return {}

    allowed = ", ".join(f'"{t}"' for t in _CANONICAL_TYPES)
    system = (
        "You are classifying file-upload fields on job application forms into "
        "canonical document types. For each label, return the canonical type "
        "from this set:\n"
        f"  {allowed}\n"
        "If no type fits the label, return an empty string. Be strict — only "
        "return a type when the label clearly refers to that document. Do not "
        "invent new types."
    )
    user = "Labels to classify (return one classification per label):\n" + "\n".join(
        f"- {lbl}" for lbl in labels
    )

    structured = llm.with_structured_output(_CanonicalDocBatch)
    try:
        result: _CanonicalDocBatch = structured.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
    except Exception as exc:
        logger.warning("determine_requirements: LLM canonical-type batch failed — %s", exc)
        return {}

    out: Dict[str, str] = {}
    valid_types = set(_CANONICAL_TYPES)
    for item in result.classifications or []:
        canonical = (item.canonical_document_type or "").strip()
        if canonical and canonical in valid_types:
            out[item.label] = canonical
    return out


def _tag_canonical_document_types(form_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return form_fields with `canonical_document_type` populated on every
    file-type field. Heuristic first, LLM fallback for unmatched labels.
    """
    if not form_fields:
        return form_fields

    file_indices: List[int] = [
        i for i, f in enumerate(form_fields) if f.get("field_type") == "file"
    ]
    if not file_indices:
        return form_fields

    # Pass 1: heuristic
    needs_llm: List[int] = []
    for idx in file_indices:
        field = form_fields[idx]
        if field.get("canonical_document_type"):
            continue
        label = field.get("label", "") or ""
        canonical = classify_label(label)
        if canonical:
            field["canonical_document_type"] = canonical
        else:
            needs_llm.append(idx)

    # Pass 2: LLM batch for unmatched labels
    if needs_llm:
        labels = [form_fields[i].get("label", "") or "" for i in needs_llm]
        llm_results = _llm_classify_labels(labels)
        for idx, label in zip(needs_llm, labels):
            canonical = llm_results.get(label, "")
            if canonical:
                form_fields[idx]["canonical_document_type"] = canonical
            else:
                form_fields[idx]["canonical_document_type"] = ""

    return form_fields


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
    # Job: use scraped requirements if available, else fall back to defaults.
    # Also tag every file-type FormField in state['form_fields'] with a
    # canonical document type so asset_mapping / human_gate_1 can dedupe
    # and gate auto-generate visibility.
    # ------------------------------------------------------------------
    if opportunity_type == "job":
        # Internal-jobs short-circuit: discovery / scrape / form-fields all
        # skipped upstream by the graph router. Emit [CV, Cover Letter]
        # directly to match the `jobs_application` non-system fields
        # (resume_file + cover_letter).
        if opportunity_data.get("employer_id") == 1:
            logger.info("determine_requirements: internal job — emitting [CV, Cover Letter] defaults")
            return {**updates, "normalized_requirements": [r.model_dump() for r in _DEFAULTS["job"]]}

        # Canonical-type tagging on form_fields (in-place mutation captured
        # below in the returned dict).
        form_fields = list(state.get("form_fields") or [])
        if form_fields:
            form_fields = _tag_canonical_document_types(form_fields)
            updates = {**updates, "form_fields": form_fields}

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
