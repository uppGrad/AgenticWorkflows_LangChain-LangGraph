"""Asset mapping (Step 6 rewrite).

Builds the categorised RequirementItem list that gate 1 surfaces to the user.
Replaces the previous heuristic that mapped requirements to user-uploaded
documents — only CVs are stored, so the `available=True` /
`tailoring_depth='light'` paths were dead in production.

Inputs (from state, in priority order):
  - state['form_fields']             jobs with non-empty extraction
  - state['normalized_requirements'] non-jobs and form-failed jobs

Output:
  - state['requirement_items']: List[RequirementItem dicts]
  - state['asset_mapping']:    same list (the JSONB column on
                               ApplicationSession is reused for stability;
                               only the dict shape inside changes)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from uppgrad_agentic.workflows.auto_apply._profile import resolve_profile  # noqa: F401
from uppgrad_agentic.workflows.auto_apply.schemas import RequirementItem
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document types the system can write/generate vs. ones the user must supply.
# These sets are also referenced by canonical_doc_types.classify_label and
# determine_requirements (must stay in lockstep).
# ---------------------------------------------------------------------------

_GENERATABLE: set[str] = {
    "CV",
    "Cover Letter",
    "Motivation Letter",
    "SOP",
    "Personal Statement",
    "Research Proposal",
    "Writing Sample",
    "References",
}

_USER_SUPPLIED: set[str] = {
    "Transcript",
    "English Proficiency Test",
    "Portfolio",
    "Certificate",
    "Passport",
    "Birth Certificate",
}


# ---------------------------------------------------------------------------
# Per-opportunity-type document defaults (floor when both inputs are empty)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, List[str]] = {
    "job": ["CV", "Cover Letter"],
    "masters": ["CV", "SOP"],
    "phd": ["CV", "SOP"],
    "scholarship": ["CV", "Cover Letter"],
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _doc_description(doc_type: str) -> str:
    if doc_type in _USER_SUPPLIED:
        return f"{doc_type} must be uploaded by you — the system cannot generate it."
    if doc_type in _GENERATABLE:
        return f"{doc_type} can be uploaded or auto-generated from your CV and profile."
    return f"{doc_type} requirement for this application."


def _build_from_form_fields(
    form_fields: List[Dict[str, Any]],
) -> List[RequirementItem]:
    """Group form fields into document / text / misc requirements.

    Documents: file fields → one item per canonical_document_type (deduped,
        keeping the required one when both required and optional appear).
    Text:      textareas / long-form questions → one item each, label as
        the question.
    Misc:      everything else (single virtual line, profile-fillable).
    """
    items: List[RequirementItem] = []
    item_id = 0

    # ── Documents: dedupe on canonical_document_type ────────────────────
    # Keyed by canonical_document_type (or label as fallback when classifier
    # produced no canonical type).
    doc_groups: Dict[str, Dict[str, Any]] = {}
    for idx, field in enumerate(form_fields):
        if field.get("field_type") != "file":
            continue
        canonical = (field.get("canonical_document_type") or "").strip()
        key = canonical or (field.get("label") or "").strip().lower() or f"_unkeyed_{idx}"
        existing = doc_groups.get(key)
        if existing is None:
            doc_groups[key] = {"index": idx, "field": field, "canonical": canonical}
        else:
            # Prefer the required version if any
            if field.get("required") and not existing["field"].get("required"):
                doc_groups[key] = {"index": idx, "field": field, "canonical": canonical}

    for group in doc_groups.values():
        field = group["field"]
        canonical = group["canonical"] or ""
        label = field.get("label") or canonical or "Document"
        items.append(
            RequirementItem(
                id=item_id,
                category="document",
                label=label,
                description=_doc_description(canonical) if canonical else f"{label} upload requirement.",
                field_type="file",
                required=bool(field.get("required")),
                document_type=canonical or None,
                question=None,
                form_field_index=group["index"],
            )
        )
        item_id += 1

    # ── Text: textareas + free-form questions ───────────────────────────
    for idx, field in enumerate(form_fields):
        ftype = field.get("field_type")
        source = field.get("expected_source")
        if ftype != "textarea" and not (ftype == "text" and source == "user_answer"):
            continue
        label = (field.get("label") or "").strip() or "Free-form question"
        items.append(
            RequirementItem(
                id=item_id,
                category="text",
                label=label,
                description=f"Free-form question on the application form: {label}",
                field_type=ftype,
                required=bool(field.get("required")),
                document_type=None,
                question=label,
                form_field_index=idx,
            )
        )
        item_id += 1

    # ── Misc: one collapsed line covering everything else ───────────────
    misc_field_indices: List[int] = []
    for idx, field in enumerate(form_fields):
        ftype = field.get("field_type")
        if ftype == "file":
            continue
        if ftype == "textarea":
            continue
        if ftype == "text" and field.get("expected_source") == "user_answer":
            continue
        misc_field_indices.append(idx)

    if misc_field_indices:
        items.append(
            RequirementItem(
                id=item_id,
                category="misc",
                label=f"Profile / identity fields ({len(misc_field_indices)})",
                description=(
                    "Other fields on the form (name, email, location, simple "
                    "selects). These can be auto-filled from your profile."
                ),
                field_type=None,
                required=any(form_fields[i].get("required") for i in misc_field_indices),
                document_type=None,
                question=None,
                form_field_index=None,
            )
        )
        item_id += 1

    return items


def _build_from_normalized_requirements(
    normalized_requirements: List[Dict[str, Any]],
) -> List[RequirementItem]:
    """Document-only items (no text/misc groups) for non-jobs and
    form-failed jobs.
    """
    items: List[RequirementItem] = []
    seen: set[str] = set()
    item_id = 0
    for req in normalized_requirements:
        if req.get("requirement_type") != "document":
            continue
        doc_type = (req.get("document_type") or "").strip()
        if not doc_type or doc_type in seen:
            continue
        seen.add(doc_type)
        items.append(
            RequirementItem(
                id=item_id,
                category="document",
                label=doc_type,
                description=_doc_description(doc_type),
                field_type=None,
                required=not bool(req.get("is_assumed", False)),
                document_type=doc_type,
                question=None,
                form_field_index=None,
            )
        )
        item_id += 1
    return items


def _build_defaults(opportunity_type: str) -> List[RequirementItem]:
    doc_types = _DEFAULTS.get(opportunity_type, _DEFAULTS["job"])
    items: List[RequirementItem] = []
    for i, doc_type in enumerate(doc_types):
        items.append(
            RequirementItem(
                id=i,
                category="document",
                label=doc_type,
                description=_doc_description(doc_type),
                field_type=None,
                required=True,
                document_type=doc_type,
                question=None,
                form_field_index=None,
            )
        )
    return items


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def asset_mapping(state: AutoApplyState) -> dict:
    updates = {"current_step": "asset_mapping", "step_history": ["asset_mapping"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    opportunity_type = state.get("opportunity_type", "")
    form_fields: List[Dict[str, Any]] = list(state.get("form_fields") or [])
    normalized_requirements: List[Dict[str, Any]] = list(state.get("normalized_requirements") or [])

    items: List[RequirementItem] = []

    if opportunity_type == "job" and form_fields:
        items = _build_from_form_fields(form_fields)
        if not items:
            # form_fields was non-empty but yielded nothing groupable —
            # fall through to normalized_requirements / defaults
            items = _build_from_normalized_requirements(normalized_requirements)
    else:
        items = _build_from_normalized_requirements(normalized_requirements)

    # Floor: nothing produced from either source — emit per-type defaults
    if not items:
        logger.info(
            "asset_mapping: no requirement items derived for %s — falling back to defaults",
            opportunity_type,
        )
        items = _build_defaults(opportunity_type)

    item_dicts = [item.model_dump() for item in items]

    logger.info(
        "asset_mapping: produced %d requirement_items (%s document, %s text, %s misc)",
        len(item_dicts),
        sum(1 for i in item_dicts if i["category"] == "document"),
        sum(1 for i in item_dicts if i["category"] == "text"),
        sum(1 for i in item_dicts if i["category"] == "misc"),
    )

    return {
        **updates,
        "requirement_items": item_dicts,
        # Reuse the asset_mapping JSONB column on ApplicationSession for
        # stability — backend stores whatever shape we put here.
        "asset_mapping": item_dicts,
    }
