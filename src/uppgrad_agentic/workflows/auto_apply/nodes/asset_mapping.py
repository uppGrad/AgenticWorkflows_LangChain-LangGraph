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
    # Keyed by canonical_document_type (or label+name as fallback when
    # classifier produced no canonical type). The name fallback is what
    # keeps Greenhouse Resume vs Cover Letter distinct when both file
    # inputs share the visible button caption "Attach": their `name`
    # attributes are still unique (resume / cover_letter).
    doc_groups: Dict[str, Dict[str, Any]] = {}
    for idx, field in enumerate(form_fields):
        if field.get("field_type") != "file":
            continue
        canonical = (field.get("canonical_document_type") or "").strip()
        label_key = (field.get("label") or "").strip().lower()
        name_key = (field.get("name") or "").strip().lower()
        key = (
            canonical
            or (f"{label_key}|{name_key}" if name_key else label_key)
            or f"_unkeyed_{idx}"
        )
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

    # ── Text: TRUE textareas only ──────────────────────────────────────
    # Spec: gate-1 surfaces individual cards ONLY for items that need an
    # upload-vs-generate-vs-skip decision. That means real essay-style
    # questions (Why us?, Additional Information). Everything else —
    # short text inputs, Yes/No comboboxes (`field_type="text"` +
    # `role="combobox"`), country pickers, sponsorship dropdowns, etc.
    # — collapses into the misc bucket and gets auto-derived in
    # application_tailoring's misc auto-fill pass for review at gate-2.
    #
    # We don't filter on expected_source here: a `<textarea>` is by
    # definition free-form prose input. If the extractor mislabels it
    # (e.g. "unknown" because the label was ambiguous), losing it to
    # misc — where the planner returns a useless mock answer — is
    # worse than treating every textarea as a text-category item.
    for idx, field in enumerate(form_fields):
        ftype = field.get("field_type")
        if ftype != "textarea":
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

    # ── Misc: everything that isn't a file or a textarea ───────────────
    misc_field_indices: List[int] = []
    for idx, field in enumerate(form_fields):
        ftype = field.get("field_type")
        if ftype == "file":
            continue
        if ftype == "textarea":
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


def _build_from_internal_application_form_spec(
    spec: List[Dict[str, Any]],
) -> List[RequirementItem]:
    """Build RequirementItems from a backend-supplied internal-application
    form spec.

    For internal jobs (employer_id == 1), the backend's opportunity-snapshot
    builder introspects the Django `jobs.Application` model and emits one
    spec entry per fillable field. This function turns those into
    RequirementItems the user reviews at gate 1.

    Spec entry shape (each entry, in order):
      {
        "key":           str,      # Application model column name
        "label":         str,      # human-readable
        "category":      "document" | "text" | "misc",
        "document_type": str,      # for category='document'
        "required":      bool,
        "field_type":    str,      # optional, hints the UI
        "description":   str,      # optional
      }

    The list order is significant — `RequirementItem.id` is assigned
    sequentially, which `finalize_internal_submission` later uses to map
    a RequirementItem back to its Application column via
    `opportunity_snapshot["application_form_spec"][item.id]["key"]`.
    """
    items: List[RequirementItem] = []
    for idx, entry in enumerate(spec or []):
        category = entry.get("category", "document")
        items.append(
            RequirementItem(
                id=idx,
                category=category,
                label=entry.get("label", f"Field {idx}"),
                description=entry.get("description", ""),
                field_type=entry.get("field_type"),
                required=bool(entry.get("required", False)),
                document_type=entry.get("document_type") if category == "document" else None,
                question=entry.get("label") if category == "text" else None,
                form_field_index=None,
            )
        )
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
    opportunity_data = state.get("opportunity_data") or {}
    form_fields: List[Dict[str, Any]] = list(state.get("form_fields") or [])
    normalized_requirements: List[Dict[str, Any]] = list(state.get("normalized_requirements") or [])

    # Internal jobs (employer_id == 1) receive an `application_form_spec`
    # in their opportunity_data describing the Django Application model's
    # fillable columns (and, in the future, employer-defined custom
    # fields per posting). When present, we honour it instead of falling
    # back to the static [CV, Cover Letter] defaults — so any new field
    # an employer adds just gets surfaced at gate 1 automatically.
    internal_spec: List[Dict[str, Any]] = list(
        opportunity_data.get("application_form_spec") or []
    )
    is_internal_job = (
        opportunity_type == "job" and opportunity_data.get("employer_id") == 1
    )

    items: List[RequirementItem] = []

    if is_internal_job and internal_spec:
        items = _build_from_internal_application_form_spec(internal_spec)
        logger.info(
            "asset_mapping: internal job — built %d items from application_form_spec",
            len(items),
        )
    elif opportunity_type == "job" and form_fields:
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
