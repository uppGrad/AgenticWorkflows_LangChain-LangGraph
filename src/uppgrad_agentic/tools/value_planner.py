"""Compute fill values for the application form's fields.

Pure function over (form_fields, profile, tailored_documents, opportunity).
No LangGraph imports, no global state. Returns a list of FormFieldFillPlan
records, one per FormField, that the playwright_filler can execute.

Decision rules (per field, in priority order):

1. file → use a path from `tailored_documents` matching the field's
          document type (Resume → CV; Cover Letter → "Cover Letter";
          Portfolio → "Portfolio"; etc.). When no path is available,
          skip with reason="no_document_available".

2. date → today's date (computed).

3. profile-mappable label → look up in the profile snapshot via
   `tools.profile_lookup`. Works for First/Last Name, Email, Phone,
   Country, City, Location, LinkedIn, GitHub, Website.

4. select / radio with options → pick the first non-placeholder option.
   When a field has expected_source=user_profile, we still try the
   profile lookup first (e.g. Country dropdown can take "Turkey" from
   profile.country and the click-pick-option layer will match the option).

5. checkbox → check it when required, leave optional ones unchecked.

6. textarea / user_answer free-text → "[Mock answer — <label>]" placeholder
   in dry-run mode (the only mode this PoC supports for now).

7. Anything else → skip with reason describing what was unhandled.

The "mock_answer" placeholder behavior is intentional for the dry-run path.
A future iteration will replace it with an LLM draft using the profile +
opportunity context as input.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from uppgrad_agentic.tools.profile_lookup import lookup as profile_lookup
from uppgrad_agentic.workflows.auto_apply.schemas import (
    FormField,
    FormFieldFillPlan,
)


# Map FormField.document_type-ish labels → keys in tailored_documents dict.
# tailored_documents shape from the existing application_tailoring node:
#   { "CV": {"content": "...", "tailoring_depth": "..."},
#     "Cover Letter": {"content": "..."},
#     ... }
# The `content` is text. The fill plan needs a FILE PATH, so the caller is
# responsible for materializing tailored content to disk first; here we just
# emit a placeholder path that points at the per-doc-type entry. The
# playwright_filler / adapter will resolve it to a real path before fill.
_DOC_LABEL_HINTS = (
    ("resume", "CV"),
    ("cv", "CV"),
    ("curriculum vitae", "CV"),
    ("cover letter", "Cover Letter"),
    ("statement of purpose", "SOP"),
    ("personal statement", "Personal Statement"),
    ("portfolio", "Portfolio"),
    ("writing sample", "Writing Sample"),
    ("research proposal", "Research Proposal"),
    ("transcript", "Transcript"),
)


def _doc_key_for_label(label: str) -> Optional[str]:
    label_l = (label or "").lower()
    for hint, key in _DOC_LABEL_HINTS:
        if hint in label_l:
            return key
    return None


def _first_real_option(options: List[str]) -> Optional[str]:
    placeholders = ("choose…", "choose...", "select", "select an option",
                    "select...", "—", "-", "")
    for opt in options or []:
        if opt and opt.lower().strip() not in placeholders:
            return opt
    return None


def _mock_answer(label: str) -> str:
    short = (label or "")[:60]
    return f"[Mock answer — {short}]"


def plan_field_value(
    field: FormField | Dict[str, Any],
    profile: Dict[str, str],
    tailored_documents: Dict[str, Any],
    opportunity_data: Dict[str, Any],
) -> FormFieldFillPlan:
    """Compute the fill plan for one FormField. Pure, no I/O."""
    # Accept dicts (DB-deserialized) and FormField pydantic models alike.
    if isinstance(field, FormField):
        f = field
    else:
        f = FormField(**field)

    field_type = f.field_type
    label = f.label
    options = f.options or []

    # File uploads — caller (adapter) supplies a path via tailored_documents
    # if available. We emit a sentinel path that the adapter swaps out OR
    # encode the doc-type so the adapter can resolve.
    if field_type == "file":
        doc_key = _doc_key_for_label(label) or f.accepts_file and "CV" or None
        if doc_key and doc_key in (tailored_documents or {}):
            entry = tailored_documents[doc_key]
            path_hint = entry.get("file_path") if isinstance(entry, dict) else None
            if path_hint:
                return FormFieldFillPlan(
                    field=f, value=path_hint, status="filled",
                    source="user_document", reason=f"tailored_documents[{doc_key!r}].file_path",
                )
            # Fall through: tailored content exists as text but no path yet
            return FormFieldFillPlan(
                field=f, value=f"<<doc:{doc_key}>>", status="filled",
                source="user_document",
                reason=f"adapter_resolves: tailored_documents[{doc_key!r}].content → tmpfile",
            )
        return FormFieldFillPlan(
            field=f, value="", status="skipped", source="no_value",
            reason=f"no_document_for_label={label!r}",
        )

    # Computed: today's date
    if field_type == "date":
        return FormFieldFillPlan(
            field=f, value=date.today().isoformat(), status="filled",
            source="computed", reason="today",
        )

    # Profile lookup — try whenever the label hints at a profile attribute,
    # regardless of expected_source (the LLM's classification is noisy).
    pv = profile_lookup(label, profile or {})
    if pv is not None:
        return FormFieldFillPlan(
            field=f, value=pv, status="filled",
            source="user_profile", reason=f"profile[{label!r}]",
        )

    # Select / radio with concrete options
    if field_type in ("select", "radio") and options:
        opt = _first_real_option(options)
        if opt is not None:
            return FormFieldFillPlan(
                field=f, value=opt, status="filled",
                source="mock", reason="first_real_option",
            )
        return FormFieldFillPlan(
            field=f, value="", status="skipped", source="no_value",
            reason="no_real_option",
        )

    # Checkbox: only check when required
    if field_type == "checkbox":
        if f.required:
            return FormFieldFillPlan(
                field=f, value="true", status="filled",
                source="mock", reason="required_checkbox",
            )
        return FormFieldFillPlan(
            field=f, value="", status="skipped", source="no_value",
            reason="optional_checkbox",
        )

    # Textarea / free-text user_answer → mock placeholder
    if field_type == "textarea" or f.expected_source == "user_answer":
        return FormFieldFillPlan(
            field=f, value=_mock_answer(label), status="filled",
            source="mock", reason="mock_answer",
        )

    # Last-resort skip
    return FormFieldFillPlan(
        field=f, value="", status="skipped", source="no_value",
        reason=f"unhandled type={field_type!r} src={f.expected_source!r}",
    )


def compute_form_values(
    form_fields: List[FormField | Dict[str, Any]],
    profile: Dict[str, str],
    tailored_documents: Dict[str, Any],
    opportunity_data: Dict[str, Any],
) -> List[FormFieldFillPlan]:
    """Build the full fill plan. Caller passes raw DB-shape dicts or pydantic
    models; we accept either."""
    return [
        plan_field_value(f, profile or {}, tailored_documents or {}, opportunity_data or {})
        for f in (form_fields or [])
    ]
