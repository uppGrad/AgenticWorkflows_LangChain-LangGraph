"""Compute fill values for the application form's fields.

Pure function over (form_fields, profile, tailored_documents, opportunity)
plus optional gate-1 outputs (tailored_answers, requirement_items,
human_review_1). No LangGraph imports, no global state. Returns a list of
FormFieldFillPlan records, one per FormField, that the playwright_filler
can execute.

Decision rules (per field, in priority order):

0. Per-field skip/ignore (user gate-1 verdict) — short-circuit:
   When `human_review_1.requirements[<owning_id>].choice ∈ {skip, ignore_for_now}`,
   emit `status=skipped, source=user_skipped` regardless of what the field
   would otherwise resolve to. The session-level kill-switch (any
   `required+ignore_for_now` → don't run auto-fill at all) lives in the
   adapter, not here — by the time we're called, that decision has been
   made and we're proceeding with whatever subset of fields the user
   wants filled.

1. Misc + `misc_strategy=ignore`:
   Form fields not pointed to by any RequirementItem (i.e. the misc-bucket
   members) are skipped wholesale when the user picks `misc_strategy=ignore`
   at gate 1.

2. file → use a path from `tailored_documents` matching the field's
          document type. When no path is available, skip with
          reason="no_document_available".

3. date → today's date (computed).

4. profile-mappable label → look up in the profile snapshot via
   `tools.profile_lookup`. Works for First/Last Name, Email, Phone,
   Country, City, Location, LinkedIn, GitHub, Website.

5. select / radio with options → pick the first non-placeholder option
   when no profile match.

6. checkbox → check it when required, leave optional ones unchecked.

7. textarea / user_answer free-text:
   a. Prefer `tailored_answers[str(field_index)]` (LLM-drafted real answer
      from gate-1 auto_generate flow).
   b. Fall back to `[Mock answer — <label>]` placeholder when no tailored
      answer exists.

8. Anything else → skip with reason describing what was unhandled.

────────────────────────────────────────────────────────────────────────
Misc handling — option B (current implementation)
────────────────────────────────────────────────────────────────────────

`misc_strategy=auto_fill` runs the per-field rules above (profile lookup,
sensible-default option, required-checkbox, etc.). Fields the planner
can't resolve via any rule become `status=skipped, source=no_value` and
appear in `FormFillResult.reports`. The adapter / frontend surface those
unfilled fields to the user post-fill ("we filled X/Y; please review the
rest"). No pre-fill classification needed.

Other options considered (for future iteration):

  Option A — Pre-classify each misc field at gate 1 and PROMOTE
  non-generatable ones to category=text RequirementItems with their own
  per-id choice. Misc would only contain generatable fields. Pro: cleaner
  UX, explicit user control over each unknown. Con: adds a generatability
  classifier (profile-lookup pre-pass + LLM fallback) before gate 1, slowing
  gate-1 latency and burning tokens before the user has even seen the page.

  Option C — Add a finer `misc_strategy` value:
    `auto_fill_generatable_only` (default) — fill profile-mappable + sensible-default;
    skip unknowns; surface skipped count at gate 2.
    `auto_fill_all` — try everything (mock answers for unknowns).
    `ignore` — current ignore behavior.
  Pro: session-level toggle. Con: user can't single-out individual unknowns.

Migration to A or C is straightforward if real-data shows users want
finer control — both can layer on top of Option B's plumbing without
changes to the FormFieldFillPlan / FormFillResult contracts.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Set

from uppgrad_agentic.tools.profile_lookup import lookup as profile_lookup
from uppgrad_agentic.workflows.auto_apply.schemas import (
    FormField,
    FormFieldFillPlan,
)


# Map FormField.document_type-ish labels → keys in tailored_documents dict.
# tailored_documents shape from application_tailoring:
#   { "CV": {"content": "...", "tailoring_depth": "..."},
#     "Cover Letter": {"content": "..."},
#     ... }
# The `content` is text. The fill plan needs a FILE PATH, so the caller is
# responsible for materializing tailored content to disk first.
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


def _build_field_choice_map(
    requirement_items: Optional[List[Dict[str, Any]]],
    human_review_1: Optional[Dict[str, Any]],
) -> Dict[int, str]:
    """form_field_index → user's gate-1 choice string.

    Only includes fields whose owning RequirementItem has a non-None
    `form_field_index` (i.e. document/text categories — misc items have
    `form_field_index=None` because they collapse multiple form fields
    into one virtual line).
    """
    if not requirement_items or not human_review_1:
        return {}
    requirements = (human_review_1 or {}).get("requirements") or {}
    out: Dict[int, str] = {}
    for item in requirement_items:
        ffi = item.get("form_field_index")
        if ffi is None:
            continue
        choice = (requirements.get(str(item.get("id"))) or {}).get("choice")
        if choice:
            out[ffi] = choice
    return out


def _build_non_misc_index_set(
    requirement_items: Optional[List[Dict[str, Any]]],
) -> Set[int]:
    """Set of form_field indices that DO have an owning non-misc RequirementItem.

    Used to derive which form fields are in the misc bucket: any field whose
    index is NOT in this set was collapsed into the misc virtual item by
    asset_mapping._build_from_form_fields.
    """
    if not requirement_items:
        return set()
    out: Set[int] = set()
    for item in requirement_items:
        ffi = item.get("form_field_index")
        if ffi is None:
            continue
        out.add(ffi)
    return out


def plan_field_value(
    field: FormField | Dict[str, Any],
    profile: Dict[str, str],
    tailored_documents: Dict[str, Any],
    opportunity_data: Dict[str, Any],
    *,
    tailored_answers: Optional[Dict[str, Any]] = None,
    field_index: Optional[int] = None,
    field_choice: Optional[str] = None,
    is_misc: bool = False,
    misc_strategy: str = "auto_fill",
    gate_2_field_answer: Optional[Dict[str, Any]] = None,
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

    # Rule 0 — User per-field opt-out from gate 1. Documents and texts
    # carry an explicit choice; both `skip` and `ignore_for_now` mean
    # "don't fill this." Optional+ignore_for_now is treated identically
    # to optional+skip per the validator's allowed-choices table.
    if field_choice in ("skip", "ignore_for_now"):
        return FormFieldFillPlan(
            field=f, value="", status="skipped", source="user_skipped",
            reason=f"user_choice={field_choice}",
        )

    # Rule 0a — Gate-2 clarifying-question verdict (per-form_field_index).
    # User had a chance to provide a direct answer OR explicitly opt out
    # for misc-bucket fields the planner couldn't safely resolve. Wins
    # over profile lookup, tailored_answers, and option-pick rules below.
    # The session-level kill-switch (any required+ignore_for_now → don't
    # auto-fill at all) lives in the adapter; if we're here with such a
    # field_answer, the per-field skip still applies.
    if gate_2_field_answer:
        choice = gate_2_field_answer.get("choice")
        if choice in ("skip", "ignore_for_now"):
            return FormFieldFillPlan(
                field=f, value="", status="skipped", source="user_skipped",
                reason=f"gate_2_choice={choice}",
            )
        answer = gate_2_field_answer.get("answer")
        if isinstance(answer, str) and answer.strip():
            return FormFieldFillPlan(
                field=f, value=answer, status="filled",
                source="user_answer", reason="gate_2_user_answer",
            )

    # Rule 1 — Misc opt-out. The misc bucket collapses non-document /
    # non-text fields into one gate-1 line; `misc_strategy=ignore` means
    # don't auto-fill any of them.
    if is_misc and misc_strategy == "ignore":
        return FormFieldFillPlan(
            field=f, value="", status="skipped", source="user_skipped",
            reason="misc_strategy=ignore",
        )

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

    # Textarea / free-text user_answer:
    #   a. Prefer the LLM-drafted real answer from gate-1 auto_generate.
    #   b. Fall back to mock placeholder when no tailored answer is present.
    if field_type == "textarea" or f.expected_source == "user_answer":
        if tailored_answers and field_index is not None:
            entry = tailored_answers.get(str(field_index))
            content: Optional[str] = None
            if isinstance(entry, dict):
                content = entry.get("content")
            elif isinstance(entry, str):
                content = entry
            if content and str(content).strip():
                return FormFieldFillPlan(
                    field=f, value=str(content), status="filled",
                    source="user_answer",
                    reason=f"tailored_answers[{field_index}]",
                )
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
    *,
    tailored_answers: Optional[Dict[str, Any]] = None,
    requirement_items: Optional[List[Dict[str, Any]]] = None,
    human_review_1: Optional[Dict[str, Any]] = None,
    human_review_2: Optional[Dict[str, Any]] = None,
) -> List[FormFieldFillPlan]:
    """Build the full fill plan.

    Backward-compatible: when `tailored_answers`, `requirement_items`,
    `human_review_1`, or `human_review_2` are not provided, the planner
    behaves as before (no per-field skip / misc_strategy / tailored_answers
    / gate-2 user answers). New callers (adapter `attempt_auto_fill`)
    should pass all four.
    """
    field_choice_map = _build_field_choice_map(requirement_items, human_review_1)
    non_misc_indices = _build_non_misc_index_set(requirement_items)
    misc_strategy = (human_review_1 or {}).get("misc_strategy", "auto_fill")
    gate_2_field_answers: Dict[str, Any] = (
        (human_review_2 or {}).get("field_answers") or {}
    )

    plans: List[FormFieldFillPlan] = []
    for idx, f in enumerate(form_fields or []):
        is_misc = bool(requirement_items) and (idx not in non_misc_indices)
        plans.append(
            plan_field_value(
                f, profile or {}, tailored_documents or {}, opportunity_data or {},
                tailored_answers=tailored_answers,
                field_index=idx,
                field_choice=field_choice_map.get(idx),
                is_misc=is_misc,
                misc_strategy=misc_strategy,
                gate_2_field_answer=gate_2_field_answers.get(str(idx)),
            )
        )
    return plans
