"""Gate 2 (Step 7 + clarifying-questions extension).

Surfaces previews of tailored documents AND tailored answers, evaluation
warnings, posting_closed, a freshly-recomputed auto_submit_feasible flag,
AND a `needs_user_answer` list of form fields the auto-fill planner cannot
safely resolve from profile + tailoring context.

Resume value:
  {
    "approved": bool,
    "attempt_auto_submit": bool,                      # gate-2 opt-in for auto-fill
    "feedback": dict,                                 # per-doc/answer feedback
    "field_answers": {                                # per-form-field user input
      "<form_field_index>": {
        "answer": "<user-provided text>",            # OR
        "choice": "skip" | "ignore_for_now"
      },
      ...
    }
  }

field_answers semantics — mirror gate 1's choice-allowed table:
  - `skip`            → form_field.required must be False; per-field skip
  - `ignore_for_now`  → always valid; on a REQUIRED form_field it triggers
                        the auto-fill kill-switch (handled in adapter)
  - `answer: str`     → user-provided text, used as-is by value_planner
                        (source=user_answer, no LLM)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from langgraph.types import interrupt

from uppgrad_agentic.tools.profile_lookup import lookup as profile_lookup
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 400
_MAX_USER_ANSWER_LEN = 5000


# Labels that look like EEO / protected-class disclosures. Surfaced in the
# needs_user_answer list with `is_eeo=True` so the frontend can render the
# right control (e.g. one-click "Prefer not to say"). Heuristic — case-
# insensitive substring match.
_EEO_LABEL_HINTS = (
    "gender", "veteran", "disability", "race", "ethnic",
    "hispanic", "latino", "sexual orientation", "transgender",
)


def _is_eeo_label(label: str) -> bool:
    label_l = (label or "").lower()
    return any(hint in label_l for hint in _EEO_LABEL_HINTS)


def _compute_auto_submit_feasible(
    requirement_items: List[Dict[str, Any]],
    requirements: Dict[str, Dict[str, Any]],
    tailored_documents: Dict[str, Any],
    tailored_answers: Dict[str, Any],
) -> bool:
    """False when any required document's auto-generation produced no
    content, OR any required upload is missing, OR any required text
    answer is empty.
    """
    for item in requirement_items:
        if not item.get("required"):
            continue
        if item.get("category") == "misc":
            continue

        idx_str = str(item["id"])
        choice_entry = requirements.get(idx_str) or {}
        choice = choice_entry.get("choice")

        if choice not in {"upload", "auto_generate"}:
            return False

        if item.get("category") == "document":
            doc_type = item.get("document_type") or item.get("label") or ""
            info = tailored_documents.get(doc_type) or {}
            if not (info.get("content") or "").strip():
                return False
        elif item.get("category") == "text":
            ffi = item.get("form_field_index")
            key = str(ffi) if ffi is not None else idx_str
            info = tailored_answers.get(key) or {}
            if not (info.get("content") or "").strip():
                return False
    return True


def _build_needs_user_answer(
    form_fields: List[Dict[str, Any]],
    requirement_items: List[Dict[str, Any]],
    tailored_answers: Dict[str, Any],
    profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """The residual: form fields the auto-fill planner cannot safely resolve.

    A field lands here when ALL of these are true:
      - It's in the misc bucket (no owning non-misc RequirementItem)
      - It has no tailored_answer keyed by its form_field_index
      - profile_lookup against its label returns None (so auto-fill won't
        grab it from the profile snapshot either)

    EEO-style labels are flagged with `is_eeo=True` so the frontend can
    render a "decline to state" control instead of a free-text input.
    """
    non_misc_indices: Set[int] = {
        item["form_field_index"] for item in (requirement_items or [])
        if item.get("form_field_index") is not None
    }
    needs: List[Dict[str, Any]] = []
    for idx, f in enumerate(form_fields or []):
        if idx in non_misc_indices:
            continue
        if str(idx) in (tailored_answers or {}):
            continue
        label = f.get("label", "") or ""
        if profile_lookup(label, profile or {}) is not None:
            continue
        needs.append({
            "form_field_index": idx,
            "label": label,
            "field_type": f.get("field_type", ""),
            "required": bool(f.get("required")),
            "options": list(f.get("options") or []),
            "is_eeo": _is_eeo_label(label),
        })
    return needs


_FIELD_ANSWER_VALID_CHOICES = {"skip", "ignore_for_now"}


def _validate_field_answers(
    payload: Dict[str, Any],
    needs_user_answer: List[Dict[str, Any]],
) -> tuple[Optional[Dict[str, Dict[str, Any]]], List[str]]:
    """Validate `field_answers` from the gate-2 resume payload.

    Same shape rules as gate-1's per-id validator:
      - `skip` only valid on non-required form fields
      - `ignore_for_now` allowed on any (required → kill-switch trigger)
      - `answer` accepted on any; max length cap
      - per-id keys must reference a form_field_index in needs_user_answer

    Returns (cleaned_dict, errors). cleaned_dict is None when the input is
    not a dict at all; errors is a list of strings.
    """
    raw = payload.get("field_answers")
    if raw is None:
        return {}, []
    if not isinstance(raw, dict):
        return None, ["field_answers must be an object keyed by form_field_index"]

    indices_in_need = {str(item["form_field_index"]): item for item in needs_user_answer}
    errors: List[str] = []
    out: Dict[str, Dict[str, Any]] = {}

    for raw_key, entry in raw.items():
        key = str(raw_key)
        item = indices_in_need.get(key)
        if item is None:
            # Allow per-form_field-index entries even if not in needs list —
            # the frontend may want to override a profile-resolved field.
            # Still validate the entry shape below.
            item = {"required": False}
        if not isinstance(entry, dict):
            errors.append(f"field_answers[{key}] must be an object")
            continue

        choice = entry.get("choice")
        answer = entry.get("answer")

        if choice is None and answer is None:
            errors.append(f"field_answers[{key}] must have either 'answer' or 'choice'")
            continue

        if choice is not None:
            if choice not in _FIELD_ANSWER_VALID_CHOICES:
                errors.append(
                    f"field_answers[{key}].choice must be one of {sorted(_FIELD_ANSWER_VALID_CHOICES)}"
                )
                continue
            if choice == "skip" and bool(item.get("required")):
                errors.append(
                    f"field_answers[{key}]: required field cannot be skipped — use ignore_for_now"
                )
                continue
            out[key] = {"choice": choice}
            continue

        if not isinstance(answer, str):
            errors.append(f"field_answers[{key}].answer must be a string")
            continue
        if len(answer) > _MAX_USER_ANSWER_LEN:
            errors.append(
                f"field_answers[{key}].answer exceeds {_MAX_USER_ANSWER_LEN} chars"
            )
            continue
        out[key] = {"answer": answer}

    return out, errors


def human_gate_2(state: AutoApplyState) -> dict:
    updates = {"current_step": "human_gate_2", "step_history": ["human_gate_2"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    tailored_documents: Dict[str, Any] = state.get("tailored_documents") or {}
    tailored_answers: Dict[str, Any] = state.get("tailored_answers") or {}
    requirement_items: List[Dict[str, Any]] = list(state.get("requirement_items") or [])
    form_fields: List[Dict[str, Any]] = list(state.get("form_fields") or [])
    profile_snapshot: Dict[str, Any] = state.get("profile_snapshot") or {}
    human_review_1 = state.get("human_review_1") or {}
    requirements: Dict[str, Dict[str, Any]] = human_review_1.get("requirements") or {}

    opportunity_data = state.get("opportunity_data") or {}
    opportunity_type = state.get("opportunity_type", "")
    evaluation_result = state.get("evaluation_result") or {}

    title = opportunity_data.get("title") or "this opportunity"
    company = (
        opportunity_data.get("company")
        or opportunity_data.get("university")
        or opportunity_data.get("provider_name")
        or ""
    )
    opportunity_title = f"{title} at {company}" if company else title

    # Document previews
    doc_previews: Dict[str, Any] = {}
    for idx, (doc_type, info) in enumerate(tailored_documents.items()):
        content = info.get("content") or ""
        doc_previews[doc_type] = {
            "id": idx,
            "content_preview": content[:_PREVIEW_CHARS],
            "tailoring_depth": info.get("tailoring_depth", ""),
            "source": info.get("source", ""),
            "llm_used": info.get("llm_used", False),
            "passes": info.get("passes", 0),
            "char_count": len(content),
        }

    # Text-answer previews
    answer_previews: Dict[str, Any] = {}
    for key, info in tailored_answers.items():
        content = info.get("content") or ""
        answer_previews[key] = {
            "question": info.get("question", ""),
            "form_field_index": info.get("form_field_index"),
            "content_preview": content[:_PREVIEW_CHARS],
            "char_count": len(content),
            "llm_used": info.get("llm_used", False),
        }

    auto_submit_feasible = _compute_auto_submit_feasible(
        requirement_items, requirements, tailored_documents, tailored_answers,
    )

    needs_user_answer = _build_needs_user_answer(
        form_fields, requirement_items, tailored_answers, profile_snapshot,
    )

    payload = interrupt(
        {
            "tailored_documents": doc_previews,
            "tailored_answers": answer_previews,
            "evaluation_warnings": list(evaluation_result.get("warnings") or []),
            "posting_closed": bool(state.get("posting_closed")),
            "auto_submit_feasible": auto_submit_feasible,
            "needs_user_answer": needs_user_answer,
            "opportunity_title": opportunity_title,
            "opportunity_type": opportunity_type,
        }
    )

    if not isinstance(payload, dict):
        payload = {}

    approved: bool = bool(payload.get("approved", False))
    attempt_auto_submit: bool = bool(payload.get("attempt_auto_submit", False))
    feedback: Dict[str, Any] = payload.get("feedback") or {}
    if not isinstance(feedback, dict):
        feedback = {}

    field_answers, fa_errors = _validate_field_answers(payload, needs_user_answer)
    if fa_errors:
        # Graph contract: invalid field_answers don't re-interrupt (the user
        # already opined on the package); they just get logged + dropped.
        # The backend serializer should reject these earlier with a 400.
        logger.warning("human_gate_2: dropping invalid field_answers — %s", fa_errors)
        field_answers = field_answers or {}

    return {
        **updates,
        "human_review_2": {
            "approved": approved,
            "attempt_auto_submit": attempt_auto_submit,
            "feedback": feedback,
            "field_answers": field_answers or {},
        },
    }
