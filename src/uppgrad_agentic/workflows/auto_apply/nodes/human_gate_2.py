"""Gate 2 (Step 7 rewrite).

Surfaces previews of tailored documents AND tailored answers, evaluation
warnings, posting_closed, and a freshly-recomputed auto_submit_feasible
flag. Resume value adds `attempt_auto_submit` (intent only — auto-submit
itself is not implemented yet).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langgraph.types import interrupt

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 400


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


def human_gate_2(state: AutoApplyState) -> dict:
    updates = {"current_step": "human_gate_2", "step_history": ["human_gate_2"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    tailored_documents: Dict[str, Any] = state.get("tailored_documents") or {}
    tailored_answers: Dict[str, Any] = state.get("tailored_answers") or {}
    requirement_items: List[Dict[str, Any]] = list(state.get("requirement_items") or [])
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

    payload = interrupt(
        {
            "tailored_documents": doc_previews,
            "tailored_answers": answer_previews,
            "evaluation_warnings": list(evaluation_result.get("warnings") or []),
            "posting_closed": bool(state.get("posting_closed")),
            "auto_submit_feasible": auto_submit_feasible,
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

    # ─── Apply user overrides from field_answers ────────────────────────
    # The frontend sends `field_answers` keyed by form_field_index for
    # (a) "Quick questions" the user filled in, and (b) edits to
    # auto-derived misc answers (Yes/No / Country / sponsorship the user
    # changed before approving). Each entry is either:
    #   {answer: "<text>"}        — explicit override
    #   {choice: "skip"}          — drop the answer entirely
    #   {choice: "ignore_for_now"} — surface as kill-switch (handled at
    #                                fill-time)
    # We merge into tailored_answers so downstream (auto-fill +
    # package_and_handoff) sees a single source of truth.
    field_answers: Dict[str, Any] = payload.get("field_answers") or {}
    if not isinstance(field_answers, dict):
        field_answers = {}
    answers_updated = dict(tailored_answers)
    form_fields_state: List[Dict[str, Any]] = list(state.get("form_fields") or [])
    for raw_idx, fa in field_answers.items():
        if not isinstance(fa, dict):
            continue
        try:
            ffi = int(raw_idx)
        except (ValueError, TypeError):
            continue
        key = str(ffi)
        choice = fa.get("choice")
        if choice == "skip":
            # User explicitly skipped — drop any auto-derived answer.
            answers_updated.pop(key, None)
            continue
        if choice == "ignore_for_now":
            # Leave any existing tailored_answer alone; the kill-switch
            # is observed at fill-time via human_review_2.feedback / the
            # auto_submit_feasible signal. No tailored_answers mutation.
            continue
        ans = fa.get("answer")
        if ans is None or not isinstance(ans, str):
            continue
        question = ""
        if 0 <= ffi < len(form_fields_state):
            question = (form_fields_state[ffi].get("label") or "")
        existing = answers_updated.get(key) or {}
        answers_updated[key] = {
            **existing,
            "content": ans.strip(),
            "question": existing.get("question") or question,
            "form_field_index": ffi,
            "llm_used": False,
            "source": "user_override",
        }

    return {
        **updates,
        "human_review_2": {
            "approved": approved,
            "attempt_auto_submit": attempt_auto_submit,
            "feedback": feedback,
            "field_answers": field_answers,
        },
        # Persist the override layer back to top-level state so
        # downstream nodes + auto_apply_adapter.attempt_auto_fill see
        # the user's final say, not the pre-edit value_planner output.
        "tailored_answers": answers_updated,
    }
