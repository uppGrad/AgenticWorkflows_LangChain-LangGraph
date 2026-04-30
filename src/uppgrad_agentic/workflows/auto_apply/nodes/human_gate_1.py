"""Gate 1 (Step 6 rewrite).

Surfaces the categorised requirement_items list to the user. The user picks
a per-item choice (upload / auto_generate / ignore_for_now / skip) plus an
optional ≤200-char user_prompt for documents, and a misc_strategy
(auto_fill / ignore) for the collapsed misc line.

Validation:
  - required document items reject "skip" and "ignore_for_now"
  - required text items reject "skip"
  - USER_SUPPLIED canonical doc types reject "auto_generate"
On invalid resume, the node returns no state changes and re-interrupts so
the backend can return a 400 to the frontend.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langgraph.types import interrupt

from uppgrad_agentic.workflows.auto_apply.nodes.asset_mapping import _USER_SUPPLIED
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resume-value contract
#
# Command(resume={
#   "requirements": {
#       "<id>": {
#           "choice": "upload" | "auto_generate" | "ignore_for_now" | "skip",
#           "uploaded_text": "<extracted text>" | null,
#           "user_prompt": "<≤200 chars>" | null   # documents only
#       },
#       ...
#   },
#   "misc_strategy": "auto_fill" | "ignore"
# })
#
# - Each per-id entry must reference an id that appears in requirement_items.
# - The misc_strategy applies to category='misc' line(s); ignored otherwise.
# - Always send a non-empty dict — LangGraph treats falsy resume values as
#   "no resume" and re-interrupts.
#
# Interrupt payload (what the frontend receives):
#   {
#       "requirement_items": [...],
#       "opportunity_type": "job",
#       "opportunity_title": "Software Engineer at Acme Corp",
#   }
# ---------------------------------------------------------------------------

_VALID_CHOICES = {"upload", "auto_generate", "ignore_for_now", "skip"}
_VALID_MISC_STRATEGIES = {"auto_fill", "ignore"}
_MAX_USER_PROMPT_LEN = 200


# Allowed choices per (category, required) tuple. Reading the table:
#
#   - `skip`           = "I don't want this." Only valid for OPTIONAL fields.
#   - `ignore_for_now` = "I'll handle this manually after handoff." Always
#                        valid; on REQUIRED items it's the signal that the
#                        backend's auto-fill module should NOT run for this
#                        session — the user is in package-and-bounce mode.
#   - `upload`         = file-side action. Not meaningful for text answers.
#   - `auto_generate`  = LLM drafts the document/answer.
#
# Misc items are NOT keyed in this table — they're collapsed into a single
# virtual line at gate 1 and routed via `misc_strategy` (auto_fill | ignore).
# Per-id misc entries shouldn't appear in normal frontend payloads; if one
# does, this loop's category check falls through and the entry is a no-op.
_ALLOWED_CHOICES: Dict[tuple, set] = {
    ("document", True):  {"upload", "auto_generate", "ignore_for_now"},
    ("document", False): {"upload", "auto_generate", "skip", "ignore_for_now"},
    ("text", True):      {"auto_generate", "ignore_for_now"},
    ("text", False):     {"auto_generate", "skip", "ignore_for_now"},
}


def _validate_resume(
    payload: Any,
    requirement_items: List[Dict[str, Any]],
) -> Optional[List[str]]:
    """Return a list of validation errors. Empty list = valid; None means
    "not even a dict — re-interrupt".
    """
    if not isinstance(payload, dict) or not payload:
        return None

    errors: List[str] = []
    requirements = payload.get("requirements")
    if not isinstance(requirements, dict):
        errors.append("requirements must be an object keyed by requirement id")
        return errors

    misc_strategy = payload.get("misc_strategy", "ignore")
    if misc_strategy not in _VALID_MISC_STRATEGIES:
        errors.append(
            f"misc_strategy must be one of {sorted(_VALID_MISC_STRATEGIES)}"
        )

    items_by_id = {str(item["id"]): item for item in requirement_items}

    for raw_id, raw in requirements.items():
        item = items_by_id.get(str(raw_id))
        if item is None:
            errors.append(f"unknown requirement id: {raw_id}")
            continue
        if not isinstance(raw, dict):
            errors.append(f"requirements[{raw_id}] must be an object")
            continue
        choice = raw.get("choice")
        if choice not in _VALID_CHOICES:
            errors.append(
                f"requirements[{raw_id}].choice must be one of {sorted(_VALID_CHOICES)}"
            )
            continue

        category = item.get("category")
        required = bool(item.get("required"))

        # Semantic check: which choices are allowed for this (category, required)
        # combination. Misc has no row here — see the table comment above.
        allowed = _ALLOWED_CHOICES.get((category, required))
        if allowed is not None and choice not in allowed:
            req_label = "required" if required else "optional"
            errors.append(
                f"requirements[{raw_id}]: choice='{choice}' not allowed for "
                f"{req_label} {category}"
            )
            continue

        # Structural rules per category — separate from the choice-allowed table.
        if category == "document":
            doc_type = item.get("document_type")
            if doc_type in _USER_SUPPLIED and choice == "auto_generate":
                errors.append(
                    f"requirements[{raw_id}]: '{doc_type}' must be uploaded — auto_generate is not allowed"
                )
            if choice == "upload":
                uploaded_text = raw.get("uploaded_text")
                if not uploaded_text or not isinstance(uploaded_text, str) or not uploaded_text.strip():
                    errors.append(
                        f"requirements[{raw_id}]: choice=upload requires non-empty uploaded_text"
                    )
            user_prompt = raw.get("user_prompt")
            if user_prompt is not None:
                if not isinstance(user_prompt, str):
                    errors.append(f"requirements[{raw_id}].user_prompt must be a string")
                elif len(user_prompt) > _MAX_USER_PROMPT_LEN:
                    errors.append(
                        f"requirements[{raw_id}].user_prompt exceeds {_MAX_USER_PROMPT_LEN} chars"
                    )

    return errors


def _compute_auto_submit_feasible(
    requirement_items: List[Dict[str, Any]],
    requirements: Dict[str, Dict[str, Any]],
) -> bool:
    """True when every required item has either a usable upload or a valid
    auto-generate selection. Misc items don't gate feasibility — they're
    auto-filled or skipped based on misc_strategy and never block submission
    on their own.
    """
    for item in requirement_items:
        if not item.get("required"):
            continue
        if item.get("category") == "misc":
            continue
        choice = (requirements.get(str(item["id"])) or {}).get("choice")
        if choice in {"upload", "auto_generate"}:
            continue
        return False
    return True


def human_gate_1(state: AutoApplyState) -> dict:
    updates = {"current_step": "human_gate_1", "step_history": ["human_gate_1"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    requirement_items: List[Dict[str, Any]] = list(state.get("requirement_items") or [])
    opportunity_data = state.get("opportunity_data") or {}
    opportunity_type = state.get("opportunity_type", "")

    title = (
        opportunity_data.get("title")
        or opportunity_data.get("name")
        or "this opportunity"
    )
    company = (
        opportunity_data.get("company")
        or opportunity_data.get("university")
        or opportunity_data.get("provider_name")
        or ""
    )
    opportunity_title = f"{title} at {company}" if company else title

    payload = interrupt(
        {
            "requirement_items": requirement_items,
            "opportunity_type": opportunity_type,
            "opportunity_title": opportunity_title,
        }
    )

    errors = _validate_resume(payload, requirement_items)
    if errors is None or errors:
        # Invalid resume — return state unchanged so the node re-interrupts.
        # The backend serializer reads this signal and returns 400.
        if errors:
            logger.warning("human_gate_1: invalid resume payload — %s", errors)
        return updates

    requirements: Dict[str, Dict[str, Any]] = payload["requirements"]
    misc_strategy: str = payload.get("misc_strategy", "ignore")

    feasible = _compute_auto_submit_feasible(requirement_items, requirements)

    return {
        **updates,
        "human_review_1": {
            "requirements": requirements,
            "misc_strategy": misc_strategy,
        },
        "auto_submit_feasible_at_gate_1": feasible,
    }
