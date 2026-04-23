from __future__ import annotations

from typing import Any, Dict, List

from langgraph.types import interrupt

from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


# ---------------------------------------------------------------------------
# Resume value contract
#
# When the graph is resumed the caller must pass a Command(resume=decisions)
# where `decisions` is a dict mapping string proposal IDs to decision objects:
#
#   decisions: {
#       "0": {"action": "accept"},
#       "1": {"action": "reject"},
#       "2": {"action": "accept", "comment": "Good catch, please apply."},
#   }
#
# - "action" is required and must be "accept" or "reject".
# - "comment" is optional free text from the user, passed through to human_review.
# - Proposals with no entry in decisions are treated as rejected.
# - Passing decisions={} rejects all proposals (user dismissed without reviewing).
#
# Interrupt payload (what the frontend receives):
#
#   {
#       "proposals": [{"id": 0, "section": ..., "rationale": ..., ...}, ...],
#       "doc_type": "CV",
#       "doc_meta": {...},
#   }
# ---------------------------------------------------------------------------


def human_gate(state: DocFeedbackState) -> dict:
    updates = {"current_step": "human_gate", "step_history": ["human_gate"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    proposals: List[Dict[str, Any]] = state.get("proposals") or []

    # Attach stable integer IDs so the frontend can reference proposals by ID
    # in its resume payload without relying on list position after any reordering.
    numbered_proposals = [{"id": i, **p} for i, p in enumerate(proposals)]

    doc_classification = state.get("doc_classification") or {}
    doc_meta = state.get("doc_meta") or {}

    # -----------------------------------------------------------------------
    # Suspend here. The graph is frozen until the caller resumes it via:
    #   graph.invoke(Command(resume=decisions), config={"configurable": {"thread_id": ...}})
    # The return value of interrupt() is whatever was passed as the resume value.
    # -----------------------------------------------------------------------
    decisions: Dict[str, Any] = interrupt(
        {
            "proposals": numbered_proposals,
            "doc_type": doc_classification.get("doc_type", "UNKNOWN"),
            "doc_meta": doc_meta,
        }
    )

    # -----------------------------------------------------------------------
    # Validate and normalise the resume value.
    # decisions may be None (e.g. user closed the dialog without submitting).
    # -----------------------------------------------------------------------
    if not isinstance(decisions, dict):
        decisions = {}

    approved_proposals: List[Dict[str, Any]] = []
    normalised_decisions: Dict[str, Dict[str, Any]] = {}

    for proposal in numbered_proposals:
        pid = str(proposal["id"])
        raw = decisions.get(pid) or {}

        # Normalise: accept a bare "accept"/"reject" string as well as a dict
        if isinstance(raw, str):
            raw = {"action": raw}

        action = (raw.get("action") or "reject").lower().strip()
        if action not in ("accept", "reject"):
            action = "reject"

        comment = (raw.get("comment") or "").strip()

        normalised_decisions[pid] = {"action": action, "comment": comment}

        if action == "accept":
            # Strip the synthetic "id" field before passing to finalize
            proposal_without_id = {k: v for k, v in proposal.items() if k != "id"}
            approved_proposals.append(proposal_without_id)

    return {
        **updates,
        "human_review": {
            "approved_proposals": approved_proposals,
            "decisions": normalised_decisions,
        },
    }
