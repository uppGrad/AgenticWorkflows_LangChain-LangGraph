"""Auto-accept proposals — drop-in replacement for `human_gate` when the
doc-feedback pipeline is being driven from auto-apply tailoring.

Auto-apply has no per-proposal review UI; the user has already chosen
"tailor my CV against this JD" at gate-1 and trusted the agent to make
sensible edits. Inside the doc-feedback graph, that translates to:

  - Accept every proposal where `requires_confirmation == False`.
  - Reject proposals where `requires_confirmation == True` — that flag
    is the model's own "I'm not sure, ask the human first" signal
    (e.g. PII removals, References-on-request, photo removal). Auto-
    apply is not the right context to override those.

Output shape matches `human_gate` exactly so `finalize` is unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List

from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


def auto_accept_proposals(state: DocFeedbackState) -> dict:
    updates = {
        "current_step": "auto_accept_proposals",
        "step_history": ["auto_accept_proposals"],
    }
    if state.get("result", {}).get("status") == "error":
        return updates

    proposals: List[Dict[str, Any]] = state.get("proposals") or []
    numbered_proposals = [{"id": i, **p} for i, p in enumerate(proposals)]

    approved_proposals: List[Dict[str, Any]] = []
    decisions: Dict[str, Dict[str, Any]] = {}

    for proposal in numbered_proposals:
        pid = str(proposal["id"])
        # `requires_confirmation` is the proposal's own caution signal —
        # respect it. Everything else flows through.
        requires_confirmation = bool(proposal.get("requires_confirmation", False))
        if requires_confirmation:
            decisions[pid] = {
                "action": "reject",
                "comment": "auto-rejected: requires_confirmation",
            }
            continue
        decisions[pid] = {"action": "accept", "comment": "auto-accepted"}
        proposal_without_id = {k: v for k, v in proposal.items() if k != "id"}
        approved_proposals.append(proposal_without_id)

    return {
        **updates,
        "human_review": {
            "approved_proposals": approved_proposals,
            "decisions": decisions,
        },
    }
