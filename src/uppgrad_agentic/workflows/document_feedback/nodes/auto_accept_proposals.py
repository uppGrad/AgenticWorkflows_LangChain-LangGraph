"""Auto-accept proposals — drop-in replacement for `human_gate` when the
doc-feedback pipeline is being driven from auto-apply tailoring.

Auto-apply has no per-proposal review UI; the user has already chosen
"tailor my CV against this JD" at gate-1 and trusted the agent to make
sensible edits. Inside the doc-feedback graph that translates to a
doc-type-aware accept rule, because `requires_confirmation` carries
different semantics on the CV vs SOP/COVER_LETTER paths:

  - **CV path:** `requires_confirmation=True` is set ONLY for PII
    removals (DOB, marital status, photo) per the synth prompt. Visa
    context can legitimately justify keeping them — the user should
    confirm. We REJECT these proposals.

  - **SOP / COVER_LETTER path:** `requires_confirmation=True` is set
    for EVERY substance rewrite, every delete, and every merge per the
    `_SYSTEM_SUBSTANCE` prompt — that's the highest-quality work the
    pipeline does and is the whole point of running it. Filtering on
    that flag would throw out 70%+ of the value. The evaluator
    (grounding validator + AI-tell budget + preserve_sentences audit)
    already enforces safety; in auto-apply the user consented at
    gate-1; we ACCEPT these proposals.

Output shape matches `human_gate` exactly so `finalize` is unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List

from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


# Doc types where `requires_confirmation=True` is set for substance
# rewrites (the whole point of the synth — accept everything that
# survived the evaluator). Anything else falls through to the safe
# default (treat the flag as a hard reject), so adding a new doc type
# without thought can't accidentally widen the auto-accept blast radius.
_DOC_TYPES_TRUST_EVALUATOR = frozenset({"SOP", "COVER_LETTER"})


def auto_accept_proposals(state: DocFeedbackState) -> dict:
    updates = {
        "current_step": "auto_accept_proposals",
        "step_history": ["auto_accept_proposals"],
    }
    if state.get("result", {}).get("status") == "error":
        return updates

    proposals: List[Dict[str, Any]] = state.get("proposals") or []
    numbered_proposals = [{"id": i, **p} for i, p in enumerate(proposals)]

    doc_type = (state.get("doc_classification") or {}).get("doc_type", "UNKNOWN")
    trust_evaluator = doc_type in _DOC_TYPES_TRUST_EVALUATOR

    approved_proposals: List[Dict[str, Any]] = []
    decisions: Dict[str, Dict[str, Any]] = {}

    for proposal in numbered_proposals:
        pid = str(proposal["id"])
        requires_confirmation = bool(proposal.get("requires_confirmation", False))

        # On SOP / COVER_LETTER the flag means "substance rewrite" — the
        # whole point of running the pipeline. Accept (the evaluator
        # already bounded grounding + AI-tells + preserve_sentences).
        # On every other path (CV most importantly) the flag means real
        # caution risk (PII removals, visa context). Reject.
        if requires_confirmation and not trust_evaluator:
            decisions[pid] = {
                "action": "reject",
                "comment": f"auto-rejected: requires_confirmation on {doc_type} path",
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
