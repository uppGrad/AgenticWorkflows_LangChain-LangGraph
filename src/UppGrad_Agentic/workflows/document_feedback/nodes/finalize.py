from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


# ---------------------------------------------------------------------------
# Step 2: Coherence smoothing prompt
# ---------------------------------------------------------------------------

_SMOOTHING_SYSTEM = """You are editing a document that was partially rewritten. \
Some sections were changed; others were left exactly as written. \
Your only task is to smooth awkward transitions between edited and unchanged passages.

Strict rules:
- Do NOT add new information, facts, or claims
- Do NOT change the meaning of any sentence
- Do NOT remove any content
- Do NOT reorder sections
- Do NOT alter section headers
- Fix only phrasing and transitions where adjacent passages feel abrupt or disconnected

Return only the complete document text — no preamble, no explanation, no markdown fences.
"""

_MAX_SMOOTH_CHARS = 12_000


# ---------------------------------------------------------------------------
# Step 1 helpers: apply accepted changes
# ---------------------------------------------------------------------------

def _is_placeholder(text: str) -> bool:
    """True when before_text is an unresolvable heuristic placeholder."""
    t = text.strip()
    return t == "" or bool(re.match(r"^\[.+\]$", t))


def _find_span(text: str, needle: str) -> Optional[Tuple[int, int]]:
    """Return (start, end) of the first occurrence of needle in text, or None."""
    idx = text.find(needle)
    if idx == -1:
        return None
    return (idx, idx + len(needle))


def _apply_changes(
    raw_text: str,
    approved: List[Dict[str, Any]],
) -> Tuple[str, List[Dict], List[Dict], List[Dict]]:
    """
    Substitute approved proposals into raw_text.

    Returns:
        (rewritten_text, applied, conflicts, could_not_apply)

    applied        — proposals successfully substituted (in document order)
    conflicts      — pairs where two proposals overlapped the same span;
                     the lower-confidence one was dropped
    could_not_apply — proposals where before_text was a placeholder or not found
    """
    could_not_apply: List[Dict] = []

    # Resolve each proposal to a (start, end, proposal) span in raw_text.
    located: List[Tuple[int, int, Dict]] = []

    for proposal in approved:
        before = proposal.get("before_text", "")
        if _is_placeholder(before):
            could_not_apply.append({
                "section": proposal.get("section", ""),
                "rationale": proposal.get("rationale", ""),
                "after_text": proposal.get("after_text", ""),
                "reason": (
                    "before_text is empty or a placeholder — cannot locate the text to "
                    "replace. Add the after_text to the relevant section manually."
                ),
            })
            continue

        span = _find_span(raw_text, before)
        if span is None:
            could_not_apply.append({
                "section": proposal.get("section", ""),
                "rationale": proposal.get("rationale", ""),
                "before_text": before[:100],
                "reason": "before_text not found in document.",
            })
            continue

        located.append((span[0], span[1], proposal))

    # Sort ascending by start so we can detect overlaps in one pass.
    located.sort(key=lambda t: t[0])

    # Overlap resolution: keep the higher-confidence proposal for each conflict.
    conflicts: List[Dict] = []
    kept: List[Tuple[int, int, Dict]] = []

    for start, end, proposal in located:
        if kept and start < kept[-1][1]:
            # This proposal overlaps the previously kept one.
            prev_start, prev_end, prev = kept[-1]
            prev_conf = float(prev.get("confidence", 0.0))
            curr_conf = float(proposal.get("confidence", 0.0))

            if curr_conf > prev_conf:
                conflicts.append(_conflict_entry(winner=proposal, loser=prev))
                kept[-1] = (start, end, proposal)
            else:
                conflicts.append(_conflict_entry(winner=prev, loser=proposal))
            # No else branch needed: prev stays in kept unchanged.
        else:
            kept.append((start, end, proposal))

    # Apply right-to-left to preserve earlier character positions.
    rewritten = raw_text
    applied: List[Dict] = []

    for start, end, proposal in reversed(kept):
        after = proposal.get("after_text", "")
        rewritten = rewritten[:start] + after + rewritten[end:]
        applied.append({
            "section": proposal.get("section", ""),
            "rationale": proposal.get("rationale", ""),
            "before": proposal.get("before_text", "")[:120],
            "after": after[:120],
        })

    applied.reverse()  # restore document order for the diff summary

    return rewritten, applied, conflicts, could_not_apply


def _conflict_entry(winner: Dict, loser: Dict) -> Dict:
    return {
        "kept": {
            "section": winner.get("section", ""),
            "rationale": winner.get("rationale", ""),
            "confidence": winner.get("confidence", 0.0),
        },
        "discarded": {
            "section": loser.get("section", ""),
            "rationale": loser.get("rationale", ""),
            "confidence": loser.get("confidence", 0.0),
        },
        "reason": "Overlapping text spans — higher-confidence proposal kept.",
    }


# ---------------------------------------------------------------------------
# Step 2: Coherence smoothing (best-effort; never blocks the workflow)
# ---------------------------------------------------------------------------

def _smooth(rewritten: str) -> Tuple[str, bool]:
    """
    Ask the LLM to smooth transitions in the rewritten document.

    Returns (smoothed_text, smoothing_applied).
    Falls back to (rewritten, False) if LLM is unavailable or the response
    looks truncated/corrupt (less than half the input length).
    """
    llm = get_llm()
    if llm is None:
        return rewritten, False

    truncated = rewritten[:_MAX_SMOOTH_CHARS]
    msgs = [
        SystemMessage(content=_SMOOTHING_SYSTEM),
        HumanMessage(content=truncated),
    ]

    try:
        response = llm.invoke(msgs)
        smoothed = (response.content or "").strip()
        # Sanity check: reject if the LLM returned something much shorter.
        if len(smoothed) < len(truncated) * 0.5:
            return rewritten, False
        return smoothed, True
    except Exception:
        return rewritten, False


# ---------------------------------------------------------------------------
# Step 3: Build diff summary
# ---------------------------------------------------------------------------

def _build_diff(
    all_proposals: List[Dict[str, Any]],
    approved_proposals: List[Dict[str, Any]],
    applied: List[Dict],
    conflicts: List[Dict],
    could_not_apply: List[Dict],
    smoothing_applied: bool,
) -> Dict[str, Any]:
    # Identify rejected proposals: those in all_proposals but not in approved_proposals.
    approved_keys = {
        (p.get("section", ""), p.get("rationale", ""))
        for p in approved_proposals
    }
    rejected = [
        {
            "section": p.get("section", ""),
            "rationale": p.get("rationale", ""),
        }
        for p in all_proposals
        if (p.get("section", ""), p.get("rationale", "")) not in approved_keys
    ]

    n_applied = len(applied)
    n_rejected = len(rejected)
    n_conflicts = len(conflicts)
    n_could_not = len(could_not_apply)

    parts: List[str] = []
    if n_applied:
        parts.append(f"{n_applied} change{'s' if n_applied != 1 else ''} applied")
    if n_rejected:
        parts.append(f"{n_rejected} rejected by user")
    if n_conflicts:
        parts.append(f"{n_conflicts} conflict{'s' if n_conflicts != 1 else ''} resolved")
    if n_could_not:
        parts.append(
            f"{n_could_not} suggestion{'s' if n_could_not != 1 else ''} "
            "could not be applied automatically"
        )
    if smoothing_applied:
        parts.append("coherence smoothing applied")
    summary = ("; ".join(parts) + ".") if parts else "No changes applied."

    return {
        "applied": applied,
        "rejected": rejected,
        "conflicts": conflicts,
        "could_not_apply": could_not_apply,
        "smoothing_applied": smoothing_applied,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def finalize(state: DocFeedbackState) -> dict:
    if state.get("result", {}).get("status") == "error":
        return {}

    raw_text = state.get("raw_text") or ""
    human_review = state.get("human_review") or {}
    approved_proposals: List[Dict[str, Any]] = human_review.get("approved_proposals") or []
    all_proposals: List[Dict[str, Any]] = state.get("proposals") or []

    # ------------------------------------------------------------------
    # Step 1: Apply accepted changes
    # ------------------------------------------------------------------
    try:
        rewritten, applied, conflicts, could_not_apply = _apply_changes(
            raw_text, approved_proposals
        )
    except Exception as e:
        # Preserve approved_proposals in details so the user doesn't lose their work.
        return {
            "result": {
                "status": "error",
                "error_code": "APPLY_FAILED",
                "user_message": (
                    "We could not apply the approved changes to your document. "
                    "Your selections have been saved in the error details."
                ),
                "details": {
                    "exception": str(e),
                    "approved_proposals": approved_proposals,
                },
            }
        }

    # ------------------------------------------------------------------
    # Step 2: Coherence smoothing (best-effort — never errors out)
    # ------------------------------------------------------------------
    final_text, smoothing_applied = _smooth(rewritten)

    # ------------------------------------------------------------------
    # Step 3: Build diff summary
    # ------------------------------------------------------------------
    diff = _build_diff(
        all_proposals=all_proposals,
        approved_proposals=approved_proposals,
        applied=applied,
        conflicts=conflicts,
        could_not_apply=could_not_apply,
        smoothing_applied=smoothing_applied,
    )

    # ------------------------------------------------------------------
    # Step 4: Write results to state
    # ------------------------------------------------------------------
    return {
        "final_document": final_text,
        "diff": diff,
        "result": {
            "status": "ok",
            "user_message": diff["summary"],
            "details": {
                "final_document": final_text,
                "diff": diff,
            },
        },
    }
