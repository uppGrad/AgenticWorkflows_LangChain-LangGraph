from __future__ import annotations

import json
import re
from typing import List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.workflows.document_feedback.schemas import EvaluationResult
from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM = """You are a quality-control reviewer for an AI document feedback system.

You will receive a list of change proposals and the original document. Evaluate each proposal for:

1. **Groundedness**: Does before_text actually appear in the document? If before_text is non-empty
   and not a placeholder, it must be a real excerpt from the document — flag any that are fabricated.
2. **Hallucinations**: Does after_text introduce names, dates, companies, degrees, or metrics that
   are not in the document or the user's profile? Flag anything invented.
3. **Specificity**: Is the rationale clear and specific? Reject vague rationales under 15 characters
   or after_text values that are unresolved placeholders like "[Add content here]".
4. **Format compliance**: Every proposal must have a non-empty section, rationale, and after_text,
   and a confidence value between 0.0 and 1.0.

Return:
- passed: true only if no critical issues were found (minor style notes do not fail)
- issues: a list of specific problem descriptions — empty if passed
- iteration: the iteration index provided

Be strict about groundedness and hallucinations; lenient about minor wording or ordering choices.
"""

_MAX_PROPOSALS_CHARS = 4000
_MAX_DOC_CHARS = 3000
_MAX_PROFILE_CHARS = 1000

# Placeholder patterns produced by the heuristic synthesizer — not hallucinations,
# just low-specificity. We note them but do not count them as groundedness failures.
_PLACEHOLDER_RE = re.compile(r"^\[.+\]$")


# LLM structured output schema — defined at module level so Pydantic builds the
# model class once, not on every evaluate_output() call.
class _EvalOut(BaseModel):
    passed: bool = Field(..., description="True only if no critical issues found")
    issues: list[str] = Field(default_factory=list, description="Specific problem descriptions")


# ---------------------------------------------------------------------------
# Heuristic checks
# ---------------------------------------------------------------------------

def _check_format(proposal: dict, index: int) -> List[str]:
    """Return a list of format-compliance issue strings for one proposal."""
    issues: List[str] = []
    prefix = f"Proposal {index + 1} ({proposal.get('section', '?')})"

    if not (proposal.get("section") or "").strip():
        issues.append(f"{prefix}: 'section' is empty.")
    if not (proposal.get("rationale") or "").strip():
        issues.append(f"{prefix}: 'rationale' is empty.")
    elif len(proposal["rationale"].strip()) < 15:
        issues.append(f"{prefix}: rationale is too short to be meaningful.")
    if not (proposal.get("after_text") or "").strip():
        issues.append(f"{prefix}: 'after_text' is empty — every proposal must supply replacement text.")
    conf = proposal.get("confidence")
    if conf is None or not (0.0 <= float(conf) <= 1.0):
        issues.append(f"{prefix}: 'confidence' is missing or out of range [0, 1].")
    if proposal.get("requires_confirmation") is None:
        issues.append(f"{prefix}: 'requires_confirmation' is missing.")
    return issues


def _check_groundedness(
    proposal: dict,
    index: int,
    raw_text: str,
) -> Tuple[List[str], bool]:
    """
    Return (issues, is_placeholder) for one proposal.
    is_placeholder is True when before_text looks like a synthesizer placeholder —
    these are low-specificity but not hallucinations.
    """
    issues: List[str] = []
    before = (proposal.get("before_text") or "").strip()
    after = (proposal.get("after_text") or "").strip()
    prefix = f"Proposal {index + 1} ({proposal.get('section', '?')})"

    # Classify whether before/after are placeholders
    before_is_placeholder = bool(_PLACEHOLDER_RE.match(before)) or before == ""
    after_is_placeholder = bool(_PLACEHOLDER_RE.match(after))

    if after_is_placeholder:
        # Flag as a specificity problem, not a hallucination
        issues.append(
            f"{prefix}: after_text '{after}' is an unresolved placeholder — "
            "synthesis should supply real replacement text."
        )

    if before and not before_is_placeholder:
        # Real text claim — verify it exists in the document
        # Allow partial match: the before_text must appear as a substring (case-insensitive)
        # for text longer than 20 chars; for shorter snippets use exact match.
        needle = before if len(before) <= 20 else before[:80]
        if needle.lower() not in raw_text.lower():
            issues.append(
                f"{prefix}: before_text not found in document — "
                f"'{before[:60]}{'...' if len(before) > 60 else ''}' "
                "may be fabricated."
            )

    return issues, before_is_placeholder or after_is_placeholder


def _check_hallucinated_facts(
    proposal: dict,
    index: int,
    raw_text: str,
    profile_snapshot: dict,
) -> List[str]:
    """
    Lightweight heuristic: look for proper nouns in after_text that don't appear
    anywhere in raw_text or the profile. This catches the most egregious fabrications.
    """
    issues: List[str] = []
    after = (proposal.get("after_text") or "").strip()
    prefix = f"Proposal {index + 1} ({proposal.get('section', '?')})"

    if _PLACEHOLDER_RE.match(after) or not after:
        return issues  # placeholder — already flagged elsewhere

    # Build a reference corpus from raw_text + profile
    profile_text = json.dumps(profile_snapshot)
    corpus = (raw_text + " " + profile_text).lower()

    # Heuristic: extract capitalised multi-word tokens (likely proper nouns)
    # e.g. "Stanford University", "Google DeepMind", "GPT-4"
    proper_noun_re = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
    candidates = proper_noun_re.findall(after)

    hallucinated = [
        name for name in candidates
        if name.lower() not in corpus
        # Allow very common words that are capitalised at sentence start
        and len(name) > 5
    ]

    if hallucinated:
        issues.append(
            f"{prefix}: after_text introduces proper noun(s) not found in the document "
            f"or profile: {', '.join(hallucinated[:4])}. Verify these are not invented."
        )

    return issues


def _heuristic_evaluate(
    proposals: List[dict],
    raw_text: str,
    profile_snapshot: dict,
    iteration: int,
) -> EvaluationResult:
    if not proposals:
        return EvaluationResult(
            passed=False,
            issues=["No proposals were generated — synthesis produced an empty list."],
            iteration=iteration,
        )

    all_issues: List[str] = []
    placeholder_count = 0
    groundedness_failures = 0

    for i, proposal in enumerate(proposals):
        all_issues.extend(_check_format(proposal, i))

        grounding_issues, is_placeholder = _check_groundedness(proposal, i, raw_text)
        if is_placeholder:
            placeholder_count += 1
        else:
            groundedness_failures += len(
                [iss for iss in grounding_issues if "not found" in iss]
            )
        all_issues.extend(grounding_issues)

        all_issues.extend(
            _check_hallucinated_facts(proposal, i, raw_text, profile_snapshot)
        )

    # Deduplicate
    seen: set[str] = set()
    unique_issues: List[str] = []
    for iss in all_issues:
        key = iss[:100]
        if key not in seen:
            seen.add(key)
            unique_issues.append(iss)

    # Passing rules:
    # - Format failures are always blocking.
    # - Groundedness failures on real (non-placeholder) text are blocking if > 20% of proposals.
    # - Placeholder proposals are a quality note but not blocking on their own
    #   (they come from the heuristic synthesizer and are expected without an LLM).
    format_failures = [i for i in unique_issues if "empty" in i or "missing" in i or "out of range" in i]
    real_proposals = len(proposals) - placeholder_count
    grounding_failure_rate = (
        groundedness_failures / max(real_proposals, 1) if real_proposals > 0 else 0.0
    )

    passed = len(format_failures) == 0 and grounding_failure_rate <= 0.20

    return EvaluationResult(
        passed=passed,
        issues=unique_issues,
        iteration=iteration,
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def evaluate_output(state: DocFeedbackState) -> dict:
    updates = {"current_step": "evaluate_output", "step_history": ["evaluate_output"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    proposals = state.get("proposals") or []
    raw_text = state.get("raw_text") or ""
    profile_snapshot = state.get("profile_snapshot") or {}
    # iteration_count tracks how many synthesis→evaluate cycles have completed.
    # Read the current value before this evaluation; we increment it in the return.
    current_iteration = state.get("iteration_count", 0)

    llm = get_llm()
    if llm is None:
        result = _heuristic_evaluate(proposals, raw_text, profile_snapshot, current_iteration)
        return {
            **updates,
            "evaluation_result": result.model_dump(),
            "iteration_count": current_iteration + 1,
        }

    proposals_text = json.dumps(proposals, indent=2)[:_MAX_PROPOSALS_CHARS]
    doc_excerpt = raw_text[:_MAX_DOC_CHARS]
    profile_text = json.dumps(profile_snapshot)[:_MAX_PROFILE_CHARS]

    structured = llm.with_structured_output(_EvalOut)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content=(
                f"Iteration: {current_iteration}\n\n"
                f"Change proposals (JSON):\n{proposals_text}\n\n"
                f"Original document (first {_MAX_DOC_CHARS} chars):\n{doc_excerpt}\n\n"
                f"User profile summary:\n{profile_text}"
            )
        ),
    ]

    try:
        out: _EvalOut = structured.invoke(msgs)
        result = EvaluationResult(
            passed=out.passed,
            issues=out.issues,
            iteration=current_iteration,
        )
    except Exception as e:
        result = _heuristic_evaluate(proposals, raw_text, profile_snapshot, current_iteration)
        result.issues.append(f"[LLM evaluator failed, used heuristic: {e}]")

    return {
        **updates,
        "evaluation_result": result.model_dump(),
        "iteration_count": current_iteration + 1,
    }
