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

You will receive a list of change proposals, the original document, the user
profile, and (for SOP / COVER_LETTER docs) the rhetoric and narrative
analysis outputs. Evaluate each proposal for:

1. **Groundedness**: Does before_text actually appear in the document? If before_text is non-empty
   and not a placeholder, it must be a real excerpt from the document — flag any that are fabricated.
2. **Hallucinations**: Does after_text introduce names, dates, companies, degrees, or metrics that
   are not in the document or the user's profile? Flag anything invented.
3. **Specificity**: Is the rationale clear and specific? Reject vague rationales under 15 characters
   or after_text values that are unresolved placeholders like "[Add content here]".
4. **Format compliance**: Every proposal must have a non-empty section, rationale, and a
   confidence value between 0.0 and 1.0. `after_text` must be non-empty UNLESS
   `action="delete"`, in which case it MUST be the empty string.
5. **Substance compliance** (SOP / COVER_LETTER ONLY): for each rhetoric
   finding with `priority: "high"` there MUST be at least one proposal whose
   before_text matches that paragraph (anchor substring is enough). If a
   high-priority finding has no targeting proposal, flag the omission as
   a substance gap. Additionally, for any proposal targeting a finding with
   non-empty `preserve_sentences`, EVERY preserve sentence MUST appear
   verbatim in after_text — flag missing preservations.
6. **Mix sanity** (SOP / COVER_LETTER ONLY): if more than 30% of proposals are
   sentence-level polish (style/grammar/keyword) while one or more high-
   priority rhetoric findings remain unaddressed, flag the mix as
   polish-dominated. Substance work comes first; polish only after.
7. **Narrative compliance** (SOP / COVER_LETTER ONLY): the proposals must
   address the document-level narrative findings.
   - For every entry in `narrative.repeated_anchors`, the resulting document
     (after applying all proposals) must NOT keep the same anchor as the
     focus of two or more paragraphs. If a repeated anchor is still being
     leaned on across proposals, flag the redundancy.
   - For every paragraph index in `narrative.paragraphs_to_delete`, there
     must be a corresponding `action="delete"` (or `action="merge"`)
     proposal targeting that paragraph. Flag missing deletions.
   - If `narrative.conclusion_commits_forward` is false, there must be a
     proposal rewriting the closing paragraph that names the target org
     AND specifies a concrete contribution. Flag a generic closing.
8. **AI-tell density** (SOP / COVER_LETTER ONLY): no `after_text` may
   contain an em-dash (—) or a double-hyphen ( -- ). The total count of
   banned phrases across all `after_text` values combined must be at most 1.
   The banned phrases are: "I believe my background", "see this opportunity
   as a chance", "continue developing myself", "I am especially motivated",
   "directly shapes", "play a meaningful role", "tapestry", "delve", "delving",
   "stands out to me", "matters to me because", and the metaphorical
   "leverage" / "navigate". Flag any violation.

Return:
- passed: true only if no critical issues were found (minor style notes do not fail)
- issues: a list of specific problem descriptions — empty if passed
- iteration: the iteration index provided

Be strict about groundedness, hallucinations, substance/narrative compliance,
and AI-tell density; lenient about minor wording or ordering choices.
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
    action = (proposal.get("action") or "rewrite").lower()

    if not (proposal.get("section") or "").strip():
        issues.append(f"{prefix}: 'section' is empty.")
    if not (proposal.get("rationale") or "").strip():
        issues.append(f"{prefix}: 'rationale' is empty.")
    elif len(proposal["rationale"].strip()) < 15:
        issues.append(f"{prefix}: rationale is too short to be meaningful.")

    # Delete proposals must have empty after_text; rewrite/merge must not.
    after_text = (proposal.get("after_text") or "").strip()
    if action == "delete":
        if after_text:
            issues.append(
                f"{prefix}: action='delete' requires after_text to be empty, "
                "but a non-empty replacement was supplied."
            )
        # Delete must also have a non-empty before_text — there must be
        # something to remove.
        if not (proposal.get("before_text") or "").strip():
            issues.append(
                f"{prefix}: action='delete' requires before_text to identify "
                "the paragraph being removed."
            )
    else:
        if not after_text:
            issues.append(
                f"{prefix}: 'after_text' is empty — every {action} proposal "
                "must supply replacement text."
            )

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


def _normalize_for_match(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# Heuristic for "polish proposal": short before_text (sentence- or
# fragment-level) AND no `[Substance` marker in the rationale. The synthesizer
# emits `[Substance/<strategy>]` for paragraph-level substance work, so the
# absence of that marker plus a sub-paragraph before_text is a reliable polish
# signal.
def _is_polish_proposal(proposal: dict) -> bool:
    before = (proposal.get("before_text") or "").strip()
    rationale = (proposal.get("rationale") or "")
    if "[Substance" in rationale:
        return False
    # Sub-paragraph length is the polish tell; paragraph-level edits will
    # have hundreds of characters of before_text.
    return len(before) < 200


def _check_substance_compliance(
    proposals: List[dict],
    doc_type: str,
    analysis_results: dict,
) -> List[str]:
    """For SOP/COVER_LETTER docs, audit proposals against the rhetoric findings.

    Three checks:
      1. Coverage — every high-priority paragraph has a targeting proposal.
      2. Preservation — preserve_sentences from each finding appear verbatim
         in the after_text of any proposal targeting that paragraph.
      3. Mix — polish proposals don't dominate while substance work is
         unaddressed.
    """
    if doc_type not in ("SOP", "COVER_LETTER"):
        return []

    rhetoric = (analysis_results or {}).get("rhetoric") or {}
    findings = rhetoric.get("paragraph_findings") or []
    if not findings:
        return []

    issues: List[str] = []

    # Build a lookup: anchor → finding (for both checks).
    high_priority = [f for f in findings if f.get("priority") == "high"]

    # 1. Coverage check.
    uncovered: List[dict] = []
    for finding in high_priority:
        anchor = finding.get("paragraph_anchor", "")
        if not anchor:
            continue
        norm_anchor = _normalize_for_match(anchor)
        targeted = any(
            norm_anchor in _normalize_for_match(p.get("before_text", ""))
            for p in proposals
        )
        if not targeted:
            uncovered.append(finding)
            section = finding.get("section", "?")
            diagnosis = finding.get("diagnosis", "lacks substance")
            issues.append(
                f"Substance gap: high-priority paragraph in section '{section}' "
                f"({diagnosis}) has no targeting proposal."
            )

    # 2. Preservation check.
    for finding in findings:
        preserve = finding.get("preserve_sentences") or []
        if not preserve:
            continue
        anchor = finding.get("paragraph_anchor", "")
        if not anchor:
            continue
        norm_anchor = _normalize_for_match(anchor)
        # Find any proposal targeting this paragraph.
        targeting = [
            p for p in proposals
            if norm_anchor in _normalize_for_match(p.get("before_text", ""))
        ]
        if not targeting:
            continue
        for prop in targeting:
            after = _normalize_for_match(prop.get("after_text", ""))
            for sentence in preserve:
                if not sentence or not isinstance(sentence, str):
                    continue
                norm_sentence = _normalize_for_match(sentence)
                if norm_sentence and norm_sentence not in after:
                    issues.append(
                        f"Substance violation: proposal targeting section "
                        f"'{finding.get('section', '?')}' dropped a "
                        f"preserve_sentence: '{sentence[:80]}{'...' if len(sentence) > 80 else ''}'"
                    )

    # 3. Mix sanity — only flag when high-priority items are uncovered.
    if uncovered:
        polish_count = sum(1 for p in proposals if _is_polish_proposal(p))
        total = len(proposals)
        if total > 0 and polish_count / total > 0.30:
            issues.append(
                f"Polish-dominated mix: {polish_count}/{total} proposals are "
                f"sentence-level polish while {len(uncovered)} high-priority "
                "paragraph(s) remain without a substance proposal. Substance "
                "work comes first; polish only after."
            )

    return issues


# Banned phrases — must mirror the synth prompt. The total count across all
# after_text values must be at most _AI_TELL_PHRASE_BUDGET.
_BANNED_PHRASES = [
    "i believe my background",
    "see this opportunity as a chance",
    "continue developing myself",
    "i am especially motivated",
    "directly shapes",
    "play a meaningful role",
    "tapestry",
    "delve",
    "delving",
    "stands out to me",
    "matters to me because",
]
_AI_TELL_PHRASE_BUDGET = 1


def _check_narrative_compliance(
    proposals: List[dict],
    doc_type: str,
    analysis_results: dict,
) -> List[str]:
    """Audit proposals against narrative findings.

    Three checks:
      1. Repeated-anchor diversity — for every entry in repeated_anchors,
         at most ONE rewrite/merge proposal may keep that anchor as the
         dominant focus of after_text. The rest must either delete or
         pivot to a different anchor.
      2. Deletion coverage — every paragraph index in paragraphs_to_delete
         must be matched by a delete or merge proposal whose before_text
         contains that paragraph.
      3. Closing commitment — when conclusion_commits_forward is false,
         there must be a proposal whose after_text names the target org
         AND avoids the generic closing patterns.
    """
    if doc_type not in ("SOP", "COVER_LETTER"):
        return []

    narrative = (analysis_results or {}).get("narrative") or {}
    if not narrative:
        return []

    issues: List[str] = []

    # 1. Repeated-anchor diversity.
    repeated = narrative.get("repeated_anchors") or []
    rewrite_or_merge = [
        p for p in proposals
        if (p.get("action") or "rewrite").lower() in ("rewrite", "merge")
    ]
    for entry in repeated:
        # Tuple was serialised as a list by Pydantic's model_dump.
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        anchor = str(entry[0]).strip()
        if not anchor:
            continue
        anchor_lower = anchor.lower()
        # Count how many rewrite/merge proposals end up dominated by the
        # same anchor. We define "dominated" as: the anchor appears in
        # after_text. If a proposal wraps the anchor in passing only, we
        # accept up to one such mention; the test is whether the anchor
        # appears as a focal noun phrase (proxied by appearing 2+ times in
        # after_text or appearing in the first 200 chars).
        focusing = 0
        for p in rewrite_or_merge:
            after = (p.get("after_text") or "").lower()
            if not after:
                continue
            if after.count(anchor_lower) >= 2 or anchor_lower in after[:200]:
                focusing += 1
        if focusing >= 2:
            issues.append(
                f"Narrative violation: anchor '{anchor}' is still the focus "
                f"of {focusing} proposals' after_text — narrative analysis "
                "flagged it as repeated; rewrites must distribute focus "
                "across different anchors."
            )

    # 2. Deletion coverage.
    paragraph_roles = narrative.get("paragraph_roles") or []
    role_index_to_anchor = {
        int(pr.get("paragraph_index", -1)): str(pr.get("paragraph_anchor", "")).strip()
        for pr in paragraph_roles
        if isinstance(pr, dict)
    }
    delete_or_merge_proposals = [
        p for p in proposals
        if (p.get("action") or "rewrite").lower() in ("delete", "merge")
    ]
    for idx in (narrative.get("paragraphs_to_delete") or []):
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue
        anchor = role_index_to_anchor.get(idx_int, "")
        if not anchor:
            continue
        norm_anchor = _normalize_for_match(anchor)
        covered = any(
            norm_anchor in _normalize_for_match(p.get("before_text", ""))
            for p in delete_or_merge_proposals
        )
        if not covered:
            issues.append(
                f"Narrative violation: paragraph {idx_int} ('{anchor[:60]}...') "
                "was flagged for deletion but no delete/merge proposal "
                "targets it. Emit an action='delete' proposal."
            )

    # 3. Closing commitment.
    if narrative.get("conclusion_commits_forward") is False:
        # The closing is the highest paragraph_index in paragraph_roles
        # whose role is "closing"; fall back to the highest index overall.
        closing_anchor = ""
        closing_indices = [
            int(pr.get("paragraph_index", -1))
            for pr in paragraph_roles
            if isinstance(pr, dict) and pr.get("role") == "closing"
        ]
        if closing_indices:
            closing_anchor = role_index_to_anchor.get(max(closing_indices), "")
        elif role_index_to_anchor:
            closing_anchor = role_index_to_anchor.get(max(role_index_to_anchor.keys()), "")
        norm_closing = _normalize_for_match(closing_anchor)
        # Find a rewrite proposal that targets this paragraph.
        targeting = [
            p for p in proposals
            if (p.get("action") or "rewrite").lower() == "rewrite"
            and norm_closing
            and norm_closing in _normalize_for_match(p.get("before_text", ""))
        ]
        if not targeting:
            issues.append(
                "Narrative violation: closing paragraph is generic "
                "(no forward commitment) but no rewrite proposal targets "
                "it. Emit a rewrite that names the target org and a "
                "concrete contribution."
            )
        else:
            # Confirm at least one targeting proposal commits forward —
            # i.e. its after_text avoids the generic patterns.
            generic_patterns = [
                re.compile(r"\bthank\s+you\s+for\s+(your\s+)?(time|consideration|considering)", re.IGNORECASE),
                re.compile(r"\bcontinue\s+developing\s+myself", re.IGNORECASE),
                re.compile(r"\bwould\s+be\s+happy\s+for\s+the\s+opportunity", re.IGNORECASE),
                re.compile(r"\bI\s+believe\s+my\s+background", re.IGNORECASE),
                re.compile(r"\bsee\s+this\s+opportunity\s+as\s+a\s+chance", re.IGNORECASE),
            ]
            forward_committing = any(
                not any(p.search(t.get("after_text") or "") for p in generic_patterns)
                for t in targeting
            )
            if not forward_committing:
                issues.append(
                    "Narrative violation: closing rewrite still uses "
                    "generic sign-off language. The new closing must "
                    "name the target org and a concrete contribution."
                )

    return issues


def _check_ai_tells(proposals: List[dict], doc_type: str) -> List[str]:
    """Em-dash + banned-phrase audit on after_text values.

    SOP/COVER_LETTER only — CV bullets sometimes legitimately use em-dashes
    in date ranges, and the banned-phrase list targets prose AI tells.
    """
    if doc_type not in ("SOP", "COVER_LETTER"):
        return []

    issues: List[str] = []

    for i, p in enumerate(proposals):
        after = p.get("after_text") or ""
        if not after.strip():
            continue
        # 1. Em-dashes / double hyphens.
        if "—" in after or " -- " in after:
            prefix = f"Proposal {i + 1} ({p.get('section', '?')})"
            issues.append(
                f"{prefix}: after_text contains an em-dash (—) or "
                "double-hyphen — these are AI-writing tells. Replace "
                "with a comma, period, semicolon, or colon."
            )

    # 2. Banned-phrase budget — counted across the entire after_text corpus.
    corpus = " ".join((p.get("after_text") or "").lower() for p in proposals)
    total_hits = 0
    hit_phrases: List[str] = []
    for phrase in _BANNED_PHRASES:
        count = corpus.count(phrase)
        if count > 0:
            total_hits += count
            hit_phrases.append(f"'{phrase}' ({count}x)")
    if total_hits > _AI_TELL_PHRASE_BUDGET:
        issues.append(
            f"Banned-phrase budget exceeded ({total_hits} > "
            f"{_AI_TELL_PHRASE_BUDGET}): {', '.join(hit_phrases)}. "
            "These are AI-writing tells — rewrite the after_text values "
            "to avoid them."
        )

    return issues


def _heuristic_evaluate(
    proposals: List[dict],
    raw_text: str,
    profile_snapshot: dict,
    iteration: int,
    doc_type: str = "",
    analysis_results: dict | None = None,
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

    # Substance compliance — SOP/COVER_LETTER only.
    substance_issues = _check_substance_compliance(proposals, doc_type, analysis_results or {})
    all_issues.extend(substance_issues)

    # Narrative compliance — SOP/COVER_LETTER only.
    narrative_issues = _check_narrative_compliance(proposals, doc_type, analysis_results or {})
    all_issues.extend(narrative_issues)

    # AI-tell density — SOP/COVER_LETTER only.
    ai_tell_issues = _check_ai_tells(proposals, doc_type)
    all_issues.extend(ai_tell_issues)

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
    # - Substance gaps + preservation violations + polish-dominated mix are
    #   blocking — they are the failure mode this evaluator was extended for.
    # - Narrative gaps (uncovered deletions, repeated anchors, generic closing)
    #   and AI-tell violations are also blocking — these are the failure modes
    #   the second-pass extension was added for.
    # - Placeholder proposals are a quality note but not blocking on their own
    #   (they come from the heuristic synthesizer and are expected without an LLM).
    format_failures = [i for i in unique_issues if "empty" in i or "missing" in i or "out of range" in i]
    real_proposals = len(proposals) - placeholder_count
    grounding_failure_rate = (
        groundedness_failures / max(real_proposals, 1) if real_proposals > 0 else 0.0
    )

    passed = (
        len(format_failures) == 0
        and grounding_failure_rate <= 0.20
        and len(substance_issues) == 0
        and len(narrative_issues) == 0
        and len(ai_tell_issues) == 0
    )

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
    doc_type = (state.get("doc_classification") or {}).get("doc_type", "")
    analysis_results = state.get("analysis_results") or {}
    # iteration_count tracks how many synthesis→evaluate cycles have completed.
    # Read the current value before this evaluation; we increment it in the return.
    current_iteration = state.get("iteration_count", 0)

    llm = get_llm()
    if llm is None:
        result = _heuristic_evaluate(
            proposals, raw_text, profile_snapshot, current_iteration,
            doc_type=doc_type, analysis_results=analysis_results,
        )
        return {
            **updates,
            "evaluation_result": result.model_dump(),
            "iteration_count": current_iteration + 1,
        }

    proposals_text = json.dumps(proposals, indent=2)[:_MAX_PROPOSALS_CHARS]
    doc_excerpt = raw_text[:_MAX_DOC_CHARS]
    profile_text = json.dumps(profile_snapshot)[:_MAX_PROFILE_CHARS]

    # For SOP/CL include the rhetoric and narrative findings so the LLM can
    # audit substance + mix (rules 5-6) and narrative compliance (rule 7).
    # Stays out of the prompt for CV — neither audit applies there.
    rhetoric_section = ""
    if doc_type in ("SOP", "COVER_LETTER"):
        rhetoric = analysis_results.get("rhetoric") or {}
        narrative = analysis_results.get("narrative") or {}
        if rhetoric:
            rhetoric_section += (
                "\n\nRhetoric analysis (drives substance + mix audit):\n"
                + json.dumps(rhetoric, indent=2)[:2500]
            )
        if narrative:
            rhetoric_section += (
                "\n\nNarrative analysis (drives narrative-compliance audit "
                "— repeated anchors, deletions, closing commitment):\n"
                + json.dumps(narrative, indent=2)[:2500]
            )

    structured = llm.with_structured_output(_EvalOut)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content=(
                f"Iteration: {current_iteration}\n"
                f"Document type: {doc_type or 'UNKNOWN'}\n\n"
                f"Change proposals (JSON):\n{proposals_text}\n\n"
                f"Original document (first {_MAX_DOC_CHARS} chars):\n{doc_excerpt}\n\n"
                f"User profile summary:\n{profile_text}"
                f"{rhetoric_section}"
            )
        ),
    ]

    try:
        out: _EvalOut = structured.invoke(msgs)
        # Always cross-check against the deterministic auditors; the LLM
        # evaluator can be lenient on its own, but substance / narrative /
        # AI-tell rules can be verified mechanically and shouldn't depend
        # on the LLM.
        substance_issues = _check_substance_compliance(proposals, doc_type, analysis_results)
        narrative_issues = _check_narrative_compliance(proposals, doc_type, analysis_results)
        ai_tell_issues = _check_ai_tells(proposals, doc_type)
        merged_issues = list(out.issues or [])
        for iss in substance_issues + narrative_issues + ai_tell_issues:
            if iss not in merged_issues:
                merged_issues.append(iss)
        passed = (
            bool(out.passed)
            and not substance_issues
            and not narrative_issues
            and not ai_tell_issues
        )
        result = EvaluationResult(
            passed=passed,
            issues=merged_issues,
            iteration=current_iteration,
        )
    except Exception as e:
        result = _heuristic_evaluate(
            proposals, raw_text, profile_snapshot, current_iteration,
            doc_type=doc_type, analysis_results=analysis_results,
        )
        result.issues.append(f"[LLM evaluator failed, used heuristic: {e}]")

    return {
        **updates,
        "evaluation_result": result.model_dump(),
        "iteration_count": current_iteration + 1,
    }
