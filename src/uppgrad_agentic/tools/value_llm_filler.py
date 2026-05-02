"""Tier 4b — post-deterministic value LLM.

Sits between value_planner (Tier 1-3, deterministic) and the playwright
filler. For plan entries the deterministic tiers couldn't fill (status=
'skipped', source='no_value'), this module asks the LLM whether the
profile + CV give a confident answer for the labelled question.

What this catches:
  * "Years of Python experience" — LLM reads CV, returns "4".
  * "Highest degree" — LLM reads profile, returns "Bachelor's in CS".
  * "What is your nationality?" — profile may have it; LLM returns it.

What this DOESN'T do:
  * Fabricate values. The structured output explicitly accepts None as
    "I don't know"; null returns are not promoted.
  * Touch sensitive fields. A deny-list filters labels that look like
    compensation, identifiers, or other compliance-y questions BEFORE
    the LLM is even called.
  * Cross the budget cap. Default 5 calls per session; remaining
    skipped entries stay skipped (and surface as gate-2 Quick Questions).

Pure function — no LangGraph, no global state. Backend adapter calls
this from `attempt_auto_fill` after `_inject_tailored_answers`. Mirrors
the value_planner / playwright_filler.tier-4 pattern of "small, focused
LLM helper".
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from uppgrad_agentic.workflows.auto_apply.schemas import (
    FormField,
    FormFieldFillPlan,
)

logger = logging.getLogger(__name__)


# ─── Deny-list ──────────────────────────────────────────────────────────────
#
# These label fragments cause the filler to skip without an LLM call. The
# common theme: any answer would be either compliance-sensitive (we should
# never guess) or compensation-related (we explicitly refuse to fabricate
# numbers, mirroring `_SYSTEM_GENERATE_TEXT` in application_tailoring).
_DENY_LABEL_FRAGMENTS = (
    # Compensation
    "salary",
    "compensation",
    "base pay",
    "hourly rate",
    "day rate",
    "expected pay",
    "expected wage",
    # Personal identifiers
    "ssn",
    "social security",
    "national id",
    "passport number",
    "id number",
    "tax id",
    # Sensitive demographics where guessing is harmful
    "date of birth",
    "birth date",
    "ethnicity",
    "race",
    "gender identity",
    "sexual orientation",
    "religion",
    "marital status",
    "disability",
    "veteran status",
    # Free-form long answers (those go through application_tailoring's
    # auto_generate path at gate 1, not here).
    "cover letter",
    "tell us why",
    "describe a time",
)


def _label_is_denied(label: str) -> bool:
    if not label:
        return False
    norm = label.lower()
    return any(frag in norm for frag in _DENY_LABEL_FRAGMENTS)


def _is_eligible_skip(plan: FormFieldFillPlan) -> bool:
    """Tier 4b only acts on entries the deterministic tiers gave up on."""
    if plan.status != "skipped":
        return False
    if plan.source != "no_value":
        return False
    label = (plan.field.label or "").strip()
    if len(label) < 3:
        return False
    return not _label_is_denied(label)


# ─── LLM contract ───────────────────────────────────────────────────────────


class FieldGuess(BaseModel):
    """Structured output from the value LLM. `value=None` means
    "I don't know" — never fabricate."""
    value: Optional[str] = Field(
        default=None,
        description=(
            "The value to enter into the field. Use null when the profile/CV "
            "does NOT clearly answer this question — never guess."
        ),
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="0.0-1.0 confidence the answer is correct given the source material.",
    )
    reason: str = Field(
        default="",
        description="One short sentence: where in the CV/profile the answer comes from.",
    )


_SYSTEM = """You are filling a single field on a job application form using the
candidate's profile and CV as the source of truth.

You are given:
  * The form field's label and its acceptable shape (text / number / select with
    options / etc.)
  * The candidate's flat profile snapshot (name, email, phone, location, skills,
    education, github, linkedin, etc.)
  * The candidate's CV text in full
  * The opportunity context (title, company, description excerpt)

Your job: return a value to enter into that field, OR null if the source
material doesn't clearly answer.

STRICT RULES:
  1. Use ONLY facts that appear in the profile or the CV. Do NOT invent
     numbers, dates, employers, qualifications, or any other facts.
  2. For "Years of <skill>" questions, count from the CV's earliest
     dated reference to that skill until present. If the CV doesn't
     give dates, return null.
  3. For yes/no questions, only answer "Yes" when the source explicitly
     supports it (e.g. "Are you authorised to work in the EU?" → answer
     only if profile.location/nationality makes it unambiguous).
  4. For select/radio fields, the value MUST be one of the provided options
     verbatim. If none of the options matches a truthful answer, return null.
  5. For checkboxes, return "true" only when the source explicitly supports
     it; otherwise null.
  6. NEVER produce compensation figures, ID numbers, demographic answers,
     or any value the candidate hasn't explicitly stated somewhere in the
     profile or CV.
  7. Keep text answers short — under 80 characters unless the field
     explicitly accepts longer (textarea). For number-style answers
     return just the number as a string.
  8. When in doubt, return null. Returning null surfaces the field as a
     "Quick Question" the user answers themselves — that's a strictly
     safer outcome than a guessed value getting submitted.
"""


# ─── CV trimming ────────────────────────────────────────────────────────────
#
# Keep the CV portion of the prompt bounded so we don't send 30k chars per
# call. 6k chars is enough to capture the header + 2-3 most recent roles +
# skills section for a typical resume, which is where 90% of the answers
# this tier wants come from.
_CV_PROMPT_CAP = 6_000


def _trim_cv(cv_text: str) -> str:
    if not cv_text:
        return ""
    text = cv_text.strip()
    if len(text) <= _CV_PROMPT_CAP:
        return text
    # Keep header + start of body. Anything past the cap is unlikely to
    # contain answer-relevant facts (older roles, references).
    return text[:_CV_PROMPT_CAP] + "\n\n[... CV truncated ...]"


def _profile_summary_for_prompt(profile: Dict[str, Any]) -> str:
    """Compact, predictable rendering of the flat profile snapshot. Skips
    keys whose values are empty / None so the LLM doesn't see noise."""
    if not profile:
        return "(no profile available)"
    keys_in_order = [
        "name", "first_name", "last_name", "email", "phone",
        "city", "country", "location",
        "linkedin", "github", "website",
        "nationality", "age", "degree_level", "gpa",
        "disciplines",
    ]
    lines = []
    for k in keys_in_order:
        v = profile.get(k)
        if v in (None, "", [], {}):
            continue
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v if x)
            if not v:
                continue
        lines.append(f"  {k}: {v}")
    return "\n".join(lines) if lines else "(profile snapshot empty)"


def _opp_summary_for_prompt(opportunity_data: Dict[str, Any]) -> str:
    title = opportunity_data.get("title") or ""
    company = (
        opportunity_data.get("company")
        or opportunity_data.get("university")
        or opportunity_data.get("provider_name")
        or ""
    )
    desc = (opportunity_data.get("description") or "")[:1200]
    return f"  title: {title}\n  organisation: {company}\n  description: {desc}"


def _build_field_block(field: FormField) -> str:
    parts = [f"label: {field.label!r}", f"type: {field.field_type}"]
    if field.required:
        parts.append("required: true")
    if field.options:
        parts.append(f"options: {field.options[:12]}")
    if field.expected_source and field.expected_source != "unknown":
        parts.append(f"hint: expected_source={field.expected_source}")
    return "\n  ".join(parts)


# ─── LLM call wrapper ───────────────────────────────────────────────────────


def _ask_llm_for_field(
    field: FormField,
    profile: Dict[str, Any],
    cv_text: str,
    opportunity_data: Dict[str, Any],
    *,
    llm,
) -> Optional[FieldGuess]:
    """Single bounded LLM call. Returns None on any error / refusal /
    null-value response so the caller falls back to "leave skipped"."""
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        structured = llm.with_structured_output(FieldGuess)
    except Exception as exc:
        logger.warning("value_llm_filler: structured-output not available — %s", exc)
        return None

    user_prompt = (
        f"=== FIELD ===\n  {_build_field_block(field)}\n\n"
        f"=== PROFILE ===\n{_profile_summary_for_prompt(profile)}\n\n"
        f"=== CV ===\n{_trim_cv(cv_text)}\n\n"
        f"=== OPPORTUNITY ===\n{_opp_summary_for_prompt(opportunity_data)}\n\n"
        "Return your guess as a FieldGuess object. value=null when the source "
        "material doesn't clearly answer."
    )

    try:
        guess: FieldGuess = structured.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_prompt),
        ])
    except Exception as exc:
        logger.warning(
            "value_llm_filler: LLM invoke failed for label=%r — %s",
            field.label, exc,
        )
        return None

    if guess.value in (None, ""):
        return None
    if guess.confidence < 0.5:
        # We treat low-confidence guesses as "I don't know" rather than
        # submitting a maybe-wrong answer.
        logger.info(
            "value_llm_filler: low-confidence skip for label=%r (conf=%.2f reason=%r)",
            field.label, guess.confidence, guess.reason,
        )
        return None
    # If the field has options, the value MUST be in them verbatim. Defends
    # against the LLM picking a synonym the form won't accept.
    if field.options and guess.value not in field.options:
        logger.info(
            "value_llm_filler: value %r not in options %r — dropping",
            guess.value, field.options[:8],
        )
        return None
    return guess


# ─── Public entrypoint ──────────────────────────────────────────────────────


def llm_fill_skipped_fields(
    plan: List[FormFieldFillPlan],
    profile: Dict[str, Any],
    cv_text: str,
    opportunity_data: Dict[str, Any],
    *,
    llm,
    budget: int = 5,
) -> List[FormFieldFillPlan]:
    """Return a NEW plan list where eligible skipped+no_value entries
    are promoted to filled+llm_inferred when the LLM gives a confident
    non-null answer.

    Bounded by `budget` LLM calls. When `llm` is None, returns the plan
    unchanged — same fallback contract as the rest of the agentic stack.
    """
    if llm is None or not plan:
        return list(plan)

    out: List[FormFieldFillPlan] = []
    calls_made = 0
    upgraded = 0
    for entry in plan:
        if not _is_eligible_skip(entry):
            out.append(entry)
            continue
        if calls_made >= budget:
            # Budget exhausted — leave the rest skipped.
            out.append(entry)
            continue
        calls_made += 1
        guess = _ask_llm_for_field(
            entry.field, profile, cv_text, opportunity_data, llm=llm,
        )
        if guess is None:
            out.append(entry)
            continue
        upgraded += 1
        out.append(
            FormFieldFillPlan(
                field=entry.field,
                value=str(guess.value),
                status="filled",
                source="llm_inferred",
                reason=(
                    f"llm_inferred conf={guess.confidence:.2f}: "
                    + (guess.reason or "")[:80]
                ),
            )
        )

    logger.info(
        "value_llm_filler: %d/%d skipped fields promoted (budget=%d, calls_used=%d)",
        upgraded, sum(1 for p in plan if _is_eligible_skip(p)),
        budget, calls_made,
    )
    return out
