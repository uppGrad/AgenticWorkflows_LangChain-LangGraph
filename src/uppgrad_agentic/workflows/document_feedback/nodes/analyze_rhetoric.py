"""Rhetorical/substance analyzer for SOP and COVER_LETTER documents.

Where the other Phase-2 analyzers measure presentation (style, structure, ATS,
keyword overlap), this one looks at *what the document is arguing*. It scores
paragraphs on four substance dimensions and flags the ones most in need of
strategic rewriting — so the synthesizer can produce paragraph-level proposals
("this paragraph reads generic; tie it to a specific Anthropic product or
recent paper") rather than only sentence-level polish.

Skipped for CV (which is bullet-driven, not argument-driven).
"""
from __future__ import annotations

import re
from typing import List, Literal

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.common.prompt_context import format_profile_brief, format_user_focus


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

Priority = Literal["high", "medium", "low"]
RewriteStrategy = Literal["augment", "restructure", "replace"]


class ParagraphFinding(BaseModel):
    paragraph_anchor: str = Field(
        ...,
        description=(
            "First ~100 characters of the paragraph, copied verbatim from the "
            "document. Used by synthesis as a stable identifier for the "
            "paragraph being critiqued."
        ),
    )
    section: str = Field(
        ...,
        description="Section the paragraph belongs to (e.g. 'Body', 'Opening').",
    )
    company_specificity: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "0 = could be sent to any company in the same industry; "
            "1 = references concrete company-specific signal (product, paper, "
            "value, recent move)."
        ),
    )
    role_specificity: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "0 = generic interest in the function; 1 = engages with what makes "
            "THIS role distinct from a sibling role at the same company."
        ),
    )
    experience_link: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "0 = abstract claims of skill/interest; 1 = a concrete past "
            "experience is tied to a specific role requirement or company need."
        ),
    )
    ownership_impact: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "0 = describes tasks done; 1 = shows decisions owned and outcomes "
            "produced (numbers, before/after, scope)."
        ),
    )
    is_generic: bool = Field(
        ...,
        description=(
            "True when the paragraph would still make sense if the company "
            "name and role were swapped for any peer."
        ),
    )
    preserve_sentences: List[str] = Field(
        default_factory=list,
        description=(
            "Verbatim sentences from this paragraph that already carry real "
            "substance and should SURVIVE the rewrite (e.g. a specific past "
            "experience cited correctly, a real owned outcome, a credible "
            "earned claim). Each entry must be copied character-for-character "
            "from the paragraph. Empty list = nothing salvageable."
        ),
    )
    differentiators: List[str] = Field(
        default_factory=list,
        description=(
            "≤3 short phrases (≤80 chars each) drawn VERBATIM from this "
            "paragraph that uniquely identify THIS candidate: named "
            "projects ('Turkish-language LLM QLoRA fine-tune'), specific "
            "technologies, ownership claims, concrete numeric outcomes. "
            "Stronger contract than preserve_sentences — must survive any "
            "rewrite of this paragraph VERBATIM, not just semantically. "
            "Generic claims ('teamwork', 'problem solving', 'passionate') "
            "do NOT belong here. Empty list = paragraph has no distinctive "
            "specifics worth preserving."
        ),
    )
    rewrite_strategy: RewriteStrategy = Field(
        ...,
        description=(
            "How the synthesizer should treat this paragraph. "
            "'augment' = paragraph has substance on at least one dimension; "
            "the rewrite should ADD a missing dimension while keeping "
            "preserve_sentences intact. "
            "'restructure' = content is partly substantive but framed weakly; "
            "rewrite reorders/relinks the existing material around the "
            "preserve_sentences. "
            "'replace' = paragraph is generic on all four dimensions; safe to "
            "rewrite from scratch."
        ),
    )
    diagnosis: str = Field(
        ...,
        description=(
            "1-2 sentences naming the specific substance gap "
            "(NOT style/grammar). Be concrete."
        ),
    )
    recommended_focus: str = Field(
        ...,
        description=(
            "The substantive angle that would strengthen this paragraph, "
            "drawing on the candidate's actual profile and the actual "
            "opportunity. Never invent details. NOT 'use stronger verbs'."
        ),
    )
    priority: Priority = Field(
        ...,
        description="High = must be rewritten; medium = should; low = nice-to-have.",
    )


class RhetoricAnalysis(BaseModel):
    paragraph_findings: List[ParagraphFinding] = Field(default_factory=list)
    overall_substance_score: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Aggregate substance score across the document. 0 = pure generic "
            "polish; 1 = every paragraph earns its claims with company-, "
            "role-, and experience-specific reasoning."
        ),
    )
    answers_why_company: bool = Field(
        ...,
        description="Does the document credibly answer 'why this company specifically'?",
    )
    answers_why_role: bool = Field(
        ...,
        description="Does the document credibly answer 'why this role specifically'?",
    )
    answers_why_you: bool = Field(
        ...,
        description=(
            "Does the document tie the candidate's actual past to the actual "
            "requirements of the role (not just claim relevant experience)?"
        ),
    )
    summary: str = Field(
        ...,
        description="One sentence overall diagnosis of substance vs. polish.",
    )
    top_priorities: List[str] = Field(
        default_factory=list,
        description=(
            "1-3 paragraph_anchor strings (verbatim from findings) that the "
            "synthesis step should target with paragraph-level rewrite "
            "proposals, ordered by impact."
        ),
    )


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

# Paragraph-level patterns that signal generic, untargeted writing.
_GENERIC_PATTERNS = [
    re.compile(r"\bI\s+am\s+writing\s+to\s+(apply|express)", re.IGNORECASE),
    re.compile(r"\b(your|the)\s+(esteemed|prestigious|reputable)\s+", re.IGNORECASE),
    re.compile(r"\bI\s+am\s+(excited|thrilled|eager|enthusiastic)\s+(to|about)", re.IGNORECASE),
    re.compile(r"\bpassion(ate)?\s+(for|about)\s+(this|the)\s+(field|industry|area)", re.IGNORECASE),
    re.compile(r"\bperfect\s+(fit|match|opportunity)\b", re.IGNORECASE),
    re.compile(r"\bdream\s+(job|company|role)\b", re.IGNORECASE),
    re.compile(r"\bgrow\s+(both\s+)?personally\s+and\s+professionally\b", re.IGNORECASE),
    re.compile(r"\b(team\s+player|hard\s+work(er|ing)|fast\s+learner|highly\s+motivated)\b", re.IGNORECASE),
    re.compile(r"\bcontribute\s+to\s+(your|the)\s+(team|organisation|organization|company|mission)\b", re.IGNORECASE),
]

_NUMERIC_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|x|users?|customers?|requests?|hours?|days?|weeks?|months?|years?|k|m|b)?", re.IGNORECASE)
_PAST_TENSE_ACTION_RE = re.compile(
    r"\b(led|built|designed|launched|shipped|reduced|increased|delivered|"
    r"architected|implemented|developed|migrated|refactored|scaled|optimi[sz]ed|"
    r"owned|drove|grew|cut|eliminated|automated|deployed)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_paragraphs(doc_sections: dict[str, str]) -> List[tuple[str, str]]:
    """Yield (section_name, paragraph_text) pairs. Paragraphs split on blank line."""
    pairs: List[tuple[str, str]] = []
    for name, text in doc_sections.items():
        if name == "Preamble":
            continue
        for chunk in re.split(r"\n\s*\n", text):
            chunk = chunk.strip()
            if len(chunk) >= 80:
                pairs.append((name, chunk))
    return pairs


def _identify_preserve_sentences(paragraph: str) -> List[str]:
    """Heuristic: a sentence is worth preserving if it pairs a past-tense action
    with a numeric outcome OR a named entity. Boilerplate-flagged sentences are
    excluded even if they match the action regex (e.g. "I am writing to express
    my passion ...")."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
    keep: List[str] = []
    for sent in sentences:
        if any(p.search(sent) for p in _GENERIC_PATTERNS):
            continue
        has_action = bool(_PAST_TENSE_ACTION_RE.search(sent))
        has_number = bool(_NUMERIC_RE.search(sent))
        # Loose proper-noun detector: capitalised token mid-sentence.
        has_named_entity = bool(re.search(r"(?<=\s)[A-Z][a-zA-Z0-9]{2,}", sent))
        if has_action and (has_number or has_named_entity):
            keep.append(sent)
    return keep


def _heuristic(
    doc_sections: dict[str, str],
    opportunity_context: dict,
) -> RhetoricAnalysis:
    company = (opportunity_context.get("organization") or opportunity_context.get("company") or "").strip()
    role = (opportunity_context.get("title") or "").strip()
    company_lower = company.lower()
    role_lower = role.lower()

    # Build a set of opportunity-derived tokens — tokens longer than 4 chars
    # from title + keywords. Hits in a paragraph indicate at least topical
    # alignment; absence is a generic-ness signal.
    opp_tokens: set[str] = set()
    for kw in opportunity_context.get("keywords") or []:
        opp_tokens.add(str(kw).lower())
    for word in re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#./-]{4,}\b", role + " " + (opportunity_context.get("description") or "")):
        opp_tokens.add(word.lower())

    pairs = _split_paragraphs(doc_sections)
    findings: List[ParagraphFinding] = []
    high_count = 0

    for section, para in pairs:
        para_lower = para.lower()

        # company_specificity: literal company mention + concrete signal
        company_hits = company_lower and company_lower in para_lower
        company_score = 0.7 if company_hits else 0.1

        # role_specificity: role title mention or opportunity-keyword overlap
        role_hits = bool(role_lower) and role_lower in para_lower
        opp_overlap = len([t for t in opp_tokens if t in para_lower])
        role_score = min(1.0, (0.5 if role_hits else 0.0) + 0.1 * opp_overlap)

        # experience_link: presence of past-tense action verbs near a
        # company/project name or numeric outcome — very rough proxy
        has_past_tense_action = bool(_PAST_TENSE_ACTION_RE.search(para))
        experience_score = 0.6 if has_past_tense_action else 0.2

        # ownership_impact: numeric outcomes
        numeric_hits = len(_NUMERIC_RE.findall(para))
        ownership_score = min(1.0, 0.3 + 0.15 * numeric_hits)

        # Sentences worth preserving even if other dimensions are weak.
        preserve = _identify_preserve_sentences(para)

        # generic-ness: matches any boilerplate pattern
        generic_hits = sum(1 for p in _GENERIC_PATTERNS if p.search(para))
        is_generic = generic_hits >= 2 or (generic_hits >= 1 and company_score < 0.3 and role_score < 0.3)

        # Rewrite strategy. If we found preservable sentences, never go full
        # 'replace' — that's exactly the failure mode the user flagged: nuking
        # a paragraph that has earned content.
        strong_dims = sum(1 for s in (company_score, role_score, experience_score, ownership_score) if s >= 0.5)
        if preserve and strong_dims >= 1:
            rewrite_strategy: RewriteStrategy = "augment"
        elif preserve:
            rewrite_strategy = "restructure"
        elif strong_dims >= 2:
            rewrite_strategy = "augment"
        elif strong_dims == 1:
            rewrite_strategy = "restructure"
        else:
            rewrite_strategy = "replace"

        # priority
        substance_avg = (company_score + role_score + experience_score + ownership_score) / 4
        if substance_avg < 0.35 or is_generic:
            priority: Priority = "high"
            high_count += 1
        elif substance_avg < 0.6:
            priority = "medium"
        else:
            priority = "low"

        diag_bits: List[str] = []
        if not company_hits:
            diag_bits.append(f"never names {company or 'the company'}")
        if role_score < 0.3:
            diag_bits.append("doesn't engage with role-specific signal")
        if experience_score < 0.4:
            diag_bits.append("no concrete past experience cited")
        if ownership_score < 0.45:
            diag_bits.append("no quantified outcome")
        if generic_hits:
            diag_bits.append(f"contains {generic_hits} generic boilerplate phrase(s)")
        diagnosis = "Paragraph " + (", ".join(diag_bits) if diag_bits else "is reasonably grounded") + "."

        recommendation = (
            f"Replace with a paragraph that ties one specific past experience "
            f"to a named requirement of the {role or 'role'}"
            + (f" at {company}" if company else "")
            + " and shows the outcome you produced."
        )

        findings.append(ParagraphFinding(
            paragraph_anchor=para[:100],
            section=section,
            company_specificity=round(company_score, 2),
            role_specificity=round(role_score, 2),
            experience_link=round(experience_score, 2),
            ownership_impact=round(ownership_score, 2),
            is_generic=is_generic,
            preserve_sentences=preserve,
            rewrite_strategy=rewrite_strategy,
            diagnosis=diagnosis,
            recommended_focus=recommendation,
            priority=priority,
        ))

    if findings:
        avg_company = sum(f.company_specificity for f in findings) / len(findings)
        avg_role = sum(f.role_specificity for f in findings) / len(findings)
        avg_exp = sum(f.experience_link for f in findings) / len(findings)
        avg_own = sum(f.ownership_impact for f in findings) / len(findings)
        overall = round((avg_company + avg_role + avg_exp + avg_own) / 4, 2)
    else:
        avg_company = avg_role = avg_exp = avg_own = 0.0
        overall = 0.0

    answers_why_company = avg_company >= 0.4
    answers_why_role = avg_role >= 0.4
    answers_why_you = avg_exp >= 0.45 and avg_own >= 0.4

    if not findings:
        summary = "No analyzable paragraphs detected."
    elif overall >= 0.6:
        summary = "Document earns most of its claims with company- and role-specific reasoning."
    elif high_count >= max(2, len(findings) // 2):
        summary = (
            "Document is well-written but reads generic — most paragraphs would "
            "fit any peer company; substance rewrites needed before polish."
        )
    else:
        summary = (
            f"{high_count} paragraph(s) lack company- or role-specific substance "
            "and should be rewritten before sentence-level polish."
        )

    high_priority = [f.paragraph_anchor for f in findings if f.priority == "high"][:3]

    return RhetoricAnalysis(
        paragraph_findings=findings,
        overall_substance_score=overall,
        answers_why_company=answers_why_company,
        answers_why_role=answers_why_role,
        answers_why_you=answers_why_you,
        summary=summary,
        top_priorities=high_priority,
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM = """You are a senior admissions reader and hiring lead reviewing the \
SUBSTANCE of an application document — NOT its grammar, sentence flow, or word \
choice. Your job is to tell the writer where the argument is generic and what \
specific angle would make it convincing.

For each paragraph (skip headers and short fragments), score four dimensions:

1. **company_specificity** (0-1): Does the paragraph engage with concrete, \
company-specific signal? A score of 0.7+ requires references to actual \
products, papers, values, recent moves, or stated mission elements. Naming \
the company in passing scores ~0.3 — that's not specificity, that's a fill-in.

2. **role_specificity** (0-1): Does the paragraph engage with what makes \
THIS role distinct? "Software engineering" is generic; "frontend work on the \
inference console" is specific. A score of 0.7+ requires the paragraph to \
make sense ONLY for this role, not a sibling role at the same company.

3. **experience_link** (0-1): Is a CONCRETE past experience tied to a \
SPECIFIC role requirement? Vague claims of relevant experience score low. A \
named project + a named requirement + a clear connection scores high.

4. **ownership_impact** (0-1): Does the paragraph show decisions OWNED and \
outcomes PRODUCED — not just tasks done? "Worked on X" is task-level; \
"chose to migrate from X to Y because Z, which reduced N by 40%" is impact.

A paragraph is GENERIC (`is_generic=true`) if it would still make sense \
unchanged if you swapped the company name and role for any peer. The classic \
signs are: "I am writing to apply...", "your esteemed organization", "I am \
passionate about...", "I would be a perfect fit", "team player", "fast \
learner", "contribute to your mission", "grow personally and professionally". \
These phrases are tells, not actual content.

For each finding, give:
- `paragraph_anchor`: first ~100 chars of the paragraph, copied VERBATIM from \
the document. This is how synthesis identifies the paragraph for rewrite.
- `preserve_sentences`: a list of VERBATIM sentences from the paragraph that \
already carry real substance and MUST survive the rewrite. A sentence belongs \
here if it cites a specific past experience correctly, names a real owned \
outcome, or makes a credible earned claim. The synthesizer will keep these \
intact. Be honest — if nothing is salvageable, return an empty list. If the \
whole paragraph is solid, list its sentences. The classic mistake is to flag \
a paragraph as "generic" because of one boilerplate sentence and rewrite it \
from scratch, destroying a real experience claim sitting in the next \
sentence. Don't do that.
- `differentiators`: ≤3 short phrases (≤80 chars) drawn VERBATIM from the \
paragraph that uniquely identify THIS candidate — named projects \
("Turkish-language LLM QLoRA fine-tune", "2D escape-room game in Unity"), \
specific technologies, ownership claims tied to a real outcome, concrete \
numeric figures. These are the SHORTEST chunks that make the candidate \
recognisable; the synthesizer must keep them VERBATIM in any rewrite of \
the paragraph (stronger contract than preserve_sentences, which allows \
paraphrase as long as semantics survive). DO NOT include generic claims: \
"teamwork", "problem solving", "passionate about technology", "I would \
contribute positively" — those apply to any candidate. Empty list when \
the paragraph has no distinctive specifics. The classic failure mode this \
field protects against: the rewrite preserves the paragraph's MEANING but \
strips the candidate's voice — turning "fine-tuned a Turkish-language LLM \
with QLoRA at HAVELSAN" into "applied modern NLP techniques in a \
professional setting". Mark the named anchor as a differentiator and the \
synth has to keep it.
- `rewrite_strategy`: 'augment' if the paragraph has substance on at least \
one dimension and just needs a missing dimension added (most common — \
preserve_sentences should be non-empty); 'restructure' if material is partly \
substantive but framed weakly; 'replace' ONLY if all four dimensions are low \
AND preserve_sentences is empty. Default to augment over replace when in \
doubt — it's safer to add than to nuke.
- `diagnosis`: 1-2 sentences naming what's actually missing — be concrete. \
"Lacks specificity" is useless; "doesn't reference any product, paper, or \
research direction at the company" is useful.
- `recommended_focus`: the substantive angle that would strengthen the \
paragraph. Draw on the candidate's actual profile and the actual opportunity. \
NEVER invent skills, experiences, or company facts not present in the inputs. \
NEVER suggest "use stronger verbs" or "add metrics" — those are style notes.
- `priority`: high if the paragraph is generic OR is a load-bearing "why" \
paragraph that fails its job; medium if it's adequate but flat; low otherwise.

`top_priorities`: 1-3 paragraph_anchor strings (verbatim from your findings) \
that synthesis should rewrite first.

`answers_why_company`, `answers_why_role`, `answers_why_you`: be honest. The \
default state of most documents is "no" on at least one of these.

Hard rules:
- DO NOT invent any company, product, paper, or candidate experience not \
present in the inputs.
- DO NOT critique grammar, passive voice, or sentence length — other \
analyzers handle that.
- A document can be "well-written" and still score low here. That is the \
WHOLE POINT. Surface polish is not what you are measuring.
"""

_MAX_DOC_CHARS = 6000
_MAX_OPP_CHARS = 2500


# ---------------------------------------------------------------------------
# Node — receives context_pack via Send
# ---------------------------------------------------------------------------

def analyze_rhetoric(context_pack: dict) -> dict:
    updates = {"step_history": ["analyze_rhetoric"]}
    doc_type = context_pack.get("doc_type", "UNKNOWN")

    # Rhetorical/substance analysis is meaningful only for argument-shaped docs.
    if doc_type not in ("SOP", "COVER_LETTER"):
        return {**updates, "analysis_results": {"rhetoric": {}}}

    doc_sections = context_pack.get("doc_sections") or {}
    opportunity_context = context_pack.get("opportunity_context") or {}

    # If there's no opportunity context, the substance critique can't ground
    # company-/role-specificity claims. Fall back to heuristic which still
    # detects generic boilerplate and missing experience-linkage.
    llm = get_llm()
    if llm is None:
        result = _heuristic(doc_sections, opportunity_context)
        return {**updates, "analysis_results": {"rhetoric": result.model_dump()}}

    # Render paragraphs so the LLM can anchor findings to verbatim slices.
    para_block: List[str] = []
    for section, para in _split_paragraphs(doc_sections):
        para_block.append(f"[Section: {section}]\n{para}")
    paragraphs_text = "\n\n---\n\n".join(para_block)[:_MAX_DOC_CHARS]

    if not paragraphs_text:
        result = _heuristic(doc_sections, opportunity_context)
        return {**updates, "analysis_results": {"rhetoric": result.model_dump()}}

    opp_text = str(opportunity_context)[:_MAX_OPP_CHARS] if opportunity_context else "(no opportunity context provided — score generic-ness only; do not fabricate company specifics)"

    user_focus = format_user_focus(context_pack.get("parsed_instructions"))
    profile_brief = format_profile_brief(context_pack.get("profile_snapshot"))

    body = (
        f"Document type: {doc_type}\n\n"
        f"Target opportunity:\n{opp_text}\n\n"
        f"Document paragraphs (each delimited by `---`):\n{paragraphs_text}"
    )
    if profile_brief:
        body += f"\n\n{profile_brief}"
    if user_focus:
        body += f"\n\n{user_focus}"

    structured = llm.with_structured_output(RhetoricAnalysis)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=body),
    ]

    try:
        result: RhetoricAnalysis = structured.invoke(msgs)
        return {**updates, "analysis_results": {"rhetoric": result.model_dump()}}
    except Exception as e:
        result = _heuristic(doc_sections, opportunity_context)
        out = result.model_dump()
        out["summary"] = (out.get("summary") or "") + f" [LLM failed, used heuristic: {e}]"
        return {**updates, "analysis_results": {"rhetoric": out}}
