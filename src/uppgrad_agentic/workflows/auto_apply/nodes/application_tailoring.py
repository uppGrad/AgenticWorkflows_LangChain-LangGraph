"""Application tailoring (Step 6 rewrite).

Consumes the gate-1 per-requirement choices in `state['human_review_1']`
and produces:
  - state['tailored_documents']   — full content per document
  - state['tailored_answers']     — per-text-question content keyed by
                                    form_field_index (string)

Branches per requirement:
  category=document, choice=upload         → PreA → T1 → LA → T2  (always 2-pass)
  category=document, choice=auto_generate  → single tailoring call
  category=text,     choice=auto_generate  → single LLM call (1500-char cap)
  choice in {ignore_for_now, skip}         → no output produced
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.tools.latex_templates import template_for
from uppgrad_agentic.workflows.auto_apply._profile import resolve_profile
from uppgrad_agentic.workflows.auto_apply.nodes.upload_pre_analysis import analyze_upload_pre
from uppgrad_agentic.workflows.auto_apply.nodes.upload_light_post_analysis import (
    analyze_upload_light_post,
)
from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState
from uppgrad_agentic.workflows.document_feedback.graph import build_auto_tailoring_graph

logger = logging.getLogger(__name__)

_MAX_SOURCE_CHARS = 6_000
_MAX_OPP_CHARS = 3_000
_MAX_TEXT_ANSWER_CHARS = 1_500

# Per-doc-type output caps (preserved from previous implementation)
_DOC_TYPE_CAPS = {
    "CV": 8000,
    "Cover Letter": 3000,
    "SOP": 6000,
    "Personal Statement": 6000,
}
_DEFAULT_CAP = 5000


def _truncate_to_cap(content: str, doc_type: str) -> str:
    cap = _DOC_TYPE_CAPS.get(doc_type, _DEFAULT_CAP)
    if len(content) <= cap:
        return content
    boundary = content.rfind("\n\n", 0, cap)
    if boundary > cap // 2:
        return content[:boundary]
    return content[:cap]


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# LaTeX helpers (Sub-PR A — tailored documents now carry both plain text
# `content` and `latex_source` for the backend's LaTeX renderer)
# ---------------------------------------------------------------------------

# Patterns kept conservative so plain-text extraction stays robust to
# LLM-introduced quirks like newlines inside arguments. We strip the most
# common scaffolding (`\\section{x}` → `x`, `\\textbf{x}` → `x`, hyperref
# `\\href{url}{label}` → `label`, the `% --- BEGIN/END BODY ---` markers,
# the preamble/postamble) without trying to be a real TeX parser.
_LATEX_PREAMBLE_RE = re.compile(r"\\documentclass.*?\\begin\{document\}", re.DOTALL)
_LATEX_POSTAMBLE_RE = re.compile(r"\\end\{document\}.*", re.DOTALL)
_LATEX_BODY_MARKERS_RE = re.compile(
    r"%\s*---\s*(BEGIN|END)\s+BODY\s*---[^\n]*\n?", re.IGNORECASE
)
_LATEX_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*\n?")
_LATEX_HREF_RE = re.compile(r"\\href\{[^}]*\}\{([^}]*)\}")
_LATEX_SIMPLE_CMD_WITH_ARG_RE = re.compile(r"\\[a-zA-Z]+\*?\{([^{}]*)\}")
_LATEX_SIMPLE_CMD_NO_ARG_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?")
_LATEX_ENV_RE = re.compile(r"\\(begin|end)\{[^}]*\}")
_LATEX_ITEM_RE = re.compile(r"^\s*\\item\s*", re.MULTILINE)


def _latex_to_plain(latex_source: str) -> str:
    """Strip LaTeX scaffolding to a plain-text approximation of the document.

    Used to populate the legacy `content` field on `tailored_documents`
    entries — `application_evaluation` reads it for length / placeholder /
    keyword checks, and the dashboard surfaces it as a quick preview.
    The PDF the user actually downloads is rendered from `latex_source`,
    not from this string.

    Best-effort, not a real TeX parser. Robust enough to handle the
    skeletons in `tools/latex_templates` plus typical LLM output shape.
    """
    if not latex_source:
        return ""
    text = latex_source
    # Drop preamble + postamble first, otherwise their commands leak into
    # the simple-command pass below.
    text = _LATEX_PREAMBLE_RE.sub("", text)
    text = _LATEX_POSTAMBLE_RE.sub("", text)
    text = _LATEX_BODY_MARKERS_RE.sub("", text)
    text = _LATEX_COMMENT_RE.sub("", text)
    # Hyperlinks: `\href{url}{label}` → `label`.
    text = _LATEX_HREF_RE.sub(lambda m: m.group(1), text)
    # Drop `\begin{X}` / `\end{X}` BEFORE the generic command-with-arg
    # pass below, otherwise that pass turns `\begin{itemize}` into
    # `itemize` and we leak the env name into the plain text.
    text = _LATEX_ENV_RE.sub("", text)
    # `\item` keeps the bullet semantics with a leading dash.
    text = _LATEX_ITEM_RE.sub("- ", text)
    # `\section{X}` / `\textbf{X}` / `\emph{X}` etc. → `X`. Run twice to
    # peel one layer of nested commands (e.g. `\textbf{\large{X}}`).
    for _ in range(2):
        text = _LATEX_SIMPLE_CMD_WITH_ARG_RE.sub(lambda m: m.group(1), text)
    # Lone commands like `\noindent`, `\\`, `\hfill` → drop.
    text = _LATEX_SIMPLE_CMD_NO_ARG_RE.sub("", text)
    # Tidy whitespace: collapse 3+ blank lines to 2.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_latex_source(raw: str) -> str:
    """Pull the LaTeX source out of an LLM response.

    Accepts either a plain `\\documentclass...\\end{document}` or a fenced
    block (```latex / ```tex / ```). Returns "" when no document marker is
    present — caller should treat that as "LaTeX generation failed".
    """
    if not raw:
        return ""
    # First strip a top-level code fence if present.
    candidate = _strip_fences(raw)
    if r"\documentclass" not in candidate or r"\end{document}" not in candidate:
        return ""
    # Trim any prose that leaked before \documentclass or after \end{document}.
    start = candidate.find(r"\documentclass")
    end_marker = candidate.find(r"\end{document}")
    if start < 0 or end_marker < 0:
        return ""
    end = end_marker + len(r"\end{document}")
    return candidate[start:end].strip()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# ─── Output format note ──────────────────────────────────────────────────────
#
# Every document-tailoring prompt below ends with a LaTeX skeleton (the
# "TEMPLATE" block). The LLM is told to emit a complete, compilable LaTeX
# source: the preamble must stay byte-for-byte identical to the template,
# only the body between the BEGIN/END BODY markers gets filled. We extract
# `latex_source` from the response and derive a plain-text `content` from
# it via `_latex_to_plain` for legacy consumers (evaluation node, dashboard
# preview).

_LATEX_OUTPUT_RULES = """
Return ONLY a complete LaTeX document. No prose around it, no markdown fences,
no "Here is your..." preamble.

Strict rules for the LaTeX you emit:
  * Keep the preamble (everything from \\documentclass through \\begin{document})
    EXACTLY as given in the TEMPLATE block — do not change packages, fonts,
    margins, or commands. The renderer only ships those packages.
  * Keep the BEGIN BODY / END BODY marker comments. Replace ONLY the
    placeholder comment between them with the actual document content.
  * Use ONLY commands defined in the preamble. Do not add \\usepackage{...}
    lines, do not call \\input or \\include, do not use shell-escape commands.
  * For CV documents, use the resume helpers defined in the preamble:
      \\section{Section Name}
      \\resumeSubHeadingListStart / \\resumeSubHeadingListEnd
      \\resumeSubheading{ORG}{LOCATION}{TITLE/DEGREE}{DATES}
      \\resumeItemListStart / \\resumeItemListEnd
      \\resumeItemPlain{Bullet text}        % unlabelled bullet
      \\resumeItem{Label}{Bullet text}      % labelled (e.g. category headers)
      \\resumeSubItem{Label}{Description}   % nested sub-bullet
    Do NOT invent additional commands.
  * For COVER LETTER / SOP / Personal Statement / Motivation Letter
    documents, the prose template has NO list helpers. NEVER use
    \\begin{itemize}, \\begin{enumerate}, \\item, or any \\resume*
    command — they don't exist in this preamble and will fail to compile.
    Render every paragraph as flowing prose, separated by ONE blank line.
  * Escape LaTeX-special characters in user content (this includes the
    URL portion of \\href{URL}{LABEL} — underscores in email addresses
    must be \\_ even inside the URL argument):
      &  →  \\&     %  →  \\%     $  →  \\$     #  →  \\#
      _  →  \\_     {  →  \\{     }  →  \\}     ~  →  \\textasciitilde{}
      ^  →  \\textasciicircum{}
  * For URLs use \\href{https://...}{label}. Plain URLs without \\href will
    misformat. For mailto links: \\href{mailto:user\\_name@example.com}{user\\_name@example.com}
    — note both arguments need _ escaped.
  * End with \\end{document}.
"""


_SYSTEM_T1 = """You are a professional application document writer running a FIRST tailoring pass.

You will be given:
  - the document type
  - the opportunity (title, organisation, description)
  - the user's profile summary
  - the user's CV (always provided as base context)
  - an UPLOADED version of the document the user wants tailored
  - a pre-tailoring analysis with top priorities
  - an optional user-provided guidance prompt for this document
  - a LaTeX TEMPLATE skeleton you must use for the output

Produce a tailored revision that addresses the top_priorities and aligns the
content with the opportunity. Preserve all factual details from the source —
do NOT fabricate roles, dates, qualifications, or achievements. Mirror the
opportunity's language where truthful.
""" + _LATEX_OUTPUT_RULES


_SYSTEM_T2 = """You are polishing a tailored application document on a SECOND, final pass.

You will receive:
  - the T1 LaTeX output (already tailored once)
  - a post-T1 analysis listing remaining structure issues, content gaps vs
    the opportunity, and content gaps vs the user's profile
  - the same opportunity / profile / CV / user_prompt context
  - the original LaTeX TEMPLATE skeleton

Address the analysis findings in place. Do not rewrite the whole document.
Preserve facts. Mirror opportunity language.
""" + _LATEX_OUTPUT_RULES


_SYSTEM_GENERATE_DOC = """You are generating a job application document from scratch.

You will be given the document type, the opportunity (title, organisation,
description), the user's profile summary, the user's CV as the source of
factual details, an optional user prompt, the canonical document type
(if known), and a LaTeX TEMPLATE skeleton you must use for the output.

Use ONLY facts present in the source material — do not invent dates,
employers, qualifications, or achievements. Structure the document
appropriately for its type. Mirror the opportunity's language where
truthful.

Do NOT include unfilled placeholders such as [Date], [Address],
[Hiring Manager Name], [Today's Date], or any other bracketed/parenthesised
fill-in markers. If you don't have a specific value, omit that line entirely
rather than emitting a placeholder.
""" + _LATEX_OUTPUT_RULES


_SYSTEM_GENERATE_TEXT = """You are answering a free-form question on a job application form.

You will be given:
  - the question (verbatim from the form)
  - the opportunity (title, organisation, description)
  - the user's profile summary
  - the user's CV

Write a single concise answer (1-2 short paragraphs, target 800-1200
characters) that directly addresses the question using truthful details
from the user's profile and CV. Do not invent facts.

If the question asks for compensation expectations (salary, base pay,
hourly rate, day rate, bonus, equity), DO NOT fabricate a specific number.
Write a brief answer indicating the user is open to discussing
compensation aligned with the role's responsibilities and the local
market, and would welcome a conversation once the team shares their
range. Do not produce a concrete figure or currency amount.

Return ONLY the answer text — no labels, no quotes, no markdown fences."""


def _opp_context(opportunity_data: Dict[str, Any], opportunity_type: str) -> str:
    title = opportunity_data.get("title", "Unknown role")
    company = (
        opportunity_data.get("company")
        or opportunity_data.get("university")
        or opportunity_data.get("provider_name")
        or "Unknown organisation"
    )
    description = (
        opportunity_data.get("description")
        or str((opportunity_data.get("data") or {}).get("description", ""))
        or ""
    )
    return (
        f"=== OPPORTUNITY ===\n"
        f"Title: {title}\n"
        f"Organisation: {company}\n"
        f"Type: {opportunity_type}\n"
        f"Description:\n{description[:_MAX_OPP_CHARS]}\n"
    )


def _profile_summary(profile: Dict[str, Any]) -> str:
    parts: List[str] = []
    if profile.get("name"):
        parts.append(f"Name: {profile['name']}")
    if profile.get("email"):
        parts.append(f"Email: {profile['email']}")
    if profile.get("location"):
        parts.append(f"Location: {profile['location']}")
    if profile.get("degree_level"):
        parts.append(f"Highest degree: {profile['degree_level']}")
    if profile.get("disciplines"):
        parts.append(f"Disciplines: {', '.join(profile['disciplines'])}")
    if profile.get("bio"):
        parts.append(f"Bio: {profile['bio']}")
    if profile.get("projects"):
        parts.append(f"Projects: {profile['projects']}")
    if profile.get("publications"):
        parts.append(f"Publications: {profile['publications']}")
    if profile.get("achievements"):
        parts.append(f"Achievements: {profile['achievements']}")
    return "\n".join(parts) if parts else "(no profile details available)"


def _cv_text(profile: Dict[str, Any]) -> str:
    doc_texts = profile.get("document_texts") or {}
    return (doc_texts.get("CV") or "")[:_MAX_SOURCE_CHARS]


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _llm_call(llm, system: str, user: str) -> Optional[str]:
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        text = (resp.content or "").strip()
        return _strip_fences(text) if text else None
    except Exception as exc:
        logger.warning("application_tailoring: LLM call failed — %s", exc)
        return None


def _session_instructions_block(user_instructions: Optional[str]) -> str:
    """Format the session-wide custom instructions for prompt injection.

    Returns "" when blank so the prompt stays clean. The block is
    deliberately separate from `=== USER GUIDANCE ===` (per-document
    user_prompt) so the LLM can weight them independently — the
    session-level instructions are global directives that apply to
    every artifact in the session.
    """
    text = (user_instructions or "").strip()
    if not text:
        return ""
    return f"=== SESSION-WIDE CUSTOM INSTRUCTIONS (apply across all artifacts) ===\n{text}\n\n"


def _t1_prompt(
    doc_type: str,
    opportunity_data: Dict[str, Any],
    opportunity_type: str,
    profile: Dict[str, Any],
    uploaded_text: str,
    pre_analysis,
    user_prompt: Optional[str],
    user_instructions: Optional[str],
) -> str:
    return (
        f"Document type: {doc_type}\n\n"
        + _session_instructions_block(user_instructions)
        + _opp_context(opportunity_data, opportunity_type) + "\n"
        + f"=== USER PROFILE ===\n{_profile_summary(profile)}\n\n"
        + f"=== USER CV ===\n{_cv_text(profile)}\n\n"
        + f"=== USER GUIDANCE ===\n{user_prompt or '(none)'}\n\n"
        + f"=== UPLOADED {doc_type.upper()} ===\n{uploaded_text[:_MAX_SOURCE_CHARS]}\n\n"
        + "=== PRE-TAILORING ANALYSIS ===\n"
        + f"Completeness: {pre_analysis.completeness}\n"
        + f"Relevance: {pre_analysis.relevance}\n"
        + f"Correctness: {pre_analysis.correctness}\n"
        + f"Overall quality: {pre_analysis.overall_quality}\n"
        + "Top priorities:\n"
        + ("\n".join(f"- {p}" for p in pre_analysis.top_priorities) or "- (none flagged)")
        + "\n\n=== TEMPLATE (return a filled-in copy of this) ===\n"
        + template_for(doc_type)
        + "\n\nProduce the T1 tailored document now."
    )


def _t2_prompt(
    doc_type: str,
    opportunity_data: Dict[str, Any],
    opportunity_type: str,
    profile: Dict[str, Any],
    t1_output: str,
    post_analysis,
    user_prompt: Optional[str],
    user_instructions: Optional[str],
) -> str:
    return (
        f"Document type: {doc_type}\n\n"
        + _session_instructions_block(user_instructions)
        + _opp_context(opportunity_data, opportunity_type) + "\n"
        + f"=== USER PROFILE ===\n{_profile_summary(profile)}\n\n"
        + f"=== USER CV ===\n{_cv_text(profile)}\n\n"
        + f"=== USER GUIDANCE ===\n{user_prompt or '(none)'}\n\n"
        + f"=== T1 OUTPUT (LaTeX, {doc_type}) ===\n{t1_output[:_MAX_SOURCE_CHARS]}\n\n"
        + "=== POST-T1 ANALYSIS ===\n"
        + "Structure issues:\n"
        + ("\n".join(f"- {s}" for s in post_analysis.structure_issues) or "- (none)")
        + "\nContent gaps vs opportunity:\n"
        + ("\n".join(f"- {s}" for s in post_analysis.content_gap_vs_opportunity) or "- (none)")
        + "\nContent gaps vs profile:\n"
        + ("\n".join(f"- {s}" for s in post_analysis.content_gap_vs_profile) or "- (none)")
        + "\n\n=== TEMPLATE (preamble must remain identical to this) ===\n"
        + template_for(doc_type)
        + "\n\nPolish the T1 document in place to address the analysis. Return the final LaTeX source."
    )


def _generate_doc_prompt(
    doc_type: str,
    canonical_type: Optional[str],
    opportunity_data: Dict[str, Any],
    opportunity_type: str,
    profile: Dict[str, Any],
    user_prompt: Optional[str],
    user_instructions: Optional[str],
) -> str:
    return (
        f"Document type: {doc_type}\n"
        + (f"Canonical type: {canonical_type}\n" if canonical_type else "")
        + "\n"
        + _session_instructions_block(user_instructions)
        + _opp_context(opportunity_data, opportunity_type) + "\n"
        + f"=== USER PROFILE ===\n{_profile_summary(profile)}\n\n"
        + f"=== USER CV (source of facts) ===\n{_cv_text(profile)}\n\n"
        + f"=== USER GUIDANCE ===\n{user_prompt or '(none)'}\n\n"
        + "=== TEMPLATE (return a filled-in copy of this) ===\n"
        + template_for(canonical_type or doc_type)
        + f"\n\nGenerate the {doc_type} now."
    )


def _generate_text_prompt(
    question: str,
    opportunity_data: Dict[str, Any],
    opportunity_type: str,
    profile: Dict[str, Any],
    user_instructions: Optional[str],
) -> str:
    return (
        f"Question: {question}\n\n"
        + _session_instructions_block(user_instructions)
        + _opp_context(opportunity_data, opportunity_type) + "\n"
        + f"=== USER PROFILE ===\n{_profile_summary(profile)}\n\n"
        + f"=== USER CV ===\n{_cv_text(profile)}\n\n"
        + "Write the answer now."
    )


# ---------------------------------------------------------------------------
# Per-requirement processing
# ---------------------------------------------------------------------------

def _split_latex_and_plain(raw: str, doc_type: str) -> tuple[str, str]:
    """Return `(content, latex_source)` from an LLM response.

    The system prompts ask for a complete LaTeX document. We pull the
    `\\documentclass...\\end{document}` span out (tolerant of stray prose
    or fenced wrappers) and derive a plain-text approximation for the
    legacy `content` field used by `application_evaluation` and dashboard
    previews. When the LLM returns no LaTeX (older model, structured-output
    miss, etc.), `latex_source` is "" and `content` falls back to the raw
    truncated string so we degrade rather than lose the work.
    """
    latex_source = _extract_latex_source(raw)
    if latex_source:
        plain = _latex_to_plain(latex_source)
    else:
        plain = _strip_fences(raw)
    return _truncate_to_cap(plain, doc_type), latex_source


# Auto-apply canonical doc types → doc-feedback DocType literals. Anything
# not in this map (e.g. "Motivation Letter" — no doc-feedback analog) falls
# back to the legacy T1→T2 path.
_DOC_FEEDBACK_TYPE_MAP = {
    "CV": "CV",
    "Cover Letter": "COVER_LETTER",
    "Motivation Letter": "COVER_LETTER",
    "SOP": "SOP",
    "Personal Statement": "SOP",
}


def _tailor_via_doc_feedback(
    doc_type: str,
    uploaded_text: str,
    user_prompt: Optional[str],
    opportunity_data: Dict[str, Any],
    opportunity_type: str,
    profile: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Drive the doc-feedback graph in auto-tailoring mode against the
    user's uploaded document. Returns a tailored_documents entry on
    success, None on any failure (caller falls back to T1→T2)."""
    df_doc_type = _DOC_FEEDBACK_TYPE_MAP.get(doc_type)
    if df_doc_type is None:
        logger.info(
            "application_tailoring: %s has no doc-feedback analog — using T1→T2",
            doc_type,
        )
        return None

    # Pre-populate the doc-feedback state. We skip Phase 0 (load_document /
    # detect_doc_type) entirely because the source text is already
    # extracted and we know the doc_type from the requirement_item.
    df_state = {
        "raw_text": uploaded_text,
        "doc_meta": {
            "file_name": f"{doc_type}.txt",
            "mime": "text/plain",
            "char_count": len(uploaded_text),
            "page_count": None,
            "extraction_warnings": [],
        },
        "doc_classification": {
            "doc_type": df_doc_type,
            "relevant": True,
            "confidence": 1.0,
            "reasons": ["pre-classified by auto_apply requirement_item"],
            "language": None,
        },
        "user_instructions": (user_prompt or "").strip(),
        "profile_snapshot": profile,
        "opportunity_context": _opportunity_context_for_doc_feedback(
            opportunity_data, opportunity_type
        ),
        "iteration_count": 0,
    }

    try:
        graph = build_auto_tailoring_graph()
        # Sub-graph runs to completion (no interrupt). thread_id is per-doc
        # so concurrent docs (CV + CL in same session) don't share state.
        config = {"configurable": {"thread_id": f"auto-tailor-{doc_type}"}}
        result = graph.invoke(df_state, config=config)
    except Exception as exc:
        logger.warning(
            "application_tailoring: doc-feedback graph crashed for %s — falling back to T1→T2 (%s)",
            doc_type, exc,
        )
        return None

    if (result.get("result") or {}).get("status") == "error":
        logger.info(
            "application_tailoring: doc-feedback ended with error for %s — falling back to T1→T2",
            doc_type,
        )
        return None

    latex_source = (result.get("final_document") or "").strip()
    if not latex_source:
        return None

    plain = _latex_to_plain(latex_source)
    diff = result.get("diff") or {}
    accepted = len((result.get("human_review") or {}).get("approved_proposals") or [])

    return {
        "content": _truncate_to_cap(plain, doc_type),
        "latex_source": latex_source,
        "tailoring_depth": "doc_feedback",
        "source": "upload",
        "llm_used": True,
        "passes": int(result.get("iteration_count") or 0),
        "doc_feedback_diff": diff,
        "doc_feedback_accepted_proposals": accepted,
    }


def _opportunity_context_for_doc_feedback(
    opportunity_data: Dict[str, Any], opportunity_type: str
) -> Dict[str, Any]:
    """Map auto_apply's opportunity_data dict to the shape doc-feedback's
    analyze_opportunity_alignment node expects. Best-effort — every field
    is optional on the doc-feedback side; missing fields just degrade the
    quality of the alignment analysis."""
    if not opportunity_data:
        return {}
    title = (
        opportunity_data.get("title")
        or opportunity_data.get("name")
        or ""
    )
    org = (
        opportunity_data.get("company")
        or opportunity_data.get("university")
        or opportunity_data.get("provider_name")
        or ""
    )
    description = (
        opportunity_data.get("description")
        or opportunity_data.get("eligibility_text")
        or ""
    )
    return {
        "opportunity_type": opportunity_type,
        "title": title,
        "organisation": org,
        "location": opportunity_data.get("location") or "",
        "description": description[:_MAX_OPP_CHARS],
    }


def _process_document(
    item: Dict[str, Any],
    choice: str,
    uploaded_text: Optional[str],
    user_prompt: Optional[str],
    opportunity_data: Dict[str, Any],
    opportunity_type: str,
    profile: Dict[str, Any],
    llm,
    user_instructions: Optional[str] = "",
) -> Optional[Dict[str, Any]]:
    """Return a tailored_documents entry, or None if nothing to produce."""
    doc_type = item.get("document_type") or item.get("label") or "Document"

    if choice == "upload":
        if not uploaded_text:
            logger.warning("application_tailoring: upload selected for %s but no text — skipping", doc_type)
            return None
        if llm is None:
            logger.warning(
                "application_tailoring: no LLM — passing uploaded %s through unchanged", doc_type
            )
            return {
                "content": _truncate_to_cap(uploaded_text, doc_type),
                "latex_source": "",
                "tailoring_depth": "light",
                "source": "upload",
                "llm_used": False,
                "passes": 0,
            }

        # Try the doc-feedback graph first — it produces grounded edits
        # (proposals must fuzzy-match source text → no hallucination) plus
        # a deterministic evaluator with retry. Falls back to the legacy
        # T1→T2 path on any failure.
        df_result = _tailor_via_doc_feedback(
            doc_type, uploaded_text, user_prompt,
            opportunity_data, opportunity_type, profile,
        )
        if df_result is not None:
            logger.info(
                "application_tailoring: %s tailored via doc-feedback graph "
                "(accepted=%s, content_chars=%d)",
                doc_type, df_result.get("doc_feedback_accepted_proposals"),
                len(df_result.get("content") or ""),
            )
            return df_result

        # PreA
        pre = analyze_upload_pre(opportunity_data, profile, uploaded_text, doc_type, user_prompt)

        # T1
        t1 = _llm_call(
            llm, _SYSTEM_T1,
            _t1_prompt(doc_type, opportunity_data, opportunity_type, profile, uploaded_text, pre, user_prompt, user_instructions),
        )
        if not t1:
            return {
                "content": _truncate_to_cap(uploaded_text, doc_type),
                "latex_source": "",
                "tailoring_depth": "light",
                "source": "upload",
                "llm_used": False,
                "passes": 0,
                "note": "T1 LLM call failed; returning upload unchanged.",
            }

        # LA
        post = analyze_upload_light_post(opportunity_data, profile, t1, doc_type, user_prompt)

        # T2
        t2 = _llm_call(
            llm, _SYSTEM_T2,
            _t2_prompt(doc_type, opportunity_data, opportunity_type, profile, t1, post, user_prompt, user_instructions),
        )
        final = t2 or t1

        content, latex_source = _split_latex_and_plain(final, doc_type)
        return {
            "content": content,
            "latex_source": latex_source,
            "tailoring_depth": "deep" if t2 else "light",
            "source": "upload",
            "llm_used": True,
            "passes": 2 if t2 else 1,
            "pre_analysis": pre.model_dump(),
            "post_analysis": post.model_dump() if t2 else None,
        }

    if choice == "auto_generate":
        if llm is None:
            logger.warning(
                "application_tailoring: no LLM — cannot auto-generate %s", doc_type
            )
            return {
                "content": "",
                "latex_source": "",
                "tailoring_depth": "generate",
                "source": "auto_generate",
                "llm_used": False,
                "passes": 0,
                "note": "Auto-generate requested but no LLM is configured.",
            }

        canonical_type = item.get("document_type")
        text = _llm_call(
            llm, _SYSTEM_GENERATE_DOC,
            _generate_doc_prompt(doc_type, canonical_type, opportunity_data, opportunity_type, profile, user_prompt, user_instructions),
        )
        if not text:
            return {
                "content": "",
                "latex_source": "",
                "tailoring_depth": "generate",
                "source": "auto_generate",
                "llm_used": False,
                "passes": 0,
                "note": "Auto-generate LLM call failed.",
            }
        content, latex_source = _split_latex_and_plain(text, doc_type)
        return {
            "content": content,
            "latex_source": latex_source,
            "tailoring_depth": "generate",
            "source": "auto_generate",
            "llm_used": True,
            "passes": 1,
        }

    # ignore_for_now / skip → nothing to produce
    return None


# ---------------------------------------------------------------------------
# Misc-bucket LLM derivation (Yes/No comboboxes, profile fields, short
# free-form questions). Replaces value_planner here because:
#   - value_planner's profile_lookup keyword-matches "country" anywhere
#     in the label, which fires incorrectly on "visa sponsorship to
#     work in the country" → returns Country=Turkey.
#   - value_planner returns "[Mock answer — …]" for short text fields
#     without options, which is useless to the user.
#   - value_planner can't reason about Yes/No questions where the
#     correct answer depends on the user's profile vs the opportunity
#     (e.g. visa sponsorship → Turkish user applying to US → Yes).
#
# value_planner is still the fill-time fallback in
# auto_apply_adapter.attempt_auto_fill — only the gate-1→gate-2 stage
# uses this LLM batch.
# ---------------------------------------------------------------------------

_MISC_DERIVE_SYSTEM = """You are filling out a job application form on behalf of a user. The user wants the form auto-filled. For each question, produce a concise answer.

You will receive:
- The user's profile (name, contact, location, education, skills, work history, etc.)
- The opportunity (title, company, location, description excerpt)
- A list of form fields, each with: form_field_index (int), label (str), field_type (str), required (bool), options (list of strings — empty if free-form).

Rules — apply per field:

1. If `options` is non-empty, your answer MUST be one of the option strings exactly. Reason about the user's profile and the opportunity to pick the right one.
   - Yes/No questions: think before answering. Examples:
     - "Do you require visa sponsorship?" — Yes if the user's nationality differs from the opportunity's country and they aren't already authorised to work there.
     - "Are you open to relocation?" — usually Yes unless the user's profile clearly indicates otherwise.
     - "Are you open to working in-person 25% of the time?" — usually Yes; only No if profile makes hybrid impossible.
     - "Have you ever interviewed at Anthropic before?" — No unless profile or CV mentions it.
     - "Have you built or owned X infrastructure?" — Yes only when the CV / profile clearly demonstrates it; otherwise No.

2. Profile-attribute questions (Country, City, Email, Phone, LinkedIn, GitHub, Website, Address) — echo the user's profile value verbatim.

3. Short open-ended questions ("When can you start?", "Do you have any deadlines?") — draft a 1–2 sentence concise honest answer based on profile and CV. Don't invent specifics.

4. If a field is hard to answer confidently, return an empty string for `answer` — leave the field blank rather than guessing.

5. Never write a paragraph. Keep answers SHORT. Single line for profile attributes, ≤2 sentences for short open-ended.

Return ONE entry per input field, in the same order. Use the field's form_field_index value verbatim."""


from pydantic import BaseModel as _BaseModel  # local alias to avoid top-of-file shuffle


class _MiscAnswer(_BaseModel):
    form_field_index: int
    answer: str = ""
    reason: str = ""


class _MiscAnswerList(_BaseModel):
    answers: List[_MiscAnswer] = []


def _build_profile_summary(profile: Dict[str, Any]) -> str:
    """Compact profile summary for the misc-derivation prompt. Mirrors
    the same shape `_generate_text_prompt` builds — keep them similar so
    the LLM treats both surfaces the same way."""
    if not profile:
        return "(profile unavailable)"
    fields = [
        ("Name", profile.get("full_name") or
            f"{profile.get('first_name','')} {profile.get('last_name','')}".strip()),
        ("Email", profile.get("email", "")),
        ("Phone", profile.get("phone", "")),
        ("Country", profile.get("country", "")),
        ("City", profile.get("city", "")),
        ("Location", profile.get("location", "")),
        ("Nationality", profile.get("nationality", "")),
        ("Work authorisation", profile.get("work_authorization", "") or profile.get("work_auth", "")),
        ("LinkedIn", profile.get("linkedin", "")),
        ("GitHub", profile.get("github", "")),
        ("Website", profile.get("website", "")),
        ("Education", profile.get("degree", "") or profile.get("education", "")),
        ("Bio", (profile.get("bio") or "")[:400]),
    ]
    rendered = "\n".join(f"- {k}: {v}" for k, v in fields if v)
    return rendered or "(profile fields not populated)"


def _build_opportunity_summary(
    opportunity_data: Dict[str, Any], opportunity_type: str
) -> str:
    if not opportunity_data:
        return "(opportunity unavailable)"
    title = opportunity_data.get("title") or opportunity_data.get("name") or ""
    company = (
        opportunity_data.get("company")
        or opportunity_data.get("university")
        or opportunity_data.get("provider_name")
        or ""
    )
    location = opportunity_data.get("location", "")
    description = (opportunity_data.get("description") or "")[:1500]
    parts = [
        ("Type", opportunity_type),
        ("Title", title),
        ("Organization", company),
        ("Location", location),
        ("Description (excerpt)", description),
    ]
    return "\n".join(f"- {k}: {v}" for k, v in parts if v)


def _derive_misc_answers_via_llm(
    *,
    misc_indices: List[int],
    form_fields: List[Dict[str, Any]],
    profile: Dict[str, Any],
    opportunity_data: Dict[str, Any],
    opportunity_type: str,
    tailored_documents: Dict[str, Any],
    tailored_answers: Dict[str, Dict[str, Any]],
    llm,
) -> int:
    """Single LLM batch call to derive answers for every misc-bucketed
    form field. Mutates `tailored_answers` in place; returns the count
    of fields that received a non-empty answer.

    Defensive: returns 0 silently if LLM is unavailable or the call
    fails. Defaults-fallback path (no form_fields, CV+Cover Letter
    only) never reaches here — `application_tailoring` gates this on
    `has_misc_item AND form_fields`.
    """
    if not misc_indices or llm is None:
        return 0

    # Build per-field input list. Cap to 30 fields per call to keep
    # token cost bounded; rare on real ATSes (Anthropic Greenhouse has
    # ~22 fields total, of which ~15 are misc).
    items: List[Dict[str, Any]] = []
    for idx in misc_indices[:30]:
        f = form_fields[idx]
        items.append({
            "form_field_index": idx,
            "label": (f.get("label") or "").strip(),
            "field_type": f.get("field_type") or "text",
            "required": bool(f.get("required")),
            "options": list(f.get("options") or [])[:20],
        })

    profile_summary = _build_profile_summary(profile)
    opp_summary = _build_opportunity_summary(opportunity_data, opportunity_type)

    user_msg = (
        f"=== PROFILE ===\n{profile_summary}\n\n"
        f"=== OPPORTUNITY ===\n{opp_summary}\n\n"
        f"=== FORM FIELDS ===\n{items!r}\n\n"
        "Return one MiscAnswer per field, in the same order, with form_field_index matching the input."
    )

    try:
        structured = llm.with_structured_output(_MiscAnswerList)
        result = structured.invoke([
            SystemMessage(content=_MISC_DERIVE_SYSTEM),
            HumanMessage(content=user_msg),
        ])
    except Exception as exc:
        logger.warning(
            "application_tailoring: misc LLM derivation call failed — %s", exc,
        )
        return 0

    answers_by_idx: Dict[int, str] = {}
    for entry in (result.answers or []):
        try:
            ffi = int(entry.form_field_index)
        except (TypeError, ValueError):
            continue
        ans = (entry.answer or "").strip()
        if ans:
            answers_by_idx[ffi] = ans

    misc_count = 0
    for idx in misc_indices:
        ans = answers_by_idx.get(idx)
        if not ans:
            continue
        f = form_fields[idx]
        # If options is non-empty, sanity-check that the LLM picked a
        # valid one. If not, normalise to the closest option (case-
        # insensitive exact match preferred). Bail out (skip writing)
        # when no option matches — better to leave it for gate-2 review
        # than to ship a bad value.
        opts = list(f.get("options") or [])
        if opts:
            lc = ans.lower()
            picked: Optional[str] = None
            for o in opts:
                if o.lower() == lc:
                    picked = o
                    break
            if picked is None:
                for o in opts:
                    if o.lower() in lc or lc in o.lower():
                        picked = o
                        break
            if picked is None:
                continue
            ans = picked
        tailored_answers[str(idx)] = {
            "content": ans,
            "question": f.get("label", "") or "",
            "form_field_index": idx,
            "llm_used": True,
            "source": "llm_misc_derivation",
        }
        misc_count += 1
    return misc_count


def _process_text(
    item: Dict[str, Any],
    choice: str,
    user_prompt: Optional[str],
    opportunity_data: Dict[str, Any],
    opportunity_type: str,
    profile: Dict[str, Any],
    llm,
    user_instructions: Optional[str] = "",
) -> Optional[Dict[str, Any]]:
    if choice != "auto_generate":
        return None

    question = item.get("question") or item.get("label") or ""
    if not question:
        return None

    if llm is None:
        return {
            "content": "",
            "question": question,
            "form_field_index": item.get("form_field_index"),
            "llm_used": False,
            "note": "Auto-generate requested but no LLM is configured.",
        }

    text = _llm_call(
        llm, _SYSTEM_GENERATE_TEXT,
        _generate_text_prompt(question, opportunity_data, opportunity_type, profile, user_instructions),
    )
    if not text:
        return {
            "content": "",
            "question": question,
            "form_field_index": item.get("form_field_index"),
            "llm_used": False,
            "note": "Text-answer LLM call failed.",
        }
    if user_prompt:
        # User_prompt is documents-only per the gate-1 contract; if the
        # backend somehow forwarded one for a text item, ignore it silently.
        pass
    return {
        "content": text[:_MAX_TEXT_ANSWER_CHARS],
        "question": question,
        "form_field_index": item.get("form_field_index"),
        "llm_used": True,
    }


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def application_tailoring(state: AutoApplyState) -> dict:
    updates = {"current_step": "application_tailoring", "step_history": ["application_tailoring"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    opportunity_type = state.get("opportunity_type", "")
    opportunity_data = state.get("opportunity_data") or {}
    requirement_items: List[Dict[str, Any]] = list(state.get("requirement_items") or [])
    human_review_1 = state.get("human_review_1") or {}
    requirements: Dict[str, Dict[str, Any]] = human_review_1.get("requirements") or {}
    user_instructions: str = (state.get("user_instructions") or "").strip()

    if not requirement_items or not requirements:
        logger.warning("application_tailoring: missing requirement_items or gate-1 requirements")
        return {**updates, "tailored_documents": {}, "tailored_answers": {}}

    profile = resolve_profile(state)
    llm = get_llm()

    tailored_documents: Dict[str, Any] = {}
    tailored_answers: Dict[str, Dict[str, Any]] = {}

    for item in requirement_items:
        idx_str = str(item["id"])
        choice_entry = requirements.get(idx_str) or {}
        choice = choice_entry.get("choice")
        if not choice or choice in {"ignore_for_now", "skip"}:
            continue

        category = item.get("category")
        user_prompt = choice_entry.get("user_prompt")
        uploaded_text = choice_entry.get("uploaded_text")

        if category == "document":
            result = _process_document(
                item, choice, uploaded_text, user_prompt,
                opportunity_data, opportunity_type, profile, llm,
                user_instructions=user_instructions,
            )
            if result is not None:
                doc_type = item.get("document_type") or item.get("label") or "Document"
                tailored_documents[doc_type] = result
        elif category == "text":
            result = _process_text(
                item, choice, user_prompt,
                opportunity_data, opportunity_type, profile, llm,
                user_instructions=user_instructions,
            )
            if result is not None:
                ffi = item.get("form_field_index")
                key = str(ffi) if ffi is not None else idx_str
                tailored_answers[key] = result
        # misc → handled below in the misc-derivation pass

    # ─── Misc auto-fill derivation ──────────────────────────────────────
    # Spec: short-input fields (Yes/No comboboxes, country pickers,
    # sponsorship dropdowns, profile attributes) are collapsed into the
    # misc bucket at gate-1. When the user picked misc_strategy="auto_fill",
    # the agent derives an answer for each *now*, before gate-2 — so
    # gate-2 can show the auto-derived values for review/edit alongside
    # tailored documents and textarea answers.
    #
    # Defensive: only fires when (a) misc items exist, (b) form_fields
    # is non-empty, (c) misc_strategy="auto_fill". The defaults-fallback
    # path (no form_fields, just CV+Cover Letter from _build_default)
    # never reaches this branch.
    misc_strategy = (human_review_1.get("misc_strategy") or "ignore").strip()
    has_misc_item = any(
        (it.get("category") == "misc") for it in requirement_items
    )
    form_fields: List[Dict[str, Any]] = list(state.get("form_fields") or [])
    if has_misc_item and misc_strategy == "auto_fill" and form_fields:
        # Determine which form_field indices are misc-bucketed. Same
        # logic as asset_mapping._build_from_form_fields: every field
        # that isn't a file (handled as document) AND isn't a true
        # textarea-with-user-answer (handled as text).
        misc_indices: List[int] = []
        already_answered_keys = set(tailored_answers.keys())
        for idx, field in enumerate(form_fields):
            ftype = field.get("field_type")
            if ftype == "file":
                continue
            if ftype == "textarea":
                continue
            # Skip if this form field already has an answer from a text
            # category item (defensive — shouldn't overlap, but a
            # malformed metadata mix shouldn't double-populate).
            if str(idx) in already_answered_keys:
                continue
            misc_indices.append(idx)

        if misc_indices:
            misc_count = _derive_misc_answers_via_llm(
                misc_indices=misc_indices,
                form_fields=form_fields,
                profile=profile,
                opportunity_data=opportunity_data,
                opportunity_type=opportunity_type,
                tailored_documents=tailored_documents,
                tailored_answers=tailored_answers,
                llm=llm,
            )
            logger.info(
                "application_tailoring: derived %d misc answer(s) (auto_fill)",
                misc_count,
            )

    logger.info(
        "application_tailoring: produced %d documents, %d text/misc answers",
        len(tailored_documents), len(tailored_answers),
    )

    return {
        **updates,
        "tailored_documents": tailored_documents,
        "tailored_answers": tailored_answers,
    }
