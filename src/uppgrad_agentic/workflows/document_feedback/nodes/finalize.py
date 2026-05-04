from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.tools.latex_compiler import compile_latex
from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LaTeX CV template — embedded to avoid packaging/data-file issues
# ---------------------------------------------------------------------------
_RESUME_TEMPLATE = r"""%-------------------------
% Resume Template (sb2nov-based)
% Tectonic-compatible — no glyphtounicode
%------------------------

\documentclass[letterpaper,11pt]{article}

\usepackage[empty]{fullpage}
\usepackage{titlesec}
\usepackage[usenames,dvipsnames]{color}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage[english]{babel}
\usepackage{tabularx}

\pagestyle{fancy}
\fancyhf{}
\fancyfoot{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Adjust margins
\addtolength{\oddsidemargin}{-0.5in}
\addtolength{\evensidemargin}{-0.5in}
\addtolength{\textwidth}{1in}
\addtolength{\topmargin}{-.5in}
\addtolength{\textheight}{1.0in}

\urlstyle{same}
\raggedbottom
\raggedright
\setlength{\tabcolsep}{0in}

% Sections formatting
\titleformat{\section}{
  \vspace{-4pt}\scshape\raggedright\large
}{}{0em}{}[\color{black}\titlerule \vspace{-5pt}]

%-------------------------
% Custom commands
\newcommand{\resumeItem}[2]{
  \item\small{
    \textbf{#1}{: #2 \vspace{-2pt}}
  }
}

\newcommand{\resumeSubheading}[4]{
  \vspace{-1pt}\item
    \begin{tabular*}{0.97\textwidth}[t]{l@{\extracolsep{\fill}}r}
      \textbf{#1} & #2 \\
      \textit{\small #3} & \textit{\small #4} \\
    \end{tabular*}\vspace{-5pt}
}

\newcommand{\resumeSubSubheading}[2]{
    \begin{tabular*}{0.97\textwidth}{l@{\extracolsep{\fill}}r}
      \textit{\small #1} & \textit{\small #2} \\
    \end{tabular*}\vspace{-5pt}
}

\newcommand{\resumeItemPlain}[1]{
  \item\small{
    {#1 \vspace{-2pt}}
  }
}

\newcommand{\resumeSubItem}[2]{\resumeItem{#1}{#2}\vspace{-4pt}}

% Project heading — for projects that have MULTIPLE bullets. Wrap with
% \resumeItemListStart / \resumeItemPlain / \resumeItemListEnd. Use this
% instead of stacking multiple \resumeSubItem calls with the same name
% (which prints the project name once per bullet) or empty-name sub-items.
\newcommand{\resumeProjectHeading}[1]{
  \vspace{-1pt}\item
    \textbf{#1}\vspace{-5pt}
}

\renewcommand{\labelitemii}{$\circ$}

\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=*]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}
\newcommand{\resumeItemListStart}{\begin{itemize}}
\newcommand{\resumeItemListEnd}{\end{itemize}\vspace{-5pt}}

%-------------------------------------------
%  DOCUMENT STARTS HERE
%-------------------------------------------

\begin{document}

%----------HEADING-----------------
% USE a centered header so long contact lines wrap instead of overflowing.
% Name on its own line, then \small contact info separated by $|$:
%   \begin{center}
%     {\Large \textbf{FULL NAME}} \\ \vspace{2pt}
%     \small LOCATION $|$ \href{mailto:EMAIL}{EMAIL} $|$ PHONE \\
%     \href{LINKEDIN_URL}{LinkedIn} $|$ \href{GITHUB_URL}{GitHub}
%   \end{center}
% Do NOT use \begin{tabular*}{\textwidth}{l@{\extracolsep{\fill}}r} for the
% header — that layout cannot wrap and truncates email/phone off the page
% when contact info is long.

%----------EDUCATION-----------------
% \section{Education}
%   \resumeSubHeadingListStart
%     \resumeSubheading{UNIVERSITY}{LOCATION}{DEGREE; GPA: X.XX}{DATE RANGE}
%   \resumeSubHeadingListEnd

%----------EXPERIENCE-----------------
% \section{Experience}
%   \resumeSubHeadingListStart
%     \resumeSubheading{COMPANY}{LOCATION}{TITLE}{DATE RANGE}
%       \resumeItemListStart
%         \resumeItemPlain{DESCRIPTION}
%       \resumeItemListEnd
%   \resumeSubHeadingListEnd

%----------PROJECTS-----------------
% \section{Projects}
%   \resumeSubHeadingListStart
%     % Single-line project (one description):
%     \resumeSubItem{PROJECT NAME}{ONE-LINE DESCRIPTION}
%     % Project with MULTIPLE bullets — use \resumeProjectHeading + bullet list.
%     % NEVER repeat the project name across multiple \resumeSubItem calls,
%     % and NEVER emit \resumeSubItem with an empty first argument.
%     \resumeProjectHeading{PROJECT NAME}
%       \resumeItemListStart
%         \resumeItemPlain{FIRST BULLET}
%         \resumeItemPlain{SECOND BULLET}
%       \resumeItemListEnd
%   \resumeSubHeadingListEnd

%----------SKILLS-----------------
% \section{Skills}
%   \resumeSubHeadingListStart
%     \item{
%       \textbf{Languages}{: LIST} \hfill
%       \textbf{Technologies}{: LIST}
%     }
%   \resumeSubHeadingListEnd

\end{document}
"""

# ---------------------------------------------------------------------------
# LaTeX generation prompts — branched by doc type
# ---------------------------------------------------------------------------
#
# CV → resume template + bullet-list helpers (`\resumeItemPlain`, etc.).
# SOP / COVER_LETTER → article template, plain paragraphs, NO bullet helpers
# at all. The previous version applied the resume prompt to every doc type,
# which made the LLM wrap each SOP/CL paragraph in `\resumeItemPlain` —
# producing a rendered PDF where every paragraph showed as a bullet point.

_LATEX_SYSTEM = r"""You are a professional resume/CV typesetter.
You receive:
  1. The plain-text of a document (extracted from a PDF). All accepted user edits have ALREADY been merged into this text — treat it as the final content.
  2. A LaTeX TEMPLATE with preamble, custom commands, and commented-out section markers.
  3. Optionally, a list of ADDITIONAL HINTS — proposals that could not be auto-applied (typically structure-level suggestions with no specific anchor in the source). Apply each hint only if you can locate a sensible spot for it; otherwise ignore it.

YOUR TASK:
  - Produce a COMPLETE, COMPILABLE LaTeX document using EXACTLY the provided template preamble and custom commands.
  - Render the document text faithfully into the template's section structure (HEADING, EDUCATION, EXPERIENCE, PROJECTS, SKILLS, etc.).
  - Use ONLY the custom commands defined in the template: \resumeSubheading, \resumeItemPlain, \resumeItem, \resumeSubItem, \resumeProjectHeading, \resumeSubHeadingListStart/End, \resumeItemListStart/End.
  - Preserve every sentence of the input text. Do NOT paraphrase, summarise, condense, or "improve" the wording — the upstream pipeline has already made all editorial decisions and any further rewriting silently overrides accepted user edits.
  - If a section from the original document is not covered by the template markers, create a new \section{} for it.

CRITICAL RULES:
  - Use EXACTLY the \documentclass, \usepackage, and \newcommand lines from the template. Do NOT add, remove, or change any packages.
  - Do NOT use \input{glyphtounicode} or \pdfgentounicode — they are incompatible with the compiler.
  - Do NOT use fontspec, moderncv, awesome-cv, or any custom .cls files.
  - Do NOT invent, fabricate, or add ANY facts, dates, skills, experiences, or achievements not present in the original document or the accepted proposals.
  - Do NOT remove any original content unless explicitly instructed by an accepted proposal.
  - Escape special LaTeX characters in user content: & % $ # _ { } ~ ^

PROJECTS SECTION SHAPE (very important — common rendering bug):
  - When a project in the source has MULTIPLE descriptions / bullet points, render it as:
        \resumeProjectHeading{PROJECT NAME}
          \resumeItemListStart
            \resumeItemPlain{FIRST BULLET}
            \resumeItemPlain{SECOND BULLET}
          \resumeItemListEnd
  - When a project has exactly ONE description line, use:
        \resumeSubItem{PROJECT NAME}{DESCRIPTION}
  - NEVER stack multiple `\resumeSubItem{NAME}{...}` calls with the same NAME — that prints the project name once per bullet.
  - NEVER emit `\resumeSubItem{}{...}` with an empty first argument — the rendered output will show as a stray "`: description`" line.
  - If the source text shows a project header followed by lines that look like sub-bullets (indented, dash-prefixed, or just continuing under the header without a new colon-prefixed name), those lines are bullets of the SAME project — group them under one `\resumeProjectHeading`.

HEADER LAYOUT (very important — most common rendering bug):
  - Render the contact header inside a \begin{center} ... \end{center} block, NOT inside a \begin{tabular*}{\textwidth}{l@{\extracolsep{\fill}}r} layout.
  - The tabular* layout does not wrap text and silently truncates long emails / phone numbers / URLs off the right edge of the page.
  - Use exactly this shape (adapt to whichever fields actually exist in the source CV — omit fields that are missing, do not fabricate):
        \begin{center}
          {\Large \textbf{FULL NAME}} \\ \vspace{2pt}
          \small LOCATION $|$ \href{mailto:EMAIL}{EMAIL} $|$ PHONE \\
          \href{LINKEDIN_URL}{LinkedIn} $|$ \href{GITHUB_URL}{GitHub}
        \end{center}
  - The contact line MUST be wrapped in \small (or \footnotesize). Default-size contacts overflow even inside center.
  - Separate contact items with $|$ (math-mode pipe). Do NOT use plain `|`.
  - Split contact info across two centered lines (with `\\`) when it's long; do not pack everything onto one line.

Return ONLY the complete LaTeX source code. No explanations, no markdown fences.
Start with \documentclass and end with \end{document}.
"""


# ---------------------------------------------------------------------------
# Prose LaTeX template — for SOP / COVER_LETTER
# ---------------------------------------------------------------------------
# Plain article class. No resume bullet helpers. Paragraphs are typeset as
# flowing paragraphs separated by blank lines (parskip), the way an SOP or
# cover letter actually reads on paper.

_PROSE_TEMPLATE = r"""%-------------------------
% Prose document template (SOP / Cover Letter)
% Tectonic-compatible.
%-------------------------

\documentclass[11pt,letterpaper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{parskip}      % blank-line paragraph breaks, no first-line indent
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.7em}

\begin{document}

% Body goes here as flowing paragraphs separated by blank lines.
% For Cover Letters, optionally start with a date and salutation, then
% paragraphs, then a closing.
% For SOPs, use \section*{...} headings ONLY when the original document
% had clear section breaks; otherwise plain paragraphs.

\end{document}
"""


_LATEX_SYSTEM_PROSE = r"""You are a professional typesetter rendering a \
Statement of Purpose or Cover Letter into LaTeX.

You receive:
  1. The plain-text of the document (extracted from a PDF). All accepted user edits — paragraph rewrites, deletes, merges — have ALREADY been merged into this text. Treat it as the final content.
  2. A minimal LaTeX TEMPLATE (article class, parskip-based prose).
  3. Optionally, a list of ADDITIONAL HINTS — proposals that could not be auto-applied (typically structure-level suggestions with no specific anchor). Apply each hint only if you can locate a sensible spot; otherwise ignore.

YOUR TASK:
  - Produce a COMPLETE, COMPILABLE LaTeX document using EXACTLY the template's preamble.
  - Render the document as FLOWING PROSE PARAGRAPHS separated by blank lines.
  - Preserve every sentence of the input text verbatim. Do NOT paraphrase, summarise, condense, reorder, or "improve" any paragraph — the upstream pipeline has already produced the final wording and any further rewriting silently overrides accepted user edits.

ABSOLUTE FORMATTING RULES — these are the most important rules:
  - DO NOT use \begin{itemize} or \begin{enumerate} or any list environment.
  - DO NOT use \item.
  - DO NOT use \resumeItem, \resumeItemPlain, \resumeSubItem, \resumeSubheading, \
\resumeSubHeadingListStart, \resumeItemListStart, or any resume-template commands. \
These commands DO NOT EXIST in this template and will fail to compile.
  - Each paragraph of the original document must render AS A PARAGRAPH — \
plain text, separated from the next paragraph by ONE BLANK LINE. Nothing more.
  - Cover letters: if the original has a date / address / salutation / closing, \
keep them as plain paragraphs at top and bottom. No tables, no fancy headers.
  - SOPs: use \section*{Heading Name} ONLY when the original document clearly \
labelled a section. Otherwise just plain paragraphs in document order. Most \
SOPs have no section headings — that is fine. Do NOT invent headings.

OTHER CRITICAL RULES:
  - Use EXACTLY the \documentclass and \usepackage lines from the template. \
Do NOT add, remove, or change any packages.
  - Do NOT use \input{glyphtounicode} or \pdfgentounicode.
  - Do NOT use fontspec, moderncv, awesome-cv, or any custom .cls files.
  - Do NOT invent, fabricate, or add ANY facts, claims, names, dates, or \
experiences not present in the input document text.
  - Do NOT remove or skip any paragraph from the input. Render every \
paragraph that appears in the input, in the order it appears.
  - DO NOT use em-dashes (—) or double-hyphens (--) in the body. The \
upstream pipeline has already stripped these; preserve that. Use commas, \
periods, or colons instead. Em-dashes are the strongest "AI-generated" tell \
and should not appear in the rendered document.
  - Escape special LaTeX characters in user content: & % $ # _ { } ~ ^
  - Curly/smart quotes are fine — they render correctly under utf8.

Return ONLY the complete LaTeX source code. No explanations, no markdown fences.
Start with \documentclass and end with \end{document}.
"""

_MAX_DOC_CHARS = 16000
_MAX_PROPOSALS_CHARS = 8000


# ---------------------------------------------------------------------------
# Deterministic proposal application
# ---------------------------------------------------------------------------
#
# Why this exists: previously the LLM was asked to BOTH apply each accepted
# proposal AND format the document as LaTeX in a single pass. The LLM would
# silently drop proposals — most often the closing-paragraph rewrite for
# SOPs/CLs — and `_build_diff` would still report them as "applied" because
# nothing verified the output. The user-visible failure mode: an accepted
# rewrite never made it into the rendered PDF.
#
# Fix: apply proposals to raw_text deterministically here, then send the
# already-edited text to the LLM for formatting only. The LLM cannot drop
# what was never in its input. Heuristic-fallback proposals with no anchor
# (empty or `[bracketed]` before_text) are passed to the LLM as hints since
# they have nowhere to attach deterministically.

def _build_match_pattern(before: str) -> Optional[re.Pattern]:
    """Compile a regex that matches `before` with whitespace + typography slack.

    Tolerances applied:
      - any run of whitespace in `before` matches any run of whitespace in
        the source (lets a paragraph survive PDF→text line-wrap reflow)
      - smart vs straight quotes (' vs ’ vs ‘, " vs ” vs “)
      - en-dash / em-dash / hyphen (-, –, —)

    Case-sensitive on purpose: a "fix capitalization" proposal must land at
    the lowercase original, not the rewrite's already-capitalised copy.
    """
    if not before.strip():
        return None
    tokens = before.split()
    if not tokens:
        return None
    parts: List[str] = []
    for token in tokens:
        chars: List[str] = []
        for ch in token:
            if ch in "'’‘":
                chars.append(r"['’‘]")
            elif ch in '"”“':
                chars.append(r'["”“]')
            elif ch in "-—–":
                chars.append(r"[-—–]")
            else:
                chars.append(re.escape(ch))
        parts.append("".join(chars))
    pattern = r"\s+".join(parts)
    try:
        return re.compile(pattern)
    except re.error:
        return None


# A "kitchen-sink" proposal bundles 3+ paragraphs into one before_text. The
# greedy non-overlapping applier accepts the kitchen-sink first (lowest start
# position) and rejects every per-paragraph proposal nested inside it as
# overlap — leaving the document with one mega-rewrite and zero of the
# targeted edits the synth also emitted. Reject such proposals so the
# per-paragraph ones can apply.
#
# Threshold is 2+ paragraph breaks (i.e. 3+ paragraphs). One break is allowed
# for every action so legitimate two-paragraph operations pass:
#   * action="merge"   — two adjacent paragraphs collapsed to one (canonical)
#   * action="rewrite" — two-paragraph restructure or swap (the schema has no
#                        explicit "reorder" action, so the synth may encode
#                        an A+B → B+A swap as a 1-break rewrite)
#   * action="delete"  — two adjacent paragraphs cut together
#
# Counting paragraph breaks: pypdf "word-per-line" wrap puts \n \n between
# every word in a wrapped paragraph, which used to trigger this guard for
# every legitimate proposal (one user run rejected 11/12 proposals on
# "before_text_spans_too_many_paragraphs"). We now split on \n\s*\n and
# count only the chunks that are substantial enough to be real paragraphs
# (≥ 30 chars) — single-word "chunks" produced by wrap collapse no longer
# count as paragraph breaks. The doc loader normalises this for raw_text
# upstream; this is defense in depth against the synth copying before_text
# from a pre-normalised view of the document, or future PDF-extraction
# quirks that bypass the normaliser.
_NEWLINE_BLOCK = re.compile(r"\n\s*\n")
_MIN_PARAGRAPH_CHARS = 30
_MAX_PARAGRAPH_BREAKS_IN_BEFORE_TEXT = 1


def _count_paragraph_breaks(before: str) -> int:
    """Number of real paragraph breaks in `before`.

    Splits on any blank-line-like separator and counts only chunks that are
    substantial (≥ _MIN_PARAGRAPH_CHARS). Single-word chunks from pypdf
    word-per-line wrap collapse to zero, real paragraph spans count as
    expected.
    """
    chunks = _NEWLINE_BLOCK.split(before)
    real_chunks = [c for c in chunks if len(c.strip()) >= _MIN_PARAGRAPH_CHARS]
    return max(0, len(real_chunks) - 1)


def _apply_proposals_to_text(
    raw_text: str,
    proposals: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply accepted proposals to raw_text deterministically.

    Returns (modified_text, applied_proposals, unapplied_entries) where each
    unapplied entry is {"proposal": ..., "reason": ...}. Reasons:
      - "no_anchor": before_text is empty or a `[placeholder]` (heuristic
        fallback only — these are passed to the LLM as hints).
      - "before_text_not_found": the regex pattern didn't match the source
        (synthesizer hallucination that escaped the upstream grounding
        check, or document text changed between synth and finalize).
      - "before_text_spans_too_many_paragraphs": before_text covers 3+
        paragraphs (2+ for non-merge); kitchen-sink proposals get dropped
        so they cannot swallow per-paragraph edits as overlap.
      - "overlap_with_earlier_proposal": this proposal's match span overlaps
        a span already claimed by an earlier-positioned proposal.
    """
    if not proposals:
        return raw_text, [], []

    located: List[Tuple[int, int, Dict[str, Any]]] = []
    unapplied: List[Dict[str, Any]] = []

    for p in proposals:
        before = (p.get("before_text") or "").strip()
        if not before or before.startswith("["):
            unapplied.append({"proposal": p, "reason": "no_anchor"})
            continue
        if _count_paragraph_breaks(before) > _MAX_PARAGRAPH_BREAKS_IN_BEFORE_TEXT:
            unapplied.append(
                {"proposal": p, "reason": "before_text_spans_too_many_paragraphs"}
            )
            continue
        pattern = _build_match_pattern(before)
        if pattern is None:
            unapplied.append({"proposal": p, "reason": "no_anchor"})
            continue
        m = pattern.search(raw_text)
        if not m:
            unapplied.append({"proposal": p, "reason": "before_text_not_found"})
            continue
        located.append((m.start(), m.end(), p))

    located.sort(key=lambda t: t[0])
    accepted: List[Tuple[int, int, Dict[str, Any]]] = []
    last_end = -1
    for start, end, p in located:
        if start < last_end:
            unapplied.append({"proposal": p, "reason": "overlap_with_earlier_proposal"})
            continue
        accepted.append((start, end, p))
        last_end = end

    text = raw_text
    applied: List[Dict[str, Any]] = []
    # Right-to-left so earlier spans aren't shifted by later substitutions.
    for start, end, p in reversed(accepted):
        action = (p.get("action") or "rewrite").lower()
        after = p.get("after_text") or ""
        if action == "delete":
            seg_end = end
            while seg_end < len(text) and text[seg_end] in "\r\n":
                seg_end += 1
            text = text[:start] + text[seg_end:]
        else:
            text = text[:start] + after + text[end:]
        applied.append(p)

    applied.reverse()
    return text, applied, unapplied


# ---------------------------------------------------------------------------
# LLM-driven LaTeX generation
# ---------------------------------------------------------------------------

def _generate_latex(
    raw_text: str,
    approved_proposals: List[Dict[str, Any]],
    doc_type: str,
) -> Tuple[str, bool, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply accepted proposals deterministically, then format as LaTeX.

    Returns (latex_source, llm_succeeded, applied_proposals, unapplied_entries).
    Falls back to a minimal template wrapping the modified text if LLM is
    unavailable. Either way, deterministic application has already happened
    so accepted edits will always be in the output text.
    """
    modified_text, applied, unapplied = _apply_proposals_to_text(
        raw_text, approved_proposals
    )
    if unapplied and approved_proposals:
        logger.warning(
            "Could not deterministically apply %d/%d accepted proposals — "
            "passing them to the LLM as hints (reasons: %s)",
            len(unapplied), len(approved_proposals),
            ", ".join(sorted({u["reason"] for u in unapplied})),
        )

    llm = get_llm()
    if llm is None:
        logger.warning("LLM unavailable — using plain-text LaTeX fallback")
        return _fallback_latex(modified_text), False, applied, unapplied

    is_prose = doc_type in ("SOP", "COVER_LETTER")
    template = _PROSE_TEMPLATE if is_prose else _RESUME_TEMPLATE
    system_prompt = _LATEX_SYSTEM_PROSE if is_prose else _LATEX_SYSTEM

    hints_text = ""
    for entry in unapplied:
        p = entry["proposal"]
        hints_text += (
            f"\n--- Hint ---\n"
            f"Section: {p.get('section', 'N/A')}\n"
            f"Rationale: {p.get('rationale', 'N/A')}\n"
            f"Suggested addition / change: {p.get('after_text', '(empty)')}\n"
        )

    human_content = (
        f"Document type: {doc_type}\n\n"
        f"=== LATEX TEMPLATE (use this preamble EXACTLY) ===\n"
        f"{template}\n\n"
        f"=== DOCUMENT TEXT (accepted edits already applied) ===\n"
        f"{modified_text[:_MAX_DOC_CHARS]}\n\n"
    )
    if hints_text:
        human_content += (
            f"=== ADDITIONAL HINTS (could not be auto-applied — apply only if you can locate the right spot) ===\n"
            f"{hints_text[:_MAX_PROPOSALS_CHARS]}\n\n"
        )
    human_content += "Generate the complete LaTeX source now, using the template above."

    msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    try:
        response = llm.invoke(msgs)
        latex = (response.content or "").strip()

        # Strip markdown fences if present
        latex = re.sub(r"^```(?:latex|tex)?\s*\n?", "", latex)
        latex = re.sub(r"\n?```\s*$", "", latex)

        # Remove glyphtounicode if LLM sneaks it in despite instructions
        latex = re.sub(r"\\input\{glyphtounicode\}\s*\n?", "", latex)
        latex = re.sub(r"\\pdfgentounicode\s*=\s*1\s*\n?", "", latex)

        # Basic sanity check
        if r"\documentclass" not in latex or r"\end{document}" not in latex:
            logger.warning("LLM output doesn't look like valid LaTeX — falling back")
            return _fallback_latex(modified_text), False, applied, unapplied

        return latex, True, applied, unapplied

    except Exception as e:
        logger.exception("LLM LaTeX generation failed: %s", e)
        return _fallback_latex(modified_text), False, applied, unapplied


# ---------------------------------------------------------------------------
# LaTeX sanitization for tectonic compatibility
# ---------------------------------------------------------------------------

def _sanitize_latex(latex: str) -> str:
    """Remove packages and commands known to break tectonic compilation."""
    # Remove problematic packages
    latex = re.sub(r"\\input\{glyphtounicode\}\s*\n?", "", latex)
    latex = re.sub(r"\\pdfgentounicode\s*=\s*1\s*\n?", "", latex)
    latex = re.sub(r"\\usepackage(\[.*?\])?\{fontspec\}\s*\n?", "", latex)
    latex = re.sub(r"\\usepackage(\[.*?\])?\{moderncv\}\s*\n?", "", latex)
    latex = re.sub(r"\\usepackage(\[.*?\])?\{awesome-cv\}\s*\n?", "", latex)
    latex = re.sub(r"\\usepackage(\[.*?\])?\{fontawesome5?\}\s*\n?", "", latex)
    # Remove fontspec-dependent commands
    latex = re.sub(r"\\setmainfont(\[.*?\])?\{.*?\}\s*\n?", "", latex)
    latex = re.sub(r"\\setsansfont(\[.*?\])?\{.*?\}\s*\n?", "", latex)
    latex = re.sub(r"\\setmonofont(\[.*?\])?\{.*?\}\s*\n?", "", latex)
    # Remove fontawesome icon commands (replace with empty)
    latex = re.sub(r"\\fa[A-Z][a-zA-Z]*", "", latex)
    return latex


# ---------------------------------------------------------------------------
# AI-tell normalize pass (SOP/COVER_LETTER prose path only)
# ---------------------------------------------------------------------------
#
# Belt-and-suspenders against AI-writing tells. The synthesizer prompt forbids
# them and the evaluator blocks on em-dashes + banned phrases, but the
# finalize LLM can still re-introduce them while reflowing the document into
# LaTeX. We strip them deterministically before compile so the rendered PDF
# stays clean even if the LLM regresses.
#
# Single-rule em-dash policy: replace with comma. Predictable; risks an
# occasional awkward comma where a colon would read better, but that's a
# smaller cost than letting an em-dash slip into the rendered PDF.

# Banned-phrase rewrites — same list as the synth/evaluator. Conservative
# substitutions that read fine in any context the original phrase appeared in.
_BANNED_PHRASE_REWRITES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bI\s+believe\s+my\s+background\s+in\b", re.IGNORECASE), "My background in"),
    (re.compile(r"\bI\s+see\s+this\s+(opportunity|position)\s+as\s+a\s+chance\s+to\b", re.IGNORECASE), r"This \1 would let me"),
    (re.compile(r"\bcontinue\s+developing\s+myself\b", re.IGNORECASE), "keep developing"),
    (re.compile(r"\bI\s+am\s+especially\s+motivated\s+by\b", re.IGNORECASE), "What draws me to this is"),
    (re.compile(r"\bdirectly\s+shapes\b", re.IGNORECASE), "shapes"),
    (re.compile(r"\bplay\s+a\s+meaningful\s+role\b", re.IGNORECASE), "contribute"),
    (re.compile(r"\btapestry\b", re.IGNORECASE), "mix"),
    (re.compile(r"\bdelving\s+into\b", re.IGNORECASE), "working on"),
    (re.compile(r"\bdelve\s+into\b", re.IGNORECASE), "work on"),
    (re.compile(r"\bstands\s+out\s+to\s+me\b", re.IGNORECASE), "interests me"),
    (re.compile(r"\bmatters\s+to\s+me\s+because\b", re.IGNORECASE), "is important because"),
]


def _normalize_ai_tells(latex_body: str) -> str:
    """Strip em-dashes and banned phrases from LaTeX prose body.

    Operates on the rendered LaTeX source after the LLM produces it; runs
    only on the SOP/COVER_LETTER prose path. The CV path is untouched
    because em-dashes in CV bullets (date ranges, "Aug 2023 — present") are
    a legitimate convention.

    Em-dashes:
      - Long em-dash `—` and double-hyphen ` -- ` → comma + space.
      - Hyphen-with-spaces ` - ` (when used as a dash) is left alone — it's
        ambiguous with compound modifiers and replacing risks worse output
        than leaving it.

    Banned phrases:
      - Conservative regex map; same list as the synth prompt and evaluator.
      - Substitutions are deliberately tame — they shouldn't change meaning,
        only register. A document that produced these phrases despite the
        synth prompt and evaluator block is already past two filters; this
        is the last line of defense.
    """
    # 1. Em-dashes. Long em-dash → comma; double-hyphen → comma. Run after
    # other rules so banned phrases match against the original wording.
    out = latex_body

    # 2. Banned phrases.
    for pattern, replacement in _BANNED_PHRASE_REWRITES:
        out = pattern.sub(replacement, out)

    # 3. Em-dash variants → comma. Handle spaced ` — ` first to avoid
    # producing double commas, then bare `—`, then ` -- ` (LaTeX en-dash
    # input that renders as em-dash via ligature).
    out = re.sub(r"\s+—\s+", ", ", out)
    out = re.sub(r"—", ", ", out)
    out = re.sub(r"\s--\s", ", ", out)

    return out


def _strip_resume_commands_for_prose(latex: str) -> str:
    """Defense in depth for SOP/COVER_LETTER output.

    The prose prompt forbids resume helpers and list environments, but if the
    LLM regresses to its training-data prior and emits `\\resumeItemPlain` or
    wraps paragraphs in itemize anyway, the prose template doesn't define
    those commands — compilation would fail (or worse, succeed via the
    fallback and still bullet everything). Unwrap them to plain paragraphs
    so the rendered PDF is prose, not a bulleted list.
    """
    # Drop list-environment delimiters; keep the content between them.
    latex = re.sub(r"\\begin\{(itemize|enumerate)\}\s*\n?", "", latex)
    latex = re.sub(r"\\end\{(itemize|enumerate)\}\s*\n?", "", latex)
    # Resume "list start/end" wrappers — same treatment.
    latex = re.sub(r"\\resume(?:Item|SubHeading)?ListStart\s*\n?", "", latex)
    latex = re.sub(r"\\resume(?:Item|SubHeading)?ListEnd\s*\n?", "", latex)
    # \resumeItemPlain{X} → X (one paragraph). Same for \item{X} variants.
    latex = re.sub(r"\\resumeItemPlain\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1\n", latex)
    latex = re.sub(r"\\resumeSubItem\s*\{([^{}]*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1: \2\n", latex)
    latex = re.sub(r"\\resumeItem\s*\{([^{}]*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1: \2\n", latex)
    # Bare \item — turn each into a paragraph break.
    latex = re.sub(r"^\s*\\item\s+", "\n", latex, flags=re.MULTILINE)
    # \resumeSubheading is too specific to safely unwrap; just drop the call.
    latex = re.sub(r"\\resumeSubheading\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\{[^{}]*\}\s*\{[^{}]*\}", "", latex)
    # \resumeProjectHeading{X} → X as a paragraph.
    latex = re.sub(r"\\resumeProjectHeading\s*\{([^{}]*)\}", r"\1\n", latex)
    return latex


def _aggressive_cleanup(latex: str) -> str:
    """More aggressive cleanup — strips anything non-essential."""
    latex = _sanitize_latex(latex)
    # Remove any \usepackage lines for unknown packages (keep only known-safe ones)
    safe_packages = {
        "fullpage", "titlesec", "color", "enumitem", "hyperref",
        "fancyhdr", "babel", "tabularx", "geometry", "inputenc",
        "fontenc", "xcolor", "array", "multicol", "multirow",
        "textcomp", "latexsym", "marvosym", "amssymb",
        # Prose-template packages
        "parskip", "microtype",
    }
    lines = latex.split("\n")
    cleaned = []
    for line in lines:
        m = re.match(r"\\usepackage(?:\[.*?\])?\{(\w+)\}", line.strip())
        if m and m.group(1) not in safe_packages:
            logger.warning("Stripping unknown package: %s", m.group(1))
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _fallback_latex(raw_text: str) -> str:
    """Minimal LaTeX wrapper when LLM is unavailable."""
    # Escape special LaTeX characters
    escaped = _escape_latex(raw_text)
    return (
        r"\documentclass[11pt,a4paper]{article}" "\n"
        r"\usepackage[utf8]{inputenc}" "\n"
        r"\usepackage[T1]{fontenc}" "\n"
        r"\usepackage[margin=1in]{geometry}" "\n"
        r"\usepackage{enumitem}" "\n"
        r"\usepackage{titlesec}" "\n"
        r"\usepackage{hyperref}" "\n"
        r"\begin{document}" "\n\n"
        f"{escaped}\n\n"
        r"\end{document}" "\n"
    )


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------------------
# Diff summary (simplified — no more "could not apply" for LLM rewrites)
# ---------------------------------------------------------------------------

def _build_diff(
    all_proposals: List[Dict[str, Any]],
    approved_proposals: List[Dict[str, Any]],
    applied_proposals: List[Dict[str, Any]],
    unapplied_entries: List[Dict[str, Any]],
    llm_succeeded: bool,
) -> Dict[str, Any]:
    """Build a summary of what was done.

    `applied_proposals` and `unapplied_entries` come from
    `_apply_proposals_to_text` and reflect what was actually merged into the
    document text — not just what the user approved.
    """
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

    applied = [
        {
            "section": p.get("section", ""),
            "rationale": p.get("rationale", ""),
            "before": (p.get("before_text") or "")[:120],
            "after": (p.get("after_text") or "")[:120],
        }
        for p in applied_proposals
    ]

    could_not_apply = [
        {
            "section": entry["proposal"].get("section", ""),
            "rationale": entry["proposal"].get("rationale", ""),
            "reason": entry["reason"],
        }
        for entry in unapplied_entries
    ]

    n_applied = len(applied)
    n_rejected = len(rejected)
    n_unapplied = len(could_not_apply)

    parts: List[str] = []
    if n_applied:
        parts.append(f"{n_applied} change{'s' if n_applied != 1 else ''} incorporated")
    if n_unapplied:
        parts.append(
            f"{n_unapplied} could not be applied automatically "
            "(passed to formatter as hints)"
        )
    if n_rejected:
        parts.append(f"{n_rejected} rejected by user")
    if llm_succeeded:
        parts.append("professional LaTeX document generated")
    else:
        parts.append("plain-text fallback used (LLM unavailable)")

    summary = ("; ".join(parts) + ".") if parts else "No changes applied."

    return {
        "applied": applied,
        "rejected": rejected,
        "conflicts": [],
        "could_not_apply": could_not_apply,
        "smoothing_applied": False,
        "latex_generated": llm_succeeded,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def finalize(state: DocFeedbackState) -> dict:
    updates = {"current_step": "finalize", "step_history": ["finalize"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    raw_text = state.get("raw_text") or ""
    human_review = state.get("human_review") or {}
    approved_proposals: List[Dict[str, Any]] = human_review.get("approved_proposals") or []
    all_proposals: List[Dict[str, Any]] = state.get("proposals") or []
    doc_type = (state.get("doc_classification") or {}).get("doc_type", "CV")

    # ------------------------------------------------------------------
    # Step 1: Apply accepted proposals deterministically + generate LaTeX
    # ------------------------------------------------------------------
    try:
        latex_source, llm_succeeded, applied, unapplied = _generate_latex(
            raw_text, approved_proposals, doc_type
        )
    except Exception as e:
        return {
            **updates,
            "result": {
                "status": "error",
                "error_code": "LATEX_GENERATION_FAILED",
                "user_message": (
                    "We could not generate the final document. "
                    "Your selections have been saved."
                ),
                "details": {
                    "exception": str(e),
                    "approved_proposals": approved_proposals,
                },
            },
        }

    # ------------------------------------------------------------------
    # Step 2: Sanitize LaTeX + Compile → PDF (with retry)
    # ------------------------------------------------------------------
    pdf_bytes = None
    if llm_succeeded:
        # Sanitize common LLM mistakes that break tectonic
        latex_source = _sanitize_latex(latex_source)
        # For SOP / COVER_LETTER, the prose template defines no resume / list
        # commands. If the LLM regressed and emitted them anyway, unwrap to
        # plain paragraphs so the PDF doesn't bullet-point every paragraph.
        # Then strip AI-writing tells (em-dashes, banned phrases) — the
        # synth prompt and evaluator block these, but the finalize LLM can
        # re-introduce them while reflowing prose into LaTeX.
        if doc_type in ("SOP", "COVER_LETTER"):
            latex_source = _strip_resume_commands_for_prose(latex_source)
            latex_source = _normalize_ai_tells(latex_source)

        pdf_bytes = compile_latex(latex_source)
        if pdf_bytes is None:
            # Retry with more aggressive cleanup
            logger.warning("First compile failed — retrying with aggressive cleanup")
            latex_source = _aggressive_cleanup(latex_source)
            pdf_bytes = compile_latex(latex_source)
            if pdf_bytes is None:
                logger.warning("LaTeX compilation failed after retry — will store source only")

    # ------------------------------------------------------------------
    # Step 3: Build diff summary
    # ------------------------------------------------------------------
    diff = _build_diff(
        all_proposals=all_proposals,
        approved_proposals=approved_proposals,
        applied_proposals=applied,
        unapplied_entries=unapplied,
        llm_succeeded=llm_succeeded,
    )

    # ------------------------------------------------------------------
    # Step 4: Write results to state
    # ------------------------------------------------------------------
    return {
        **updates,
        "final_document": latex_source,
        "final_pdf_bytes": pdf_bytes,
        "diff": diff,
        "result": {
            "status": "ok",
            "user_message": diff["summary"],
            "details": {
                "final_document": latex_source,
                "diff": diff,
                "pdf_compiled": pdf_bytes is not None,
            },
        },
    }
