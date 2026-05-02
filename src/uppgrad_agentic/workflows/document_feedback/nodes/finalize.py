from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

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
% USE: \begin{tabular*}{\textwidth}{l@{\extracolsep{\fill}}r}
%        \textbf{\Large FULL NAME} & Email: \href{mailto:EMAIL}{EMAIL}\\
%        \href{WEBSITE}{WEBSITE} & Mobile: PHONE \\
%      \end{tabular*}

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
%     \resumeSubItem{PROJECT NAME}{DESCRIPTION}
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
  1. The full plain-text of a document (extracted from a PDF).
  2. A list of ACCEPTED change proposals (each with before_text, after_text, rationale).
  3. A LaTeX TEMPLATE with preamble, custom commands, and commented-out section markers.

YOUR TASK:
  - Produce a COMPLETE, COMPILABLE LaTeX document using EXACTLY the provided template preamble and custom commands.
  - Fill in the content sections (HEADING, EDUCATION, EXPERIENCE, PROJECTS, SKILLS, etc.) using the original document text AND the accepted changes.
  - Use ONLY the custom commands defined in the template: \resumeSubheading, \resumeItemPlain, \resumeItem, \resumeSubItem, \resumeSubHeadingListStart/End, \resumeItemListStart/End.
  - Incorporate every accepted change proposal (replace before_text with after_text).
  - If a section from the original document is not covered by the template markers, create a new \section{} for it.

CRITICAL RULES:
  - Use EXACTLY the \documentclass, \usepackage, and \newcommand lines from the template. Do NOT add, remove, or change any packages.
  - Do NOT use \input{glyphtounicode} or \pdfgentounicode — they are incompatible with the compiler.
  - Do NOT use fontspec, moderncv, awesome-cv, or any custom .cls files.
  - Do NOT invent, fabricate, or add ANY facts, dates, skills, experiences, or achievements not present in the original document or the accepted proposals.
  - Do NOT remove any original content unless explicitly instructed by an accepted proposal.
  - Escape special LaTeX characters in user content: & % $ # _ { } ~ ^

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
  1. The full plain-text of the document (extracted from a PDF).
  2. A list of ACCEPTED change proposals (each with before_text, after_text, rationale).
  3. A minimal LaTeX TEMPLATE (article class, parskip-based prose).

YOUR TASK:
  - Produce a COMPLETE, COMPILABLE LaTeX document using EXACTLY the template's preamble.
  - Render the document as FLOWING PROSE PARAGRAPHS separated by blank lines.
  - Incorporate every accepted change proposal (replace before_text with after_text).

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
experiences not present in the original document or the accepted proposals.
  - Do NOT remove any original content unless an accepted proposal replaces it.
  - Each proposal carries an `action` field which controls what to do:
    * `action="rewrite"` (default): the proposal's `before_text` matches a \
paragraph in the original — REPLACE that paragraph with `after_text`.
    * `action="delete"`: the proposal's `before_text` matches a paragraph in \
the original — OMIT that paragraph entirely from your output. `after_text` \
will be empty for delete proposals; do NOT include any placeholder text in \
its place.
    * `action="merge"`: the proposal's `before_text` is two paragraphs from \
the original concatenated. Replace BOTH source paragraphs with the single \
merged `after_text`.
  - Apply proposals once each. Do NOT duplicate paragraphs. Do NOT keep a \
paragraph that an accepted delete-proposal flagged for removal.
  - DO NOT use em-dashes (—) or double-hyphens (--) in the body. The \
proposals were authored to avoid them; preserve that. Use commas, periods, \
or colons instead. Em-dashes are the strongest "AI-generated" tell and \
should not appear in the rendered document.
  - Escape special LaTeX characters in user content: & % $ # _ { } ~ ^
  - Curly/smart quotes are fine — they render correctly under utf8.

Return ONLY the complete LaTeX source code. No explanations, no markdown fences.
Start with \documentclass and end with \end{document}.
"""

_MAX_DOC_CHARS = 16000
_MAX_PROPOSALS_CHARS = 8000


# ---------------------------------------------------------------------------
# LLM-driven LaTeX generation
# ---------------------------------------------------------------------------

def _generate_latex(
    raw_text: str,
    approved_proposals: List[Dict[str, Any]],
    doc_type: str,
) -> Tuple[str, bool]:
    """Ask the LLM to generate a complete LaTeX document.

    Returns (latex_source, llm_succeeded).
    Falls back to a minimal template wrapping raw_text if LLM is unavailable.
    """
    llm = get_llm()
    if llm is None:
        logger.warning("LLM unavailable — using plain-text LaTeX fallback")
        return _fallback_latex(raw_text), False

    # Build the proposals text
    proposals_text = ""
    for i, p in enumerate(approved_proposals, 1):
        proposals_text += (
            f"\n--- Proposal {i} ---\n"
            f"Section: {p.get('section', 'N/A')}\n"
            f"Rationale: {p.get('rationale', 'N/A')}\n"
            f"Before: {p.get('before_text', '(empty)')}\n"
            f"After: {p.get('after_text', '(empty)')}\n"
        )

    # CV → resume template + bullet-list helpers.
    # SOP / COVER_LETTER → prose template, no list helpers (avoids the
    # bug where every paragraph rendered as a bullet point because the
    # resume prompt instructed the LLM to wrap content in \resumeItemPlain).
    is_prose = doc_type in ("SOP", "COVER_LETTER")
    template = _PROSE_TEMPLATE if is_prose else _RESUME_TEMPLATE
    system_prompt = _LATEX_SYSTEM_PROSE if is_prose else _LATEX_SYSTEM

    human_content = (
        f"Document type: {doc_type}\n\n"
        f"=== LATEX TEMPLATE (use this preamble EXACTLY) ===\n"
        f"{template}\n\n"
        f"=== ORIGINAL DOCUMENT TEXT ===\n"
        f"{raw_text[:_MAX_DOC_CHARS]}\n\n"
        f"=== ACCEPTED CHANGE PROPOSALS ({len(approved_proposals)} total) ===\n"
        f"{proposals_text[:_MAX_PROPOSALS_CHARS]}\n\n"
        f"Generate the complete LaTeX source now, using the template above."
    )

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
            return _fallback_latex(raw_text), False

        return latex, True

    except Exception as e:
        logger.exception("LLM LaTeX generation failed: %s", e)
        return _fallback_latex(raw_text), False


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
    llm_succeeded: bool,
) -> Dict[str, Any]:
    """Build a summary of what was done."""
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
        for p in approved_proposals
    ]

    n_applied = len(applied)
    n_rejected = len(rejected)

    parts: List[str] = []
    if n_applied:
        parts.append(f"{n_applied} change{'s' if n_applied != 1 else ''} incorporated")
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
        "could_not_apply": [],
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
    # Step 1: Generate LaTeX document with accepted changes
    # ------------------------------------------------------------------
    try:
        latex_source, llm_succeeded = _generate_latex(
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
