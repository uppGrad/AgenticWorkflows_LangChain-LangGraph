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
# LaTeX generation prompt
# ---------------------------------------------------------------------------

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

_MAX_DOC_CHARS = 8000
_MAX_PROPOSALS_CHARS = 4000


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

    human_content = (
        f"Document type: {doc_type}\n\n"
        f"=== LATEX TEMPLATE (use this preamble and commands EXACTLY) ===\n"
        f"{_RESUME_TEMPLATE}\n\n"
        f"=== ORIGINAL DOCUMENT TEXT ===\n"
        f"{raw_text[:_MAX_DOC_CHARS]}\n\n"
        f"=== ACCEPTED CHANGE PROPOSALS ({len(approved_proposals)} total) ===\n"
        f"{proposals_text[:_MAX_PROPOSALS_CHARS]}\n\n"
        f"Generate the complete LaTeX source now, using the template above."
    )

    msgs = [
        SystemMessage(content=_LATEX_SYSTEM),
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


def _aggressive_cleanup(latex: str) -> str:
    """More aggressive cleanup — strips anything non-essential."""
    latex = _sanitize_latex(latex)
    # Remove any \usepackage lines for unknown packages (keep only known-safe ones)
    safe_packages = {
        "fullpage", "titlesec", "color", "enumitem", "hyperref",
        "fancyhdr", "babel", "tabularx", "geometry", "inputenc",
        "fontenc", "xcolor", "array", "multicol", "multirow",
        "textcomp", "latexsym", "marvosym", "amssymb",
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
    if state.get("result", {}).get("status") == "error":
        return {}

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
            }
        }

    # ------------------------------------------------------------------
    # Step 2: Sanitize LaTeX + Compile → PDF (with retry)
    # ------------------------------------------------------------------
    pdf_bytes = None
    if llm_succeeded:
        # Sanitize common LLM mistakes that break tectonic
        latex_source = _sanitize_latex(latex_source)

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
