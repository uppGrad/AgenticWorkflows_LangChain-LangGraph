"""LaTeX preambles + skeletons used by `application_tailoring`.

The tailoring node passes one of these as a "skeleton" the LLM must respect
when it emits the document. Each template is a complete, self-contained
LaTeX source string that compiles standalone — the LLM fills in marked
sections (CV bullets, paragraphs of the cover letter, etc.) without
touching the preamble.

The CV and prose templates are sourced from the document-feedback
workflow's `finalize.py` (`_RESUME_TEMPLATE` and `_PROSE_TEMPLATE`). Doing
this means:
  * Auto-apply's CV LaTeX prompt has the SAME custom commands the LLM
    has been trained on (via the doc-feedback path) — fewer syntax
    errors, fewer fall-throughs to the ReportLab plain-text fallback.
  * Both workflows produce typeset CVs with the same visual style.
  * If doc-feedback's templates evolve (the teammate has been
    iterating), one diff lands changes in both workflows — but to keep
    the cross-workflow coupling explicit, we copy rather than import.
    Re-sync points are flagged here and in the doc-feedback finalize node.

Design constraints (mirrored from doc-feedback):
  * Tectonic-installable packages only (no fontspec, moderncv, awesome-cv).
  * No `\\input{glyphtounicode}` / `\\pdfgentounicode` (Tectonic
    incompatibility).
  * No `\\input`, `\\include`, `\\write18`, or shell-escape commands —
    the backend renders with `--untrusted` (sandboxed mode).

Per-doc-type templates:
  * `CV_TEMPLATE` — sb2nov-based resume layout with `\\resumeItemPlain`
    / `\\resumeSubheading` / `\\resumeItemListStart` helpers. Compatible
    with the doc-feedback path's CV finalizer.
  * `COVER_LETTER_TEMPLATE` — article + parskip prose. Mirrors
    doc-feedback's `_PROSE_TEMPLATE`.
  * `GENERIC_TEMPLATE` — fallback for SOP / Personal Statement /
    Motivation Letter / Research Proposal: same prose template since
    they all read as flowing paragraphs.

The placeholders the LLM fills are explicit `% --- BEGIN BODY ---` /
`% --- END BODY ---` markers; we keep them in the rendered output so
future edits can locate where the model wrote.
"""
from __future__ import annotations

from typing import Optional


# ─── Templates ──────────────────────────────────────────────────────────────
#
# CV_TEMPLATE is sourced from `workflows/document_feedback/nodes/finalize.py`
# `_RESUME_TEMPLATE` (sb2nov-derived). When that file changes, sync here.
# The body markers + heading skeleton are appended so the LLM has explicit
# fill instructions consistent with the cover-letter / generic templates.

CV_TEMPLATE = r"""%-------------------------
% Auto-generated CV — preamble fixed by uppgrad. Body filled by LLM.
% Sourced from workflows/document_feedback/nodes/finalize.py:_RESUME_TEMPLATE
% (sb2nov-based). Tectonic-compatible — no glyphtounicode.
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

% --- BEGIN BODY ---
%% LLM: fill the resume here. Use the custom commands defined above:
%%
%% Header (top of doc):
%%   \begin{tabular*}{\textwidth}{l@{\extracolsep{\fill}}r}
%%     \textbf{\Large FULL NAME} & Email: \href{mailto:EMAIL}{EMAIL}\\
%%     \href{WEBSITE}{WEBSITE} & Mobile: PHONE \\
%%   \end{tabular*}
%%
%% Sections (Education / Experience / Projects / Skills, etc.):
%%   \section{Section Name}
%%     \resumeSubHeadingListStart
%%       \resumeSubheading{ORG / SCHOOL}{LOCATION}{TITLE / DEGREE}{DATES}
%%         \resumeItemListStart
%%           \resumeItemPlain{Bullet text describing achievement.}
%%         \resumeItemListEnd
%%     \resumeSubHeadingListEnd
%%
%% Skills (categorised):
%%   \section{Skills}
%%     \resumeSubHeadingListStart
%%       \item{
%%         \textbf{Languages}{: Python, Go} \hfill
%%         \textbf{Tools}{: Docker, Kubernetes}
%%       }
%%     \resumeSubHeadingListEnd
%%
%% Projects (use \resumeProjectHeading for multi-bullet projects;
%% \resumeSubItem only for single-line projects). NEVER stack multiple
%% \resumeSubItem with the same name, and NEVER use an empty first arg.
%%   \section{Projects}
%%     \resumeSubHeadingListStart
%%       \resumeSubItem{ONE-LINE PROJECT}{Single description.}
%%       \resumeProjectHeading{MULTI-BULLET PROJECT NAME}
%%         \resumeItemListStart
%%           \resumeItemPlain{First bullet.}
%%           \resumeItemPlain{Second bullet.}
%%         \resumeItemListEnd
%%     \resumeSubHeadingListEnd
% --- END BODY ---

\end{document}
"""


# ─── Prose template (Cover Letter / SOP / Personal Statement / etc.) ─────────
#
# Sourced from `workflows/document_feedback/nodes/finalize.py:_PROSE_TEMPLATE`.
# Plain article class with parskip — paragraphs render as flowing prose with
# blank-line breaks. NO list helpers; NEVER mix the resume-template custom
# commands here (the LLM wrapping each cover-letter paragraph in
# `\resumeItemPlain` was a real symptom in the doc-feedback path before this
# template existed).

_PROSE_TEMPLATE = r"""%-------------------------
% Auto-generated prose document (cover letter / SOP / motivation letter)
% Preamble fixed by uppgrad. Body filled by LLM.
% Sourced from workflows/document_feedback/nodes/finalize.py:_PROSE_TEMPLATE
% Tectonic-compatible.
%-------------------------

\documentclass[11pt,letterpaper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{parskip}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.7em}

\begin{document}

% --- BEGIN BODY ---
%% LLM: emit the document body here as FLOWING PARAGRAPHS separated by
%% blank lines. NEVER use \begin{itemize}, \begin{enumerate}, \item, or
%% any of the resume-template commands (\resumeItem, \resumeItemPlain,
%% \resumeSubheading, etc.). Those don't exist in this preamble and will
%% fail to compile.
%%
%% Cover letter shape:
%%   <date>
%%   <recipient address (optional)>
%%   <salutation>
%%
%%   <opening paragraph>
%%
%%   <body paragraph 1>
%%
%%   <body paragraph 2>
%%
%%   <closing paragraph>
%%
%%   <sign-off>
%%   <name>
%%
%% SOP shape: usually no headings — flowing paragraphs in document order.
%% Use \section*{Heading} ONLY when the source document has clear section
%% breaks. Most SOPs have no headings; that's fine.
% --- END BODY ---

\end{document}
"""


COVER_LETTER_TEMPLATE = _PROSE_TEMPLATE
GENERIC_TEMPLATE = _PROSE_TEMPLATE


# Map canonical document types → templates. The key matches the values in
# `_GENERATABLE` / `_USER_SUPPLIED` in nodes/asset_mapping.py and the
# `document_type` carried on RequirementItem.
_BY_DOC_TYPE = {
    "CV": CV_TEMPLATE,
    "Cover Letter": COVER_LETTER_TEMPLATE,
    "Motivation Letter": COVER_LETTER_TEMPLATE,
    "SOP": GENERIC_TEMPLATE,
    "Personal Statement": GENERIC_TEMPLATE,
    "Research Proposal": GENERIC_TEMPLATE,
    "Writing Sample": GENERIC_TEMPLATE,
    "References": GENERIC_TEMPLATE,
}


def template_for(doc_type: Optional[str]) -> str:
    """Return the LaTeX skeleton best matching `doc_type`. Falls back to
    `GENERIC_TEMPLATE` for anything unrecognised (or `None`)."""
    if not doc_type:
        return GENERIC_TEMPLATE
    return _BY_DOC_TYPE.get(doc_type.strip(), GENERIC_TEMPLATE)
