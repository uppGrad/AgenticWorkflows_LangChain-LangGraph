"""LaTeX preambles + skeletons used by `application_tailoring`.

The tailoring node passes one of these as a "skeleton" the LLM must respect
when it emits the document. Each template is a complete, self-contained
LaTeX source string that compiles standalone — the LLM fills in marked
sections (CV bullets, paragraphs of the cover letter, etc.) without
touching the preamble.

Design constraints:
  * Only Tectonic-installable packages. Stick to what ships in the default
    TeX Live release so the backend image stays small. Currently:
    `geometry`, `enumitem`, `hyperref`, `parskip`, `xcolor`, `titlesec`.
    NO `moderncv`, `awesome-cv`, `fontawesome` — those drag in too many deps.
  * No `\\input`, `\\include`, `\\write18`, or shell-escape commands. The
    backend renders with `--no-shell-escape`; calls in the LLM output that
    need them will fail at compile time, which we want.
  * Preambles must declare a Latin-script-friendly font. Default Computer
    Modern is fine; the user's tailored content is plain Unicode-friendly.

Per-doc-type templates:
  * `CV_TEMPLATE` — resume layout: contact line, sections (Education,
    Experience, Projects, Skills), bullet lists with `enumitem`.
  * `COVER_LETTER_TEMPLATE` — letter-style article with sender block, date,
    addressee, salutation, paragraphs, sign-off.
  * `GENERIC_TEMPLATE` — fallback for SOP / Personal Statement / Motivation
    Letter / Research Proposal: article class with reasonable margins.

The placeholders the LLM fills are explicit `% --- BEGIN BODY ---` /
`% --- END BODY ---` markers; we keep them in the rendered output so future
edits can locate where the model wrote.
"""
from __future__ import annotations

from typing import Optional


# ─── Templates ──────────────────────────────────────────────────────────────

CV_TEMPLATE = r"""% Auto-generated CV — preamble fixed by uppgrad. Body filled by LLM.
\documentclass[11pt,a4paper]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{parskip}
\setlist[itemize]{leftmargin=*,nosep,topsep=2pt}
\titleformat{\section}{\large\bfseries\color{black!85}}{}{0em}{}[\titlerule]
\titlespacing*{\section}{0pt}{8pt}{4pt}
\pagenumbering{gobble}
\begin{document}
% --- BEGIN BODY ---
%% LLM: replace this comment with the CV. Use \section{...} for each
%% top-level section (Contact, Summary, Experience, Education, Projects,
%% Skills). Use \begin{itemize} ... \end{itemize} for bullets. Keep
%% inline links via \href{url}{label}.
% --- END BODY ---
\end{document}
"""


COVER_LETTER_TEMPLATE = r"""% Auto-generated cover letter — preamble fixed by uppgrad. Body filled by LLM.
\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage[hidelinks]{hyperref}
\usepackage{parskip}
\pagenumbering{gobble}
\begin{document}
% --- BEGIN BODY ---
%% LLM: emit a complete cover letter here — sender contact block at top,
%% date, addressee, salutation, 2-4 paragraphs, sign-off. Use plain
%% paragraphs separated by blank lines (parskip handles spacing). No
%% \section{...} here — cover letters read better without headings.
% --- END BODY ---
\end{document}
"""


GENERIC_TEMPLATE = r"""% Auto-generated document — preamble fixed by uppgrad. Body filled by LLM.
\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{titlesec}
\usepackage[hidelinks]{hyperref}
\usepackage{parskip}
\titleformat{\section}{\large\bfseries}{}{0em}{}
\titlespacing*{\section}{0pt}{8pt}{4pt}
\pagenumbering{arabic}
\begin{document}
% --- BEGIN BODY ---
%% LLM: emit the document body here. Use \section{...} when the document
%% has clear sections (SOP, Research Proposal); otherwise plain paragraphs.
% --- END BODY ---
\end{document}
"""


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
