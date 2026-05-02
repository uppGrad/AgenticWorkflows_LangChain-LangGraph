"""Document-level narrative analyzer for SOP and COVER_LETTER documents.

Where `analyze_rhetoric` scores paragraphs in isolation, this analyzer audits
the document as a whole: which paragraphs add new information, which lean on
the same anchor, whether the conclusion commits forward, and where transitions
break. The synthesizer uses this to collapse redundancy (delete / merge
proposals), distribute evidence across paragraphs, and force a purposeful
closing — the failure modes that "targeted but flat" pipelines exhibit.

Skipped for CV (bullet-driven, not argument-driven).
"""
from __future__ import annotations

import re
from typing import List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.common.prompt_context import format_profile_brief, format_user_focus
from uppgrad_agentic.workflows.document_feedback.schemas import (
    NarrativeAnalysis,
    ParagraphRole,
)


# ---------------------------------------------------------------------------
# Paragraph splitting (mirrors analyze_rhetoric so indices line up)
# ---------------------------------------------------------------------------

def _split_paragraphs(doc_sections: dict[str, str]) -> List[Tuple[str, str]]:
    """Yield (section_name, paragraph_text) pairs. Paragraphs split on blank line.

    Filters out short fragments (< 80 chars) the way analyze_rhetoric does, so
    paragraph indices line up between the two analyzers.
    """
    pairs: List[Tuple[str, str]] = []
    for name, text in doc_sections.items():
        if name == "Preamble":
            continue
        for chunk in re.split(r"\n\s*\n", text):
            chunk = chunk.strip()
            if len(chunk) >= 80:
                pairs.append((name, chunk))
    return pairs


# ---------------------------------------------------------------------------
# Heuristic fallback — used when no LLM provider is wired
# ---------------------------------------------------------------------------

# Anchor candidates: capitalised multi-word tokens that look like proper-noun
# evidence (project names, internships, employers). Single capitalised tokens
# are too noisy at sentence start.
_ANCHOR_RE = re.compile(r"\b([A-Z][A-Za-z0-9+#./-]{2,}(?:\s+[A-Z][A-Za-z0-9+#./-]{2,}){0,3})\b")
# Common English words that the regex would otherwise pick up at sentence start.
_ANCHOR_STOPWORDS = {
    "I", "The", "This", "That", "There", "These", "Those", "However", "Although",
    "Before", "After", "During", "Through", "Throughout", "Computer", "Engineering",
    "Software", "Engineer", "Bachelor", "Master", "PhD",
}

_GENERIC_CLOSING_PATTERNS = [
    re.compile(r"\bthank\s+you\s+for\s+(your\s+)?(time|consideration|considering)", re.IGNORECASE),
    re.compile(r"\b(continue\s+developing|grow\s+(both\s+)?personally\s+and\s+professionally)", re.IGNORECASE),
    re.compile(r"\bwould\s+be\s+(happy|delighted|thrilled)\s+for\s+the\s+opportunity", re.IGNORECASE),
    re.compile(r"\bI\s+(believe|think)\s+my\s+background", re.IGNORECASE),
    re.compile(r"\bsee\s+this\s+(opportunity|position)\s+as\s+a\s+chance", re.IGNORECASE),
]


def _extract_anchors(paragraph: str) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for match in _ANCHOR_RE.findall(paragraph):
        token = match.strip()
        if token in _ANCHOR_STOPWORDS:
            continue
        if token.lower() in seen:
            continue
        seen.add(token.lower())
        out.append(token)
    return out


def _heuristic(
    doc_sections: dict[str, str],
    opportunity_context: dict,
) -> NarrativeAnalysis:
    pairs = _split_paragraphs(doc_sections)
    company = (opportunity_context.get("organization") or opportunity_context.get("company") or "").strip().lower()
    role = (opportunity_context.get("title") or "").strip().lower()

    paragraph_roles: List[ParagraphRole] = []
    anchor_to_paragraphs: dict[str, List[int]] = {}

    n = len(pairs)
    for i, (_section, para) in enumerate(pairs):
        anchors = _extract_anchors(para)
        for a in anchors:
            anchor_to_paragraphs.setdefault(a.lower(), []).append(i)

        # Crude role assignment: hook = first paragraph, closing = last,
        # everything else evidence (heuristic doesn't differentiate further).
        if i == 0:
            role_label = "hook"
        elif i == n - 1:
            role_label = "closing"
        elif anchors:
            role_label = "evidence"
        else:
            role_label = "motivation"

        paragraph_roles.append(ParagraphRole(
            paragraph_index=i,
            paragraph_anchor=para[:100],
            role=role_label,
            anchor_examples=anchors[:5],
            adds_new_information=bool(anchors),
        ))

    repeated_anchors: List[Tuple[str, List[int]]] = [
        (anchor, idxs) for anchor, idxs in anchor_to_paragraphs.items()
        if len(idxs) >= 2
    ]

    # Conclusion check: last paragraph must mention org name AND avoid generic
    # closing patterns to count as committing forward.
    if pairs:
        last_para = pairs[-1][1]
        last_lower = last_para.lower()
        mentions_org = bool(company and company in last_lower) or bool(role and role in last_lower)
        generic_hits = sum(1 for p in _GENERIC_CLOSING_PATTERNS if p.search(last_para))
        commits_forward = mentions_org and generic_hits == 0
        if commits_forward:
            audit = "Closing names the target and avoids generic sign-off patterns."
        elif not mentions_org:
            audit = "Closing fails to name the target organisation or role."
        else:
            audit = (
                "Closing reads as a generic sign-off (thank-you / "
                "continue-developing language) rather than a forward commitment."
            )
    else:
        commits_forward = False
        audit = "Document is empty."

    # Deletion candidates: paragraphs with no anchors AND not the hook/closing.
    paragraphs_to_delete = [
        pr.paragraph_index for pr in paragraph_roles
        if pr.role in ("motivation",) and not pr.anchor_examples
    ]

    # Diversity score: ratio of unique anchors to anchor occurrences.
    total_occurrences = sum(len(v) for v in anchor_to_paragraphs.values())
    if total_occurrences == 0:
        diversity = 0.5  # nothing to measure
    else:
        diversity = round(
            min(1.0, len(anchor_to_paragraphs) / total_occurrences),
            2,
        )

    target_paragraph_count = max(1, n - len(paragraphs_to_delete))

    if not pairs:
        summary = "No analyzable paragraphs detected."
    elif repeated_anchors:
        summary = (
            f"{len(repeated_anchors)} anchor(s) reused across paragraphs and "
            f"{len(paragraphs_to_delete)} paragraph(s) add no new information."
        )
    elif paragraphs_to_delete:
        summary = f"{len(paragraphs_to_delete)} paragraph(s) add no new information and should be cut or merged."
    else:
        summary = "Narrative reads as distributed; no major redundancy detected by heuristic."

    return NarrativeAnalysis(
        paragraph_roles=paragraph_roles,
        repeated_anchors=repeated_anchors,
        progression_breaks=[],
        conclusion_commits_forward=commits_forward,
        conclusion_audit=audit,
        paragraphs_to_delete=paragraphs_to_delete,
        paragraphs_to_merge=[],
        target_paragraph_count=target_paragraph_count,
        evidence_diversity_score=diversity,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM = """You are a senior admissions reader and hiring lead reviewing an \
application document at the WHOLE-DOCUMENT level. The per-paragraph rhetoric \
analyzer already scored each paragraph in isolation. Your job is the layer \
above: does the document tell a single, sharpening story, or is it a \
collection of individually-OK paragraphs that repeat themselves, drift, and \
end with a generic sign-off?

For each paragraph in the document (skip headers and short fragments), \
produce a `ParagraphRole` entry:

- `paragraph_index`: 0-indexed position in the document.
- `paragraph_anchor`: first ~100 characters copied VERBATIM from the document.
- `role`: one of:
    * `hook` — the opening that earns the reader's attention.
    * `motivation` — why the candidate cares about this field/role.
    * `evidence` — a concrete past experience tied to a requirement.
    * `fit` — match between candidate and role/team.
    * `commitment` — forward-looking promise of contribution.
    * `closing` — sign-off paragraph.
    * `redundant` — adds nothing the rest of the document doesn't already \
say. This label is the most important one to apply correctly: a paragraph \
that just restates the candidate's interest or repeats "I would contribute \
positively" is `redundant`, even if the sentences are individually \
well-formed.
- `anchor_examples`: named projects, internships, employers, courses, or \
specific skills that the paragraph uses as PRIMARY evidence. Drawn verbatim \
from the paragraph. Empty when the paragraph has no concrete evidence.
- `adds_new_information`: TRUE only if the paragraph contributes something \
not already covered by earlier paragraphs. Default to FALSE for "I believe \
my background...", "I see this opportunity as a chance...", boilerplate \
motivation paragraphs that don't cite a specific experience, and any \
paragraph that just rephrases an earlier point.

Then produce document-level findings:

- `repeated_anchors`: any anchor (project / internship / employer name) used \
as the FOCUS of two or more paragraphs. Each entry is \
(anchor_name, [paragraph_indices_where_it's_focus]). The classic failure \
mode this catches: a candidate uses the same Unity project as the focus of \
the hook AND the projects paragraph AND the motivation paragraph — by the \
third mention it stops landing. Flag every case.
- `progression_breaks`: pairs (i, i+1, reason) where the transition from \
paragraph i to i+1 is weak or the ordering is illogical (e.g. fit before any \
evidence; two evidence paragraphs with no thematic link). One sentence per \
reason. Return empty list when the flow is fine.
- `conclusion_commits_forward`: TRUE only if the closing paragraph names the \
target organisation/role AND specifies a concrete contribution / fit tied \
back to a hook anchor. "Thank you for considering my application — I would \
be happy for the opportunity to continue developing myself" is a FALSE: it \
names no specific contribution and reads as boilerplate.
- `conclusion_audit`: 1-2 sentences naming what the closing is missing or \
(if it works) what makes it work.
- `paragraphs_to_delete`: paragraph indices to cut entirely. Strong \
candidates: `role='redundant'`, `adds_new_information=false`, generic \
"motivation" paragraphs that cite no specific experience, generic "fit" \
paragraphs that say only "I believe my background... would allow me to \
contribute positively". Be willing to recommend cuts — most application \
documents are 20-40% too long.
- `paragraphs_to_merge`: pairs (src_idx, dst_idx) where src should be folded \
into dst because they cover the same ground.
- `target_paragraph_count`: the recommended paragraph count after applying \
deletes and merges. A tight SOP / cover letter is 4-6 paragraphs.
- `evidence_diversity_score`: 0 = the same anchor is leaned on across the \
document; 1 = each paragraph that needs an anchor draws on a different one.
- `summary`: ONE sentence diagnosing narrative health. Different angle from \
the per-paragraph rhetoric summary — focus on redundancy, progression, and \
closing.

Hard rules:
- DO NOT critique grammar, sentence flow, or word choice — other analyzers \
handle that.
- DO NOT invent anchors not present in the document.
- DO NOT mark a paragraph as `redundant` if it cites a unique experience or \
makes a unique argument, even if it's poorly framed — that's a per-paragraph \
substance issue, not a redundancy issue.
- BE WILLING to recommend deletions. The pipeline can produce delete \
proposals; if you don't flag them, the synthesizer won't cut anything.
"""

_MAX_DOC_CHARS = 6000
_MAX_OPP_CHARS = 2000


# ---------------------------------------------------------------------------
# Node — receives context_pack via Send
# ---------------------------------------------------------------------------

def analyze_narrative(context_pack: dict) -> dict:
    updates = {"step_history": ["analyze_narrative"]}
    doc_type = context_pack.get("doc_type", "UNKNOWN")

    # Document-level narrative analysis is meaningful only for argument-shaped
    # docs. CVs are bullet-driven; redundancy is handled at the bullet level
    # by analyze_content_gaps.
    if doc_type not in ("SOP", "COVER_LETTER"):
        return {**updates, "analysis_results": {"narrative": {}}}

    doc_sections = context_pack.get("doc_sections") or {}
    opportunity_context = context_pack.get("opportunity_context") or {}

    llm = get_llm()
    if llm is None:
        result = _heuristic(doc_sections, opportunity_context)
        return {**updates, "analysis_results": {"narrative": result.model_dump()}}

    pairs = _split_paragraphs(doc_sections)
    if not pairs:
        result = _heuristic(doc_sections, opportunity_context)
        return {**updates, "analysis_results": {"narrative": result.model_dump()}}

    para_block: List[str] = []
    for i, (section, para) in enumerate(pairs):
        para_block.append(f"[Paragraph {i} | Section: {section}]\n{para}")
    paragraphs_text = "\n\n---\n\n".join(para_block)[:_MAX_DOC_CHARS]

    opp_text = (
        str(opportunity_context)[:_MAX_OPP_CHARS]
        if opportunity_context
        else "(no opportunity context — assess generic-ness only; do not fabricate company specifics)"
    )

    user_focus = format_user_focus(context_pack.get("parsed_instructions"))
    profile_brief = format_profile_brief(context_pack.get("profile_snapshot"))

    body = (
        f"Document type: {doc_type}\n\n"
        f"Target opportunity:\n{opp_text}\n\n"
        f"Document paragraphs (each delimited by `---`, indices match `paragraph_index`):\n"
        f"{paragraphs_text}"
    )
    if profile_brief:
        body += f"\n\n{profile_brief}"
    if user_focus:
        body += f"\n\n{user_focus}"

    structured = llm.with_structured_output(NarrativeAnalysis)
    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=body),
    ]

    try:
        result: NarrativeAnalysis = structured.invoke(msgs)
        return {**updates, "analysis_results": {"narrative": result.model_dump()}}
    except Exception as e:
        result = _heuristic(doc_sections, opportunity_context)
        out = result.model_dump()
        out["summary"] = (out.get("summary") or "") + f" [LLM failed, used heuristic: {e}]"
        return {**updates, "analysis_results": {"narrative": out}}
