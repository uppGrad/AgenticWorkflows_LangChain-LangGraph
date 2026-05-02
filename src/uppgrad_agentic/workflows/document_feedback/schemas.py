# src/uppgrad_agentic/workflows/document_feedback/schemas.py
from __future__ import annotations

from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


DocType = Literal["CV", "SOP", "COVER_LETTER", "UNKNOWN"]
ProposalAction = Literal["rewrite", "delete", "merge"]
ParagraphRoleLabel = Literal[
    "hook",
    "motivation",
    "evidence",
    "fit",
    "commitment",
    "closing",
    "redundant",
]


class DocTypeClassification(BaseModel):
    doc_type: DocType = Field(..., description="Document type classification")
    relevant: bool = Field(..., description="Whether this looks like an application-related document")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list, description="Short reasons/signals found")
    language: Optional[str] = Field(default=None, description="Detected language, if confident")


class ChangeProposal(BaseModel):
    section: str = Field(..., description="Document section the change applies to (e.g. 'Experience', 'Introduction')")
    rationale: str = Field(..., description="Why this change is recommended")
    before_text: str = Field(..., description="Original text to be replaced")
    after_text: str = Field(..., description="Proposed replacement text. Empty when action='delete'.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in this proposal")
    requires_confirmation: bool = Field(..., description="Whether user must explicitly approve before applying")
    action: ProposalAction = Field(
        default="rewrite",
        description=(
            "What to do with before_text. 'rewrite' replaces it with after_text "
            "(default). 'delete' removes the paragraph entirely (after_text "
            "should be empty). 'merge' collapses two adjacent paragraphs — "
            "before_text holds both source paragraphs concatenated, after_text "
            "holds the merged result."
        ),
    )


class EvaluationResult(BaseModel):
    passed: bool = Field(..., description="Whether the proposals passed quality checks")
    issues: List[str] = Field(default_factory=list, description="Descriptions of any groundedness or format problems found")
    iteration: int = Field(..., description="Which evaluation iteration this result belongs to (0-indexed)")


# ---------------------------------------------------------------------------
# Narrative analysis (SOP / COVER_LETTER) — whole-document scope
# ---------------------------------------------------------------------------
#
# `analyze_rhetoric` operates paragraph-by-paragraph, so it cannot see when the
# same project/internship is repeated as the focus of three paragraphs, or when
# a paragraph adds nothing the rest of the document doesn't already say. This
# schema captures the document-level concerns that drive sharper storytelling:
# anchor diversity, paragraph progression, and a conclusion that commits
# forward instead of restating.


class ParagraphRole(BaseModel):
    paragraph_index: int = Field(
        ...,
        description="0-indexed position of the paragraph in the document.",
    )
    paragraph_anchor: str = Field(
        ...,
        description=(
            "First ~100 characters of the paragraph, copied verbatim. Used by "
            "synthesis to locate the paragraph; mirrors analyze_rhetoric's "
            "anchoring contract."
        ),
    )
    role: ParagraphRoleLabel = Field(
        ...,
        description=(
            "Function of this paragraph in the document's argument. "
            "'hook' = opening that earns attention; "
            "'motivation' = why the candidate cares about this field/role; "
            "'evidence' = a concrete past experience tied to a requirement; "
            "'fit' = match between candidate and role/team; "
            "'commitment' = forward-looking promise of contribution; "
            "'closing' = sign-off paragraph; "
            "'redundant' = adds nothing new — candidate for delete or merge."
        ),
    )
    anchor_examples: List[str] = Field(
        default_factory=list,
        description=(
            "Named projects, internships, employers, or skills the paragraph "
            "uses as primary evidence. Drawn verbatim from the paragraph. "
            "Empty list = paragraph has no concrete evidence."
        ),
    )
    adds_new_information: bool = Field(
        ...,
        description=(
            "True if the paragraph contributes something not already covered "
            "by earlier paragraphs. False = candidate for deletion or merging."
        ),
    )


class NarrativeAnalysis(BaseModel):
    paragraph_roles: List[ParagraphRole] = Field(default_factory=list)

    repeated_anchors: List[Tuple[str, List[int]]] = Field(
        default_factory=list,
        description=(
            "Anchors (project/internship names) used as paragraph focus in 2+ "
            "paragraphs. Each entry is (anchor_name, [paragraph_indices]). "
            "These are exactly the redundancies the synthesizer must collapse."
        ),
    )
    progression_breaks: List[Tuple[int, int, str]] = Field(
        default_factory=list,
        description=(
            "Pairs (i, i+1, reason) flagging weak transitions or paragraph "
            "ordering issues — e.g. a fit-paragraph appearing before any "
            "evidence-paragraph, or two evidence-paragraphs with no thematic "
            "link. Reason is one sentence."
        ),
    )
    conclusion_commits_forward: bool = Field(
        ...,
        description=(
            "True if the closing paragraph names the target organisation AND "
            "specifies a concrete contribution / fit tied to a hook anchor. "
            "False if it falls back into generic thank-you / 'continue "
            "developing myself' language."
        ),
    )
    conclusion_audit: str = Field(
        ...,
        description=(
            "1-2 sentences naming what the closing paragraph is missing. "
            "If conclusion_commits_forward is true, this can describe what "
            "makes it work."
        ),
    )
    paragraphs_to_delete: List[int] = Field(
        default_factory=list,
        description=(
            "Paragraph indices that should be deleted entirely — typically "
            "those with role='redundant' or adds_new_information=False. "
            "Synthesis should emit a delete proposal for each."
        ),
    )
    paragraphs_to_merge: List[Tuple[int, int]] = Field(
        default_factory=list,
        description=(
            "Pairs (src_idx, dst_idx) where src should be folded into dst. "
            "Synthesis should emit a merge proposal collapsing the two."
        ),
    )
    target_paragraph_count: int = Field(
        ...,
        description=(
            "Recommended paragraph count after deletes + merges. Drives the "
            "shape the synthesizer should aim for."
        ),
    )
    evidence_diversity_score: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "0 = the same anchor is leaned on across the document; 1 = each "
            "paragraph that needs an anchor draws on a different one."
        ),
    )
    summary: str = Field(
        ...,
        description=(
            "One-sentence diagnosis of narrative health (separate from "
            "rhetoric.summary which covers per-paragraph substance)."
        ),
    )
    candidate_voice_signals: List[str] = Field(
        default_factory=list,
        description=(
            "≤5 short phrases (≤80 chars each), drawn verbatim or near-"
            "verbatim from the document, capturing what makes THIS "
            "candidate's positioning distinctive vs. a generic applicant — "
            "role-specific motivation, ownership mindset, product framing. "
            "Synth must keep ≥60% of these (substring match) in the post-"
            "application document; the evaluator blocks otherwise. Generic "
            "claims ('teamwork', 'problem solving') do NOT belong here. "
            "Empty list = no distinctive signals detected."
        ),
    )
