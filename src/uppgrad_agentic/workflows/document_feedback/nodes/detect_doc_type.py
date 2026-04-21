from __future__ import annotations

from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.workflows.document_feedback.schemas import DocTypeClassification
from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


MAX_CHARS_FOR_CLASSIFY = 12_000


SYSTEM = """You classify uploaded documents for a career application assistant.

Return:
- doc_type: CV | SOP | COVER_LETTER | UNKNOWN
- relevant: whether the document appears to be an application-related document (CV/SOP/cover letter or close variants)
- confidence: 0-1
- reasons: short signals you observed (e.g., 'Education section', 'Work Experience', 'Dear Hiring Manager', 'Statement of Purpose')
- language: optional

Be conservative: mark relevant=false only if it clearly looks unrelated (e.g., a book summary, random article, story, lecture notes).
"""


def _heuristic_classify(text: str, user_instructions: str) -> DocTypeClassification:
    t = text.lower()
    u = (user_instructions or "").lower()

    reasons: List[str] = []
    cv_signals = ["experience", "education", "skills", "projects", "languages", "certifications"]
    cover_signals = ["dear", "hiring manager", "sincerely", "i am writing", "position", "apply"]
    sop_signals = ["statement of purpose", "research interests", "motivation", "graduate", "msc", "phd", "my goal"]

    cv_hits = sum(1 for s in cv_signals if s in t)
    cover_hits = sum(1 for s in cover_signals if s in t)
    sop_hits = sum(1 for s in sop_signals if s in t)

    # instruction hints
    if "cv" in u or "resume" in u:
        cv_hits += 2
        reasons.append("User instructions mention CV/resume")
    if "cover" in u or "hiring manager" in u:
        cover_hits += 2
        reasons.append("User instructions mention cover letter")
    if "sop" in u or "statement of purpose" in u:
        sop_hits += 2
        reasons.append("User instructions mention SOP")

    # decide
    best = max(cv_hits, cover_hits, sop_hits)
    if best == 0:
        # conservative: UNKNOWN but still possibly relevant
        return DocTypeClassification(
            doc_type="UNKNOWN",
            relevant=False if len(t) > 2000 and ("chapter" in t or "book" in t or "summary" in t) else True,
            confidence=0.55 if len(t) > 0 else 0.2,
            reasons=reasons or ["No strong CV/SOP/cover signals detected"],
            language=None,
        )

    if best == cv_hits:
        return DocTypeClassification(
            doc_type="CV",
            relevant=True,
            confidence=min(0.65 + 0.05 * cv_hits, 0.9),
            reasons=reasons + ["Detected CV-like section keywords"],
            language=None,
        )
    if best == cover_hits:
        return DocTypeClassification(
            doc_type="COVER_LETTER",
            relevant=True,
            confidence=min(0.65 + 0.05 * cover_hits, 0.9),
            reasons=reasons + ["Detected cover-letter style phrases"],
            language=None,
        )
    return DocTypeClassification(
        doc_type="SOP",
        relevant=True,
        confidence=min(0.65 + 0.05 * sop_hits, 0.9),
        reasons=reasons + ["Detected SOP/graduate motivation signals"],
        language=None,
    )


def detect_doc_type_and_relevance(state: DocFeedbackState) -> dict:
    if state.get("result", {}).get("status") == "error":
        return {}  # already failed upstream

    raw = state.get("raw_text", "")
    user_instructions = state.get("user_instructions", "") or ""
    snippet = raw[:MAX_CHARS_FOR_CLASSIFY]

    llm = get_llm()
    if llm is None:
        cls = _heuristic_classify(snippet, user_instructions)
        return {"doc_classification": cls.model_dump()}

    # Structured output with Pydantic schema
    structured = llm.with_structured_output(DocTypeClassification)

    msg = [
        SystemMessage(content=SYSTEM),
        HumanMessage(
            content=(
                f"User instructions (may be empty):\n{user_instructions}\n\n"
                f"Document text (truncated):\n{snippet}"
            )
        ),
    ]

    try:
        cls: DocTypeClassification = structured.invoke(msg)
        return {"doc_classification": cls.model_dump()}
    except Exception as e:
        # fallback if model/structured parsing fails
        cls = _heuristic_classify(snippet, user_instructions)
        out = cls.model_dump()
        out["reasons"] = (out.get("reasons") or []) + [f"LLM classify failed; used heuristic fallback: {e}"]
        return {"doc_classification": out}
