"""Sub-PR A — `application_tailoring` now emits both `content` (plain text)
and `latex_source` per tailored document.

Locks the contract that:
  * `_extract_latex_source` recovers a valid `\\documentclass...\\end{document}`
    span from raw LLM output (fenced or unfenced) and returns "" when nothing
    valid is present.
  * `_latex_to_plain` strips LaTeX scaffolding to a sensible plain-text
    approximation suitable for `application_evaluation` length / placeholder
    checks.
  * `_process_document` populates BOTH fields on every tailored entry —
    auto-generate, upload, no-LLM fallback, LLM-failure fallback.
  * Per-doc-type templates are routed correctly.

The matching backend renderer + frontend download buttons ship in Sub-PR B
and Sub-PR C.
"""
from unittest.mock import patch

from uppgrad_agentic.tools.latex_templates import (
    CV_TEMPLATE,
    COVER_LETTER_TEMPLATE,
    GENERIC_TEMPLATE,
    template_for,
)
from uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring import (
    _extract_latex_source,
    _latex_to_plain,
    _process_document,
    _split_latex_and_plain,
)


# ─── template_for ────────────────────────────────────────────────────────────

def test_template_for_cv_returns_resume_layout():
    """CV uses the sb2nov resume template (sourced from doc-feedback's
    finalize node). Spot-check the custom commands the LLM is told to
    use are defined in the preamble."""
    t = template_for("CV")
    assert t is CV_TEMPLATE
    assert r"\resumeItem" in t
    assert r"\resumeSubheading" in t
    assert r"\resumeItemListStart" in t


def test_template_for_cover_letter_returns_prose_layout():
    """Cover letters use the doc-feedback prose template — parskip-based
    article, no resume-helper definitions (which would fail to compile
    in this preamble) and no list environments. The placeholder comment
    inside the body markers DOES reference the resume commands as a
    "do NOT use these" warning to the LLM, which is fine; we just
    check that the resume helpers are not DEFINED here."""
    t = template_for("Cover Letter")
    assert t is COVER_LETTER_TEMPLATE
    # No `\newcommand` defining the resume helpers — the prose preamble
    # would reject the `\resumeItem*` calls at compile time, which is
    # the whole point of the doc-type split.
    assert r"\newcommand{\resumeItem" not in t
    assert r"\newcommand{\resumeSubheading" not in t
    assert r"parskip" in t


def test_template_for_motivation_letter_routes_to_cover_letter():
    """Motivation Letter is conceptually a cover letter; share the template."""
    assert template_for("Motivation Letter") is COVER_LETTER_TEMPLATE


def test_template_for_unknown_falls_back_to_generic():
    assert template_for("Some Random Doc") is GENERIC_TEMPLATE
    assert template_for(None) is GENERIC_TEMPLATE
    assert template_for("") is GENERIC_TEMPLATE


def test_cv_and_prose_templates_share_doc_feedback_lineage():
    """Sanity check that the templates we copy from doc-feedback's
    finalize node still match the contract (preamble fixed, body
    placeholder marked). If finalize.py drifts, this test will need
    re-syncing — leave a clear path for the maintainer."""
    cv = template_for("CV")
    prose = template_for("SOP")
    for tpl in (cv, prose):
        assert r"\documentclass" in tpl
        assert r"\begin{document}" in tpl
        assert r"\end{document}" in tpl
        assert "% --- BEGIN BODY ---" in tpl
        assert "% --- END BODY ---" in tpl


# ─── _extract_latex_source ───────────────────────────────────────────────────

def test_extract_latex_from_unfenced_response():
    raw = (
        r"\documentclass[11pt]{article}" "\n"
        r"\begin{document}" "\n"
        "Hello.\n"
        r"\end{document}"
    )
    assert _extract_latex_source(raw).startswith(r"\documentclass")
    assert _extract_latex_source(raw).endswith(r"\end{document}")


def test_extract_latex_from_fenced_response():
    raw = (
        "```latex\n"
        r"\documentclass[11pt]{article}" "\n"
        r"\begin{document}" "\n"
        "Hello.\n"
        r"\end{document}" "\n"
        "```"
    )
    out = _extract_latex_source(raw)
    assert out.startswith(r"\documentclass")
    assert out.endswith(r"\end{document}")
    assert "```" not in out


def test_extract_latex_trims_prose_around_document():
    """Sometimes the LLM leaks 'Here is your document:' before, or
    'Hope this helps!' after. Strip both."""
    raw = (
        "Sure — here's your document:\n\n"
        r"\documentclass{article}" "\n"
        r"\begin{document}" "\n"
        r"Real content." "\n"
        r"\end{document}" "\n"
        "Let me know if you need changes."
    )
    out = _extract_latex_source(raw)
    assert out.startswith(r"\documentclass")
    assert out.endswith(r"\end{document}")
    assert "Sure" not in out
    assert "changes." not in out


def test_extract_latex_returns_empty_when_no_documentclass():
    """Fall-through: if the LLM emitted only plain text, signal failure
    with empty string so caller can store the plain text in `content`."""
    assert _extract_latex_source("Just plain text, no document.") == ""
    assert _extract_latex_source("") == ""


def test_extract_latex_returns_empty_when_no_end_document():
    raw = r"\documentclass{article}" "\n" r"\begin{document}" "\nUnfinished"
    assert _extract_latex_source(raw) == ""


# ─── _latex_to_plain ─────────────────────────────────────────────────────────

def test_latex_to_plain_strips_preamble_and_postamble():
    src = (
        r"\documentclass{article}" "\n"
        r"\usepackage{geometry}" "\n"
        r"\begin{document}" "\n"
        "Body content here.\n"
        r"\end{document}"
    )
    plain = _latex_to_plain(src)
    assert "Body content here." in plain
    assert r"\documentclass" not in plain
    assert r"\usepackage" not in plain
    assert r"\end{document}" not in plain


def test_latex_to_plain_keeps_section_text_drops_command():
    src = (
        r"\begin{document}" "\n"
        r"\section{Experience}" "\n"
        r"\textbf{Senior Engineer}, Acme Corp, 2024--present" "\n"
        r"\end{document}"
    )
    plain = _latex_to_plain(src)
    assert "Experience" in plain
    assert "Senior Engineer" in plain
    assert r"\section" not in plain
    assert r"\textbf" not in plain


def test_latex_to_plain_resolves_href_to_label():
    src = (
        r"\begin{document}" "\n"
        r"Visit \href{https://uppgrad.com}{our site} for more." "\n"
        r"\end{document}"
    )
    plain = _latex_to_plain(src)
    assert "our site" in plain
    assert "https://uppgrad.com" not in plain


def test_latex_to_plain_keeps_itemize_bullets():
    src = (
        r"\begin{document}" "\n"
        r"\begin{itemize}" "\n"
        r"  \item Built distributed systems" "\n"
        r"  \item Led team of 5" "\n"
        r"\end{itemize}" "\n"
        r"\end{document}"
    )
    plain = _latex_to_plain(src)
    assert "- Built distributed systems" in plain
    assert "- Led team of 5" in plain
    assert "itemize" not in plain


def test_latex_to_plain_drops_body_marker_comments():
    src = (
        r"\begin{document}" "\n"
        "% --- BEGIN BODY ---\n"
        "Real content.\n"
        "% --- END BODY ---\n"
        r"\end{document}"
    )
    plain = _latex_to_plain(src)
    assert "Real content." in plain
    assert "BEGIN BODY" not in plain
    assert "END BODY" not in plain


def test_latex_to_plain_handles_empty_input():
    assert _latex_to_plain("") == ""
    assert _latex_to_plain(None) == ""


# ─── _split_latex_and_plain ──────────────────────────────────────────────────

def test_split_returns_both_when_latex_present():
    raw = (
        r"\documentclass{article}" "\n"
        r"\begin{document}" "\n"
        "Cover letter body content.\n"
        r"\end{document}"
    )
    content, latex = _split_latex_and_plain(raw, "Cover Letter")
    assert "Cover letter body content." in content
    assert latex.startswith(r"\documentclass")
    assert latex.endswith(r"\end{document}")


def test_split_falls_back_to_plain_when_no_latex():
    """If the LLM ignores the format directive and returns plain prose,
    we keep the prose in `content` and flag the missing latex with ''."""
    raw = "This LLM forgot LaTeX and just wrote prose. " * 3
    content, latex = _split_latex_and_plain(raw, "Cover Letter")
    assert latex == ""
    assert "prose" in content


def test_split_truncates_to_per_doc_cap():
    """Per-doc-type caps still apply to the plain-text content."""
    long_body = "x" * 20_000
    raw = (
        r"\documentclass{article}" "\n"
        r"\begin{document}" "\n"
        + long_body + "\n"
        r"\end{document}"
    )
    content, _ = _split_latex_and_plain(raw, "Cover Letter")
    assert len(content) <= 3000  # _DOC_TYPE_CAPS["Cover Letter"]


# ─── _process_document end-to-end ────────────────────────────────────────────

def _fake_llm_returning(latex_or_text):
    """Mimic `_llm_call` by patching it directly."""
    return latex_or_text


def test_process_document_auto_generate_populates_latex_source():
    item = {"id": 0, "category": "document", "label": "Cover Letter",
            "document_type": "Cover Letter"}
    fake = (
        r"\documentclass{article}" "\n"
        r"\begin{document}" "\n"
        "Dear Hiring Team, I'm thrilled to apply.\n"
        r"\end{document}"
    )
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring._llm_call",
        return_value=fake,
    ):
        out = _process_document(
            item=item, choice="auto_generate", uploaded_text=None,
            user_prompt=None, opportunity_data={"title": "SWE"},
            opportunity_type="job",
            profile={"name": "Stu", "uploaded_documents": {}},
            llm=object(),  # truthy non-None
            user_instructions="",
        )
    assert out is not None
    assert out["content"]  # plain text populated
    assert out["latex_source"].startswith(r"\documentclass")
    assert out["latex_source"].endswith(r"\end{document}")
    assert out["source"] == "auto_generate"


def test_process_document_no_llm_returns_empty_latex_source():
    item = {"id": 0, "category": "document", "label": "Cover Letter",
            "document_type": "Cover Letter"}
    out = _process_document(
        item=item, choice="auto_generate", uploaded_text=None,
        user_prompt=None, opportunity_data={"title": "SWE"},
        opportunity_type="job",
        profile={"name": "Stu", "uploaded_documents": {}},
        llm=None,  # no LLM configured
        user_instructions="",
    )
    assert out["content"] == ""
    assert out["latex_source"] == ""
    assert out["llm_used"] is False


def test_process_document_upload_no_llm_passthrough_has_empty_latex():
    item = {"id": 0, "category": "document", "label": "CV",
            "document_type": "CV"}
    out = _process_document(
        item=item, choice="upload",
        uploaded_text="Existing CV text body.",
        user_prompt=None, opportunity_data={"title": "SWE"},
        opportunity_type="job",
        profile={"name": "Stu", "uploaded_documents": {}},
        llm=None, user_instructions="",
    )
    assert "Existing CV text body." in out["content"]
    assert out["latex_source"] == ""  # passthrough — no LaTeX generated


def test_process_document_upload_path_extracts_latex_from_t2_output():
    """Upload path tries doc-feedback first; on its failure, falls back to
    T1 → LA → T2. This test pins the legacy fallback by forcing
    `_tailor_via_doc_feedback` to return None."""
    item = {"id": 0, "category": "document", "label": "CV",
            "document_type": "CV"}
    t1_latex = (
        r"\documentclass{article}" "\n" r"\begin{document}" "\n"
        "T1 draft of the CV.\n" r"\end{document}"
    )
    t2_latex = (
        r"\documentclass{article}" "\n" r"\begin{document}" "\n"
        "T2 polished CV.\n" r"\end{document}"
    )
    seq = [t1_latex, t2_latex]
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring._tailor_via_doc_feedback",
        return_value=None,  # force legacy T1→T2 path
    ), patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring._llm_call",
        side_effect=lambda *a, **k: seq.pop(0),
    ), patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring.analyze_upload_pre",
        return_value=type("PreA", (), {
            "completeness": "x", "relevance": "x", "correctness": "x",
            "overall_quality": "needs_revision", "top_priorities": [],
            "model_dump": lambda self: {},
        })(),
    ), patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring.analyze_upload_light_post",
        return_value=type("PostA", (), {
            "structure_issues": [], "content_gap_vs_opportunity": [],
            "content_gap_vs_profile": [],
            "model_dump": lambda self: {},
        })(),
    ):
        out = _process_document(
            item=item, choice="upload", uploaded_text="Original CV text",
            user_prompt=None, opportunity_data={"title": "SWE"},
            opportunity_type="job",
            profile={"name": "Stu", "uploaded_documents": {}},
            llm=object(), user_instructions="",
        )
    assert "T2 polished CV." in out["content"]
    assert "T2 polished CV." in out["latex_source"]
    assert out["passes"] == 2
    assert out["tailoring_depth"] == "deep"
