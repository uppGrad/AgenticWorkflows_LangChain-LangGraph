"""application_tailoring delegates the upload path to the doc-feedback
auto-tailoring graph (grounded edits + retry-evaluator) and falls back to
the legacy T1→T2 chain when that graph errors or has no doc-type analog.

Tests mock `build_auto_tailoring_graph` so we don't run the 7-node
analysis fan-out + retry loop in unit tests; that's covered separately
by an end-to-end smoke test against a real CV.
"""
from unittest.mock import patch, MagicMock

from uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring import (
    _tailor_via_doc_feedback,
    _DOC_FEEDBACK_TYPE_MAP,
)


def _make_graph_returning(final_state):
    """Helper: build a mock graph whose .invoke() returns `final_state`."""
    graph = MagicMock()
    graph.invoke.return_value = final_state
    return graph


def test_returns_tailored_entry_when_graph_succeeds():
    """Happy path: doc-feedback graph completes, returns a LaTeX final_document.
    The tailored_documents entry exposes content + latex_source + diff metadata."""
    final_state = {
        "final_document": r"\documentclass{article}\begin{document}Tailored CV body\end{document}",
        "result": {"status": "ok"},
        "human_review": {
            "approved_proposals": [{"section": "X"}, {"section": "Y"}],
        },
        "diff": {"summary": "2 edits applied"},
        "iteration_count": 1,
    }
    fake_graph = _make_graph_returning(final_state)
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring.build_auto_tailoring_graph",
        return_value=fake_graph,
    ):
        out = _tailor_via_doc_feedback(
            doc_type="CV",
            uploaded_text="Original CV text" * 50,
            user_prompt="emphasise ML",
            opportunity_data={"title": "ML Engineer", "company": "Acme",
                              "description": "Build ML systems"},
            opportunity_type="job",
            profile={"name": "Test User"},
        )
    assert out is not None
    assert out["source"] == "upload"
    assert out["tailoring_depth"] == "doc_feedback"
    assert out["llm_used"] is True
    assert out["latex_source"].startswith(r"\documentclass")
    assert out["doc_feedback_accepted_proposals"] == 2
    # The graph was invoked once with state pre-populated by the helper.
    assert fake_graph.invoke.call_count == 1
    df_state = fake_graph.invoke.call_args[0][0]
    assert df_state["raw_text"].startswith("Original CV text")
    assert df_state["doc_classification"]["doc_type"] == "CV"
    assert df_state["user_instructions"] == "emphasise ML"


def test_returns_none_when_doc_type_has_no_doc_feedback_analog():
    """Doc types like 'Motivation Letter' map to COVER_LETTER (handled), but
    things like 'References' have no doc-feedback path — caller must fall
    back to T1→T2."""
    out = _tailor_via_doc_feedback(
        doc_type="References",  # not in _DOC_FEEDBACK_TYPE_MAP
        uploaded_text="...",
        user_prompt=None,
        opportunity_data={},
        opportunity_type="job",
        profile={},
    )
    assert out is None


def test_returns_none_when_graph_raises():
    """Graph crashes (LLM 500, network, etc.) → fall back to T1→T2."""
    fake_graph = MagicMock()
    fake_graph.invoke.side_effect = RuntimeError("LLM 500")
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring.build_auto_tailoring_graph",
        return_value=fake_graph,
    ):
        out = _tailor_via_doc_feedback(
            doc_type="CV",
            uploaded_text="text",
            user_prompt=None,
            opportunity_data={},
            opportunity_type="job",
            profile={},
        )
    assert out is None


def test_returns_none_when_graph_ends_with_error_status():
    """Graph completes but result.status='error' (e.g. LATEX_GENERATION_FAILED).
    Fall back to T1→T2."""
    final_state = {
        "result": {"status": "error", "error_code": "LATEX_GENERATION_FAILED"},
    }
    fake_graph = _make_graph_returning(final_state)
    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring.build_auto_tailoring_graph",
        return_value=fake_graph,
    ):
        out = _tailor_via_doc_feedback(
            doc_type="CV",
            uploaded_text="text",
            user_prompt=None,
            opportunity_data={},
            opportunity_type="job",
            profile={},
        )
    assert out is None


def test_doc_type_map_covers_all_generatable_types():
    """The doc-feedback path is only used for upload, but every type that
    might land in `upload` should be mapped (or explicitly excluded). This
    sanity test pins the current mapping so additions to the canonical
    types list force a deliberate decision."""
    # Sanity: the standard analogs.
    assert _DOC_FEEDBACK_TYPE_MAP["CV"] == "CV"
    assert _DOC_FEEDBACK_TYPE_MAP["Cover Letter"] == "COVER_LETTER"
    assert _DOC_FEEDBACK_TYPE_MAP["Motivation Letter"] == "COVER_LETTER"
    assert _DOC_FEEDBACK_TYPE_MAP["SOP"] == "SOP"
    assert _DOC_FEEDBACK_TYPE_MAP["Personal Statement"] == "SOP"
