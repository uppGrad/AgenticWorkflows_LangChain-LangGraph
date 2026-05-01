"""User-level custom instructions threading (P7).

Locks the contract that lets the user's free-form session-wide guidance
(`state['user_instructions']`) reach the LLM during tailoring. The
backend captures this from the start-session form and injects it into
the initial graph state; this test asserts it survives through to the
prompts the four LLM call sites build.
"""
from unittest.mock import patch

from uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring import (
    _generate_doc_prompt,
    _generate_text_prompt,
    _session_instructions_block,
    _t1_prompt,
    _t2_prompt,
    application_tailoring,
)


def test_session_instructions_block_returns_empty_for_blank():
    assert _session_instructions_block("") == ""
    assert _session_instructions_block(None) == ""
    assert _session_instructions_block("   ") == ""


def test_session_instructions_block_emits_separator_and_text():
    out = _session_instructions_block("Keep cover letter under 300 words.")
    assert "SESSION-WIDE CUSTOM INSTRUCTIONS" in out
    assert "Keep cover letter under 300 words." in out


def test_generate_doc_prompt_includes_instructions():
    prompt = _generate_doc_prompt(
        doc_type="Cover Letter",
        canonical_type="Cover Letter",
        opportunity_data={"title": "SWE", "company": "Acme"},
        opportunity_type="job",
        profile={"name": "Stu", "uploaded_documents": {}},
        user_prompt=None,
        user_instructions="Emphasise distributed-systems work.",
    )
    assert "SESSION-WIDE CUSTOM INSTRUCTIONS" in prompt
    assert "Emphasise distributed-systems work." in prompt


def test_generate_text_prompt_includes_instructions():
    prompt = _generate_text_prompt(
        question="Why are you a fit?",
        opportunity_data={"title": "SWE", "company": "Acme"},
        opportunity_type="job",
        profile={"name": "Stu", "uploaded_documents": {}},
        user_instructions="Mention Python and Go.",
    )
    assert "SESSION-WIDE CUSTOM INSTRUCTIONS" in prompt
    assert "Mention Python and Go." in prompt


def test_t1_t2_prompts_include_instructions():
    class _PreA:
        completeness = "x"
        relevance = "x"
        correctness = "x"
        overall_quality = "needs_revision"
        top_priorities = []

    class _PostA:
        structure_issues = []
        content_gap_vs_opportunity = []
        content_gap_vs_profile = []

    t1 = _t1_prompt(
        "Cover Letter", {"title": "SWE"}, "job",
        {"name": "Stu", "uploaded_documents": {}}, "uploaded text",
        _PreA(), user_prompt=None, user_instructions="Be concise.",
    )
    t2 = _t2_prompt(
        "Cover Letter", {"title": "SWE"}, "job",
        {"name": "Stu", "uploaded_documents": {}}, "t1 output",
        _PostA(), user_prompt=None, user_instructions="Be concise.",
    )
    assert "Be concise." in t1
    assert "Be concise." in t2


def test_no_session_instructions_omits_block():
    """Empty user_instructions must not leave behind a stray header."""
    prompt = _generate_doc_prompt(
        doc_type="CV",
        canonical_type="CV",
        opportunity_data={"title": "SWE"},
        opportunity_type="job",
        profile={"name": "Stu", "uploaded_documents": {}},
        user_prompt=None,
        user_instructions="",
    )
    assert "SESSION-WIDE CUSTOM INSTRUCTIONS" not in prompt


def test_application_tailoring_node_threads_instructions_to_llm_call():
    """End-to-end: state.user_instructions reaches the prompt that
    `_llm_call` ultimately sees. Captures the user-prompt arg of the LLM
    call and asserts the instructions text is embedded."""
    state = {
        "opportunity_type": "job",
        "opportunity_data": {"title": "SWE", "company": "Acme"},
        "user_instructions": "Highlight my Python ML coursework.",
        "profile_snapshot": {"name": "Stu", "email": "s@x.com",
                              "uploaded_documents": {}},
        "requirement_items": [
            {"id": 0, "category": "document", "label": "Cover Letter",
             "document_type": "Cover Letter", "required": False},
        ],
        "human_review_1": {
            "requirements": {"0": {"choice": "auto_generate"}},
            "misc_strategy": "ignore",
        },
    }

    captured = {}

    class _FakeLLM:
        def invoke(self, prompt):
            # pydantic-LangChain LLMs return a Message-like object with .content
            class _Msg:
                content = "Generated cover letter text."
            captured["prompt"] = prompt
            return _Msg()

    with patch(
        "uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring.get_llm",
        return_value=_FakeLLM(),
    ):
        out = application_tailoring(state)

    assert "tailored_documents" in out
    # The prompt is a list of {role, content} or a single str depending on
    # the LLM wrapper. Stringify and look for the instructions.
    prompt_repr = str(captured.get("prompt", ""))
    assert "Highlight my Python ML coursework." in prompt_repr
