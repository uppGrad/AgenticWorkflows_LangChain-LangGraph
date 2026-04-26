from uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring import _truncate_to_cap


def test_cv_capped_at_8000():
    out = _truncate_to_cap("X" * 12000, "CV")
    assert len(out) <= 8000


def test_cover_letter_capped_at_3000():
    out = _truncate_to_cap("Y" * 5000, "Cover Letter")
    assert len(out) <= 3000


def test_sop_capped_at_6000():
    out = _truncate_to_cap("Z" * 10000, "SOP")
    assert len(out) <= 6000


def test_personal_statement_capped_at_6000():
    out = _truncate_to_cap("Z" * 10000, "Personal Statement")
    assert len(out) <= 6000


def test_unknown_doc_type_uses_default_5000():
    out = _truncate_to_cap("Q" * 8000, "Reference Letter")
    assert len(out) <= 5000


def test_does_not_truncate_short_content():
    out = _truncate_to_cap("short", "CV")
    assert out == "short"


def test_truncates_at_paragraph_boundary_when_past_half():
    text = ("A" * 2000) + "\n\n" + ("B" * 4000)
    out = _truncate_to_cap(text, "Cover Letter")  # cap=3000
    assert len(out) <= 3000
    # Boundary at 2000 > cap//2 (1500) → truncate at boundary
    assert out.endswith("A" * 2000)
    assert "B" not in out


def test_hard_cuts_when_boundary_too_early():
    # Boundary at 100 is < cap//2 (1500) → falls through to hard-cut at cap
    text = ("A" * 100) + "\n\n" + ("B" * 5000)
    out = _truncate_to_cap(text, "Cover Letter")  # cap=3000
    assert len(out) == 3000
