"""Internal-jobs `application_form_spec` branch in asset_mapping.

Locks in the contract that lets the backend dictate the gate-1
RequirementItem list for internal opportunities (employer_id == 1) by
attaching a spec on the opportunity snapshot. This is the foundation
for future employer-defined custom fields per posting — when employers
can add screening questions to their internal listings, those
just become extra entries in the spec without any agentic code change.
"""
from uppgrad_agentic.workflows.auto_apply.nodes.asset_mapping import asset_mapping


def _state(spec, *, employer_id=1):
    return {
        "opportunity_type": "job",
        "opportunity_id": "1",
        "opportunity_data": {
            "id": 1,
            "employer_id": employer_id,
            "application_form_spec": spec,
        },
        "form_fields": [],
        "normalized_requirements": [],
    }


def test_internal_spec_emits_items_in_order():
    spec = [
        {"key": "resume_file", "label": "CV / Resume", "category": "document",
         "document_type": "CV", "required": True},
        {"key": "cover_letter", "label": "Cover Letter", "category": "document",
         "document_type": "Cover Letter", "required": False},
        {"key": "additional_information", "label": "Additional info",
         "category": "text", "required": False},
    ]
    out = asset_mapping(_state(spec))
    items = out["requirement_items"]
    assert len(items) == 3
    assert [i["id"] for i in items] == [0, 1, 2]
    assert items[0]["category"] == "document" and items[0]["document_type"] == "CV"
    assert items[0]["required"] is True
    assert items[1]["category"] == "document" and items[1]["document_type"] == "Cover Letter"
    assert items[2]["category"] == "text" and items[2]["question"] == "Additional info"


def test_internal_spec_ids_align_with_spec_index_for_finalize_lookup():
    """RequirementItem.id is sequential and matches the spec index.
    finalize_internal_submission relies on this to map an item back to
    its Application column key."""
    spec = [
        {"key": "resume_file", "label": "CV", "category": "document",
         "document_type": "CV", "required": True},
        {"key": "cover_letter", "label": "Cover Letter", "category": "document",
         "document_type": "Cover Letter", "required": False},
    ]
    out = asset_mapping(_state(spec))
    items = out["requirement_items"]
    # spec[item.id]["key"] gives the Application column for a given item.
    assert spec[items[0]["id"]]["key"] == "resume_file"
    assert spec[items[1]["id"]]["key"] == "cover_letter"


def test_non_internal_job_ignores_spec_even_if_present():
    """A non-internal job (employer_id != 1) shouldn't honour an
    application_form_spec — that's an internal-only concept."""
    spec = [
        {"key": "x", "label": "X", "category": "document",
         "document_type": "CV", "required": True},
    ]
    state = _state(spec, employer_id=None)
    state["normalized_requirements"] = [
        {"requirement_type": "document", "document_type": "CV",
         "is_assumed": True, "confidence": 0.9},
    ]
    out = asset_mapping(state)
    items = out["requirement_items"]
    # Should fall through to normalized_requirements path → 1 doc item
    # (CV) labelled by document_type, not "X" from the spec.
    assert len(items) == 1
    assert items[0]["label"] == "CV"


def test_internal_job_without_spec_falls_through_to_normalized():
    """Backwards compat: if an internal job somehow doesn't carry a spec
    (e.g. legacy opportunity row), we fall through to the existing
    normalized_requirements path."""
    state = {
        "opportunity_type": "job",
        "opportunity_id": "1",
        "opportunity_data": {"id": 1, "employer_id": 1},  # no spec
        "form_fields": [],
        "normalized_requirements": [
            {"requirement_type": "document", "document_type": "CV",
             "is_assumed": True, "confidence": 0.9},
            {"requirement_type": "document", "document_type": "Cover Letter",
             "is_assumed": True, "confidence": 0.8},
        ],
    }
    out = asset_mapping(state)
    items = out["requirement_items"]
    assert len(items) == 2
    assert {i["document_type"] for i in items} == {"CV", "Cover Letter"}


def test_internal_spec_text_item_carries_question_for_tailoring():
    """Text items in the spec become RequirementItems where
    `question` == label, so application_tailoring's text path picks
    up the right prompt."""
    spec = [
        {"key": "additional_information", "label": "Why are you a fit for this role?",
         "category": "text", "required": True,
         "description": "The hiring team will read this first."},
    ]
    out = asset_mapping(_state(spec))
    item = out["requirement_items"][0]
    assert item["category"] == "text"
    assert item["question"] == "Why are you a fit for this role?"
    assert item["required"] is True
