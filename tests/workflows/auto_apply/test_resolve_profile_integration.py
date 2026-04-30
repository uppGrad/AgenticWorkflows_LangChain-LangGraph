def test_asset_mapping_imports_resolve_profile():
    """Kept as a structural smoke test even though asset_mapping no longer
    consumes profile after the gate-1 remodel — the import documents the
    intended dependency direction."""
    from uppgrad_agentic.workflows.auto_apply.nodes import asset_mapping as am
    assert hasattr(am, "resolve_profile"), "asset_mapping must import resolve_profile"


def test_application_tailoring_imports_resolve_profile():
    from uppgrad_agentic.workflows.auto_apply.nodes import application_tailoring as at
    assert hasattr(at, "resolve_profile"), "application_tailoring must import resolve_profile"


def test_application_tailoring_uses_state_profile_snapshot():
    """application_tailoring must thread `state` into resolve_profile so the
    backend-injected `profile_snapshot` is honoured (rather than the in-repo
    stub)."""
    from uppgrad_agentic.workflows.auto_apply.nodes import application_tailoring as at

    captured = {}

    def spy(state):
        captured["state"] = state
        return {
            "name": "Snapshot User", "email": "s@x.com",
            "document_texts": {"CV": "hi"},
        }

    real = at.resolve_profile
    at.resolve_profile = spy
    try:
        at.application_tailoring({
            "profile_snapshot": {"y": 2},
            "opportunity_data": {},
            "opportunity_type": "job",
            "requirement_items": [
                {"id": 0, "category": "document", "label": "CV",
                 "description": "", "field_type": None, "required": True,
                 "document_type": "CV", "question": None, "form_field_index": None},
            ],
            "human_review_1": {
                "requirements": {
                    "0": {"choice": "auto_generate", "user_prompt": None},
                },
                "misc_strategy": "ignore",
            },
        })
        assert captured["state"]["profile_snapshot"] == {"y": 2}
    finally:
        at.resolve_profile = real
