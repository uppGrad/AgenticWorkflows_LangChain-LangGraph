def test_asset_mapping_imports_resolve_profile():
    from uppgrad_agentic.workflows.auto_apply.nodes import asset_mapping as am
    assert hasattr(am, "resolve_profile"), "asset_mapping must import resolve_profile"


def test_application_tailoring_imports_resolve_profile():
    from uppgrad_agentic.workflows.auto_apply.nodes import application_tailoring as at
    assert hasattr(at, "resolve_profile"), "application_tailoring must import resolve_profile"


def test_asset_mapping_uses_state_profile_snapshot():
    """Confirm asset_mapping calls resolve_profile(state) — i.e., it passes state through."""
    from uppgrad_agentic.workflows.auto_apply.nodes import asset_mapping as am

    captured = {}

    def spy(state):
        captured["state"] = state
        return {
            "name": "Snapshot User", "email": "s@x.com",
            "uploaded_documents": {"CV": True},
            "document_texts": {"CV": "hi"},
        }

    real = am.resolve_profile
    am.resolve_profile = spy
    try:
        am.asset_mapping({
            "profile_snapshot": {"x": 1},
            "normalized_requirements": [
                {"requirement_type": "document", "document_type": "CV",
                 "is_assumed": False, "confidence": 0.9},
            ],
            "opportunity_data": {},
            "opportunity_type": "job",
        })
        assert captured["state"]["profile_snapshot"] == {"x": 1}
    finally:
        am.resolve_profile = real


def test_application_tailoring_uses_state_profile_snapshot():
    from uppgrad_agentic.workflows.auto_apply.nodes import application_tailoring as at

    captured = {}

    def spy(state):
        captured["state"] = state
        return {
            "name": "Snapshot User", "email": "s@x.com",
            "uploaded_documents": {"CV": True},
            "document_texts": {"CV": "hi"},
        }

    real = at.resolve_profile
    at.resolve_profile = spy
    try:
        at.application_tailoring({
            "profile_snapshot": {"y": 2},
            "asset_mapping": [],
            "normalized_requirements": [],
            "opportunity_data": {},
            "opportunity_type": "job",
            "human_review_1": {
                "confirmed_mappings": {
                    "CV": {"tailoring_depth": "none",
                           "source_document": "CV", "available": True}
                },
            },
        })
        assert captured["state"]["profile_snapshot"] == {"y": 2}
    finally:
        at.resolve_profile = real
