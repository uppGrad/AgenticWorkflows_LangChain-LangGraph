from __future__ import annotations

from typing import Any, Dict, List

from langgraph.types import interrupt

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState


# ---------------------------------------------------------------------------
# Resume value contract
#
# When the graph is resumed the caller must pass Command(resume=selections)
# where `selections` is a dict keyed by string asset indices ("0", "1", ...):
#
#   selections: {
#       "0": {
#           "source_document": "CV",       # may override the mapped source
#           "tailoring_depth": "light",    # may override the mapped depth
#           "skip": false,                 # true to exclude this document
#           "content": null,               # text if user uploaded a new version
#       },
#       "1": {
#           "source_document": "CV",
#           "tailoring_depth": "generate",
#           "skip": false,
#           "content": null,
#       },
#   }
#
# - Entries with no key in selections keep their original asset_mapping values.
# - "skip": true removes the document from the tailoring step entirely.
# - "content" carries user-uploaded document text that replaces the stored source.
# - Passing selections={"confirm": True} confirms all mappings as-is (do NOT pass an
#   empty dict — LangGraph treats falsy resume values as "no resume" and re-interrupts).
#
# Interrupt payload (what the frontend receives):
#
#   {
#       "asset_mapping": [{"id": 0, ...AssetMap fields...}, ...],
#       "opportunity_type": "job",
#       "opportunity_title": "Software Engineer at Acme Corp",
#   }
# ---------------------------------------------------------------------------

_VALID_DEPTHS = {"none", "light", "deep", "generate"}


def human_gate_1(state: AutoApplyState) -> dict:
    if state.get("result", {}).get("status") == "error":
        return {}

    asset_mapping: List[Dict[str, Any]] = state.get("asset_mapping") or []
    opportunity_data = state.get("opportunity_data") or {}
    opportunity_type = state.get("opportunity_type", "")

    title = (
        opportunity_data.get("title")
        or opportunity_data.get("name")
        or "this opportunity"
    )
    company = (
        opportunity_data.get("company")
        or opportunity_data.get("university")
        or opportunity_data.get("provider_name")
        or ""
    )
    opportunity_title = f"{title} at {company}" if company else title

    # Attach stable indices so the frontend can reference entries by position
    indexed_mapping = [{"id": i, **m} for i, m in enumerate(asset_mapping)]

    # -------------------------------------------------------------------
    # Suspend. Resumes via Command(resume=selections).
    # -------------------------------------------------------------------
    selections: Dict[str, Any] = interrupt(
        {
            "asset_mapping": indexed_mapping,
            "opportunity_type": opportunity_type,
            "opportunity_title": opportunity_title,
        }
    )

    # -------------------------------------------------------------------
    # Validate and normalise the resume value
    # -------------------------------------------------------------------
    if not isinstance(selections, dict):
        selections = {}

    confirmed_mappings: Dict[str, Dict[str, Any]] = {}
    additional_uploads: Dict[str, str] = {}

    for entry in indexed_mapping:
        idx = str(entry["id"])
        doc_type = entry.get("requirement_type", "")
        raw = selections.get(idx) or {}

        if isinstance(raw, str):
            # Bare string is treated as a skip flag ("skip") or confirm ("confirm")
            raw = {"skip": raw.lower() == "skip"}

        skip = bool(raw.get("skip", False))

        # Allow overrides for source_document and tailoring_depth
        source_document = raw.get("source_document") or entry.get("source_document", "")
        tailoring_depth = raw.get("tailoring_depth") or entry.get("tailoring_depth", "light")
        if tailoring_depth not in _VALID_DEPTHS:
            tailoring_depth = entry.get("tailoring_depth", "light")

        # User-uploaded content for this document (overrides stored source)
        content = raw.get("content") or None
        if content and isinstance(content, str) and content.strip():
            additional_uploads[doc_type] = content.strip()

        confirmed_mappings[doc_type] = {
            "source_document": source_document,
            "tailoring_depth": tailoring_depth,
            "skip": skip,
            "available": entry.get("available", False),
            "notes": entry.get("notes", ""),
        }

    return {
        "human_review_1": {
            "confirmed_mappings": confirmed_mappings,
            "additional_uploads": additional_uploads,
        }
    }
