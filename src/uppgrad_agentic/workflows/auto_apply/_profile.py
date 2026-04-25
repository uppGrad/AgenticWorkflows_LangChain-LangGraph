from __future__ import annotations

from typing import Any, Dict


def resolve_profile(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return the user profile dict for this graph run.

    Prefers state['profile_snapshot'] (injected by backend adapter); falls back
    to the in-repo stub for CLI / local-dev mode.
    """
    snapshot = state.get("profile_snapshot")
    if snapshot:
        return snapshot
    from uppgrad_agentic.workflows.auto_apply.nodes.eligibility_and_readiness import _get_stub_profile
    return _get_stub_profile()
