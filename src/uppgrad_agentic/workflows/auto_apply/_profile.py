from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def resolve_profile(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return the user profile dict for this graph run.

    Prefers state['profile_snapshot'] (injected by backend adapter); falls back
    to the in-repo stub for CLI / local-dev mode.

    The fallback path emits a WARNING log: this should never happen in
    production, where the backend adapter always injects a real snapshot.
    Seeing this warning in prod logs indicates a missing integration step.
    """
    snapshot = state.get("profile_snapshot")
    if snapshot:
        return snapshot
    logger.warning(
        "resolve_profile: state['profile_snapshot'] absent — falling back to in-repo "
        "stub profile. This is expected only in CLI/local-dev mode; in production "
        "the backend adapter must inject a real profile_snapshot."
    )
    from uppgrad_agentic.workflows.auto_apply.nodes.eligibility_and_readiness import _get_stub_profile
    return _get_stub_profile()
