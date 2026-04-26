from __future__ import annotations

import logging
from typing import Any

from uppgrad_agentic.workflows.auto_apply.graph import build_graph

logger = logging.getLogger(__name__)


def cancel_session(thread_id: str, checkpointer: Any) -> None:
    """Inject a cancel marker into the graph's checkpoint for the given thread_id.

    Uses LangGraph's update_state API. The currently-executing node finishes
    normally; subsequent nodes see result.status == 'error' (error_code='CANCELLED')
    via the existing top-of-node short-circuit pattern, draining the graph to END.
    The background thread running the graph terminates naturally as a result.

    Idempotent and best-effort: calling this on a thread that is already
    cancelled (or has never run) does not raise.
    """
    try:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        graph.update_state(config, {
            "result": {
                "status": "error",
                "error_code": "CANCELLED",
                "user_message": "Cancelled by user.",
            },
        })
        logger.info("cancel_session: marker injected for thread_id=%s", thread_id)
    except Exception as exc:
        logger.warning("cancel_session: failed to inject marker for %s — %s", thread_id, exc)
