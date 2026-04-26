from langgraph.checkpoint.memory import MemorySaver

from uppgrad_agentic.workflows.auto_apply.control import cancel_session
from uppgrad_agentic.workflows.auto_apply.graph import build_graph


def _drive_graph_to_first_interrupt(checkpointer, thread_id: str):
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    initial = {"opportunity_type": "job", "opportunity_id": "job-001"}
    try:
        for _ in graph.stream(initial, config=config, stream_mode="values"):
            pass
    except Exception:
        pass
    return graph, config


def test_cancel_writes_error_marker_to_state():
    checkpointer = MemorySaver()
    graph, config = _drive_graph_to_first_interrupt(checkpointer, "test-cancel-1")

    cancel_session("test-cancel-1", checkpointer)

    final_state = graph.get_state(config).values
    assert final_state.get("result", {}).get("status") == "error"
    assert final_state["result"]["error_code"] == "CANCELLED"


def test_cancel_is_idempotent():
    checkpointer = MemorySaver()
    _drive_graph_to_first_interrupt(checkpointer, "test-cancel-2")

    cancel_session("test-cancel-2", checkpointer)
    cancel_session("test-cancel-2", checkpointer)   # second call must not raise


def test_cancel_unknown_thread_id_does_not_raise():
    checkpointer = MemorySaver()
    # Cancel a thread that has never run — should be a silent no-op (best-effort)
    cancel_session("nonexistent-thread", checkpointer)
