from uppgrad_agentic.workflows.auto_apply.graph import build_graph


def test_human_gate_0_routes_to_eligibility_or_error():
    graph = build_graph()
    g = graph.get_graph()
    edges_from_gate_0 = [e for e in g.edges if e.source == "human_gate_0"]
    targets = {e.target for e in edges_from_gate_0}
    # Must NOT terminate at END directly
    assert "__end__" not in targets, f"human_gate_0 still terminates at END: {targets}"
    # Must route to eligibility re-check
    assert "eligibility_and_readiness" in targets, f"missing eligibility loop: {targets}"
