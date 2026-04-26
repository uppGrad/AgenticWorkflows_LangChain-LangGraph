from uppgrad_agentic.workflows.auto_apply.graph import build_graph


def test_graph_includes_discover_apply_url_node():
    g = build_graph().get_graph()
    assert "discover_apply_url" in g.nodes


def test_load_opportunity_routes_to_discover_for_jobs():
    g = build_graph().get_graph()
    edges = [e for e in g.edges if e.source == "load_opportunity"]
    targets = {e.target for e in edges}
    assert "discover_apply_url" in targets


def test_discover_routes_to_scrape():
    g = build_graph().get_graph()
    edges = [e for e in g.edges if e.source == "discover_apply_url"]
    targets = {e.target for e in edges}
    assert "scrape_application_page" in targets
