from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from uppgrad_agentic.workflows.auto_apply.state import AutoApplyState
from uppgrad_agentic.workflows.auto_apply.nodes.load_opportunity import load_opportunity
from uppgrad_agentic.workflows.auto_apply.nodes.scrape_application_page import scrape_application_page
from uppgrad_agentic.workflows.auto_apply.nodes.evaluate_scrape import evaluate_scrape
from uppgrad_agentic.workflows.auto_apply.nodes.determine_requirements import determine_requirements
from uppgrad_agentic.workflows.auto_apply.nodes.eligibility_and_readiness import eligibility_and_readiness
from uppgrad_agentic.workflows.auto_apply.nodes.end_with_explanation import end_with_explanation
from uppgrad_agentic.workflows.auto_apply.nodes.human_gate_0 import human_gate_0
from uppgrad_agentic.workflows.auto_apply.nodes.asset_mapping import asset_mapping
from uppgrad_agentic.workflows.auto_apply.nodes.human_gate_1 import human_gate_1
from uppgrad_agentic.workflows.auto_apply.nodes.application_tailoring import application_tailoring
from uppgrad_agentic.workflows.auto_apply.nodes.application_evaluation import application_evaluation
from uppgrad_agentic.workflows.auto_apply.nodes.human_gate_2 import human_gate_2
from uppgrad_agentic.workflows.auto_apply.nodes.submit_internal import submit_internal
from uppgrad_agentic.workflows.auto_apply.nodes.package_and_handoff import package_and_handoff
from uppgrad_agentic.workflows.auto_apply.nodes.record_application import record_application

MAX_EVAL_ITERATIONS = 2


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _route_after_load(state: AutoApplyState) -> str:
    if state.get("result", {}).get("status") == "error":
        return "end_with_error"
    if state.get("opportunity_type") == "job":
        return "scrape_application_page"
    return "determine_requirements"


def _route_after_scrape(state: AutoApplyState) -> str:
    if state.get("result", {}).get("status") == "error":
        return "end_with_error"
    return "evaluate_scrape"


def _route_after_evaluate_scrape(state: AutoApplyState) -> str:
    if state.get("result", {}).get("status") == "error":
        return "end_with_error"
    return "determine_requirements"


def _route_after_eligibility(state: AutoApplyState) -> str:
    if state.get("result", {}).get("status") == "error":
        return "end_with_error"

    decision = (state.get("eligibility_result") or {}).get("decision", "manual_review")
    if decision == "ineligible":
        return "end_with_explanation"
    if decision == "pending":
        return "human_gate_0"
    # ready or manual_review both proceed to asset mapping
    return "asset_mapping"


def _route_after_app_evaluation(state: AutoApplyState) -> str:
    if state.get("result", {}).get("status") == "error":
        return "end_with_error"

    evaluation = state.get("evaluation_result") or {}
    passed = evaluation.get("passed", True)
    iteration_count = state.get("iteration_count", 0)

    if passed or iteration_count >= MAX_EVAL_ITERATIONS:
        return "human_gate_2"
    return "application_tailoring"


def _route_after_gate_0(state: AutoApplyState) -> str:
    if state.get("result", {}).get("status") == "error":
        return "end_with_error"
    return "eligibility_and_readiness"


def _route_after_gate2(state: AutoApplyState) -> str:
    if state.get("result", {}).get("status") == "error":
        return "end_with_error"

    human_review_2 = state.get("human_review_2") or {}
    if not human_review_2.get("approved", False):
        # User cancelled at final review — terminate gracefully without submitting
        return "end_with_error"

    # route_by_source logic: internal job (employer_id == 1) vs everything else
    opportunity_data = state.get("opportunity_data") or {}
    employer_id = opportunity_data.get("employer_id")
    if state.get("opportunity_type") == "job" and employer_id == 1:
        return "submit_internal"
    return "package_and_handoff"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    g = StateGraph(AutoApplyState)

    # Opportunity Intelligence phase
    g.add_node("load_opportunity", load_opportunity)
    g.add_node("scrape_application_page", scrape_application_page)
    g.add_node("evaluate_scrape", evaluate_scrape)
    g.add_node("determine_requirements", determine_requirements)

    # Eligibility phase
    g.add_node("eligibility_and_readiness", eligibility_and_readiness)
    g.add_node("end_with_explanation", end_with_explanation)
    g.add_node("human_gate_0", human_gate_0)

    # Asset Mapping phase
    g.add_node("asset_mapping", asset_mapping)
    g.add_node("human_gate_1", human_gate_1)

    # Application Tailoring + Evaluation loop
    g.add_node("application_tailoring", application_tailoring)
    g.add_node("application_evaluation", application_evaluation)

    # Final human gate
    g.add_node("human_gate_2", human_gate_2)

    # Submission phase
    g.add_node("submit_internal", submit_internal)
    g.add_node("package_and_handoff", package_and_handoff)
    g.add_node("record_application", record_application)

    # Error terminals
    g.add_node("end_with_error", lambda state: {})

    # -----------------------------------------------------------------------
    # Edges — Opportunity Intelligence
    # -----------------------------------------------------------------------
    g.add_edge(START, "load_opportunity")

    g.add_conditional_edges(
        "load_opportunity",
        _route_after_load,
        {
            "scrape_application_page": "scrape_application_page",
            "determine_requirements": "determine_requirements",
            "end_with_error": "end_with_error",
        },
    )

    g.add_conditional_edges(
        "scrape_application_page",
        _route_after_scrape,
        {
            "evaluate_scrape": "evaluate_scrape",
            "end_with_error": "end_with_error",
        },
    )

    g.add_conditional_edges(
        "evaluate_scrape",
        _route_after_evaluate_scrape,
        {
            "determine_requirements": "determine_requirements",
            "end_with_error": "end_with_error",
        },
    )

    # -----------------------------------------------------------------------
    # Edges — Eligibility
    # -----------------------------------------------------------------------
    g.add_edge("determine_requirements", "eligibility_and_readiness")

    g.add_conditional_edges(
        "eligibility_and_readiness",
        _route_after_eligibility,
        {
            "end_with_explanation": "end_with_explanation",
            "human_gate_0": "human_gate_0",
            "asset_mapping": "asset_mapping",
            "end_with_error": "end_with_error",
        },
    )

    # human_gate_0 routes back to eligibility re-check after profile update,
    # or to end_with_error when the iteration cap fires inside the node.
    g.add_conditional_edges(
        "human_gate_0",
        _route_after_gate_0,
        {
            "eligibility_and_readiness": "eligibility_and_readiness",
            "end_with_error": "end_with_error",
        },
    )

    # -----------------------------------------------------------------------
    # Edges — Asset Mapping
    # -----------------------------------------------------------------------
    g.add_edge("asset_mapping", "human_gate_1")
    g.add_edge("human_gate_1", "application_tailoring")

    # -----------------------------------------------------------------------
    # Edges — Tailoring + Evaluation loop
    # -----------------------------------------------------------------------
    g.add_edge("application_tailoring", "application_evaluation")

    g.add_conditional_edges(
        "application_evaluation",
        _route_after_app_evaluation,
        {
            "human_gate_2": "human_gate_2",
            "application_tailoring": "application_tailoring",
            "end_with_error": "end_with_error",
        },
    )

    # -----------------------------------------------------------------------
    # Edges — Final gate + route_by_source + submission
    # -----------------------------------------------------------------------
    g.add_conditional_edges(
        "human_gate_2",
        _route_after_gate2,
        {
            "submit_internal": "submit_internal",
            "package_and_handoff": "package_and_handoff",
            "end_with_error": "end_with_error",
        },
    )

    g.add_edge("submit_internal", "record_application")
    g.add_edge("package_and_handoff", "record_application")
    g.add_edge("record_application", END)

    # -----------------------------------------------------------------------
    # Edges — terminals
    # -----------------------------------------------------------------------
    g.add_edge("end_with_explanation", END)
    g.add_edge("end_with_error", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
