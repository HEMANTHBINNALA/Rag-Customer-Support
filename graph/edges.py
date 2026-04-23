"""
edges.py — Conditional routing logic for the LangGraph workflow.

Edge functions inspect the current GraphState and return the name
of the next node to execute. LangGraph uses these return values to
traverse the graph dynamically.
"""

from graph.state import GraphState


def route_after_evaluate(state: GraphState) -> str:
    """
    After evaluate_node: decide whether to generate or escalate.

    Returns:
        "escalate_node"  — if retrieval confidence is too low or
                           a sensitive keyword was detected
        "generate_node"  — if we have good enough context to try LLM
        "error_node"     — if a system error occurred upstream
    """
    if state.get("error"):
        return "error_node"

    if state.get("needs_escalation", False):
        return "escalate_node"

    return "generate_node"


def route_after_validate(state: GraphState) -> str:
    """
    After validate_node: decide whether to format or escalate.

    Returns:
        "escalate_node" — if the LLM expressed uncertainty in its answer
        "format_node"   — if the LLM gave a confident answer
        "error_node"    — if LLM generation itself failed
    """
    if state.get("error"):
        return "error_node"

    if state.get("needs_escalation", False):
        return "escalate_node"

    return "format_node"
