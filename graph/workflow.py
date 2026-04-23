"""
workflow.py — Assembles and compiles the full LangGraph RAG workflow.

Graph structure:
    START
      └──→ retrieve_node
               └──→ evaluate_node
                        │
                        ├── [needs_escalation] ──→ escalate_node ──→ END
                        ├── [error]            ──→ error_node    ──→ END
                        └── [ok]               ──→ generate_node
                                                         └──→ validate_node
                                                                  │
                                                                  ├── [needs_escalation] ──→ escalate_node ──→ END
                                                                  ├── [error]            ──→ error_node    ──→ END
                                                                  └── [ok]               ──→ format_node   ──→ END
"""

from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.nodes import (
    retrieve_node,
    evaluate_node,
    generate_node,
    validate_node,
    format_node,
    escalate_node,
    error_node,
)
from graph.edges import route_after_evaluate, route_after_validate


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph RAG workflow.

    Returns:
        A compiled LangGraph CompiledGraph ready for .invoke() calls.
    """
    graph = StateGraph(GraphState)

    # ── Register nodes ──────────────────────────────────────────────
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("evaluate_node", evaluate_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("validate_node", validate_node)
    graph.add_node("format_node", format_node)
    graph.add_node("escalate_node", escalate_node)
    graph.add_node("error_node", error_node)

    # ── Entry point ─────────────────────────────────────────────────
    graph.set_entry_point("retrieve_node")

    # ── Linear edges ────────────────────────────────────────────────
    graph.add_edge("retrieve_node", "evaluate_node")
    graph.add_edge("generate_node", "validate_node")

    # ── Conditional edges ───────────────────────────────────────────
    graph.add_conditional_edges(
        "evaluate_node",
        route_after_evaluate,
        {
            "generate_node": "generate_node",
            "escalate_node": "escalate_node",
            "error_node": "error_node",
        },
    )

    graph.add_conditional_edges(
        "validate_node",
        route_after_validate,
        {
            "format_node": "format_node",
            "escalate_node": "escalate_node",
            "error_node": "error_node",
        },
    )

    # ── Terminal edges ──────────────────────────────────────────────
    graph.add_edge("format_node", END)
    graph.add_edge("escalate_node", END)
    graph.add_edge("error_node", END)

    return graph.compile()


# Module-level compiled graph singleton
_compiled_graph = None


def get_graph():
    """Return the compiled graph (lazy singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
        print("[workflow] LangGraph RAG workflow compiled successfully.")
    return _compiled_graph


def run_query(query: str) -> dict:
    """
    Run a user query through the full RAG graph and return the final state.

    Args:
        query: The user's question string.

    Returns:
        Final GraphState dict with 'final_answer' populated.
    """
    graph = get_graph()

    initial_state: GraphState = {
        "query": query,
        "retrieved_chunks": [],
        "context_str": "",
        "confidence_score": 0.0,
        "needs_escalation": False,
        "escalation_reason": "",
        "llm_response": "",
        "final_answer": "",
        "error": None,
    }

    final_state = graph.invoke(initial_state)
    return final_state
