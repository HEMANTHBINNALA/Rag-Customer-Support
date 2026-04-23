"""
state.py — Defines the shared state TypedDict for the LangGraph workflow.

Every node in the graph reads from and writes to this single state object.
Using TypedDict makes the state explicit, type-safe, and inspectable.
"""

from typing import TypedDict, List, Dict, Optional


class GraphState(TypedDict):
    """
    Shared state passed between every node in the LangGraph workflow.

    Fields are populated progressively as the graph executes:
      - query             → set by the caller before graph.invoke()
      - retrieved_chunks  → set by retrieve_node
      - context_str       → set by retrieve_node (formatted context)
      - confidence_score  → set by retrieve_node (best similarity score)
      - needs_escalation  → set/updated by evaluate_node and validate_node
      - escalation_reason → set when needs_escalation = True
      - llm_response      → set by generate_node
      - final_answer      → set by format_node (clean, user-facing answer)
      - error             → set if any node raises an exception
    """

    query: str
    retrieved_chunks: List[Dict]          # Each: {chunk_id, text, score, source_doc, page_number}
    context_str: str                      # Pre-formatted context for the prompt
    confidence_score: float               # Best cosine similarity score from retrieval
    needs_escalation: bool                # Routing flag: True → escalate_node
    escalation_reason: str                # Human-readable reason for escalation
    llm_response: str                     # Raw LLM output
    final_answer: str                     # Final, formatted answer shown to user
    error: Optional[str]                  # Error message if something went wrong
