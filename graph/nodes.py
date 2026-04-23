"""
nodes.py — All LangGraph node functions for the RAG workflow.

Each node takes a GraphState dict, performs its operation,
and returns a partial state update dict. LangGraph merges
the update into the running state automatically.

Node execution order:
    retrieve_node → evaluate_node → [generate_node → validate_node → format_node]
                                  ↘ escalate_node (if flagged)
"""

from graph.state import GraphState
from retrieval.retriever import retrieve, format_context
from generation.prompt_builder import build_prompt, build_no_context_prompt
from generation.llm_client import call_llm
from hitl.escalation import log_escalation
from config import ESCALATION_KEYWORDS, UNCERTAINTY_PHRASES


# ─── Node 1: Retrieve ────────────────────────────────────────────────────────

def retrieve_node(state: GraphState) -> dict:
    """
    Embed the user query and fetch top-k similar chunks from ChromaDB.

    Populates: retrieved_chunks, context_str, confidence_score
    """
    try:
        query = state["query"]
        chunks, confidence = retrieve(query)
        context = format_context(chunks)

        return {
            "retrieved_chunks": chunks,
            "context_str": context,
            "confidence_score": confidence,
            "error": None,
        }
    except Exception as exc:
        return {
            "retrieved_chunks": [],
            "context_str": "",
            "confidence_score": 0.0,
            "error": f"Retrieval failed: {exc}",
        }


# ─── Node 2: Evaluate ────────────────────────────────────────────────────────

def evaluate_node(state: GraphState) -> dict:
    """
    Decide if we have enough confidence to attempt generation.

    Escalation triggers:
      - No chunks retrieved
      - Best similarity score below threshold
      - Query contains a sensitive escalation keyword

    Populates: needs_escalation, escalation_reason
    """
    from config import CONFIDENCE_THRESHOLD

    # Propagate errors from retrieve_node
    if state.get("error"):
        return {
            "needs_escalation": True,
            "escalation_reason": f"System error during retrieval: {state['error']}",
        }

    chunks = state["retrieved_chunks"]
    score = state["confidence_score"]
    query_lower = state["query"].lower()

    # Check sensitive keywords FIRST (highest priority)
    for kw in ESCALATION_KEYWORDS:
        if kw in query_lower:
            return {
                "needs_escalation": True,
                "escalation_reason": (
                    f"Query contains sensitive keyword '{kw}' "
                    "and requires human review."
                ),
            }

    # No chunks returned
    if not chunks:
        return {
            "needs_escalation": True,
            "escalation_reason": "No relevant documentation found for this query.",
        }

    # Low confidence score
    if score < CONFIDENCE_THRESHOLD:
        return {
            "needs_escalation": True,
            "escalation_reason": (
                f"Retrieval confidence too low ({score:.2f} < {CONFIDENCE_THRESHOLD}). "
                "Results may not be relevant."
            ),
        }

    return {"needs_escalation": False, "escalation_reason": ""}


# ─── Node 3: Generate ────────────────────────────────────────────────────────

def generate_node(state: GraphState) -> dict:
    """
    Send the retrieved context + query to the LLM and get a response.

    Uses build_no_context_prompt if context_str is empty (shouldn't
    happen here, but defensive programming).

    Populates: llm_response
    """
    try:
        query = state["query"]
        context_str = state["context_str"]

        if context_str:
            messages = build_prompt(query, context_str)
        else:
            messages = build_no_context_prompt(query)

        response = call_llm(messages)

        return {"llm_response": response, "error": None}
    except Exception as exc:
        return {
            "llm_response": "",
            "error": f"LLM generation failed: {exc}",
        }


# ─── Node 4: Validate ────────────────────────────────────────────────────────

def validate_node(state: GraphState) -> dict:
    """
    Check the LLM's own output for uncertainty signals.

    If the LLM itself says it doesn't know, escalate rather than
    surfacing a vague non-answer to the user.

    May update: needs_escalation, escalation_reason
    """
    if state.get("error"):
        return {
            "needs_escalation": True,
            "escalation_reason": f"LLM generation error: {state['error']}",
        }

    response_lower = state["llm_response"].lower()

    for phrase in UNCERTAINTY_PHRASES:
        if phrase in response_lower:
            return {
                "needs_escalation": True,
                "escalation_reason": (
                    f"LLM response indicates uncertainty (contains: '{phrase}'). "
                    "Routing to human agent."
                ),
            }

    # LLM is confident — no escalation needed
    return {"needs_escalation": False}


# ─── Node 5: Format ──────────────────────────────────────────────────────────

def format_node(state: GraphState) -> dict:
    """
    Clean and finalize the LLM's response for display to the user.

    Strips excessive whitespace, adds a source citation footer
    if retrieved chunks are available.

    Populates: final_answer
    """
    response = state.get("llm_response", "").strip()

    # Build a compact citation footer
    chunks = state.get("retrieved_chunks", [])
    if chunks:
        sources = {}
        for c in chunks:
            doc = c["source_doc"]
            page = c["page_number"]
            sources.setdefault(doc, set()).add(page)

        citation_parts = []
        for doc, pages in sources.items():
            sorted_pages = sorted(pages)
            pages_str = ", ".join(str(p) for p in sorted_pages)
            citation_parts.append(f"{doc} (page{'s' if len(sorted_pages) > 1 else ''} {pages_str})")

        citation = "\n\n[Sources: " + " | ".join(citation_parts) + "]"
        response += citation

    return {"final_answer": response}


# ─── Node 6: Escalate ────────────────────────────────────────────────────────

def escalate_node(state: GraphState) -> dict:
    """
    Handle escalation to a human agent.

    Logs the escalation with full context and returns a user-friendly
    message explaining that a human agent has been notified.

    Populates: final_answer
    """
    reason = state.get("escalation_reason", "Reason not specified.")
    query = state.get("query", "")

    # Log it for the human agent queue
    log_escalation(
        query=query,
        reason=reason,
        retrieved_chunks=state.get("retrieved_chunks", []),
        llm_response=state.get("llm_response", ""),
    )

    user_message = (
        "🙋 **Your query has been escalated to a human support agent.**\n\n"
        f"**Reason:** {reason}\n\n"
        "A member of our team will review your question and get back to you shortly. "
        "Thank you for your patience."
    )

    return {"final_answer": user_message}


# ─── Node 7: Error Handler ───────────────────────────────────────────────────

def error_node(state: GraphState) -> dict:
    """
    Catch-all for unexpected system errors.

    Returns a safe, user-facing message instead of a raw traceback.

    Populates: final_answer
    """
    error_detail = state.get("error", "An unknown error occurred.")
    print(f"[error_node] Handling system error: {error_detail}")

    return {
        "final_answer": (
            "⚠️ **We encountered a technical issue while processing your request.**\n\n"
            "Our team has been notified. Please try again in a moment, "
            "or contact support directly if the issue persists."
        )
    }
