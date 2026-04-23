"""
retriever.py — Retrieves relevant chunks from ChromaDB for a user query.

Combines embedding-based similarity search with a confidence score
filter so the graph layer knows whether retrieval was good enough.
"""

from typing import List, Dict, Tuple
from embeddings.embed_model import embed_query
from embeddings.vector_store import query_similar
from config import TOP_K_RESULTS, CONFIDENCE_THRESHOLD


def retrieve(query: str) -> Tuple[List[Dict], float]:
    """
    Embed the query, search ChromaDB, and return results with confidence.

    Args:
        query: The user's question string.

    Returns:
        A tuple of:
          - chunks: List of matching chunk dicts (may be empty),
                    each containing text, score, source_doc, page_number.
          - confidence_score: Float in [0, 1] representing the max cosine
                              similarity found. 0.0 if no results.

    The caller (LangGraph evaluate_node) uses confidence_score to decide
    whether to answer or escalate.
    """
    # 1. Embed the query
    query_vec = embed_query(query)

    # 2. Search ChromaDB
    raw_results = query_similar(query_embedding=query_vec, top_k=TOP_K_RESULTS)

    if not raw_results:
        print("[retriever] No results returned from vector store.")
        return [], 0.0

    # 3. Filter out chunks below confidence threshold
    filtered = [r for r in raw_results if r["score"] >= CONFIDENCE_THRESHOLD]

    # Best score across all raw results (used for routing decision)
    best_score = max(r["score"] for r in raw_results)

    print(
        f"[retriever] Query: '{query[:60]}...' | "
        f"Raw hits: {len(raw_results)} | "
        f"Above threshold ({CONFIDENCE_THRESHOLD}): {len(filtered)} | "
        f"Best score: {best_score:.4f}"
    )

    return filtered, round(best_score, 4)


def format_context(chunks: List[Dict]) -> str:
    """
    Format a list of retrieved chunks into a single context string
    that can be injected into the LLM prompt.

    Each chunk is labelled with its source and page number for
    transparency and citation purposes.

    Args:
        chunks: List of chunk dicts from retrieve().

    Returns:
        A multi-line string formatted as:
            [Source: doc.pdf | Page: 3]
            <chunk text>

            [Source: doc.pdf | Page: 5]
            <chunk text>
    """
    if not chunks:
        return ""

    parts = []
    for chunk in chunks:
        header = f"[Source: {chunk['source_doc']} | Page: {chunk['page_number']}]"
        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n".join(parts)
