"""
vector_store.py — ChromaDB CRUD operations.

Manages the ChromaDB collection that stores chunk embeddings plus
their metadata. Supports:
  - Adding new chunks (during ingestion)
  - Querying for similar chunks (during retrieval)
  - Checking if collection already exists (to skip re-ingestion)
  - Clearing the collection (for re-ingestion from scratch)
"""

from typing import List, Dict, Optional
import chromadb
from config import CHROMA_DB_PATH, CHROMA_COLLECTION


# ─── Client Singleton ────────────────────────────────────────────────────────

_client: Optional[chromadb.PersistentClient] = None


def _get_client():
    """Return the ChromaDB persistent client (lazy singleton)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        print(f"[vector_store] ChromaDB client initialized at '{CHROMA_DB_PATH}'")
    return _client


def _get_collection() -> chromadb.Collection:
    """Return (or create) the support docs collection."""
    client = _get_client()
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )
    return collection


# ─── Write Operations ────────────────────────────────────────────────────────


def add_chunks(chunks: List[Dict], embeddings: List[List[float]]) -> None:
    """
    Insert chunks and their embeddings into ChromaDB.

    Automatically batches large inserts to stay within ChromaDB's
    maximum batch size limit (5461 rows per upsert call).

    Args:
        chunks:     List of chunk dicts from chunker.chunk_pages().
        embeddings: Parallel list of embedding vectors from embed_model.embed_texts().
    """
    collection = _get_collection()

    BATCH_SIZE = 2000  # Well below ChromaDB's 5461 hard limit

    total = len(chunks)
    batches = (total + BATCH_SIZE - 1) // BATCH_SIZE  # ceil division

    for batch_num in range(batches):
        start = batch_num * BATCH_SIZE
        end   = min(start + BATCH_SIZE, total)

        batch_chunks     = chunks[start:end]
        batch_embeddings = embeddings[start:end]

        ids       = [c["chunk_id"] for c in batch_chunks]
        documents = [c["text"] for c in batch_chunks]
        metadatas = [
            {
                "source_doc":  c["source_doc"],
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"],
            }
            for c in batch_chunks
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )
        print(
            f"[vector_store] Upserted batch {batch_num + 1}/{batches} "
            f"({end - start} chunks, total so far: {end}/{total})"
        )

    print(
        f"[vector_store] Done — {total} chunks stored in '{CHROMA_COLLECTION}'"
    )



def clear_collection() -> None:
    """Delete and recreate the collection (full re-ingestion)."""
    client = _get_client()
    try:
        client.delete_collection(name=CHROMA_COLLECTION)
        print(f"[vector_store] Cleared collection '{CHROMA_COLLECTION}'")
    except Exception:
        pass  # Collection didn't exist yet


def get_chunk_count() -> int:
    """Return the number of chunks currently in the collection."""
    collection = _get_collection()
    return collection.count()


# ─── Read Operations ─────────────────────────────────────────────────────────


def query_similar(
    query_embedding: List[float],
    top_k: int = 4,
) -> List[Dict]:
    """
    Find the top-k most similar chunks to a query embedding.

    Args:
        query_embedding: The embedded query vector.
        top_k:           Number of results to return.

    Returns:
        List of result dicts, sorted by relevance (most relevant first):
            {
                "chunk_id":    str,
                "text":        str,
                "score":       float,  # cosine similarity (0–1)
                "source_doc":  str,
                "page_number": int,
                "chunk_index": int,
            }
    """
    collection = _get_collection()

    if collection.count() == 0:
        print("[vector_store] WARNING: Collection is empty. Run ingestion first.")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns distances (lower = more similar for cosine).
    # Convert distance → similarity score: score = 1 - distance
    output: List[Dict] = []
    for i, doc_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        score = max(0.0, 1.0 - distance)  # Clamp to [0, 1]
        metadata = results["metadatas"][0][i]

        output.append(
            {
                "chunk_id": doc_id,
                "text": results["documents"][0][i],
                "score": round(score, 4),
                "source_doc": metadata.get("source_doc", ""),
                "page_number": metadata.get("page_number", -1),
                "chunk_index": metadata.get("chunk_index", -1),
            }
        )

    return output
