"""
embed_model.py — Loads and wraps the sentence-transformers embedding model.

Uses sentence-transformers/all-MiniLM-L6-v2 by default:
- 384-dimensional embeddings
- Runs fully locally (no API calls)
- Fast enough for hundreds of chunks on CPU
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

# Module-level singleton — loaded once and reused across calls
_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """
    Return the loaded embedding model (lazy singleton).

    The model is downloaded from HuggingFace Hub on first call
    and cached locally (~90 MB for MiniLM-L6-v2).
    """
    global _model
    if _model is None:
        print(f"[embed_model] Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("[embed_model] Model loaded successfully.")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (each is a list of floats, length 384).
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return [emb.tolist() for emb in embeddings]


def embed_query(query: str) -> List[float]:
    """
    Generate a single embedding for a user query string.

    Args:
        query: The user's question text.

    Returns:
        A single embedding vector (list of 384 floats).
    """
    model = get_model()
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding[0].tolist()
