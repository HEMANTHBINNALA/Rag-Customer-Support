"""embeddings package — embedding model and ChromaDB vector store."""
from .embed_model import embed_texts, embed_query, get_model
from .vector_store import add_chunks, query_similar, clear_collection, get_chunk_count

__all__ = [
    "embed_texts",
    "embed_query",
    "get_model",
    "add_chunks",
    "query_similar",
    "clear_collection",
    "get_chunk_count",
]
