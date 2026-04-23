"""
config.py — Centralized configuration for the RAG Support Assistant.

All settings are loaded from the .env file (or environment variables).
Import this module anywhere in the project to access settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file if present


# ─── LLM Settings ───────────────────────────────────────────────────────────

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "gemini"

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ─── Embedding Settings ──────────────────────────────────────────────────────

EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ─── ChromaDB Settings ───────────────────────────────────────────────────────

CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "support_docs")

# ─── Chunking Settings ───────────────────────────────────────────────────────

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ─── Retrieval Settings ──────────────────────────────────────────────────────

TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

# ─── HITL / Escalation Settings ─────────────────────────────────────────────

ESCALATION_LOG_DIR: str = os.getenv("ESCALATION_LOG_DIR", "./escalation_logs")

ESCALATION_KEYWORDS: list[str] = [
    "billing",
    "refund",
    "cancel",
    "legal",
    "account deletion",
    "fraud",
    "complaint",
    "lawsuit",
    "chargeback",
    "dispute",
]

UNCERTAINTY_PHRASES: list[str] = [
    "i don't know",
    "i'm not sure",
    "i cannot find",
    "not mentioned in the document",
    "please contact support",
    "not available in",
    "no information",
]
