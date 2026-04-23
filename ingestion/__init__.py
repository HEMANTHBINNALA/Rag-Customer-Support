"""ingestion package — PDF loading, CSV loading, cleaning, and chunking."""
from .pdf_loader import load_pdf
from .text_cleaner import clean_pages
from .chunker import chunk_pages
from .csv_loader import load_single_qna_csv, load_multi_csv, load_csv_auto

__all__ = [
    "load_pdf",
    "clean_pages",
    "chunk_pages",
    "load_single_qna_csv",
    "load_multi_csv",
    "load_csv_auto",
]
