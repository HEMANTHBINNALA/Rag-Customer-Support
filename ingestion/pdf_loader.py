"""
pdf_loader.py — Extracts raw text from PDF files using PyMuPDF.

Handles multi-page PDFs and returns per-page text with metadata.
Assumption: PDFs are text-based (not scanned images).
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict


def load_pdf(pdf_path: str) -> List[Dict]:
    """
    Load a PDF file and extract text page by page.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of dicts, one per page:
            {
                "page_number": int,
                "text": str,
                "source_doc": str
            }

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If PyMuPDF fails to open the document.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {exc}") from exc

    pages: List[Dict] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # plain text extraction
        pages.append(
            {
                "page_number": page_num + 1,  # 1-indexed
                "text": text,
                "source_doc": path.name,
            }
        )

    doc.close()
    print(f"[pdf_loader] Loaded {len(pages)} pages from '{path.name}'")
    return pages
