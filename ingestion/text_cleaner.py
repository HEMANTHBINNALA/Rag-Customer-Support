"""
text_cleaner.py — Removes noise from extracted PDF text.

Strips headers, footers, page number artifacts, and excess whitespace
so that the chunker works with clean, meaningful content.
"""

import re
from typing import List, Dict


# Patterns to aggressively strip from extracted text
_NOISE_PATTERNS = [
    r"^\s*\d+\s*$",               # Standalone page numbers
    r"^\s*page\s+\d+\s*$",        # "Page 3" lines
    r"^\s*confidential\s*$",      # Common watermark text
    r"^\s*all rights reserved.*$", # Copyright boilerplate
    r"-{3,}",                      # Long dashes / dividers
    r"={3,}",                      # Repeated equals signs
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _NOISE_PATTERNS]


def clean_text(raw_text: str) -> str:
    """
    Clean a single string of raw PDF text.

    Steps:
    1. Remove lines that match noise patterns
    2. Collapse multiple blank lines into one
    3. Strip leading/trailing whitespace

    Args:
        raw_text: Raw string extracted from a PDF page.

    Returns:
        Cleaned string with noise removed.
    """
    text = raw_text

    # Apply each noise-removal pattern
    for pattern in _COMPILED_PATTERNS:
        text = pattern.sub("", text)

    # Collapse multiple newlines into at most 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def clean_pages(pages: List[Dict]) -> List[Dict]:
    """
    Clean a list of page dicts returned by pdf_loader.load_pdf().

    Skips pages where the cleaned text is empty (e.g., image-only pages).

    Args:
        pages: List of {"page_number", "text", "source_doc"} dicts.

    Returns:
        Filtered and cleaned list of page dicts.
    """
    cleaned: List[Dict] = []
    for page in pages:
        clean = clean_text(page["text"])
        if clean:  # Skip empty pages
            cleaned.append(
                {
                    "page_number": page["page_number"],
                    "text": clean,
                    "source_doc": page["source_doc"],
                }
            )

    skipped = len(pages) - len(cleaned)
    print(
        f"[text_cleaner] Cleaned {len(cleaned)} pages "
        f"({skipped} empty/noise-only pages skipped)"
    )
    return cleaned
