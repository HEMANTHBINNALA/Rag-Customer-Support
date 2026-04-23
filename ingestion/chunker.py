"""
chunker.py — Splits cleaned page text into overlapping chunks.

Uses LangChain's RecursiveCharacterTextSplitter which tries to split
on paragraph breaks → sentence breaks → word breaks in that order,
preserving semantic coherence as much as possible.
"""

import uuid
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """
    Split a list of cleaned page dicts into overlapping text chunks.

    Each chunk dict carries full provenance metadata so we can trace
    every retrieved piece of text back to its exact source and page.

    Args:
        pages: List of {"page_number", "text", "source_doc"} dicts
               (output of text_cleaner.clean_pages).

    Returns:
        List of chunk dicts:
            {
                "chunk_id":    str   — unique UUID for this chunk,
                "text":        str   — the chunk content,
                "source_doc":  str   — PDF filename,
                "page_number": int   — original page,
                "chunk_index": int   — position within the whole doc,
                "token_count": int   — approximate word-level token count,
            }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[Dict] = []
    global_index = 0

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            all_chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk_text.strip(),
                    "source_doc": page["source_doc"],
                    "page_number": page["page_number"],
                    "chunk_index": global_index,
                    "token_count": len(chunk_text.split()),
                }
            )
            global_index += 1

    print(
        f"[chunker] Created {len(all_chunks)} chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return all_chunks
