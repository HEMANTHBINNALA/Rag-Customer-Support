"""
ingest_pipeline.py — Unified ingestion pipeline for PDF and CSV sources.

Supported input types:
  --pdf   path/to/doc.pdf           → PDF ingestion (single file)
  --csv   path/to/single_qna.csv   → single-turn Q&A CSV
  --csv   path/to/multi_questions.csv  (must also pass --answers)
  --answers path/to/multi_answers.csv  → multi-turn Q&A join

Optional flags:
  --clear       Wipe the existing ChromaDB collection first.
  --max-rows N  Only ingest the first N rows (useful for large CSVs).
                Default: 5000 rows (about 50 MB of content — safe and fast).
  --category X  Only ingest rows where Category == X (CSV only).

Usage examples:
    # Ingest a PDF
    python ingest_pipeline.py --pdf docs/faq.pdf

    # Ingest single_qna.csv (first 5000 rows)
    python ingest_pipeline.py --csv "C:/Users/LENOVO/Downloads/archive/single_qna.csv"

    # Ingest single_qna.csv, Automotive category only
    python ingest_pipeline.py --csv "C:/Users/LENOVO/Downloads/archive/single_qna.csv" --category Automotive

    # Ingest multi Q&A (joined)
    python ingest_pipeline.py \\
        --csv "C:/Users/LENOVO/Downloads/archive/multi_questions.csv" \\
        --answers "C:/Users/LENOVO/Downloads/archive/multi_answers.csv" \\
        --max-rows 3000 --clear
"""

import argparse
import sys
import time
from pathlib import Path

from ingestion.text_cleaner import clean_pages
from ingestion.chunker import chunk_pages
from embeddings.embed_model import embed_texts
from embeddings.vector_store import add_chunks, clear_collection, get_chunk_count


# ─── PDF Ingestion ────────────────────────────────────────────────────────────

def _ingest_pdf(pdf_path: str) -> list:
    from ingestion.pdf_loader import load_pdf
    print(f"\n[Step 1] Loading PDF: {pdf_path}")
    pages = load_pdf(pdf_path)
    print("\n[Step 2] Cleaning extracted text...")
    return clean_pages(pages)


# ─── CSV Ingestion ────────────────────────────────────────────────────────────

def _ingest_csv_single(csv_path: str, max_rows: int, category_filter: str = None) -> list:
    from ingestion.csv_loader import load_single_qna_csv

    print(f"\n[Step 1] Loading CSV: {csv_path}")
    print(f"         Mode: single Q&A | Max rows: {max_rows}")
    if category_filter:
        print(f"         Category filter: '{category_filter}'")

    pages = load_single_qna_csv(csv_path, max_rows=max_rows)

    # Apply category filter if requested
    if category_filter:
        before = len(pages)
        pages = [
            p for p in pages
            if category_filter.lower() in p["text"].lower().split("\n")[0].lower()
        ]
        print(f"         Category filter: {before} → {len(pages)} rows kept")

    return pages  # CSV pages are already clean — skip text_cleaner


def _ingest_csv_multi(questions_path: str, answers_path: str, max_rows: int) -> list:
    from ingestion.csv_loader import load_multi_csv

    print(f"\n[Step 1] Loading multi Q&A CSV pair:")
    print(f"         Questions : {questions_path}")
    print(f"         Answers   : {answers_path}")
    print(f"         Max questions: {max_rows}")

    pages = load_multi_csv(questions_path, answers_path, max_questions=max_rows)
    return pages


# ─── Shared Pipeline Steps ────────────────────────────────────────────────────

def _embed_and_store(pages: list, source_type: str) -> None:
    """Steps 3–5: chunk → embed → store."""

    if not pages:
        print("❌ ERROR: No usable content found. Aborting.")
        sys.exit(1)

    # For CSV, pages are already clean formatted strings — chunk them
    print(f"\n[Step 3] Splitting into chunks...")
    chunks = chunk_pages(pages)

    print(f"\n[Step 4] Generating embeddings for {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    print(f"\n[Step 5] Storing in ChromaDB...")
    add_chunks(chunks, embeddings)


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def run_ingestion(
    pdf_path: str = None,
    csv_path: str = None,
    answers_path: str = None,
    clear: bool = False,
    max_rows: int = 5000,
    category_filter: str = None,
) -> None:
    start = time.time()

    print("\n" + "=" * 65)
    print("  RAG Support Assistant — Ingestion Pipeline")
    print("=" * 65)

    # ── Step 0: Optional clear ──────────────────────────────────────
    if clear:
        print("\n[Step 0] Clearing existing ChromaDB collection...")
        clear_collection()

    # ── Dispatch by source type ─────────────────────────────────────
    if pdf_path:
        pages = _ingest_pdf(pdf_path)
        _embed_and_store(pages, source_type="pdf")

    elif csv_path and answers_path:
        # Multi Q&A mode (questions + answers CSVs)
        pages = _ingest_csv_multi(csv_path, answers_path, max_rows)
        _embed_and_store(pages, source_type="csv-multi")

    elif csv_path:
        # Single Q&A mode
        pages = _ingest_csv_single(csv_path, max_rows, category_filter)
        _embed_and_store(pages, source_type="csv-single")

    else:
        print("❌ ERROR: Must provide --pdf or --csv.")
        sys.exit(1)

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - start
    total = get_chunk_count()

    print("\n" + "=" * 65)
    print("  [DONE] Ingestion Complete!")
    print(f"  Total chunks in ChromaDB : {total}")
    print(f"  Time elapsed             : {elapsed:.1f}s")
    print("=" * 65 + "\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PDF or CSV data into the RAG ChromaDB vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pdf", type=str, help="Path to a PDF file.")
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to a CSV file (single Q&A or multi_questions.csv).",
    )
    parser.add_argument(
        "--answers",
        type=str,
        default=None,
        help="Path to multi_answers.csv (required when using multi Q&A mode).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        default=False,
        help="Clear existing ChromaDB data before ingestion.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="Maximum number of rows/pages to ingest (default: 5000).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter CSV rows by category name (e.g. 'Automotive').",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.pdf and not args.csv:
        parser.error("Must specify either --pdf or --csv.")

    if args.pdf and not Path(args.pdf).exists():
        print(f"❌ PDF not found: {args.pdf}")
        sys.exit(1)

    if args.csv and not Path(args.csv).exists():
        print(f"❌ CSV not found: {args.csv}")
        sys.exit(1)

    if args.answers and not Path(args.answers).exists():
        print(f"❌ Answers CSV not found: {args.answers}")
        sys.exit(1)

    run_ingestion(
        pdf_path=args.pdf,
        csv_path=args.csv,
        answers_path=args.answers,
        clear=args.clear,
        max_rows=args.max_rows,
        category_filter=args.category,
    )
