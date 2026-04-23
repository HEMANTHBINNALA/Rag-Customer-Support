"""
csv_loader.py — Loads Q&A data from CSV files and converts them into
the same page-dict format that the rest of the ingestion pipeline expects.

Supports three CSV formats found in the archive:

1. single_qna.csv  — columns: Question, Answer, Category, Asin
   Each row becomes one "page" containing a formatted Q&A block.

2. multi_questions.csv + multi_answers.csv — joined on QuestionID.
   All answers for a question are combined into one "page".

3. Generic fallback — auto-detects question/answer columns by name.
"""

import csv
import os
from typing import List, Dict, Optional


# ── Known column mappings ────────────────────────────────────────────────────
_SINGLE_QNA_COLS = {
    "question": "Question",
    "answer": "Answer",
    "category": "Category",
    "id": "Asin",
}

_MULTI_Q_COLS = {
    "id": "QuestionID",
    "question": "QuestionText",
    "category": "Category",
}

_MULTI_A_COLS = {
    "id": "QuestionID",
    "answer": "AnswerText",
    "score": "AnswerScore",
}

# Generic fallback: try these column names (case-insensitive)
_GENERIC_QUESTION_NAMES = ["question", "questiontext", "q", "query", "title"]
_GENERIC_ANSWER_NAMES   = ["answer", "answertext", "a", "response", "body", "text"]


def _safe_read_csv(filepath: str, max_rows: Optional[int] = None) -> List[Dict]:
    """Read a CSV robustly, handling encoding issues."""
    rows = []
    with open(filepath, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            rows.append(row)
    return rows


def _detect_columns(fieldnames: List[str]):
    """Auto-detect question/answer columns from fieldnames."""
    lower_map = {col.lower().replace("_", ""): col for col in fieldnames}

    q_col = next(
        (lower_map[k] for k in _GENERIC_QUESTION_NAMES if k in lower_map), None
    )
    a_col = next(
        (lower_map[k] for k in _GENERIC_ANSWER_NAMES if k in lower_map), None
    )
    cat_col = lower_map.get("category", None)

    return q_col, a_col, cat_col


def load_single_qna_csv(filepath: str, max_rows: Optional[int] = None) -> List[Dict]:
    """
    Load single_qna.csv (Question + Answer per row).

    Args:
        filepath: Path to single_qna.csv.
        max_rows: Cap at this many rows (useful for large files).

    Returns:
        List of page dicts: {page_number, text, source_doc}
    """
    rows = _safe_read_csv(filepath, max_rows)
    fieldnames = list(rows[0].keys()) if rows else []

    # Try known columns first, fall back to auto-detect
    if _SINGLE_QNA_COLS["question"] in fieldnames:
        q_col   = _SINGLE_QNA_COLS["question"]
        a_col   = _SINGLE_QNA_COLS["answer"]
        cat_col = _SINGLE_QNA_COLS["category"]
    else:
        q_col, a_col, cat_col = _detect_columns(fieldnames)

    if not q_col or not a_col:
        raise ValueError(
            f"Cannot find question/answer columns in {filepath}. "
            f"Found: {fieldnames}"
        )

    source_name = os.path.basename(filepath)
    pages = []

    for i, row in enumerate(rows):
        question = (row.get(q_col) or "").strip()
        answer   = (row.get(a_col) or "").strip()
        category = (row.get(cat_col) or "General").strip() if cat_col else "General"

        if not question or not answer:
            continue  # Skip incomplete rows

        text = (
            f"Category: {category}\n"
            f"Q: {question}\n"
            f"A: {answer}"
        )

        pages.append(
            {
                "page_number": i + 1,
                "text": text,
                "source_doc": source_name,
            }
        )

    print(
        f"[csv_loader] single_qna: loaded {len(pages)} Q&A pairs "
        f"from '{source_name}'"
    )
    return pages


def load_multi_csv(
    questions_filepath: str,
    answers_filepath: str,
    max_questions: Optional[int] = None,
) -> List[Dict]:
    """
    Load multi_questions.csv + multi_answers.csv, joining on QuestionID.

    All answers for a question are combined into one page so the LLM
    can synthesize a comprehensive answer from multiple perspectives.

    Args:
        questions_filepath: Path to multi_questions.csv.
        answers_filepath:   Path to multi_answers.csv.
        max_questions:      Cap at this many questions.

    Returns:
        List of page dicts.
    """
    # Load questions
    q_rows = _safe_read_csv(questions_filepath, max_questions)
    q_map: Dict[str, Dict] = {}
    for row in q_rows:
        qid = row.get(_MULTI_Q_COLS["id"], "").strip()
        if qid:
            q_map[qid] = {
                "question": row.get(_MULTI_Q_COLS["question"], "").strip(),
                "category": row.get(_MULTI_Q_COLS["category"], "General").strip(),
            }

    # Load answers — build qid → [answers] map
    print(f"[csv_loader] Loading answers from {os.path.basename(answers_filepath)}...")
    a_map: Dict[str, List[str]] = {}
    with open(answers_filepath, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get(_MULTI_A_COLS["id"], "").strip()
            if qid not in q_map:
                continue  # Only keep answers for loaded questions
            answer = row.get(_MULTI_A_COLS["answer"], "").strip()
            if answer:
                a_map.setdefault(qid, []).append(answer)

    # Merge into pages
    source_name = (
        os.path.basename(questions_filepath).replace("questions", "qa")
    )
    pages = []

    for i, (qid, q_data) in enumerate(q_map.items()):
        question = q_data["question"]
        category = q_data["category"]
        answers  = a_map.get(qid, [])

        if not question or not answers:
            continue

        # Combine multiple answers
        answers_text = "\n".join(
            f"  - {ans}" for ans in answers[:5]  # Cap at 5 answers per question
        )
        text = (
            f"Category: {category}\n"
            f"Q: {question}\n"
            f"A:\n{answers_text}"
        )

        pages.append(
            {
                "page_number": i + 1,
                "text": text,
                "source_doc": source_name,
            }
        )

    print(
        f"[csv_loader] multi Q&A: loaded {len(pages)} questions with answers "
        f"(joined from {os.path.basename(questions_filepath)} + "
        f"{os.path.basename(answers_filepath)})"
    )
    return pages


def load_csv_auto(filepath: str, max_rows: Optional[int] = None) -> List[Dict]:
    """
    Auto-detect CSV format and load accordingly.

    Falls back to generic column detection for unknown CSV schemas.

    Args:
        filepath: Path to any CSV file.
        max_rows: Row cap.

    Returns:
        List of page dicts.
    """
    rows = _safe_read_csv(filepath, max_rows=5)  # Peek at columns
    if not rows:
        raise ValueError(f"CSV file is empty: {filepath}")

    fieldnames = list(rows[0].keys())

    # Known format detection
    if _SINGLE_QNA_COLS["question"] in fieldnames and _SINGLE_QNA_COLS["answer"] in fieldnames:
        return load_single_qna_csv(filepath, max_rows)

    # Generic fallback
    q_col, a_col, cat_col = _detect_columns(fieldnames)
    if not q_col or not a_col:
        raise ValueError(
            f"Could not auto-detect question/answer columns in '{filepath}'. "
            f"Columns found: {fieldnames}"
        )

    # Re-load all rows and convert
    all_rows = _safe_read_csv(filepath, max_rows)
    source_name = os.path.basename(filepath)
    pages = []

    for i, row in enumerate(all_rows):
        question = (row.get(q_col) or "").strip()
        answer   = (row.get(a_col) or "").strip()
        category = (row.get(cat_col) or "General").strip() if cat_col else "General"

        if not question or not answer:
            continue

        text = f"Category: {category}\nQ: {question}\nA: {answer}"
        pages.append({"page_number": i + 1, "text": text, "source_doc": source_name})

    print(f"[csv_loader] generic: loaded {len(pages)} rows from '{source_name}'")
    return pages
