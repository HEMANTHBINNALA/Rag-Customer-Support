"""
escalation.py — HITL (Human-in-the-Loop) escalation logic.

When the RAG system cannot answer a query with sufficient confidence
(or detects a sensitive topic), this module:
  1. Logs the full escalation context to a timestamped JSON file
  2. Prints a summary for immediate visibility
  3. (Extension point) Can send notifications to a real agent queue

Escalation log format (one JSON file per event):
    escalation_logs/
        2026-04-22_11-30-00_abc123.json
"""

import json
import uuid
import os
from datetime import datetime
from typing import List, Dict
from config import ESCALATION_LOG_DIR


def log_escalation(
    query: str,
    reason: str,
    retrieved_chunks: List[Dict],
    llm_response: str = "",
) -> str:
    """
    Log an escalation event to disk and return the log file path.

    Args:
        query:            The user's original question.
        reason:           Why the system is escalating (from evaluate/validate node).
        retrieved_chunks: What was retrieved (may be empty).
        llm_response:     Any partial LLM output (may be empty string).

    Returns:
        Path to the written log file.
    """
    # Ensure the log directory exists
    os.makedirs(ESCALATION_LOG_DIR, exist_ok=True)

    # Build the escalation record
    event_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{event_id}.json"
    filepath = os.path.join(ESCALATION_LOG_DIR, filename)

    record = {
        "event_id": event_id,
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "escalation_reason": reason,
        "retrieved_chunks_count": len(retrieved_chunks),
        "retrieved_chunks": [
            {
                "text": c.get("text", "")[:300],  # Truncate for readability
                "source_doc": c.get("source_doc", ""),
                "page_number": c.get("page_number", -1),
                "score": c.get("score", 0.0),
            }
            for c in retrieved_chunks
        ],
        "llm_response_preview": llm_response[:500] if llm_response else "",
        "status": "pending",  # Can be updated to "resolved" by the agent
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    # Console summary for immediate visibility
    print("\n" + "=" * 60)
    print("🚨 ESCALATION LOGGED")
    print(f"   Event ID  : {event_id}")
    print(f"   Timestamp : {record['timestamp']}")
    print(f"   Query     : {query[:80]}")
    print(f"   Reason    : {reason}")
    print(f"   Log file  : {filepath}")
    print("=" * 60 + "\n")

    return filepath


def get_all_escalations() -> List[Dict]:
    """
    Load and return all escalation log records from disk.

    Returns:
        List of escalation record dicts, sorted by timestamp (newest first).
    """
    if not os.path.exists(ESCALATION_LOG_DIR):
        return []

    records = []
    for fname in os.listdir(ESCALATION_LOG_DIR):
        if fname.endswith(".json"):
            fpath = os.path.join(ESCALATION_LOG_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    records.append(json.load(f))
            except Exception:
                pass  # Skip corrupted files

    # Sort newest first
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records


def mark_resolved(event_id: str) -> bool:
    """
    Mark an escalation event as resolved by a human agent.

    Args:
        event_id: The 8-char event ID from the log filename.

    Returns:
        True if found and updated, False otherwise.
    """
    if not os.path.exists(ESCALATION_LOG_DIR):
        return False

    for fname in os.listdir(ESCALATION_LOG_DIR):
        if event_id in fname and fname.endswith(".json"):
            fpath = os.path.join(ESCALATION_LOG_DIR, fname)
            with open(fpath, "r+", encoding="utf-8") as f:
                record = json.load(f)
                record["status"] = "resolved"
                record["resolved_at"] = datetime.now().isoformat()
                f.seek(0)
                json.dump(record, f, indent=2)
                f.truncate()
            print(f"[escalation] Event {event_id} marked as resolved.")
            return True

    return False
