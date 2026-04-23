"""
main.py — Interactive CLI for the RAG Customer Support Assistant.

Run after ingestion is complete:
    python main.py

Supports:
  - Multi-turn conversation loop
  - Debug mode (shows retrieved chunks and scores)
  - Clean exit with Ctrl+C or typing 'quit'
"""

import sys
from graph.workflow import run_query
from embeddings.vector_store import get_chunk_count


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       RAG Customer Support Assistant (LangGraph + ChromaDB)  ║
║       Type your question below. Type 'quit' to exit.         ║
╚══════════════════════════════════════════════════════════════╝
"""

DEBUG_HELP = "  Tip: Type 'debug on' / 'debug off' to toggle retrieval details.\n"


def print_state_debug(state: dict) -> None:
    """Print retrieval details for debugging."""
    print("\n── DEBUG INFO ─────────────────────────────────────────────")
    print(f"  Confidence score : {state.get('confidence_score', 0):.4f}")
    print(f"  Needs escalation : {state.get('needs_escalation', False)}")
    if state.get("escalation_reason"):
        print(f"  Escalation reason: {state['escalation_reason']}")
    chunks = state.get("retrieved_chunks", [])
    if chunks:
        print(f"  Retrieved chunks : {len(chunks)}")
        for i, c in enumerate(chunks, 1):
            print(
                f"    [{i}] score={c['score']:.4f} | "
                f"{c['source_doc']} p.{c['page_number']} | "
                f"{c['text'][:80].strip()}..."
            )
    print("────────────────────────────────────────────────────────────\n")


def main():
    print(BANNER)
    print(DEBUG_HELP)

    # Check ChromaDB has data
    chunk_count = get_chunk_count()
    if chunk_count == 0:
        print(
            "⚠️  WARNING: ChromaDB is empty. Run ingestion first:\n"
            "   python ingest_pipeline.py --pdf path/to/document.pdf\n"
        )
    else:
        print(f"✅ ChromaDB ready — {chunk_count} chunks loaded.\n")

    debug_mode = False

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ("quit", "exit", "q"):
                print("\nGoodbye! 👋\n")
                break
            elif user_input.lower() == "debug on":
                debug_mode = True
                print("  [Debug mode ON]\n")
                continue
            elif user_input.lower() == "debug off":
                debug_mode = False
                print("  [Debug mode OFF]\n")
                continue

            # Run through the RAG graph
            print("\nAssistant: (thinking...)")
            try:
                final_state = run_query(user_input)
            except Exception as exc:
                print(f"\n❌ Unexpected error: {exc}\n")
                continue

            answer = final_state.get("final_answer", "No response generated.")
            print(f"\nAssistant:\n{answer}\n")

            if debug_mode:
                print_state_debug(final_state)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye! 👋\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
