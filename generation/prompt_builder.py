"""
prompt_builder.py — Constructs the system and user prompts for the LLM.

The prompt is carefully engineered to:
  1. Ground the LLM strictly in the retrieved context (no hallucination)
  2. Tell it to admit when the answer isn't in the context
  3. Keep answers concise and support-appropriate in tone
"""

SYSTEM_PROMPT = """You are a helpful, professional customer support assistant.
Your job is to answer user questions based ONLY on the context provided below.

Rules you must follow:
- Only use information from the provided context to answer.
- If the answer is not clearly present in the context, respond with:
  "I'm not sure based on the available documentation. Please contact our support team."
- Do NOT fabricate information or draw from general knowledge.
- Keep your answer concise, clear, and professional (2–5 sentences max unless detail is necessary).
- If the context contains relevant page numbers, mention them in your answer.
"""


def build_prompt(query: str, context_str: str) -> list[dict]:
    """
    Build the messages list for the LLM chat API.

    Args:
        query:       The user's question.
        context_str: Pre-formatted context string from retriever.format_context().

    Returns:
        A list of message dicts in OpenAI/Groq/Gemini chat format:
            [
                {"role": "system", "content": "..."},
                {"role": "user",   "content": "..."},
            ]
    """
    user_message = f"""Context from documentation:
{context_str}

---
User question: {query}

Please answer based strictly on the context above."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def build_no_context_prompt(query: str) -> list[dict]:
    """
    Build a prompt for the case where retrieval returned no usable context.
    Instructs the LLM to gracefully admit it cannot help.

    Args:
        query: The user's question.

    Returns:
        Messages list ready to send to the LLM.
    """
    user_message = (
        f"The user asked: '{query}'\n\n"
        "No relevant documentation was found to answer this question. "
        "Politely tell the user you cannot find the answer in the available documentation "
        "and suggest they contact the support team directly."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
