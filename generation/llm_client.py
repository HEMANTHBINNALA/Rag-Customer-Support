"""
llm_client.py — Unified LLM client supporting Groq and Google Gemini.

Selects the provider at runtime based on config.LLM_PROVIDER.
Both providers use the same interface: send messages, get a string response.

Retry logic: one automatic retry with 2s backoff on timeout/connection errors.
"""

import time
from typing import List, Dict
from config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
)


def _call_groq(messages: List[Dict]) -> str:
    """Call the Groq API using the langchain-groq wrapper."""
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1024,
    )
    response = llm.invoke(messages)
    return response.content.strip()


def _call_gemini(messages: List[Dict]) -> str:
    """Call Google Gemini using the langchain-google-genai wrapper."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        max_output_tokens=1024,
    )
    response = llm.invoke(messages)
    return response.content.strip()


def call_llm(messages: List[Dict], retries: int = 1) -> str:
    """
    Send a prompt to the configured LLM and return the response string.

    Args:
        messages: Chat messages list (from prompt_builder).
        retries:  Number of retry attempts on failure (default 1).

    Returns:
        LLM response as a plain string.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    provider = LLM_PROVIDER.lower()
    last_error = None

    for attempt in range(retries + 1):
        try:
            if provider == "groq":
                return _call_groq(messages)
            elif provider == "gemini":
                return _call_gemini(messages)
            else:
                raise ValueError(
                    f"Unknown LLM_PROVIDER '{provider}'. Use 'groq' or 'gemini'."
                )
        except Exception as exc:
            last_error = exc
            print(
                f"[llm_client] Attempt {attempt + 1} failed: {exc}. "
                f"{'Retrying in 2s...' if attempt < retries else 'No more retries.'}"
            )
            if attempt < retries:
                time.sleep(2)

    raise RuntimeError(
        f"LLM call failed after {retries + 1} attempts. Last error: {last_error}"
    )
