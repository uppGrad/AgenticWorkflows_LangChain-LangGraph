# src/uppgrad_agentic/common/llm.py
from __future__ import annotations

import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel


def get_llm() -> Optional[BaseChatModel]:
    """
    Returns a chat model if environment variables are configured.
    If not configured, returns None (workflow can use heuristic fallback).
    """
    provider = os.getenv("UPPGRAD_LLM_PROVIDER", "").lower().strip()

    # Example: OpenAI via langchain-openai
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:
            raise RuntimeError("langchain-openai not installed, but UPPGRAD_LLM_PROVIDER=openai.") from e

        model = os.getenv("UPPGRAD_OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)

    # Add more providers later (anthropic, azure, etc.)
    return None
