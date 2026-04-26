# src/uppgrad_agentic/common/llm.py
from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


def get_llm() -> Optional[BaseChatModel]:
    """
    Returns a chat model if environment variables are configured.
    If not configured, returns None (workflow can use heuristic fallback).

    Provider selection:
      1. Explicit: UPPGRAD_LLM_PROVIDER=openai  →  use OpenAI
      2. Auto-detect: OPENAI_API_KEY is set      →  use OpenAI
      3. Otherwise: return None (heuristic mode)
    """
    provider = os.getenv("UPPGRAD_LLM_PROVIDER", "").lower().strip()

    # Auto-detect provider from available API keys
    if not provider:
        key = os.getenv("OPENAI_API_KEY")
        if key:
            provider = "openai"
            logger.info("Auto-detected LLM provider: openai (OPENAI_API_KEY found, length=%d)", len(key))
        else:
            # Log ALL env var keys to help debug
            all_keys = sorted(os.environ.keys())
            logger.warning(
                "No LLM provider configured. OPENAI_API_KEY not found in env. "
                "Available env vars: %s",
                ", ".join(all_keys),
            )
            return None

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            logger.info("langchain_openai imported successfully")
        except ImportError as e:
            logger.error("langchain-openai not installed but provider=openai: %s", e)
            return None

        model = os.getenv("UPPGRAD_OPENAI_MODEL", "gpt-5.4-mini")
        logger.info("Using OpenAI LLM: model=%s", model)
        return ChatOpenAI(model=model, temperature=0)

    # Add more providers later (anthropic, azure, etc.)
    logger.warning("Unknown LLM provider: %s — returning None", provider)
    return None


def get_search_provider():
    """Return a SearchProvider instance or None if not configured.

    Mirrors get_llm() opt-in pattern. Callers MUST handle None by returning
    a degraded result, never by raising.
    """
    provider_name = os.getenv("UPPGRAD_SEARCH_PROVIDER", "").lower()
    if provider_name != "brave":
        return None
    api_key = os.getenv("BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        return None
    from uppgrad_agentic.tools.search import BraveSearchProvider
    return BraveSearchProvider(api_key=api_key)
