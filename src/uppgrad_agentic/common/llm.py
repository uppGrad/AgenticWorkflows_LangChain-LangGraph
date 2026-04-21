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
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
            logger.info("Auto-detected LLM provider: openai (OPENAI_API_KEY found)")
        else:
            logger.debug("No LLM provider configured and no API keys found — using heuristic fallback")
            return None

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            logger.error("langchain-openai not installed but provider=openai: %s", e)
            return None

        model = os.getenv("UPPGRAD_OPENAI_MODEL", "gpt-4o-mini")
        logger.info("Using OpenAI LLM: model=%s", model)
        return ChatOpenAI(model=model, temperature=0)

    # Add more providers later (anthropic, azure, etc.)
    logger.warning("Unknown LLM provider: %s — returning None", provider)
    return None
