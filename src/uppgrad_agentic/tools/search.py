from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    url: str = Field(...)
    title: str = Field(default="")
    snippet: str = Field(default="")


class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, count: int = 3) -> List[SearchResult]:
        """Run a web search; return at most `count` results. Never raises."""
        raise NotImplementedError


class BraveSearchProvider(SearchProvider):
    _ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
    _TIMEOUT = 10.0

    def __init__(self, api_key: str):
        self._api_key = api_key

    def search(self, query: str, count: int = 3) -> List[SearchResult]:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        params = {"q": query, "count": min(count, 20)}
        try:
            resp = httpx.get(self._ENDPOINT, headers=headers, params=params, timeout=self._TIMEOUT)
        except httpx.HTTPError as exc:
            logger.warning("brave search: network error — %s", exc)
            return []
        if resp.status_code != 200:
            logger.warning("brave search: HTTP %s — %s", resp.status_code, resp.text[:200])
            return []
        try:
            payload = resp.json()
        except ValueError:
            logger.warning("brave search: non-JSON response")
            return []
        web_results = (payload.get("web") or {}).get("results") or []
        out: List[SearchResult] = []
        for item in web_results[:count]:
            url = item.get("url") or ""
            if not url:
                continue
            out.append(SearchResult(
                url=url,
                title=item.get("title") or "",
                snippet=item.get("description") or "",
            ))
        return out
