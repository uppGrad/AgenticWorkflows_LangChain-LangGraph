import httpx
import pytest
import respx

from uppgrad_agentic.tools.search import (
    BraveSearchProvider, SearchResult, SearchProvider,
)


def test_search_result_model():
    r = SearchResult(url="https://x.com/job/1", title="Engineer", snippet="...")
    assert r.url == "https://x.com/job/1"


def test_search_provider_is_abstract():
    with pytest.raises(TypeError):
        SearchProvider()


@respx.mock
def test_brave_provider_returns_results():
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {"url": "https://boards.greenhouse.io/acme/jobs/1",
                         "title": "SWE @ Acme", "description": "Apply now"},
                        {"url": "https://example.com/2", "title": "Other", "description": "..."},
                    ]
                }
            },
        )
    )
    provider = BraveSearchProvider(api_key="test")
    results = provider.search('"SWE" "Acme"', count=3)
    assert len(results) == 2
    assert results[0].url == "https://boards.greenhouse.io/acme/jobs/1"


@respx.mock
def test_brave_provider_returns_empty_on_429():
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        return_value=httpx.Response(429, json={"error": "rate limited"})
    )
    assert BraveSearchProvider(api_key="test").search('"x"', count=3) == []


@respx.mock
def test_brave_provider_returns_empty_on_network_error():
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        side_effect=httpx.ConnectError("boom")
    )
    assert BraveSearchProvider(api_key="test").search('"x"', count=3) == []


@respx.mock
def test_brave_provider_truncates_to_count():
    respx.get("https://api.search.brave.com/res/v1/web/search").mock(
        return_value=httpx.Response(200, json={
            "web": {"results": [{"url": f"https://x.com/{i}", "title": str(i), "description": ""}
                                for i in range(10)]}
        })
    )
    assert len(BraveSearchProvider(api_key="test").search('"x"', count=3)) == 3
