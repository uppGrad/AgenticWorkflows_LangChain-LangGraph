from uppgrad_agentic.common.llm import get_search_provider


def test_returns_none_when_no_provider_configured(monkeypatch):
    monkeypatch.delenv("UPPGRAD_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    assert get_search_provider() is None


def test_returns_brave_provider_when_configured(monkeypatch):
    monkeypatch.setenv("UPPGRAD_SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    p = get_search_provider()
    assert p is not None
    assert p.__class__.__name__ == "BraveSearchProvider"


def test_returns_none_when_brave_key_missing(monkeypatch):
    monkeypatch.setenv("UPPGRAD_SEARCH_PROVIDER", "brave")
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    assert get_search_provider() is None


def test_unknown_provider_returns_none(monkeypatch):
    monkeypatch.setenv("UPPGRAD_SEARCH_PROVIDER", "google")
    assert get_search_provider() is None
