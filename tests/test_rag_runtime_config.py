import pytest

from framevitals.rag_index import (
    Fact,
    _embedding_concurrency,
    _rag_backend_override,
    retrieve,
)


def test_rag_prefers_framevitals_environment_names(monkeypatch):
    monkeypatch.setenv("DATALENS_RAG_BACKEND", "ollama")
    monkeypatch.setenv("FRAMEVITALS_RAG_BACKEND", "tfidf")
    assert _rag_backend_override() == "tfidf"


def test_rag_legacy_environment_names_remain_compatible(monkeypatch):
    monkeypatch.delenv("FRAMEVITALS_RAG_BACKEND", raising=False)
    monkeypatch.setenv("DATALENS_RAG_BACKEND", "tfidf")
    assert _rag_backend_override() == "tfidf"


def test_invalid_embedding_concurrency_cannot_break_import_or_execution(monkeypatch):
    monkeypatch.setenv("FRAMEVITALS_RAG_EMBED_CONCURRENCY", "not-an-int")
    assert _embedding_concurrency() == 8

    monkeypatch.setenv("FRAMEVITALS_RAG_EMBED_CONCURRENCY", "100000")
    assert _embedding_concurrency() == 32


def test_retrieve_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k"):
        retrieve("question", [Fact(path="x", text="x: 1")], k=0)
