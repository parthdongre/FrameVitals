"""
RAG Fact Index
==============
Flatten an analysis result into atomic facts, then retrieve the top-k facts
relevant to a free-form question.

Two retrieval backends:
    1. Ollama embeddings (``nomic-embed-text`` by default) when reachable.
    2. TF-IDF cosine similarity (sklearn) as the always-available fallback.

The retrieval API stays identical regardless of backend, so the agent layer
never has to care which one is in play.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


_DEFAULT_EMBED_MODEL = "nomic-embed-text"
_DEFAULT_EMBED_CONCURRENCY = 8
_MAX_EMBED_CONCURRENCY = 32


def _environment_value(
    name: str,
    *,
    legacy_name: str | None = None,
    default: str = "",
) -> str:
    """Read a FrameVitals setting with an optional 0.x compatibility alias."""
    value = os.environ.get(name)
    if value is not None:
        return value
    if legacy_name is not None:
        legacy_value = os.environ.get(legacy_name)
        if legacy_value is not None:
            return legacy_value
    return default


def _embedding_model() -> str:
    value = _environment_value(
        "FRAMEVITALS_OLLAMA_EMBED_MODEL",
        legacy_name="OLLAMA_EMBED_MODEL",
        default=_DEFAULT_EMBED_MODEL,
    ).strip()
    return value or _DEFAULT_EMBED_MODEL


def _embedding_concurrency() -> int:
    raw = _environment_value(
        "FRAMEVITALS_RAG_EMBED_CONCURRENCY",
        legacy_name="DATALENS_RAG_EMBED_CONCURRENCY",
        default=str(_DEFAULT_EMBED_CONCURRENCY),
    ).strip()
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_EMBED_CONCURRENCY
    return max(1, min(value, _MAX_EMBED_CONCURRENCY))


def _rag_backend_override() -> str:
    return _environment_value(
        "FRAMEVITALS_RAG_BACKEND",
        legacy_name="DATALENS_RAG_BACKEND",
    ).strip().lower()


@dataclass
class Fact:
    path: str
    text: str
    value: Any = None
    embedding: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {"path": self.path, "text": self.text, "value": self.value}


# Paths excluded from the fact index — too noisy or too verbose to embed.
_EXCLUDE_PATH_FRAGMENTS = (
    "profile.preview",
    "profile.correlations.",
    "charts",
    "ai_report.text",
    "explainability.summary_chart_path",
    "anomalies_v2.top_rows.",
    "deep_statistics_v2.numeric_statistics.",
    "deep_statistics_v2.categorical_statistics.",
)


def _is_excluded(path: str) -> bool:
    return any(fragment in path for fragment in _EXCLUDE_PATH_FRAGMENTS)


def _summarize_value(value: Any, max_len: int = 220) -> str:
    """Return a short leaf representation suitable for retrieval."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)
    if isinstance(value, str):
        normalized = value.strip()
        return (
            normalized
            if len(normalized) <= max_len
            else normalized[:max_len] + "…"
        )
    return str(value)[:max_len]


def _humanize_path(path: str) -> str:
    return path.replace(".", " ").replace("_", " ")


def _walk(obj: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    """Yield ``(path, value)`` leaves from a nested dict/list."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            sub_path = f"{path}.{key}" if path else str(key)
            if _is_excluded(sub_path):
                continue
            yield from _walk(value, sub_path)
    elif isinstance(obj, list):
        if not obj:
            yield path, []
            return
        if all(not isinstance(item, (dict, list)) for item in obj):
            yield path, obj[:10]
            return
        for index, item in enumerate(obj[:5]):
            yield from _walk(item, f"{path}[{index}]")
    else:
        yield path, obj


def _build_fact_text(path: str, value: Any) -> str:
    return f"{_humanize_path(path)}: {_summarize_value(value)}"


def build_fact_index(analysis_result: dict) -> list[Fact]:
    """Flatten an analysis result into retrieval facts."""
    facts: list[Fact] = []
    for path, value in _walk(analysis_result):
        text = _build_fact_text(path, value)
        if len(text) < 3:
            continue
        facts.append(Fact(path=path, text=text, value=value))
    return facts


def _embed_with_ollama(texts: list[str]) -> np.ndarray | None:
    """Return an ``(n, d)`` embedding array, or ``None`` if unavailable."""
    try:
        import ollama
    except Exception:
        return None

    if not texts:
        return np.zeros((0, 0), dtype=float)

    model = _embedding_model()
    if len(texts) == 1:
        try:
            response = ollama.embeddings(model=model, prompt=texts[0])
            return np.asarray([response["embedding"]], dtype=float)
        except Exception:
            return None

    from concurrent.futures import ThreadPoolExecutor

    embeddings: list[list[float] | None] = [None] * len(texts)

    def _one(indexed_text: tuple[int, str]) -> tuple[int, list[float] | None]:
        index, text = indexed_text
        try:
            response = ollama.embeddings(model=model, prompt=text)
            return index, list(response["embedding"])
        except Exception:
            return index, None

    workers = min(_embedding_concurrency(), len(texts))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, vector in pool.map(_one, enumerate(texts)):
                embeddings[index] = vector
    except Exception:
        return None

    if any(vector is None for vector in embeddings):
        return None
    return np.asarray(embeddings, dtype=float)


def _embed_with_tfidf(
    corpus: list[str],
    queries: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return TF-IDF corpus and query vectors."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=4096,
        ngram_range=(1, 2),
        lowercase=True,
        token_pattern=r"(?u)\b[\w\-]{2,}\b",
    )
    vectorizer.fit(corpus + queries)
    return (
        vectorizer.transform(corpus).toarray(),
        vectorizer.transform(queries).toarray(),
    )


def _cosine_top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    """Return indices of the ``k`` highest cosine similarities."""
    if matrix.size == 0 or k <= 0:
        return []

    norms = np.linalg.norm(matrix, axis=1)
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm == 0:
        return []

    safe_norms = np.where(norms > 0, norms, 1.0)
    similarities = (matrix @ query_vec) / (safe_norms * query_norm)
    similarities = np.where(norms > 0, similarities, 0.0)
    if k >= len(similarities):
        order = np.argsort(-similarities)
    else:
        order = np.argpartition(-similarities, k - 1)[:k]
        order = order[np.argsort(-similarities[order])]
    return order.tolist()


def retrieve(question: str, facts: list[Fact], k: int = 8) -> dict:
    """Retrieve the ``k`` facts most relevant to ``question``.

    ``FRAMEVITALS_RAG_BACKEND=tfidf`` skips Ollama entirely. The old
    ``DATALENS_RAG_BACKEND`` name remains a compatibility fallback for the 0.x
    series but is no longer the primary configuration surface.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    if not question or not facts:
        return {"backend": "none", "facts": [], "k": 0}

    corpus = [fact.text for fact in facts]
    use_ollama_first = _rag_backend_override() != "tfidf"

    corpus_embeddings = None
    query_embedding = None
    backend = "tfidf"

    if use_ollama_first:
        corpus_embeddings = _embed_with_ollama(corpus)
        if corpus_embeddings is not None:
            query_array = _embed_with_ollama([question])
            if query_array is not None and query_array.shape[0] == 1:
                query_embedding = query_array[0]
                backend = "ollama"
            else:
                corpus_embeddings = None

    if corpus_embeddings is None or query_embedding is None:
        backend = "tfidf"
        corpus_embeddings, query_array = _embed_with_tfidf(corpus, [question])
        query_embedding = query_array[0]

    indices = _cosine_top_k(query_embedding, corpus_embeddings, k)
    selected = [facts[index] for index in indices]
    return {
        "backend": backend,
        "k": len(selected),
        "facts": [fact.to_dict() for fact in selected],
    }


def render_facts_block(retrieved: dict, max_chars: int = 4000) -> str:
    """Format retrieved facts as a compact prompt block."""
    items = retrieved.get("facts", [])
    if not items:
        return "(no relevant facts)"

    lines = []
    used = 0
    for fact in items:
        line = f"- {fact['text']}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)
