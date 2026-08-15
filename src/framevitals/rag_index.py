"""
RAG Fact Index (WS-10)
======================
Flatten an analysis result into atomic facts, then retrieve the top-k facts
relevant to a free-form question.

Two retrieval backends:
    1. Ollama embeddings (`nomic-embed-text` by default) — preferred when reachable.
    2. TF-IDF cosine similarity (sklearn) — always-available fallback.

The retrieval API stays identical regardless of backend, so the agent layer
never has to care which one is in play.

Public API:
    facts = build_fact_index(analysis_result)
    top   = retrieve(question, facts, k=8)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Number of concurrent Ollama embedding requests. Embedding ~1k facts one at
# a time over localhost is ~25s; parallelising at 8 workers brings that under
# 4s on a typical machine. Tunable via env so a constrained box (e.g. one
# with a small GPU) can dial it back.
_EMBED_CONCURRENCY = int(os.environ.get("DATALENS_RAG_EMBED_CONCURRENCY", "8"))


# ---------------------------------------------------------------------------
# Fact dataclass
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    path: str
    text: str
    value: Any = None
    embedding: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {"path": self.path, "text": self.text, "value": self.value}


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------

# Paths excluded from the fact index — too noisy or too verbose to embed.
_EXCLUDE_PATH_FRAGMENTS = (
    "profile.preview",
    "profile.correlations.",
    "charts",                     # raw chart paths aren't useful as text facts
    "ai_report.text",             # already a free-form summary, redundant
    "explainability.summary_chart_path",
    "anomalies_v2.top_rows.",     # individual anomaly rows are noisy
    "deep_statistics_v2.numeric_statistics.",
    "deep_statistics_v2.categorical_statistics.",
)


def _is_excluded(path: str) -> bool:
    return any(frag in path for frag in _EXCLUDE_PATH_FRAGMENTS)


def _summarize_value(value: Any, max_len: int = 220) -> str:
    """Short string form of a leaf value, suitable for embedding."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        # Round floats for stability
        if isinstance(value, float):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return str(value)
    if isinstance(value, str):
        v = value.strip()
        return v if len(v) <= max_len else v[:max_len] + "…"
    return str(value)[:max_len]


def _humanize_path(path: str) -> str:
    """Turn dotted path into a readable label for retrieval."""
    return path.replace(".", " ").replace("_", " ")


def _walk(obj: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    """Yield (path, value) leaves from a nested dict/list."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            sub_path = f"{path}.{key}" if path else str(key)
            if _is_excluded(sub_path):
                continue
            yield from _walk(val, sub_path)
    elif isinstance(obj, list):
        # Summarize lists of dicts / scalars compactly rather than yielding every
        # element. This keeps the index focused on aggregate facts.
        if len(obj) == 0:
            yield path, []
            return

        # All scalars -> single fact with a short list
        if all(not isinstance(item, (dict, list)) for item in obj):
            yield path, obj[:10]
            return

        # List of dicts -> yield a few summarized children
        for i, item in enumerate(obj[:5]):
            yield from _walk(item, f"{path}[{i}]")
    else:
        yield path, obj


def _build_fact_text(path: str, value: Any) -> str:
    label = _humanize_path(path)
    summary = _summarize_value(value)
    return f"{label}: {summary}"


def build_fact_index(analysis_result: dict) -> list[Fact]:
    """Flatten the analysis result into a list of Fact objects."""
    facts: list[Fact] = []
    for path, value in _walk(analysis_result):
        text = _build_fact_text(path, value)
        if not text or len(text) < 3:
            continue
        facts.append(Fact(path=path, text=text, value=value))
    return facts


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

def _embed_with_ollama(texts: list[str]) -> np.ndarray | None:
    """Return (n, d) array of embeddings, or None if Ollama is unavailable.

    Uses a small thread pool so embedding a large corpus (~1k facts) doesn't
    serialize into one ~25s wall-clock waterfall of HTTP round-trips. Each
    individual /api/embeddings call is cheap (~10-30ms on localhost) but the
    server happily handles them concurrently.
    """
    try:
        import ollama
    except Exception:
        return None

    if not texts:
        return np.zeros((0, 0), dtype=float)

    # Single-shot fast path — avoids spinning up a pool for query-side calls.
    if len(texts) <= 1:
        try:
            resp = ollama.embeddings(model=_EMBED_MODEL, prompt=texts[0])
            return np.asarray([resp["embedding"]], dtype=float)
        except Exception:
            return None

    from concurrent.futures import ThreadPoolExecutor

    embeddings: list[list[float] | None] = [None] * len(texts)

    def _one(idx_text: tuple[int, str]) -> tuple[int, list[float] | None]:
        i, t = idx_text
        try:
            resp = ollama.embeddings(model=_EMBED_MODEL, prompt=t)
            return i, list(resp["embedding"])
        except Exception:
            return i, None

    workers = max(1, min(_EMBED_CONCURRENCY, len(texts)))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, vec in pool.map(_one, list(enumerate(texts))):
                embeddings[i] = vec
    except Exception:
        return None

    if any(v is None for v in embeddings):
        # If even one call failed we cannot trust the matrix shape; fall back
        # so the caller picks TF-IDF instead of seeing a ragged ndarray.
        return None
    return np.asarray(embeddings, dtype=float)


def _embed_with_tfidf(corpus: list[str], queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """TF-IDF fallback. Returns (corpus_vectors, query_vectors)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        max_features=4096,
        ngram_range=(1, 2),
        lowercase=True,
        token_pattern=r"(?u)\b[\w\-]{2,}\b",
    )
    vec.fit(corpus + queries)
    return vec.transform(corpus).toarray(), vec.transform(queries).toarray()


def _cosine_top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    """Return indices of the k highest cosine similarities."""
    if matrix.size == 0:
        return []

    norms = np.linalg.norm(matrix, axis=1)
    qnorm = float(np.linalg.norm(query_vec))
    if qnorm == 0:
        return []
    safe = np.where(norms > 0, norms, 1.0)
    sims = (matrix @ query_vec) / (safe * qnorm)
    sims = np.where(norms > 0, sims, 0.0)
    if k >= len(sims):
        order = np.argsort(-sims)
    else:
        order = np.argpartition(-sims, k)[:k]
        order = order[np.argsort(-sims[order])]
    return order.tolist()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(question: str, facts: list[Fact], k: int = 8) -> dict:
    """
    Retrieve the k facts most relevant to `question`.

    Returns:
        {"backend": "ollama" | "tfidf", "facts": [Fact.to_dict(), ...], "k": int}

    Set ``DATALENS_RAG_BACKEND=tfidf`` to skip Ollama embeddings entirely. The
    TF-IDF path retrieves from a 1k-fact corpus in well under 100ms and is a
    perfectly sensible default for batch / test / CI runs where every second
    counts. Ollama embeddings produce slightly better neighbours but cost a
    multi-second waterfall through the embedding model.
    """
    if not question or not facts:
        return {"backend": "none", "facts": [], "k": 0}

    corpus = [f.text for f in facts]

    forced = os.environ.get("DATALENS_RAG_BACKEND", "").strip().lower()
    use_ollama_first = forced != "tfidf"

    corpus_emb = None
    query_emb = None
    backend = "tfidf"

    if use_ollama_first:
        corpus_emb = _embed_with_ollama(corpus)
        if corpus_emb is not None:
            q_arr = _embed_with_ollama([question])
            if q_arr is not None and q_arr.shape[0] == 1:
                query_emb = q_arr[0]
                backend = "ollama"
            else:
                corpus_emb = None  # fall through to TF-IDF

    if corpus_emb is None or query_emb is None:
        backend = "tfidf"
        corpus_emb, query_arr = _embed_with_tfidf(corpus, [question])
        query_emb = query_arr[0]

    indices = _cosine_top_k(query_emb, corpus_emb, k)
    selected = [facts[i] for i in indices]

    return {
        "backend": backend,
        "k": len(selected),
        "facts": [f.to_dict() for f in selected],
    }


def render_facts_block(retrieved: dict, max_chars: int = 4000) -> str:
    """Format retrieved facts as a compact text block for prompt injection."""
    items = retrieved.get("facts", [])
    if not items:
        return "(no relevant facts)"

    lines = []
    used = 0
    for f in items:
        line = f"- {f['text']}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)
