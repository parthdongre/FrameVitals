"""
Text / NLP Profiling (WS-6)
============================
Auto-detects free-text columns and produces per-column linguistic profiles.

A column is considered "text-like" when:
    - dtype is object/string AND
    - at least 30% of non-empty values have ≥ 2 tokens AND
    - average length is ≥ 8 characters AND
    - it isn't already flagged as ID-like / categorical (high cardinality + word-y).

For each text column we compute:
    - basic stats: avg / min / max length, total tokens, vocab size,
      lexical diversity (TTR), avg sentence length
    - top unigrams + bigrams (stopwords filtered when nltk is available)
    - regex pattern hits: emails, URLs, phone-ish numbers, mentions, hashtags
    - heuristic language guess (ASCII fraction + stopword overlap)
    - sentiment-lite: ratio of words on a small positive/negative seed list
    - optional 2-D LSA preview: TF-IDF (top 1024 features) -> TruncatedSVD(2)

Public entry point:
    profile_text_columns(df) -> dict
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Tokenization + stopwords
# ---------------------------------------------------------------------------

# Lightweight built-in stopword list as fallback; NLTK list is preferred.
_FALLBACK_STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "her", "him", "his", "i", "if", "in", "into", "is", "it",
    "its", "of", "on", "or", "she", "that", "the", "their", "them", "then",
    "there", "they", "this", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "will", "with", "you", "your", "but", "not", "no", "so",
    "do", "does", "did", "can", "could", "would", "should", "may", "might",
    "shall", "i'm", "im", "u", "us", "our", "than", "too", "very", "just",
    "about", "after", "before", "because", "any", "all", "some", "such", "more",
    "most", "other", "only",
}

_TOKEN_RE = re.compile(r"\b[\w\-']+\b", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_PATTERN_PROBES: dict[str, re.Pattern] = {
    "emails": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "urls": re.compile(r"https?://[^\s)]+|www\.[^\s)]+"),
    "phones": re.compile(r"(?:\+?\d[\d\-.\s()]{6,}\d)"),
    "mentions": re.compile(r"(?:^|\s)@[\w_]+"),
    "hashtags": re.compile(r"(?:^|\s)#[\w_]+"),
    "monetary": re.compile(r"(?:[$€£¥₹]|USD|EUR|GBP)\s*\d[\d,.]*"),
}


def _load_stopwords() -> set[str]:
    """Try to use NLTK's English stopwords; fall back to the built-in set."""
    try:
        import nltk  # type: ignore
        from nltk.corpus import stopwords  # type: ignore

        try:
            return set(stopwords.words("english"))
        except LookupError:
            try:
                nltk.download("stopwords", quiet=True)
                return set(stopwords.words("english"))
            except Exception:
                return set(_FALLBACK_STOPWORDS)
    except Exception:
        return set(_FALLBACK_STOPWORDS)


_STOPWORDS = _load_stopwords()


# ---------------------------------------------------------------------------
# Sentiment-lite seed lexicon
# ---------------------------------------------------------------------------

_POSITIVE_SEEDS = {
    "good", "great", "excellent", "amazing", "wonderful", "love", "best",
    "happy", "fantastic", "perfect", "thank", "thanks", "win", "won",
    "improvement", "better", "fast", "smooth", "clean", "useful", "helpful",
    "nice", "awesome", "positive",
}
_NEGATIVE_SEEDS = {
    "bad", "terrible", "awful", "worst", "hate", "broken", "slow", "buggy",
    "crash", "fail", "failed", "error", "issue", "problem", "useless",
    "negative", "wrong", "ugly", "annoying", "disappointed", "angry", "sad",
    "poor",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, ndigits: int = 4) -> float | int | None:
    if value is None:
        return None
    try:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return None
            return round(v, ndigits)
        return value
    except Exception:
        return None


def _ascii_ratio(text: str) -> float:
    if not text:
        return 1.0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return ascii_chars / len(text)


def _tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


def _filter_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1 and not t.isdigit()]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _is_text_like(series: pd.Series, max_unique_ratio: float = 0.99) -> bool:
    """Heuristic: looks like free text rather than a category or ID."""
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return False

    s = series.dropna().astype(str)
    if s.empty:
        return False

    sample = s.head(500)
    avg_len = float(sample.str.len().mean())
    multi_token_ratio = float(sample.str.split().str.len().fillna(0).gt(1).mean())

    if avg_len < 8 or multi_token_ratio < 0.3:
        return False

    # Avoid super-high cardinality columns that look more like IDs/hashes
    n_unique = int(s.nunique())
    if n_unique / max(len(s), 1) >= max_unique_ratio and avg_len < 20:
        return False

    return True


def detect_text_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if _is_text_like(df[col])]


# ---------------------------------------------------------------------------
# Per-column profile
# ---------------------------------------------------------------------------

def _length_stats(series: pd.Series) -> dict:
    s = series.dropna().astype(str)
    if s.empty:
        return {"avg": 0, "min": 0, "max": 0, "median": 0}
    lens = s.str.len()
    return {
        "avg": _safe_float(lens.mean()),
        "min": int(lens.min()),
        "max": int(lens.max()),
        "median": _safe_float(lens.median()),
    }


def _vocabulary_stats(token_lists: list[list[str]]) -> dict:
    flat = [t for tokens in token_lists for t in tokens]
    if not flat:
        return {"total_tokens": 0, "vocab_size": 0, "lexical_diversity": 0}
    vocab = set(flat)
    return {
        "total_tokens": int(len(flat)),
        "vocab_size": int(len(vocab)),
        "lexical_diversity": _safe_float(len(vocab) / len(flat)),
    }


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _top_terms(token_lists: list[list[str]], n: int, top_k: int) -> list[dict]:
    counter: Counter = Counter()
    for tokens in token_lists:
        filtered = _filter_tokens(tokens)
        if n == 1:
            counter.update(filtered)
        else:
            counter.update(_ngrams(filtered, n))
    out = []
    for term, count in counter.most_common(top_k):
        if isinstance(term, tuple):
            term_str = " ".join(term)
        else:
            term_str = term
        out.append({"term": term_str, "count": int(count)})
    return out


def _pattern_hits(series: pd.Series) -> dict:
    s = series.dropna().astype(str)
    if s.empty:
        return {"counts": {}, "examples": {}}
    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for name, pattern in _PATTERN_PROBES.items():
        rows_with_match = 0
        sample_examples: list[str] = []
        for text in s:
            match = pattern.search(text)
            if match:
                rows_with_match += 1
                if len(sample_examples) < 3:
                    sample_examples.append(match.group(0).strip()[:80])
        counts[name] = rows_with_match
        if sample_examples:
            examples[name] = sample_examples
    return {"counts": counts, "examples": examples, "total_rows": int(len(s))}


def _language_guess(series: pd.Series) -> dict:
    """Cheap language guess based on ASCII fraction + English stopword overlap."""
    s = series.dropna().astype(str).head(200)
    if s.empty:
        return {"language": "unknown", "confidence": 0.0}

    blob = " ".join(s.tolist())
    ascii_frac = _ascii_ratio(blob)
    tokens = _tokenize(blob)
    if not tokens:
        return {"language": "unknown", "confidence": 0.0}
    overlap = sum(1 for t in tokens if t in _STOPWORDS)
    overlap_ratio = overlap / max(len(tokens), 1)

    if ascii_frac > 0.95 and overlap_ratio > 0.05:
        return {"language": "english", "confidence": _safe_float(min(1.0, overlap_ratio * 4 + ascii_frac * 0.2))}
    if ascii_frac < 0.6:
        return {"language": "non-latin (likely)", "confidence": _safe_float(1.0 - ascii_frac)}
    return {"language": "latin (non-english likely)", "confidence": _safe_float(ascii_frac * 0.5)}


def _sentiment_lite(token_lists: list[list[str]]) -> dict:
    pos_total = 0
    neg_total = 0
    n_tokens = 0
    for tokens in token_lists:
        for t in tokens:
            n_tokens += 1
            if t in _POSITIVE_SEEDS:
                pos_total += 1
            elif t in _NEGATIVE_SEEDS:
                neg_total += 1
    if n_tokens == 0:
        return {"positive_ratio": 0.0, "negative_ratio": 0.0, "polarity": 0.0, "n_tokens": 0}
    pos_ratio = pos_total / n_tokens
    neg_ratio = neg_total / n_tokens
    polarity = pos_ratio - neg_ratio
    return {
        "positive_ratio": _safe_float(pos_ratio),
        "negative_ratio": _safe_float(neg_ratio),
        "polarity": _safe_float(polarity),
        "n_tokens": int(n_tokens),
    }


def _lsa_projection(series: pd.Series, max_rows: int = 500) -> dict:
    """TF-IDF + TruncatedSVD(2) preview. Best-effort, returns sample points."""
    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:
        return {"available": False, "reason": f"sklearn unavailable: {exc}"}

    s = series.dropna().astype(str).head(max_rows)
    if len(s) < 10:
        return {"available": False, "reason": "n<10"}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vec = TfidfVectorizer(max_features=1024, stop_words="english", ngram_range=(1, 2))
            tfidf = vec.fit_transform(s.tolist())
            n_components = min(2, max(1, tfidf.shape[1] - 1))
            if n_components < 2:
                return {"available": False, "reason": "vocabulary too small for 2-D projection"}
            svd = TruncatedSVD(n_components=2, random_state=42)
            embedding = svd.fit_transform(tfidf)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    explained = [float(v) for v in svd.explained_variance_ratio_]
    points = [
        {"x": _safe_float(embedding[i, 0]), "y": _safe_float(embedding[i, 1]),
         "preview": s.iloc[i][:80]}
        for i in range(min(len(s), 200))
    ]

    return {
        "available": True,
        "method": "TF-IDF (1-2 grams, max 1024 features) + TruncatedSVD(2)",
        "explained_variance_ratio": explained,
        "n_points": len(points),
        "points": points,
    }


def _profile_one_column(series: pd.Series) -> dict:
    s = series.dropna().astype(str)
    n = int(len(s))
    if n == 0:
        return {"available": False, "reason": "no non-null values"}

    tokens_per_row = [_tokenize(text) for text in s]
    sentence_lengths = []
    for text in s.head(1000):
        parts = _SENTENCE_RE.split(text)
        for p in parts:
            wc = len(_tokenize(p))
            if wc > 0:
                sentence_lengths.append(wc)

    vocab_stats = _vocabulary_stats(tokens_per_row)
    avg_sentence_len = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0

    return {
        "available": True,
        "n_rows": n,
        "length_stats": _length_stats(s),
        "tokens": vocab_stats,
        "avg_sentence_length": _safe_float(avg_sentence_len),
        "language": _language_guess(s),
        "top_unigrams": _top_terms(tokens_per_row, n=1, top_k=15),
        "top_bigrams": _top_terms(tokens_per_row, n=2, top_k=12),
        "patterns": _pattern_hits(s),
        "sentiment_lite": _sentiment_lite(tokens_per_row),
        "lsa": _lsa_projection(s),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def profile_text_columns(df: pd.DataFrame, max_columns: int = 5) -> dict:
    """
    Detect text-like columns in `df` and profile up to `max_columns` of them.

    Returns a JSON-safe dict with shape:
        {
          "available": bool,
          "detected_columns": [...],
          "stopwords_source": "nltk" | "fallback",
          "profiles": { col_name: {...}, ... }
        }
    """
    detected = detect_text_columns(df)
    if not detected:
        return {
            "available": False,
            "reason": "No text-like columns detected.",
            "detected_columns": [],
            "profiles": {},
        }

    selected = detected[:max_columns]
    profiles: dict[str, dict] = {}

    for col in selected:
        try:
            profiles[col] = _profile_one_column(df[col])
        except Exception as exc:
            profiles[col] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    stopwords_source = "nltk" if len(_STOPWORDS) > len(_FALLBACK_STOPWORDS) else "fallback"

    return {
        "available": True,
        "detected_columns": detected,
        "profiled_columns": selected,
        "stopwords_source": stopwords_source,
        "profiles": profiles,
    }
