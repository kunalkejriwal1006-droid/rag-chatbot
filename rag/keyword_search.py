"""BM25 keyword index - the "keyword search" half of hybrid search.

Persisted separately from Qdrant (plain pickle) since Qdrant's local mode
only supports vector search well; BM25 gives exact-term recall (policy
codes, UIN numbers, named add-ons like "Zero Depreciation") that embedding
similarity alone can miss.
"""
import pickle
import re

from rank_bm25 import BM25Okapi

from rag import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())


def build_index(payloads: list[dict]) -> None:
    corpus_tokens = [_tokenize(p["text"]) for p in payloads]
    bm25 = BM25Okapi(corpus_tokens)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "payloads": payloads}, f)


_cache = None


def _load():
    global _cache
    if _cache is None:
        with open(config.BM25_INDEX_PATH, "rb") as f:
            _cache = pickle.load(f)
    return _cache


def search(query: str, top_k: int, filters: dict | None = None) -> list[dict]:
    data = _load()
    bm25: BM25Okapi = data["bm25"]
    payloads: list[dict] = data["payloads"]

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(payloads)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked:
        if scores[idx] <= 0:
            break
        p = payloads[idx]
        if filters and any(p.get(k) != v for k, v in filters.items() if v):
            continue
        results.append({"score": float(scores[idx]), **p})
        if len(results) >= top_k:
            break
    return results
