"""Hybrid retrieval: vector search + BM25 keyword search, combined via
Reciprocal Rank Fusion (RRF), plus optional metadata filtering and
parent-section context expansion.
"""
import re

from rag import config, embeddings, keyword_search, vectorstore

RRF_K = 60

# "list every add-on" style queries need most of a category, not just the
# single best-matching chunk, so bump top_k when this matches.
ENUMERATION_INTENT_RE = re.compile(
    r"\b(list|enumerate|name)\s+(all|every|each)\b"
    r"|\ball\s+(the\s+)?(add-?ons?|covers?|coverages?|exclusions?|sections?|"
    r"benefits?|endorsements?)\b"
    r"|\bwhat\s+are\s+all\b"
    r"|\bhow\s+many\s+(add-?ons?|covers?|exclusions?)\s+(are\s+there|does)\b",
    re.IGNORECASE,
)


def hybrid_search(
    query: str,
    top_k: int | None = None,
    filters: dict | None = None,
    expand_context: bool = False,
) -> list[dict]:
    top_k = top_k or config.TOP_K
    if ENUMERATION_INTENT_RE.search(query):
        top_k = max(top_k, config.ENUMERATION_TOP_K)
    fetch_k = max(top_k * 3, 20)

    qvec = embeddings.embed_query(query)
    vector_results = vectorstore.search(qvec, fetch_k, filters)
    keyword_results = keyword_search.search(query, fetch_k, filters)

    scores: dict[str, float] = {}
    payload_map: dict[str, dict] = {}
    for rank, r in enumerate(vector_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        payload_map[cid] = r
    for rank, r in enumerate(keyword_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        payload_map.setdefault(cid, r)

    ranked_ids = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_k]
    results = [payload_map[cid] for cid in ranked_ids]

    if expand_context:
        results = [_expand_to_parent_section(r) for r in results]

    return results


def _expand_to_parent_section(chunk: dict, window: int = 4) -> dict:
    """Pull in neighboring chunks (by chunk_index) so the LLM sees full context.

    Uses index proximity rather than matching on section name - table rows
    each have a unique section name, and generic names like "Conditions"
    are shared across unrelated add-ons.
    """
    src = chunk.get("source_file", "")
    center_idx = chunk.get("chunk_index", 0)
    if not src:
        return chunk

    neighbors = vectorstore.scroll_by_index_range(src, center_idx, window=window)
    neighbors = sorted(neighbors, key=lambda c: c.get("chunk_index", 0))
    if neighbors:
        chunk = dict(chunk)
        chunk["text"] = "\n\n".join(c["text"] for c in neighbors)
    return chunk
