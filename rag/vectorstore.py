"""Local (embedded) Qdrant wrapper - no server/Docker required.

Note: Qdrant's local file-based mode takes an exclusive lock on the storage
directory. Only one process (ingest.py OR cli.py/streamlit_app.py) can have
it open at a time - close one before starting the other.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from rag import config

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=str(config.QDRANT_PATH))
    return _client


def collection_exists() -> bool:
    return get_client().collection_exists(config.COLLECTION_NAME)


def recreate_collection(dim: int) -> None:
    client = get_client()
    if client.collection_exists(config.COLLECTION_NAME):
        client.delete_collection(config.COLLECTION_NAME)
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def ensure_collection(dim: int) -> None:
    client = get_client()
    if not client.collection_exists(config.COLLECTION_NAME):
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def upsert(ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
    client = get_client()
    points = [
        PointStruct(id=_hash_id(i), vector=v, payload={**p, "chunk_id": i})
        for i, v, p in zip(ids, vectors, payloads)
    ]
    client.upsert(collection_name=config.COLLECTION_NAME, points=points)


def _hash_id(chunk_id: str) -> int:
    """Qdrant point ids must be int or UUID; derive a stable int from our string chunk_id."""
    import hashlib

    return int(hashlib.sha256(chunk_id.encode()).hexdigest()[:16], 16)


def _build_filter(filters: dict | None) -> Filter | None:
    if not filters:
        return None
    conditions = [
        FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items() if v
    ]
    return Filter(must=conditions) if conditions else None


def search(vector: list[float], top_k: int, filters: dict | None = None) -> list[dict]:
    client = get_client()
    results = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=vector,
        limit=top_k,
        query_filter=_build_filter(filters),
        with_payload=True,
    )
    return [{"score": p.score, **p.payload} for p in results.points]


def scroll_by(source_file: str, section: str, limit: int = 20) -> list[dict]:
    """Fetch all chunks for a given (source_file, section) - used for parent-section expansion."""
    client = get_client()
    flt = Filter(
        must=[
            FieldCondition(key="source_file", match=MatchValue(value=source_file)),
            FieldCondition(key="section", match=MatchValue(value=section)),
        ]
    )
    points, _ = client.scroll(
        collection_name=config.COLLECTION_NAME,
        scroll_filter=flt,
        limit=limit,
        with_payload=True,
    )
    return [p.payload for p in points]


def scroll_by_index_range(
    source_file: str,
    center_index: int,
    window: int = 4,
) -> list[dict]:
    """Fetch chunks from the same source file within +/-window of center_index."""
    from qdrant_client.models import Range, FieldCondition as FC, Filter as F
    client = get_client()
    lo = max(0, center_index - window)
    hi = center_index + window
    flt = F(
        must=[
            FC(key="source_file", match=MatchValue(value=source_file)),
            FC(key="chunk_index", range=Range(gte=lo, lte=hi)),
        ]
    )
    points, _ = client.scroll(
        collection_name=config.COLLECTION_NAME,
        scroll_filter=flt,
        limit=(window * 2) + 3,
        with_payload=True,
    )
    return [p.payload for p in points]


def scroll_all(limit: int = 100000) -> list[dict]:
    """Fetch every stored chunk's payload - used to build the BM25 keyword index."""
    client = get_client()
    points, _ = client.scroll(
        collection_name=config.COLLECTION_NAME, limit=limit, with_payload=True
    )
    return [p.payload for p in points]
