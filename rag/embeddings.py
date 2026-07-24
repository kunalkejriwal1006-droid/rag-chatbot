"""Local embedding backend (BAAI/bge-base-en-v1.5 via sentence-transformers).

Runs on CPU, no API key or rate limits. BGE is asymmetric - queries need an
instruction prefix, documents don't (per BAAI's model card).
"""
from sentence_transformers import SentenceTransformer

from rag import config

_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_documents_cached(ids: list[str], texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    model = _get_model()
    vector = model.encode([_QUERY_INSTRUCTION + text], normalize_embeddings=True)[0]
    return vector.tolist()
