"""Local embedding backend (BAAI/bge-base-en-v1.5 via sentence-transformers).

Gemini's embed_content endpoint on the provided API key proved to be
persistently rate-limited in practice (repeated 429s even on tiny batches),
which made bulk ingestion impractical. Generation (the actual LLM step) via
Gemini worked fine, so only embeddings were switched - see rag/generation.py
for the Gemini LLM call.

BAAI/bge-* is explicitly called out as an "excellent open source" embedding
choice in RAG-Implementations.docx section 5, so this is a supported
alternative from the original design, not an ad-hoc substitution. Running
locally means no API calls, no rate limits, and no per-token cost.

BGE models are asymmetric: queries need an instruction prefix, documents
don't (see BAAI's model card).
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
    """Embed chunk texts for storage. Named *_cached for interface parity with
    the previous Gemini-backed implementation; local embedding is fast enough
    that no on-disk cache is needed, it just runs the whole batch."""
    model = _get_model()
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    model = _get_model()
    vector = model.encode([_QUERY_INSTRUCTION + text], normalize_embeddings=True)[0]
    return vector.tolist()
