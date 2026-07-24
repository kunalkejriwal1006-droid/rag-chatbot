"""Central configuration loaded from .env."""
from pathlib import Path
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val not in (None, "") else default


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434")

# Read fallback backend order, e.g. "gemini,groq,ollama"
LLM_FALLBACK_ORDER = [
    b.strip().lower() 
    for b in _get("LLM_FALLBACK_ORDER", "gemini,groq,ollama").split(",")
    if b.strip()
]

# Model definitions per backend
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-3.5-flash")
GROQ_MODEL = _get("GROQ_MODEL", "llama-3.1-8b-instant")
OLLAMA_MODEL = _get("OLLAMA_MODEL", "llama3.1:8b")

EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

PDF_DIR = (PROJECT_ROOT / _get("PDF_DIR", ".")).resolve()
DATA_DIR = (PROJECT_ROOT / _get("DATA_DIR", "./data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

QDRANT_PATH = DATA_DIR / "qdrant_db"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
MANIFEST_PATH = DATA_DIR / "chunks_manifest.jsonl"

COLLECTION_NAME = _get("COLLECTION_NAME", "insurance_motor")

CHUNK_MIN_TOKENS = int(_get("CHUNK_MIN_TOKENS", "200"))
CHUNK_MAX_TOKENS = int(_get("CHUNK_MAX_TOKENS", "600"))
TABLE_ROW_SPLIT_THRESHOLD = int(_get("TABLE_ROW_SPLIT_THRESHOLD", "8"))

TOP_K = int(_get("TOP_K", "8"))
ENUMERATION_TOP_K = int(_get("ENUMERATION_TOP_K", "30"))
