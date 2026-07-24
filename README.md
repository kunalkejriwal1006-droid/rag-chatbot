# Insurance Broker RAG System

A retrieval-augmented Q&A system over Bajaj Allianz / Bajaj General Insurance
two-wheeler motor policy documents (policy wordings, brochures, and add-on
endorsement libraries). Ask it questions in plain English — coverage terms,
add-on eligibility, NCB slabs, exclusions, claim procedures — and it answers
using only the ingested PDFs, with page-level citations.

Built following the design in [`RAG-Implementations.docx`](RAG-Implementations.docx):
semantic (not fixed-size) chunking, rich per-chunk metadata, table-aware
chunking, and hybrid (vector + keyword) retrieval.

## How it's different from the original plan

The doc's plan assumed OpenAI embeddings + Gemini/GPT-5 for generation. Two
things changed during setup, both due to the specific Gemini API key
provided:

- **Embeddings run locally** (`BAAI/bge-base-en-v1.5` via
  `sentence-transformers`), not through Gemini's `embed_content` API. That
  endpoint was persistently rate-limited on the provided key (repeated 429s
  even on 5-item batches), which made bulk embedding impractical. BGE is
  explicitly listed as an "excellent open source" option in the design doc,
  so this is a supported alternative, not an improvised one. It also means
  ingestion has **no API cost and no rate limits** — re-ingesting all 829
  chunks takes about 3 minutes on CPU.
- **Generation still uses Gemini** (`gemini-2.5-flash`) — that endpoint
  worked fine in testing, so the actual "LLM" part of the RAG system is
  Gemini as requested.

## Architecture

```
PDFs  →  parse (PyMuPDF text + pdfplumber tables)
      →  semantic chunk (heading-aware sections, sentence-safe token budgeting,
                          table decision matrix)
      →  metadata enrichment (insurer, product, document_type, section, UIN, page)
      →  embed locally (BAAI/bge-base-en-v1.5)
      →  store: Qdrant (vectors + metadata) + BM25 (keyword index)

Query →  hybrid retrieval (vector search + BM25, fused via Reciprocal Rank Fusion)
      →  optional metadata filters / parent-section expansion
      →  Gemini (gemini-2.5-flash) generates a grounded, cited answer
```

| Concern | Choice |
|---|---|
| PDF text extraction | PyMuPDF (`fitz`) |
| Table detection | pdfplumber |
| Chunking | Custom heading-aware + sentence-safe semantic chunker (`rag/chunking.py`) |
| Embeddings | `BAAI/bge-base-en-v1.5`, local, via `sentence-transformers` |
| Vector store | Qdrant, embedded/local mode (no server or Docker needed) |
| Keyword search | BM25 (`rank_bm25`) |
| Retrieval | Hybrid vector+keyword via Reciprocal Rank Fusion, metadata filters, parent-section expansion |
| LLM (generation) | Gemini `gemini-2.5-flash` |

## Project layout

```
rag/
  config.py          Loads .env - paths, model names, chunking/retrieval params
  metadata.py         Per-PDF metadata (insurer, product, document_type, UIN)
  pdf_parser.py       Per-page text + table extraction
  chunking.py          Semantic chunking (sections, sentences, tables)
  embeddings.py       Local BGE embedding wrapper
  vectorstore.py      Embedded Qdrant wrapper
  keyword_search.py   BM25 index build/query
  retrieval.py         Hybrid retrieval (RRF) + parent-section expansion
  generation.py         Gemini answer generation with citation rules
  ingest.py            Ingestion pipeline orchestrator (CLI entry point)
cli.py                  Interactive query CLI
streamlit_app.py        Web chat UI
data/                   Generated: Qdrant storage, BM25 index, chunk manifest (gitignored)
*.pdf                    Source documents (13 Bajaj Allianz two-wheeler PDFs)
```

## Setup

### 1. Python environment

**Important (Windows only):** `sentence-transformers` pulls in `torch`,
whose package files have very long paths. If your Python install lives
under a deeply nested path (e.g. the Windows Store Python under
`AppData\Local\Packages\...`), `pip install` can fail with an `OSError`
about a path being too long. The fix used in this setup was a virtualenv at
a short path:

```powershell
python -m venv C:\rag_venv
C:\rag_venv\Scripts\pip install -r requirements.txt
```

All commands below use `C:\rag_venv\Scripts\python.exe`. If your own Python
install doesn't hit the long-path issue, a normal `venv` in the project
folder works too — just adjust the interpreter path.

### 2. Configure `.env`

Copy `.env.example` to `.env` (already done in this project) and set:

```
GEMINI_API_KEY=your_key_here       # used only for answer generation
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
GENERATION_MODEL=gemini-2.5-flash
```

Get a Gemini key at [aistudio.google.com](https://aistudio.google.com/apikey).
If generation calls start failing with 429s, it's the same style of
rate-limiting described above — waiting a bit or swapping the key resolves it.

### 3. Ingest the PDFs

```powershell
C:\rag_venv\Scripts\python.exe -m rag.ingest --rebuild
```

This parses all PDFs in the project folder, chunks them, embeds every chunk
locally, and writes to `data/qdrant_db` (vectors) and `data/bm25_index.pkl`
(keyword index). A human-readable `data/chunks_manifest.jsonl` is also
written — one line per chunk — useful for spot-checking chunk quality.

Current corpus: 13 PDFs → **829 chunks** (621 prose, 71 whole-table, 137
table-row), taking about 3 minutes end-to-end on CPU.

Useful flags:

```powershell
# Only re-ingest specific files (fast smoke test before a full run)
python -m rag.ingest --rebuild --files "BAJAJ ALLIANZ GENERAL INSURANCE CO_1.pdf,BAJAJ ALLIANZ GENERAL INSURANCE CO_4.pdf"

# Add new PDFs without wiping existing data (drop --rebuild)
python -m rag.ingest
```

Drop new PDFs straight into the project folder and re-run ingestion — no
code changes needed. For best metadata quality, add an entry for the new
file in `rag/metadata.py`'s `DOCUMENT_METADATA` dict (insurer/product/
document_type/UIN); otherwise it falls back to reasonable heuristics.

## Using it

**Important:** Qdrant's local storage mode locks its folder to one process
at a time. Don't run ingestion, the CLI, and Streamlit simultaneously —
finish or stop one before starting another.

### CLI

```powershell
# One-off question
C:\rag_venv\Scripts\python.exe cli.py "What is Zero Depreciation cover and how many times can it be used?"

# Interactive chat
C:\rag_venv\Scripts\python.exe cli.py
```

Interactive-mode commands:

```
/filter product=Two Wheeler Package Policy    set a metadata filter
/filter document_type=Policy Wording          (valid keys: product, document_type,
                                                vehicle_type, chunk_type, section)
/filter clear                                 clear all filters
/filters                                       show active filters
/expand on|off                                  toggle parent-section context expansion
/help
/exit
```

### Web UI (Streamlit)

```powershell
C:\rag_venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Opens a chat interface with sidebar filters (product, document type) and a
"Sources" panel per answer showing exact file + page citations.

## What "semantic chunking" means here

Per the design doc's guidance to avoid fixed-size chunks:

1. **Structural pass** — each PDF is split into sections using heading
   heuristics: numbered "Section N:" headers, ALL-CAPS headers, known IRDAI
   policy section names (Exclusions, Definitions, Claims Procedure, ...),
   and add-on titles that precede a "UIN:" line (the pattern used
   throughout the endorsement libraries). A chunk never crosses a section
   boundary.
2. **Sentence-safe budgeting** — within a section, sentences are grouped
   into ~200-600 token chunks without ever splitting mid-sentence.
3. **Tables are chunked separately**, following the doc's decision matrix:
   small lookup tables (add-on comparison matrices, premium slabs) become
   one chunk in natural language ("Zero Dep: Private Car: Yes; EV: Yes; Max
   age: 7 years"); large row-oriented tables become one chunk per row so
   each fact is independently retrievable.

Every chunk carries metadata: `insurer`, `product`, `lob`, `vehicle_type`,
`document_type`, `section`, `chunk_type`, `page_start`/`page_end`, `uin`,
`source_file`. This is what powers the `/filter` commands and the
Streamlit sidebar.

## Known limitations

- **Broad enumeration queries are weak.** Something like "list every add-on
  cover available" wants ~80 separate sections but only the top-K (default
  8) most relevant chunks are retrieved. Ask about a specific add-on
  ("what does Engine Protector cover?") instead, or raise `top_k` /
  `TOP_K` in `.env` for broader questions.
- **Heading detection is heuristic**, not a layout-aware PDF parser. It
  occasionally mis-tags a boilerplate line (e.g. an Ombudsman office city
  name in a list) as a section heading. This produces a few noisy
  low-value chunks but doesn't affect retrieval of real content.
- **Gemini generation** is used for answers; if you see 429 errors there,
  it's the same rate-limiting behavior observed with embeddings on this
  particular key — retry after a short wait, or use a different key.
- **Single-writer Qdrant.** Local mode takes an exclusive file lock; only
  one of ingest/CLI/Streamlit can be open at a time.

## Extending to more insurers/products

The metadata schema (`rag/metadata.py`) and collection design were built to
extend beyond this single-insurer, single-LOB corpus: add new insurers by
extending `DOCUMENT_METADATA`, and the `insurer`/`product`/`lob`/
`vehicle_type` filters already work for narrowing retrieval once more
insurers are ingested into the same Qdrant collection.
