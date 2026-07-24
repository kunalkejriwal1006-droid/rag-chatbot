"""Ingestion pipeline: PDFs -> parse -> semantic chunk -> embed -> store.

Usage:
    python -m rag.ingest [--rebuild] [--pdf-dir PATH] [--files a.pdf,b.pdf]

--rebuild drops and recreates the Qdrant collection first (use this after
changing chunking logic, or the first time you run ingestion).
--files restricts ingestion to specific filenames (comma-separated) - handy
for a quick smoke test before committing to embedding the full corpus.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from rag import chunking, config, embeddings, keyword_search, metadata, pdf_parser, vectorstore


def _sample_text(pages, n_pages: int = 2) -> str:
    return "\n".join(p.text for p in pages[:n_pages])


def ingest(pdf_dir: Path, rebuild: bool, files: list[str] | None = None) -> None:
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if files:
        wanted = set(files)
        pdf_files = [p for p in pdf_files if p.name in wanted]
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {pdf_dir}")

    all_payload_items = []  # [{"id": ..., "payload": {...}}]
    for pdf_path in pdf_files:
        t0 = time.time()
        pages = pdf_parser.parse_pdf(pdf_path)
        base_meta = metadata.get_base_metadata(pdf_path.name, _sample_text(pages))
        records = chunking.chunk_document(
            pages,
            min_tokens=config.CHUNK_MIN_TOKENS,
            max_tokens=config.CHUNK_MAX_TOKENS,
            table_row_split_threshold=config.TABLE_ROW_SPLIT_THRESHOLD,
        )
        items = chunking.build_chunk_payloads(pdf_path.name, base_meta, records)
        all_payload_items.extend(items)
        print(
            f"  {pdf_path.name}: {len(pages)} pages -> {len(items)} chunks "
            f"({time.time() - t0:.1f}s)"
        )

    if not all_payload_items:
        print("No chunks produced - aborting.")
        return

    ids = [it["id"] for it in all_payload_items]
    payloads = [it["payload"] for it in all_payload_items]
    for cid, p in zip(ids, payloads):
        p["chunk_id"] = cid

    print(f"\nTotal chunks: {len(payloads)}")
    type_counts = Counter(p["chunk_type"] for p in payloads)
    doctype_counts = Counter(p["document_type"] for p in payloads)
    avg_tokens = sum(p["token_estimate"] for p in payloads) / len(payloads)
    print(f"  by chunk_type: {dict(type_counts)}")
    print(f"  by document_type: {dict(doctype_counts)}")
    print(f"  avg token_estimate: {avg_tokens:.0f}")

    print(f"\nEmbedding chunks locally with {config.EMBEDDING_MODEL}...")
    texts = [p["text"] for p in payloads]
    t0 = time.time()
    vectors = embeddings.embed_documents_cached(ids, texts)
    print(f"  have {len(vectors)} chunk embeddings ({time.time() - t0:.1f}s this run, "
          f"dim={len(vectors[0])})")

    if rebuild:
        print("Rebuilding Qdrant collection...")
        vectorstore.recreate_collection(dim=len(vectors[0]))
    else:
        vectorstore.ensure_collection(dim=len(vectors[0]))

    print("Upserting into Qdrant...")
    vectorstore.upsert(ids, vectors, payloads)

    print("Building BM25 keyword index...")
    keyword_search.build_index(payloads)

    print(f"Writing manifest to {config.MANIFEST_PATH}...")
    with open(config.MANIFEST_PATH, "w", encoding="utf-8") as f:
        for cid, p in zip(ids, payloads):
            f.write(json.dumps({"id": cid, **p}, ensure_ascii=False) + "\n")

    print("\nIngestion complete.")


def main():
    parser = argparse.ArgumentParser(description="Ingest insurance PDFs into the RAG store.")
    parser.add_argument("--pdf-dir", type=str, default=None, help="Override PDF directory")
    parser.add_argument(
        "--rebuild", action="store_true", help="Drop and recreate the vector collection first"
    )
    parser.add_argument(
        "--files", type=str, default=None, help="Comma-separated filenames to restrict ingestion to"
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir).resolve() if args.pdf_dir else config.PDF_DIR
    files = [f.strip() for f in args.files.split(",")] if args.files else None
    ingest(pdf_dir, rebuild=args.rebuild, files=files)


if __name__ == "__main__":
    main()
