"""PDF parsing: per-page text (PyMuPDF) + per-page table detection (pdfplumber).

PyMuPDF gives cleaner running text (better ligature/whitespace handling) so
it is used for the prose that feeds section/paragraph chunking. pdfplumber
is used only for its grid-line table detector, since insurance PDFs put
their highest-value facts (premium slabs, IDV tables, add-on matrices) in
actual tables per RAG-Implementations.docx.
"""
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import pdfplumber


@dataclass
class PageData:
    page_no: int  # 1-indexed
    text: str
    tables: list = field(default_factory=list)  # list of list[list[str]]


def _extract_tables_for_page(page) -> list:
    try:
        raw_tables = page.extract_tables()
    except Exception:
        return []
    cleaned = []
    for t in raw_tables:
        rows = [[(c or "").strip() for c in row] for row in t]
        # Drop rows that are entirely empty (common pdfplumber artifact)
        rows = [r for r in rows if any(cell for cell in r)]
        # Ignore trivial/false-positive tables (need at least 2 rows x 2 cols)
        if len(rows) >= 2 and len(rows[0]) >= 2:
            cleaned.append(rows)
    return cleaned


def parse_pdf(path: Path) -> list[PageData]:
    """Return one PageData per page with text and any detected tables."""
    doc = fitz.open(path)
    texts = [doc[i].get_text() for i in range(len(doc))]
    doc.close()

    tables_per_page = [[] for _ in texts]
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= len(tables_per_page):
                    break
                tables_per_page[i] = _extract_tables_for_page(page)
    except Exception:
        # If pdfplumber chokes on a malformed PDF, degrade gracefully to
        # text-only extraction rather than aborting ingestion.
        pass

    return [
        PageData(page_no=i + 1, text=texts[i], tables=tables_per_page[i])
        for i in range(len(texts))
    ]
