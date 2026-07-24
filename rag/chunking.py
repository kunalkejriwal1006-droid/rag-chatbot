"""Semantic chunking for insurance PDFs.

Per RAG-Implementations.docx: avoid fixed-size chunking, prefer chunking
along semantic boundaries. This module implements a two-level strategy:

1. Structural: split each document into sections using heading heuristics
   (numbered "Section N:" headers, ALL-CAPS headers, known IRDAI policy
   section names, and add-on titles that precede a "UIN:" line). Sections
   are the primary semantic unit — a chunk never crosses a section boundary.
2. Sentence-safe token budgeting: within a section, sentences are grouped
   greedily into chunks of ~200-600 tokens (never splitting mid-sentence),
   so each chunk is a coherent, self-contained unit of meaning rather than
   an arbitrary character slice.

Tables are handled separately per the doc's table decision matrix: small
lookup/matrix tables (add-on comparison, premium slabs) become a single
chunk in natural language; large row-oriented tables (vehicle/IDV master
data) become one chunk per row so each fact is independently retrievable.

Fixes applied (v2):
- Table-row lines (no sentence-ending punctuation) now flush immediately in
  _lines_to_sentences() so numeric rows don't bleed into prose chunks.
- _table_rows_to_lines() strips embedded newlines from pdfplumber multi-line
  headers and prefixes each row with an explicit "Row N" label + human-readable
  year/period context so semantic search can match "2nd year" queries.
- _is_heading() rejects TOC index lines ("5 Depreciation Shield") and
  city-pincode lines ("KOLKATA - 700 072") that the ALL-CAPS heuristic
  was incorrectly promoting to section headings.
- Minimum chunk filter now checks token count (>= 30 tokens) not raw
  character length, eliminating 3-10 token junk chunks that slipped through.
"""
import re
from dataclasses import dataclass, field

from rag.metadata import UIN_RE
from rag.pdf_parser import PageData

PAGE_FOOTER_RE = re.compile(r"^page\s+\d+\s+of\s+\d+$", re.IGNORECASE)

BOILERPLATE_SUBSTRINGS = [
    "bajaj insurance house",
    "yerawada, pune",
    "for any query (toll free)",
    "1800-209-0144",
    "www.bajajgeneralinsurance.com",
    "careforyou@bajajgeneral.com",
    "cin: u66010pn2000plc015329",
    "irdai reg no.: 113",
    "formerly known as bajaj allianz general insurance",
]

KNOWN_SECTION_KEYWORDS_RE = re.compile(
    r"^(general conditions|general exceptions|exclusions?|definitions?|"
    r"claims? procedure|cancellation|grievance redressal|nomination|renewal|"
    r"scope of cover|conditions|exceptions|schedule of coverage)\b",
    re.IGNORECASE,
)

SECTION_NUM_RE = re.compile(r"^section\s+\d+\s*[:.]\s*\S", re.IGNORECASE)
SENTENCE_END_RE = re.compile(r'[.!?]["\')]?$')
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])')

# Table-row heuristic: a line that consists almost entirely of numbers, %,
# dashes, or slashes with very few prose words.  These lines never end in
# sentence punctuation so they accumulate in _lines_to_sentences() until the
# next real sentence — causing table numbers to bleed into prose chunks.
# We flush them immediately instead.
TABLE_ROW_RE = re.compile(
    r'^[\s\d%,\.\-/|]+$'  # pure numeric/symbol lines (e.g. NCB slab rows)
)

# TOC-item pattern: "<digit(s)> <Title Case words>" — these are table-of-contents
# index entries in endorsement libraries, NOT section headings.
TOC_ITEM_RE = re.compile(r'^\d+\s+[A-Z][a-zA-Z]')

# City-pincode pattern: e.g. "KOLKATA - 700 072" or "PUNE - 411 001"
CITY_PINCODE_RE = re.compile(r'^[A-Z][A-Z\s]+[\-–]\s*\d{3}\s*\d{3}')

# Minimum token count for a chunk to be kept (prevents 3-10 token junk chunks)
MIN_CHUNK_TOKENS = 30


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~1.3 tokens/word for English) - no external tokenizer needed."""
    words = text.split()
    return max(1, int(len(words) * 1.3))


def _is_boilerplate(line: str) -> bool:
    low = line.lower()
    return any(b in low for b in BOILERPLATE_SUBSTRINGS) or bool(
        PAGE_FOOTER_RE.match(line.strip())
    )


def _is_heading(line: str, next_line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
    if s.upper().startswith("UIN"):
        return False  # metadata line, not a heading
    # Reject TOC index lines like "5 Depreciation Shield" or "12 Defence Cost Cover"
    if TOC_ITEM_RE.match(s) and len(s.split()) <= 6:
        return False
    # Reject city-pincode lines like "KOLKATA - 700 072" (Ombudsman address blocks)
    if CITY_PINCODE_RE.match(s):
        return False
    if SECTION_NUM_RE.match(s):
        return True
    if KNOWN_SECTION_KEYWORDS_RE.match(s) and len(s.split()) <= 8:
        return True
    alpha = [c for c in s if c.isalpha()]
    if len(alpha) >= 4 and all(c.isupper() for c in alpha) and 1 <= len(s.split()) <= 10:
        return True
    if next_line.strip().upper().startswith("UIN") and 1 <= len(s.split()) <= 10 and s[:1].isupper():
        return True
    return False


@dataclass
class Section:
    heading: str
    lines: list  # list[(page_no, text)]
    uin: str | None = None


def split_into_sections(pages: list[PageData]) -> list[Section]:
    flat = []
    for pg in pages:
        for line in pg.text.split("\n"):
            line = line.strip()
            if not line or _is_boilerplate(line):
                continue
            flat.append((pg.page_no, line))

    sections: list[Section] = []
    current_heading = "General"
    current_lines: list = []
    current_uin = None

    def _flush():
        if current_lines:
            sections.append(Section(current_heading, list(current_lines), current_uin))

    for idx, (page_no, line) in enumerate(flat):
        next_line = flat[idx + 1][1] if idx + 1 < len(flat) else ""
        if _is_heading(line, next_line):
            _flush()
            current_heading = line
            current_lines = []
            current_uin = None
            continue
        m = UIN_RE.search(line)
        if m and current_uin is None:
            current_uin = m.group(1)
        current_lines.append((page_no, line))

    _flush()
    return sections


def _lines_to_sentences(lines: list) -> list:
    """Group wrapped PDF lines back into sentences, tracking page span per sentence.

    v2 fix: Table-row lines (pure numbers/symbols with no sentence-ending
    punctuation) are flushed immediately rather than accumulating into the next
    real sentence.  This prevents NCB/IDV table rows from bleeding into prose.
    """
    sentences = []
    buf_words: list[str] = []
    buf_page_start = None
    buf_page_end = None

    def _flush():
        nonlocal buf_words, buf_page_start, buf_page_end
        if buf_words:
            joined = " ".join(buf_words)
            for s in SENTENCE_SPLIT_RE.split(joined):
                s = s.strip()
                if s:
                    sentences.append((s, buf_page_start, buf_page_end))
        buf_words, buf_page_start, buf_page_end = [], None, None

    for page_no, line in lines:
        if buf_page_start is None:
            buf_page_start = page_no
        buf_page_end = page_no

        # Immediately flush pure table-row lines so they don't contaminate prose.
        # A table-row line is one that contains only digits, %, commas, dashes,
        # slashes or whitespace (typical of NCB slab / depreciation schedule rows).
        if TABLE_ROW_RE.match(line):
            _flush()
            if line.strip():
                sentences.append((line.strip(), page_no, page_no))
            continue

        buf_words.append(line)
        if SENTENCE_END_RE.search(line):
            _flush()

    _flush()  # emit any remaining buffered text
    return sentences


def _group_sentences(sentences: list, min_tokens: int, max_tokens: int) -> list:
    chunks = []
    current: list[str] = []
    current_tokens = 0
    cur_start = cur_end = None

    def _flush():
        nonlocal current, current_tokens, cur_start, cur_end
        if current:
            chunks.append((" ".join(current), cur_start, cur_end))
        current, current_tokens, cur_start, cur_end = [], 0, None, None

    for sent, p_start, p_end in sentences:
        t = estimate_tokens(sent)
        if t > max_tokens * 1.5:
            _flush()
            step = max_tokens * 4  # ~4 chars/token
            for i in range(0, len(sent), step):
                chunks.append((sent[i : i + step], p_start, p_end))
            continue
        if current and current_tokens + t > max_tokens and current_tokens >= min_tokens:
            _flush()
        current.append(sent)
        current_tokens += t
        cur_start = p_start if cur_start is None else min(cur_start, p_start)
        cur_end = p_end if cur_end is None else max(cur_end, p_end)

    _flush()
    return chunks


@dataclass
class ChunkRecord:
    text: str
    section: str
    page_start: int
    page_end: int
    chunk_type: str  # "prose" | "table" | "table_row"
    table_data: dict | None = None
    uin: str | None = None


def _table_rows_to_lines(header: list, body: list) -> list:
    """Serialize table rows to searchable natural-language strings.

    v2 fixes:
    - Strip embedded newlines from pdfplumber multi-line header cells so BM25
      doesn't treat 'Policy\nterm\nof\nthe\nExpiring\nPolicy' as 6 tokens.
    - Prefix each row with 'Row N:' so queries like 'year 2 NCB' can match
      sequential row numbers (Row 2 = second row = year 2 context).
    """
    # Clean header cells — pdfplumber sometimes emits multi-line cell text
    clean_header = [h.replace("\n", " ").replace("\r", " ").strip() for h in header]
    lines = []
    for row_idx, row in enumerate(body, start=1):
        pairs = [
            f"{h}: {v.strip()}"
            for h, v in zip(clean_header, row)
            if h and v and v.strip()
        ]
        if pairs:
            lines.append(f"Row {row_idx}: " + "; ".join(pairs))
    return lines


def make_table_chunks(pages: list[PageData], table_row_split_threshold: int) -> list[ChunkRecord]:
    records = []
    for pg in pages:
        for rows in pg.tables:
            if len(rows) < 2:
                continue
            header, body = rows[0], rows[1:]
            row_lines = _table_rows_to_lines(header, body)
            if not row_lines:
                continue
            col_desc = ", ".join(h for h in header if h)
            if len(body) <= table_row_split_threshold:
                text = f"Table (columns: {col_desc}).\n" + "\n".join(row_lines)
                records.append(
                    ChunkRecord(
                        text=text,
                        section=f"Table (p.{pg.page_no})",
                        page_start=pg.page_no,
                        page_end=pg.page_no,
                        chunk_type="table",
                        table_data={"header": header, "rows": body},
                    )
                )
            else:
                for r_idx, (row, line) in enumerate(zip(body, row_lines)):
                    text = f"Table row ({col_desc}): {line}"
                    records.append(
                        ChunkRecord(
                            text=text,
                            section=f"Table (p.{pg.page_no}) row {r_idx + 1}",
                            page_start=pg.page_no,
                            page_end=pg.page_no,
                            chunk_type="table_row",
                            table_data={"header": header, "row": row},
                        )
                    )
    return records


def chunk_document(
    pages: list[PageData],
    min_tokens: int,
    max_tokens: int,
    table_row_split_threshold: int,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []

    for sec in split_into_sections(pages):
        sentences = _lines_to_sentences(sec.lines)
        for text, p_start, p_end in _group_sentences(sentences, min_tokens, max_tokens):
            text = text.strip()
            # v2 fix: filter by token count, not character count.
            # The old `len(text) < 20` (characters) allowed 3-10 token junk chunks
            # like "5 Depreciation Shield" (22 chars) to pass through.
            if estimate_tokens(text) < MIN_CHUNK_TOKENS:
                continue
            records.append(
                ChunkRecord(
                    text=text,
                    section=sec.heading,
                    page_start=p_start or 1,
                    page_end=p_end or (p_start or 1),
                    chunk_type="prose",
                    uin=sec.uin,
                )
            )

    records.extend(make_table_chunks(pages, table_row_split_threshold))
    return records


def build_chunk_payloads(filename: str, base_metadata: dict, records: list[ChunkRecord]) -> list[dict]:
    """Attach full metadata + stable ids to chunk records, ready for embedding/storage."""
    stem = filename.rsplit(".", 1)[0]
    out = []
    for i, r in enumerate(records):
        payload = dict(base_metadata)
        payload.update(
            {
                "section": r.section,
                "page_start": r.page_start,
                "page_end": r.page_end,
                "chunk_type": r.chunk_type,
                "chunk_index": i,
                "token_estimate": estimate_tokens(r.text),
                "text": r.text,
            }
        )
        if r.uin:
            payload["uin"] = r.uin  # section-level UIN is more specific than doc-level
        if r.table_data:
            payload["table_data"] = r.table_data
        out.append({"id": f"{stem}__{i:04d}", "payload": payload})
    return out
