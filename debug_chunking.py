from rag.pdf_parser import parse_pdf
from rag.chunking import chunk_document
from pathlib import Path

# 1. Choose a PDF file
pdf_path = "BAJAJ ALLIANZ GENERAL INSURANCE CO_1.pdf"

# 2. Parse the PDF
pages = parse_pdf(Path(pdf_path))

# 3. Run the chunker
chunks = chunk_document(
    pages, 
    min_tokens=200, 
    max_tokens=600, 
    table_row_split_threshold=8
)

# 4. Print the output
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ---")
    print(f"Section: {chunk.section}")
    print(f"Type: {chunk.chunk_type}")
    print(f"Text: {chunk.text[:200]}...") # Print first 200 chars