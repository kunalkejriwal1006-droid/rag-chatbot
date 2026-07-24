"""Verify all calculator fixes work correctly."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from rag import calculator

test_queries = [
    # Original failing query
    "Premium is 5000 rs , i m on 2nd yr , what is the ncb benifits for me",
    # Abbreviations
    "ncb for 5000 rs premium 3rd yr",
    # Typos and informal language
    "what ncb discount do i get, premium 8000, 4th year, 0 claims",
    # Currency before number
    "ncb benefit rs 6000 premium 2nd year",
    # IDV calculation
    "what is idv for 2 year old bike price is 80000",
    # IDV with rs
    "idv for my 3 year old two wheeler cost rs 95000",
    # General NCB question (no numbers — should return None, fall through to RAG)
    "what is NCB and how does it work?",
    # Edge case: year but no premium
    "ncb for 3rd year",
]

for q in test_queries:
    result = calculator.detect_and_calculate(q)
    print(f"Query: {q!r}")
    if result:
        print(f"  ✓ calc_type={result['calc_type']}")
        print(f"  {result['formatted_answer'][:150]}")
    else:
        print(f"  → None (will fall through to RAG retrieval)")
    print()
