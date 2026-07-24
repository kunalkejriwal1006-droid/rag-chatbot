"""LLM answer generation, grounded strictly in retrieved context.

Supports dynamic fallback across three backends:
  1. "gemini" — Google Gemini API (preferred)
  2. "groq"   — Groq cloud API (fast fallback)
  3. "ollama" — Ollama local inference (slow fallback)
"""

import logging
from rag import calculator, config

logger = logging.getLogger("rag.generation")

# ---------------------------------------------------------------------------
# Lazy Initializers for Clients
# ---------------------------------------------------------------------------

_gemini_client = None
_groq_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured")
        # pyrefly: ignore [missing-import]
        from google import genai
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not configured")
        # pyrefly: ignore [missing-import]
        from groq import Groq as _GroqClient
        _groq_client = _GroqClient(api_key=config.GROQ_API_KEY)
    return _groq_client


SYSTEM_INSTRUCTION = """You are an assistant for an insurance brokerage, helping brokers quickly find accurate facts about Bajaj Allianz / Bajaj General Insurance two-wheeler motor policies.

Rules:
- Answer ONLY using the provided context chunks. Do not use outside knowledge about insurance.
- Group your factual claims into coherent paragraphs or bullet points.
- If the context genuinely does not contain relevant information, say so plainly instead of guessing — but try hard to find the answer first.
- When multiple documents disagree or cover different product variants, point out the difference rather than silently picking one.
- Be concise. Structure multi-point answers as bullet points.
- Cite the source document and page number for key facts (e.g., "per CO_7, p.5").

For calculation questions:
- If a CALCULATION RESULT block is present in the context, use those exact numbers in your answer — do not re-derive or contradict them.
- If the user asks a calculation question and no CALCULATION RESULT is present, look for relevant tables or figures (NCB slabs, depreciation schedules, premium rates, add-on pricing, etc.) in the context and perform the calculation step-by-step.
- Always show your work step-by-step when calculating:
  1. Identify the relevant values from the context.
  2. State the formula or rule being applied.
  3. Show the arithmetic.
  4. State the final answer clearly.
"""


# ---------------------------------------------------------------------------
# Individual Backend Callers
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str) -> str:
    client = _get_gemini_client()
    # pyrefly: ignore [missing-import]
    from google.genai import types as genai_types
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION, temperature=0.1
        ),
    )
    return resp.text


def _call_groq(prompt: str) -> str:
    client = _get_groq_client()
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def _call_ollama(prompt: str) -> str:
    # pyrefly: ignore [missing-import]
    import ollama as _ollama_lib
    resp = _ollama_lib.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.1},
    )
    return resp["message"]["content"]


# ---------------------------------------------------------------------------
# LLM call abstraction with dynamic fallback
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """Try to get response using configured backends in order of fallback list."""
    errors = []
    for backend in config.LLM_FALLBACK_ORDER:
        try:
            if backend == "gemini":
                logger.info(f"Attempting LLM call via Gemini ({config.GEMINI_MODEL})...")
                return _call_gemini(prompt)
            elif backend == "groq":
                logger.info(f"Attempting LLM call via Groq ({config.GROQ_MODEL})...")
                return _call_groq(prompt)
            elif backend == "ollama":
                logger.info(f"Attempting LLM call via Ollama ({config.OLLAMA_MODEL})...")
                return _call_ollama(prompt)
            else:
                logger.warning(f"Unknown backend in fallback list: {backend}")
        except Exception as e:
            err_msg = f"{backend.upper()} failed: {str(e)}"
            logger.warning(err_msg)
            errors.append(err_msg)

    # If all backends failed
    raise RuntimeError(
        "All LLM backends in fallback order failed. Errors:\n" + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_context(chunks: list[dict]) -> str:
    """Formats context chunks for the LLM, keeping citations clean."""
    formatted_blocks = []
    used_sources = []

    for chunk in chunks:
        formatted_blocks.append(chunk["text"])
        src = chunk.get("source_file", "Unknown")
        page = chunk.get("page_start", 1)
        used_sources.append(f"{src}, p.{page}")

    context_text = "\n\n".join(formatted_blocks)
    unique_sources = ", ".join(list(set(used_sources)))

    return f"Context:\n{context_text}\n\nSources for reference: {unique_sources}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_answer(query: str, chunks: list[dict]) -> dict:
    # --- Deterministic math short-circuit -----------------------------------
    calc = calculator.detect_and_calculate(query)
    calc_block = ""
    if calc:
        calc_block = (
            "\n\n--- CALCULATION RESULT (authoritative, do not contradict) ---\n"
            + calc["formatted_answer"]
            + "\n--- END CALCULATION RESULT ---\n\n"
        )

    if not chunks and not calc:
        return {
            "answer": "I couldn't find anything relevant in the ingested documents for that question.",
            "sources": [],
        }

    context = _format_context(chunks) if chunks else "No additional document context retrieved."
    prompt = f"Context:\n{calc_block}{context}\n\nQuestion: {query}\n\nAnswer:"

    answer = _call_llm(prompt)

    sources = sorted(
        {(c.get("source_file"), c.get("page_start")) for c in chunks},
        key=lambda x: (x[0] or "", x[1] or 0),
    )
    return {"answer": answer, "sources": sources}
