"""Interactive CLI for querying the insurance RAG system.

Usage:
    python cli.py                      # interactive chat
    python cli.py "your question here" # single-shot

In interactive mode, type /help for filter commands.
"""
import sys

from rag import config, generation, retrieval

HELP_TEXT = """
Commands:
  /filter key=value   Set a metadata filter (e.g. /filter document_type=Policy Wording)
                       Valid keys: product, document_type, vehicle_type, chunk_type, section
  /filter clear        Clear all filters
  /filters              Show active filters
  /expand on|off        Toggle parent-section context expansion (default: off)
  /help                  Show this message
  /exit or /quit         Leave
"""


def ask(query: str, filters: dict, expand: bool) -> None:
    # This searches your local Qdrant/BM25 index
    chunks = retrieval.hybrid_search(
        query, top_k=config.TOP_K, filters=filters or None, expand_context=expand
    )

    result = generation.generate_answer(query, chunks)
    print("\n" + result["answer"] + "\n")
    if result["sources"]:
        print("Sources:")
        for src, page in result["sources"]:
            print(f"  - {src}, p.{page}")
        print()
    if result["backend"]:
        print(f"(answered via {result['backend']})\n")


def repl() -> None:
    filters: dict = {}
    expand = False
    print("Insurance RAG CLI. Type /help for commands, /exit to quit.\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        if line == "/help":
            print(HELP_TEXT)
            continue
        if line == "/filters":
            print(filters or "(none)")
            continue
        if line.startswith("/filter"):
            arg = line[len("/filter") :].strip()
            if arg == "clear":
                filters = {}
                print("Filters cleared.")
            elif "=" in arg:
                k, v = arg.split("=", 1)
                filters[k.strip()] = v.strip()
                print(f"Filter set: {k.strip()}={v.strip()}")
            else:
                print("Usage: /filter key=value  or  /filter clear")
            continue
        if line.startswith("/expand"):
            arg = line[len("/expand") :].strip().lower()
            expand = arg == "on"
            print(f"Context expansion: {'on' if expand else 'off'}")
            continue

        ask(line, filters, expand)


def main() -> None:
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]), filters={}, expand=False)
    else:
        repl()


if __name__ == "__main__":
    main()
