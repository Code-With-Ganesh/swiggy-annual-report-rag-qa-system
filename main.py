# main.py - CLI entry point for the Swiggy Report QA system
# run this to ask questions about the annual report interactively

import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.rag_pipeline import build_db, index_exists
from src.embeddings import load_model
from src.vector_store import load_from_disk
from src.query_engine import init_gemini, ask

INDEX_DIR = "vector_db"
COOLDOWN = 5  # seconds between queries

def main():
    print("="*50)
    print("  SWIGGY ANNUAL REPORT QA SYSTEM")
    print("="*50)

    # load or build the index
    if index_exists(INDEX_DIR):
        print("\nFound existing index, loading...")
        index, chunks = load_from_disk(INDEX_DIR)
    else:
        print("\nNo index found, building from PDF (this takes a while)...")
        index, chunks = build_db()

    # load embedding model for encoding questions
    print("\nLoading embedding model...")
    embed_model = load_model()

    # setup gemini
    gemini = init_gemini()

    print("\n" + "="*50)
    print("Ready! Ask anything about the Swiggy Annual Report.")
    print("Type 'quit' to exit.")
    print("="*50)

    last_time = 0.0

    while True:
        print()
        q = input("Your question: ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # basic cooldown so we dont hit rate limits
        elapsed = time.time() - last_time
        if elapsed < COOLDOWN and last_time > 0:
            wait = COOLDOWN - elapsed
            print(f"Waiting {wait:.0f}s (rate limit)...")
            time.sleep(wait)

        try:
            result = ask(q, gemini, index, chunks, embed_model, top_k=5)
            last_time = time.time()

            print(f"\nAnswer: {result['answer']}")
            print(f"Source pages: {result['sources']}")

            # show context snippets
            print("\n--- Context ---")
            for i, c in enumerate(result["context"], 1):
                snippet = c["text"][:200].replace("\n", " ")
                print(f"\n[{i}] Page {c['page_num']} (score: {c['score']:.2f})")
                print(f"    {snippet}...")
            print("-"*40)

        except Exception as e:
            print(f"\nError: {e}")
            print("Try a different question.")

if __name__ == "__main__":
    main()
