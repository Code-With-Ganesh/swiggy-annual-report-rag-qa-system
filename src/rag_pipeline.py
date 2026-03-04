# rag_pipeline.py - runs the full indexing pipeline: PDF -> chunks -> embeddings -> FAISS

import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pdf_loader import load_pdf
from text_chunker import make_chunks
from embeddings import load_model, get_embeddings
from vector_store import create_index, save_to_disk, load_from_disk

PDF_PATH = os.path.join("data", "Swiggy Annual-Report-FY-2023-24.pdf")
INDEX_DIR = "vector_db"

def build_db(pdf_path=PDF_PATH, save_dir=INDEX_DIR):
    """runs the whole pipeline end to end and saves the index"""
    print("="*50)
    print("BUILDING VECTOR DATABASE")
    print("="*50)

    # extract text
    print("\n[1/4] Loading PDF...")
    pages = load_pdf(pdf_path)

    # chunk it
    print("\n[2/4] Chunking...")
    chunks = make_chunks(pages)

    # embed
    print("\n[3/4] Generating embeddings...")
    model = load_model()
    embs = get_embeddings(chunks, model)

    # build faiss index and save
    print("\n[4/4] Building index...")
    index = create_index(embs)
    save_to_disk(index, chunks, folder=save_dir)

    print(f"\nDone! {len(pages)} pages, {len(chunks)} chunks, saved to {save_dir}/")
    return index, chunks

def index_exists(folder=INDEX_DIR):
    """checks if we already have a built index"""
    p1 = os.path.join(folder, "faiss_index.bin")
    p2 = os.path.join(folder, "chunks_metadata.json")
    return os.path.exists(p1) and os.path.exists(p2)

if __name__ == "__main__":
    build_db()
