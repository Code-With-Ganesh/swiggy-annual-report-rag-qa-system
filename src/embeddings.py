# embeddings.py - generates vector embeddings using sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

EMBED_MODEL = "all-MiniLM-L6-v2"  # 384-dim, lightweight

def load_model(name=EMBED_MODEL):
    print(f"Loading embedding model: {name}...")
    model = SentenceTransformer(name)
    print("Embedding model loaded.")
    return model

def get_embeddings(chunks, model):
    """encodes chunk texts into numpy vectors for FAISS"""
    texts = [c["text"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    vecs = model.encode(texts, show_progress_bar=True, batch_size=32)
    vecs = np.array(vecs, dtype="float32")  # FAISS needs float32
    print(f"Embeddings shape: {vecs.shape}")
    return vecs


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, "src")
    from pdf_loader import load_pdf
    from text_chunker import make_chunks

    pages = load_pdf(os.path.join("data", "Swiggy Annual-Report-FY-2023-24.pdf"))
    chunks = make_chunks(pages)
    model = load_model()
    embs = get_embeddings(chunks, model)
    print(f"\nSample (first 10 values): {embs[0][:10]}")
