# vector_store.py - FAISS index operations (build, save, load, search)

import faiss
import numpy as np
import json, os

def create_index(embeddings):
    """builds a FAISS L2 index from the embedding matrix"""
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings)
    print(f"FAISS index built: {idx.ntotal} vectors, dim={dim}")
    return idx

def save_to_disk(index, chunks, folder="vector_db"):
    """persists the FAISS index and chunk metadata to files"""
    os.makedirs(folder, exist_ok=True)
    faiss.write_index(index, os.path.join(folder, "faiss_index.bin"))
    with open(os.path.join(folder, "chunks_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved index + metadata to {folder}/ ({index.ntotal} vectors, {len(chunks)} chunks)")

def load_from_disk(folder="vector_db"):
    """loads a previously saved index and metadata"""
    idx_path = os.path.join(folder, "faiss_index.bin")
    meta_path = os.path.join(folder, "chunks_metadata.json")
    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"No index found in '{folder}/'. Build it first.")
    index = faiss.read_index(idx_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {index.ntotal} vectors and {len(chunks)} chunks from {folder}/")
    return index, chunks

def find_similar(query_vec, index, chunks, top_k=5):
    """finds the top_k most similar chunks to the query vector"""
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            entry = chunks[idx].copy()
            entry["score"] = float(distances[0][i])
            results.append(entry)
    return results


if __name__ == "__main__":
    # quick sanity check with random data
    dim = 384
    dummy_vecs = np.random.rand(10, dim).astype("float32")
    dummy_chunks = [{"chunk_id": i, "text": f"test chunk {i}", "page_num": i+1} for i in range(10)]

    index = create_index(dummy_vecs)
    save_to_disk(index, dummy_chunks, folder="vector_db_test")
    loaded_idx, loaded_chunks = load_from_disk(folder="vector_db_test")

    q = np.random.rand(1, dim).astype("float32")
    hits = find_similar(q, loaded_idx, loaded_chunks, top_k=3)
    for h in hits:
        print(f"  Chunk {h['chunk_id']} (Page {h['page_num']}), score: {h['score']:.4f}")

    import shutil
    shutil.rmtree("vector_db_test")
    print("Test done.")
