import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.retrieval.embeddings import EmbeddingRetriever


def main():
    # Input: deine Chunk-JSON
    in_path = os.path.join(project_root, "data", "processed", "verwandlung_chunks.json")

    # Output: Embeddings + ein kleines Meta-JSON (optional)
    out_emb_path = os.path.join(project_root, "data", "processed", "verwandlung.embeddings.npy")

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    texts = [c["content"] for c in chunks]

    retriever = EmbeddingRetriever(
        model_name="intfloat/multilingual-e5-base",
        device=None,  # auto: cuda wenn verfügbar
    )
    retriever.fit(texts, meta=chunks, batch_size=64)

    # Speichern
    assert retriever._emb is not None
    np.save(out_emb_path, retriever._emb)

    print(f"Saved embeddings: {out_emb_path} | shape={retriever._emb.shape}")


if __name__ == "__main__":
    main()
