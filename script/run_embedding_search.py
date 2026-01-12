import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.retrieval.embeddings import EmbeddingRetriever


def main():
    chunks_path = os.path.join(project_root, "data", "processed", "verwandlung_chunks.json")
    emb_path = os.path.join(project_root, "data", "processed", "verwandlung.embeddings.npy")

    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    texts = [c["content"] for c in chunks]

    emb = np.load(emb_path)
    if emb.shape[0] != len(texts):
        raise ValueError(f"Embeddings rows ({emb.shape[0]}) != chunks ({len(texts)})")

    retriever = EmbeddingRetriever(model_name="intfloat/multilingual-e5-base")
    retriever.load_embeddings(embeddings=emb, texts=texts, meta=chunks)

    print(f"Embedding Retrieval bereit – {len(chunks)} Chunks im Index.")
    print("Tippe eine Query (oder 'exit').\n")

    while True:
        query = input("Query: ").strip()
        if not query or query.lower() in ("exit", "quit", ":q"):
            break

        print("-" * 60)
        results = retriever.search(query, top_k=10)
        for r in results:
            c = r.meta
            print(f"Score={r.score:.3f} | Chunk-ID={c['id']} | Start={c['start_index']} | Wörter={c.get('word_count','?')}")
            print(r.text.replace("\n", " ")[:500])
            if len(r.text) > 500:
                print("...")
            print("-" * 60)


if __name__ == "__main__":
    main()
