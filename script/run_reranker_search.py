import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.retrieval.embeddings import EmbeddingRetriever
from src.retrieval.reranker import CrossEncoderReranker


def main():
    chunks_path = os.path.join(project_root, "data", "processed", "verwandlung_chunks.json")
    emb_path = os.path.join(project_root, "data", "processed", "verwandlung.embeddings.npy")

    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Nicht gefunden: {chunks_path}")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Nicht gefunden: {emb_path} (erst build_embeddings.py laufen lassen)")

    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    texts = [c["content"] for c in chunks]

    emb = np.load(emb_path)
    if emb.shape[0] != len(texts):
        raise ValueError(f"Embeddings rows ({emb.shape[0]}) != chunks ({len(texts)})")

    # Stage 1: Bi-Encoder (Embeddings) -> Kandidatenpool
    bi = EmbeddingRetriever(model_name="intfloat/multilingual-e5-base")
    bi.load_embeddings(embeddings=emb, texts=texts, meta=chunks)

    # Stage 2: Cross-Encoder Reranker
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )

    print("Embedding+Rerank Retrieval – 'exit' zum Beenden.\n")

    while True:
        query = input("Query: ").strip()
        if not query or query.lower() in ("exit", "quit", ":q"):
            break

        # 1) Bi-Encoder Top-N (z.B. 50)
        first_stage = bi.search(query, top_k=50)

        cand_texts = [r.text for r in first_stage]
        cand_meta = [r.meta for r in first_stage]
        cand_indices = [r.index for r in first_stage]

        # 2) Cross-Encoder rerank Top-N -> Top-K
        reranked = reranker.rerank(
            query=query,
            candidate_texts=cand_texts,
            candidate_meta=cand_meta,
            candidate_indices=cand_indices,
            top_k=10,
            batch_size=32,
        )

        print("-" * 80)
        for rank, r in enumerate(reranked, start=1):
            c = r.meta
            print(
                f"{rank}. RerankScore={r.score:.3f} | Chunk-ID={c['id']} "
                f"| Start={c['start_index']} | Wörter={c.get('word_count','?')}"
            )
            print(r.text.replace("\n", " ")[:500])
            if len(r.text) > 500:
                print("...")
            print("-" * 80)


if __name__ == "__main__":
    main()
