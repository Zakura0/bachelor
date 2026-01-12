import json
import os
import sys
from collections import defaultdict

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.retrieval.nli import NLIVerifier
from src.retrieval.tfidf import TfidfRetriever
from src.retrieval.embeddings import EmbeddingRetriever
from src.retrieval.reranker import CrossEncoderReranker


def rrf_fuse(rank_lists, k=60):
    """
    Reciprocal Rank Fusion.
    rank_lists: list of lists of indices, ordered best->worst
    returns: dict index -> rrf_score
    """
    scores = defaultdict(float)
    for ranked in rank_lists:
        for r, idx in enumerate(ranked, start=1):
            scores[idx] += 1.0 / (k + r)
    return scores


def main():
    chunks_path = os.path.join(project_root, "data", "processed", "verwandlung_chunks.json")
    emb_path = os.path.join(project_root, "data", "processed", "verwandlung.embeddings.npy")

    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    texts = [c["content"] for c in chunks]

    emb = np.load(emb_path)
    if emb.shape[0] != len(texts):
        raise ValueError("Embeddings passen nicht zur Chunk-Anzahl.")

    # --- Retriever initialisieren ---
    tfidf = TfidfRetriever(ngram_range=(1, 2))
    tfidf.fit(texts, meta=chunks)

    dense = EmbeddingRetriever(model_name="intfloat/multilingual-e5-base")
    dense.load_embeddings(embeddings=emb, texts=texts, meta=chunks)

    reranker = CrossEncoderReranker(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    K_SPARSE = 200
    K_DENSE = 200
    TOP_OUT = 10

    print("Hybrid (TF-IDF + Embeddings) -> Cross-Encoder Rerank. 'exit' zum Beenden.\n")

    while True:
        query = input("Query: ").strip()
        if not query or query.lower() in ("exit", "quit", ":q"):
            break

        # 1) Kandidaten aus beiden Retrievern
        sparse_res = tfidf.search(query, top_k=K_SPARSE)
        dense_res = dense.search(query, top_k=K_DENSE)

        sparse_ranked = [r.index for r in sparse_res]
        dense_ranked = [r.index for r in dense_res]

        # 2) RRF fusion score
        fused = rrf_fuse([sparse_ranked, dense_ranked], k=60)

        # 3) Candidate-Set: Union, sortiert nach fused score
        candidate_indices = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)

        # Begrenzen, damit Reranking nicht zu teuer wird:
        candidate_indices = candidate_indices[:300]  # typischerweise 200–500

        cand_texts = [texts[i] for i in candidate_indices]
        cand_meta = [chunks[i] for i in candidate_indices]

        # 4) Rerank
        reranked = reranker.rerank(
            query=query,
            candidate_texts=cand_texts,
            candidate_meta=cand_meta,
            candidate_indices=candidate_indices,
            top_k=TOP_OUT,
            batch_size=32
        )
        
        nli = NLIVerifier(model_name="joeddav/xlm-roberta-large-xnli")

        top_for_nli = reranked[:30]
        premises = [r.text for r in top_for_nli]
        meta = [r.meta for r in top_for_nli]
        idxs = [r.index for r in top_for_nli]

        nli_ranked = nli.score_entailment(
            hypothesis=query,
            premises=premises,
            indices=idxs,
            meta=meta,
            batch_size=16
        )

        # Ausgabe Top-10 nach entailment
        for rank, r in enumerate(nli_ranked[:10], start=1):
            c = r.meta
            print(f"{rank}. entail={r.entailment:.3f} ({r.label}) | Chunk-ID={c['id']} | Start={c['start_index']}")
            print(r.text.replace("\n"," ")[:500])
            print("-"*80)


if __name__ == "__main__":
    main()
