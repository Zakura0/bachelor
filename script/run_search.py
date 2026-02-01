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

    tfidf = TfidfRetriever(ngram_range=(1, 2))
    tfidf.fit(texts, meta=chunks)

    embeddings = EmbeddingRetriever(model_name="intfloat/multilingual-e5-base")
    embeddings.load_embeddings(embeddings=emb, texts=texts, meta=chunks)

    reranker = CrossEncoderReranker(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    K_TFIDF = 200
    K_EMBEDDINGS = 200
    TOP_OUT = 10

    print("Programm gestartet. 'exit' zum Beenden.\n")

    while True:
        query = input("Query: ").strip()
        if not query or query.lower() == "exit":
            break

        tfidf_res = tfidf.search(query, top_k=K_TFIDF)
        embeddings_res = embeddings.search(query, top_k=K_EMBEDDINGS)

        tfidf_ranked = [r.index for r in tfidf_res]
        embeddings_ranked = [r.index for r in embeddings_res]
        
        print(f"\n{len(tfidf_ranked)} TF-IDF Ergebnisse, {len(embeddings_ranked)} Embedding Ergebnisse.")
        print("Führe Reciprocal Rank Fusion durch...")

        fused = rrf_fuse([tfidf_ranked, embeddings_ranked], k=60)
        candidate_indices = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)
        candidate_indices = candidate_indices[:300]

        cand_texts = [texts[i] for i in candidate_indices]
        cand_meta = [chunks[i] for i in candidate_indices]
        
        print(f"\n{len(candidate_indices)} Kandidaten für Reranking ausgewählt.")
        print("Starte Reranking...")

        reranked = reranker.rerank(
            query=query,
            candidate_texts=cand_texts,
            candidate_meta=cand_meta,
            candidate_indices=candidate_indices,
            top_k=TOP_OUT,
            batch_size=32
        )
        
        print("Reranking abgeschlossen. Starte NLI-Verifikation...\n")
        
        nli = NLIVerifier(model_name="joeddav/xlm-roberta-large-xnli")

        texts = [r.text for r in reranked]
        meta = [r.meta for r in reranked]
        idxs = [r.index for r in reranked]

        nli_ranked = nli.score_entailment(
            hypothesis=query,
            premises=texts,
            indices=idxs,
            meta=meta,
            batch_size=16
        )
        
        for rank, r in enumerate(nli_ranked[:10], start=1):
            c = r.meta
            print(f"{rank}. entail={r.entailment:.3f} ({r.label}) | Chunk-ID={c['id']} | Start={c['start_index']}")
            print(r.text.replace("\n"," ")[:500])
            print("-"*80)


if __name__ == "__main__":
    main()
