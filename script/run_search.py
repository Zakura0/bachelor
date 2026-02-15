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

from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()


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


class SearchPipeline:
    """Reusable search pipeline for queries."""    
    def __init__(self, chunks_path: str, emb_path: str):
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = data["chunks"]
        self.texts = [c["content"] for c in self.chunks]

        emb = np.load(emb_path)
        if emb.shape[0] != len(self.texts):
            raise ValueError("Embeddings passen nicht zur Chunk-Anzahl.")

        self.tfidf = TfidfRetriever(ngram_range=(1, 2))
        self.tfidf.fit(self.texts, meta=self.chunks)

        self.embeddings = EmbeddingRetriever(model_name="intfloat/multilingual-e5-base")
        self.embeddings.load_embeddings(embeddings=emb, texts=self.texts, meta=self.chunks)

        self.reranker = CrossEncoderReranker(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        self.nli = NLIVerifier(model_name="joeddav/xlm-roberta-large-xnli")

    def search(self, query: str, top_k: int = 10, verbose: bool = False):
        """
        Run a search query through the full pipeline.
        Returns list of NLI results with text, index, meta, entailment score.
        """
        K_TFIDF = 200
        K_EMBEDDINGS = 200

        tfidf_res = self.tfidf.search(query, top_k=K_TFIDF)
        embeddings_res = self.embeddings.search(query, top_k=K_EMBEDDINGS)

        tfidf_ranked = [r.index for r in tfidf_res]
        embeddings_ranked = [r.index for r in embeddings_res]
        
        if verbose:
            print(f"\n{len(tfidf_ranked)} TF-IDF Ergebnisse, {len(embeddings_ranked)} Embedding Ergebnisse.")
            print("Führe Reciprocal Rank Fusion durch...")

        fused = rrf_fuse([tfidf_ranked, embeddings_ranked], k=60)
        candidate_indices = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)
        candidate_indices = candidate_indices[:300]

        cand_texts = [self.texts[i] for i in candidate_indices]
        cand_meta = [self.chunks[i] for i in candidate_indices]
        
        if verbose:
            print(f"{len(candidate_indices)} Kandidaten für Reranking ausgewählt.")
            print("Starte Reranking...")

        reranked = self.reranker.rerank(
            query=query,
            candidate_texts=cand_texts,
            candidate_meta=cand_meta,
            candidate_indices=candidate_indices,
            top_k=30,
            batch_size=32
        )
        
        top_reranked = reranked[:30]
        
        if verbose:
            print("Reranking abgeschlossen. Starte NLI-Verifikation...")
        
        nli_texts = [r.text for r in top_reranked]
        nli_meta = [r.meta for r in top_reranked]
        nli_idxs = [r.index for r in top_reranked]

        nli_ranked = self.nli.score_entailment(
            hypothesis=query,
            premises=nli_texts,
            indices=nli_idxs,
            meta=nli_meta,
            batch_size=16
        )
        
        if verbose:
            print("NLI-Verifikation abgeschlossen.")

        return nli_ranked[:top_k]


def run_search_interactive(chunks_path: str, emb_path: str):
    """Interactive search loop."""
    pipeline = SearchPipeline(chunks_path, emb_path)

    print("Zusammenfassungssuche. 'exit' zum Beenden.\n")

    while True:
        query = input("Query: ").strip()
        if not query or query.lower() == "exit":
            break

        results = pipeline.search(query, top_k=10, verbose=True)
        
        print("\n Top-Ergebnisse:\n")
        
        COLOR_CYAN = "\033[96m"
        COLOR_YELLOW = "\033[93m"
        COLOR_RESET = "\033[0m"
        
        for rank, r in enumerate(results, start=1):
            c = r.meta
            color = COLOR_CYAN if rank % 2 == 1 else COLOR_YELLOW
            print(f"{color}{rank}. Chunk-ID={c['id']} | Start={c['start_index']} | {r.entailment:.3f} ({r.label[0]})")
            print(r.text.replace("\n"," ")[:500])
            print("-"*80 + COLOR_RESET)


def main():
    if len(sys.argv) < 3:
        print("Bitte gib Chunks- und Embeddings-Pfad an.")
        sys.exit(1)
    run_search_interactive(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
