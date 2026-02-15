import json
import os
import sys
from collections import defaultdict, namedtuple

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.retrieval.nli import NLIVerifier
from src.retrieval.tfidf import TfidfRetriever
from src.retrieval.embeddings import EmbeddingRetriever
from src.retrieval.reranker import CrossEncoderReranker

from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

SearchResult = namedtuple('SearchResult', ['text', 'index', 'meta', 'score'])


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


class AblationSearchPipeline:
    """Configurable search pipeline for ablation studies."""
    
    def __init__(self, chunks_path: str, emb_path: str, 
                 use_tfidf: bool = True,
                 use_embeddings: bool = True, 
                 use_reranker: bool = True,
                 use_nli: bool = True):
        """
        Initialize search pipeline with configurable components.
        
        Args:
            chunks_path: Path to chunks JSON
            emb_path: Path to embeddings NPY
            use_tfidf: Whether to use TF-IDF retrieval
            use_embeddings: Whether to use embedding-based retrieval
            use_reranker: Whether to use cross-encoder reranking
            use_nli: Whether to use NLI verification
        """
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = data["chunks"]
        self.texts = [c["content"] for c in self.chunks]
        self.use_tfidf = use_tfidf
        self.use_embeddings = use_embeddings
        self.use_reranker = use_reranker
        self.use_nli = use_nli
        
        if not use_tfidf and not use_embeddings:
            raise ValueError("At least one of TF-IDF or embeddings must be enabled")
        
        emb = np.load(emb_path)
        if emb.shape[0] != len(self.texts):
            raise ValueError("Embeddings passen nicht zur Chunk-Anzahl.")

        if self.use_tfidf:
            self.tfidf = TfidfRetriever(ngram_range=(1, 2))
            self.tfidf.fit(self.texts, meta=self.chunks)
        else:
            self.tfidf = None
            
        if self.use_embeddings:
            self.embeddings = EmbeddingRetriever(model_name="intfloat/multilingual-e5-base")
            self.embeddings.load_embeddings(embeddings=emb, texts=self.texts, meta=self.chunks)
        else:
            self.embeddings = None
            
        if self.use_reranker:
            self.reranker = CrossEncoderReranker(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        else:
            self.reranker = None
            
        if self.use_nli:
            self.nli = NLIVerifier(model_name="joeddav/xlm-roberta-large-xnli")
        else:
            self.nli = None

    def get_config_name(self) -> str:
        """Return a descriptive name for the current configuration."""
        components = []
        if self.use_tfidf:
            components.append("TFIDF")
        if self.use_embeddings:
            components.append("EMB")
        if self.use_reranker:
            components.append("RERANK")
        if self.use_nli:
            components.append("NLI")
        return "+".join(components) if components else "NONE"

    def search(self, query: str, top_k: int = 10, verbose: bool = False):
        """
        Run a search query through the configured pipeline.
        Returns list of results with text, index, meta, and score/entailment.
        """
        K_INITIAL = 200
        
        rank_lists = []
        
        if self.use_tfidf:
            tfidf_res = self.tfidf.search(query, top_k=K_INITIAL)
            tfidf_ranked = [r.index for r in tfidf_res]
            rank_lists.append(tfidf_ranked)
            if verbose:
                print(f"{len(tfidf_ranked)} TF-IDF Ergebnisse.")
        
        if self.use_embeddings:
            embeddings_res = self.embeddings.search(query, top_k=K_INITIAL)
            embeddings_ranked = [r.index for r in embeddings_res]
            rank_lists.append(embeddings_ranked)
            if verbose:
                print(f"{len(embeddings_ranked)} Embedding Ergebnisse.")
    
        if len(rank_lists) > 1:
            if verbose:
                print("Führe RRF durch...")
            fused = rrf_fuse(rank_lists, k=60)
            candidate_indices = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)
        else:
            candidate_indices = rank_lists[0]
        
        candidate_indices = candidate_indices[:300]
        
        if self.use_reranker:
            cand_texts = [self.texts[i] for i in candidate_indices]
            cand_meta = [self.chunks[i] for i in candidate_indices]
            
            if verbose:
                print(f"Starte Reranking mit {len(candidate_indices)} Kandidaten...")
            
            reranked = self.reranker.rerank(
                query=query,
                candidate_texts=cand_texts,
                candidate_meta=cand_meta,
                candidate_indices=candidate_indices,
                top_k=30,
                batch_size=32
            )
            
            candidate_indices = [r.index for r in reranked[:30]]
            
            if verbose:
                print("Reranking abgeschlossen.")
        
        if self.use_nli:
            cand_texts = [self.texts[i] for i in candidate_indices]
            cand_meta = [self.chunks[i] for i in candidate_indices]
            
            if verbose:
                print("Starte NLI-Verifikation...")
            
            nli_ranked = self.nli.score_entailment(
                hypothesis=query,
                premises=cand_texts,
                indices=candidate_indices,
                meta=cand_meta,
                batch_size=16
            )
            
            if verbose:
                print("NLI-Verifikation abgeschlossen.")
            
            return nli_ranked[:top_k]
        else:
            results = []
            for idx in candidate_indices[:top_k]:
                results.append(SearchResult(
                    text=self.texts[idx],
                    index=idx,
                    meta=self.chunks[idx],
                    score=1.0
                ))
            return results


def run_search_interactive(chunks_path: str, emb_path: str, 
                          use_tfidf: bool = True,
                          use_embeddings: bool = True,
                          use_reranker: bool = True,
                          use_nli: bool = True):
    """Interactive search loop with configurable pipeline."""
    pipeline = AblationSearchPipeline(
        chunks_path, emb_path,
        use_tfidf=use_tfidf,
        use_embeddings=use_embeddings,
        use_reranker=use_reranker,
        use_nli=use_nli
    )
    
    print(f"Ablation Search Pipeline: {pipeline.get_config_name()}")
    print("'exit' zum Beenden.\n")

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
            
            if hasattr(r, 'entailment'):
                print(f"{color}{rank}. Chunk-ID={c['id']} | Start={c['start_index']} | {r.entailment:.3f} ({r.label[0]})")
            else:
                print(f"{color}{rank}. Chunk-ID={c['id']} | Start={c['start_index']}")
            
            print(r.text.replace("\n"," ")[:500])
            print("-"*80 + COLOR_RESET)


def main():
    if len(sys.argv) < 3:
        print("Usage: python ablation_search.py <chunks_path> <embeddings_path> [--no-tfidf] [--no-embeddings] [--no-reranker] [--no-nli]")
        print("\nExample: python ablation_search.py data/processed/verwandlung_chunks.json data/processed/verwandlung.embeddings.npy --no-nli")
        sys.exit(1)
    
    chunks_path = sys.argv[1]
    emb_path = sys.argv[2]
    
    use_tfidf = "--no-tfidf" not in sys.argv
    use_embeddings = "--no-embeddings" not in sys.argv
    use_reranker = "--no-reranker" not in sys.argv
    use_nli = "--no-nli" not in sys.argv
    
    run_search_interactive(chunks_path, emb_path, use_tfidf, use_embeddings, use_reranker, use_nli)


if __name__ == "__main__":
    main()
