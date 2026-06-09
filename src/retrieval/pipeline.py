"""
Konfigurierbare Such-Pipeline.
Kombiniert TF-IDF, Embeddings, RRF, Cross-Encoder Reranker, NLI und LLM-Reranker.
Alle Parameter werden aus config.py gelesen.
"""
import json
import os
import sys
from collections import defaultdict, namedtuple
from typing import List

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import (
    EMBEDDING_MODEL, RERANKER_MODEL, NLI_MODEL, LLM_MODEL,
    K_RETRIEVAL, K_RRF, K_RERANKER, RRF_K,
)
from src.retrieval.tfidf import TfidfRetriever
from src.retrieval.embeddings import EmbeddingRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.nli import NLIVerifier
from src.retrieval.llm_reranker import LLMReranker

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()


SearchResult = namedtuple("SearchResult", ["text", "index", "meta", "score"])


# Vordefinierte Pipeline-Konfigurationen
# 1 = TF-IDF only
# 2 = Embeddings only
# 3 = TF-IDF + Embeddings (RRF)
# 4 = TF-IDF + Embeddings + Reranker
# 5 = TF-IDF + Embeddings + Reranker + NLI
# 6 = TF-IDF + Embeddings + LLM
# 7 = TF-IDF + Embeddings + Reranker + LLM
PIPELINE_PRESETS = {
    1: dict(use_tfidf=True,  use_embeddings=False, use_reranker=False, use_nli=False, use_llm=False),
    2: dict(use_tfidf=False, use_embeddings=True,  use_reranker=False, use_nli=False, use_llm=False),
    3: dict(use_tfidf=True,  use_embeddings=True,  use_reranker=False, use_nli=False, use_llm=False),
    4: dict(use_tfidf=True,  use_embeddings=True,  use_reranker=True,  use_nli=False, use_llm=False),
    5: dict(use_tfidf=True,  use_embeddings=True,  use_reranker=True,  use_nli=True,  use_llm=False),
    6: dict(use_tfidf=True,  use_embeddings=True,  use_reranker=False, use_nli=False, use_llm=True),
    7: dict(use_tfidf=True,  use_embeddings=True,  use_reranker=True,  use_nli=False, use_llm=True),
}


def rrf_fuse(rank_lists: List[List[int]], k: int = RRF_K) -> dict:
    """
    Reciprocal Rank Fusion.
    rank_lists: list of lists of indices, ordered best→worst
    returns: dict index → rrf_score
    """
    scores = defaultdict(float)
    for ranked in rank_lists:
        for r, idx in enumerate(ranked, start=1):
            scores[idx] += 1.0 / (k + r)
    return scores


class SearchPipeline:
    """
    Konfigurierbare Such-Pipeline.

    Pipeline-Stufen (Parameter aus config.py):
      TF-IDF / Embeddings  →  K_RETRIEVAL Kandidaten je
      RRF Fusion           →  K_RRF Kandidaten
      Cross-Encoder        →  K_RERANKER Kandidaten
      NLI / LLM            →  top_k Ergebnisse

    preset: einer der Schlüssel aus PIPELINE_PRESETS, z.B. "tfidf+emb+rerank+llm"
    """

    def __init__(self, chunks_path: str, emb_path: str, preset: int):
        if preset not in PIPELINE_PRESETS:
            raise ValueError(f"Unbekanntes Preset '{preset}'. Verfügbar: 1–{len(PIPELINE_PRESETS)}")

        flags = PIPELINE_PRESETS[preset]
        use_tfidf      = flags["use_tfidf"]
        use_embeddings = flags["use_embeddings"]
        use_reranker   = flags["use_reranker"]
        use_nli        = flags["use_nli"]
        use_llm        = flags["use_llm"]

        self.preset         = preset
        self.use_tfidf      = use_tfidf
        self.use_embeddings = use_embeddings
        self.use_reranker   = use_reranker
        self.use_nli        = use_nli
        self.use_llm        = use_llm

        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = data["chunks"]
        self.texts  = [c["content"] for c in self.chunks]

        emb = np.load(emb_path)
        if emb.shape[0] != len(self.texts):
            raise ValueError("Embeddings passen nicht zur Chunk-Anzahl.")

        self.tfidf = TfidfRetriever(ngram_range=(1, 2)) if use_tfidf else None
        if self.tfidf:
            self.tfidf.fit(self.texts, meta=self.chunks)

        self.embeddings = None
        if use_embeddings:
            self.embeddings = EmbeddingRetriever(model_name=EMBEDDING_MODEL)
            self.embeddings.load_embeddings(embeddings=emb, texts=self.texts, meta=self.chunks)

        self.reranker = CrossEncoderReranker(model_name=RERANKER_MODEL) if use_reranker else None
        self.nli      = NLIVerifier(model_name=NLI_MODEL)               if use_nli      else None
        self.llm      = LLMReranker(model=LLM_MODEL)                    if use_llm      else None

    def get_config_name(self) -> str:
        parts = []
        if self.use_tfidf:      parts.append("TFIDF")
        if self.use_embeddings: parts.append("EMB")
        if self.use_reranker:   parts.append("RERANK")
        if self.use_nli:        parts.append("NLI")
        if self.use_llm:        parts.append("LLM")
        return "+".join(parts)

    def search(self, query: str, top_k: int = 10, verbose: bool = False):
        """Suche und gib top_k Ergebnisse zurück."""
        rank_lists = []

        if self.use_tfidf:
            res = self.tfidf.search(query, top_k=K_RETRIEVAL)
            rank_lists.append([r.index for r in res])
            if verbose:
                print(f"{len(rank_lists[-1])} TF-IDF Ergebnisse.")

        if self.use_embeddings:
            res = self.embeddings.search(query, top_k=K_RETRIEVAL)
            rank_lists.append([r.index for r in res])
            if verbose:
                print(f"{len(rank_lists[-1])} Embedding Ergebnisse.")

        if len(rank_lists) > 1:
            if verbose:
                print("Führe RRF durch...")
            fused = rrf_fuse(rank_lists)
            candidate_indices = sorted(fused, key=lambda i: fused[i], reverse=True)
        else:
            candidate_indices = rank_lists[0]

        candidate_indices = candidate_indices[:K_RRF]

        if self.use_reranker:
            if verbose:
                print(f"Starte Reranking mit {len(candidate_indices)} Kandidaten...")
            reranked = self.reranker.rerank(
                query=query,
                candidate_texts=[self.texts[i] for i in candidate_indices],
                candidate_meta=[self.chunks[i] for i in candidate_indices],
                candidate_indices=candidate_indices,
                top_k=K_RERANKER,
                batch_size=32,
            )
            candidate_indices = [r.index for r in reranked[:K_RERANKER]]
            if verbose:
                print("Reranking abgeschlossen.")

        if self.use_nli:
            if verbose:
                print("Starte NLI-Verifikation...")
            top_results = self.nli.score_entailment(
                hypothesis=query,
                premises=[self.texts[i] for i in candidate_indices],
                indices=candidate_indices,
                meta=[self.chunks[i] for i in candidate_indices],
                batch_size=16,
            )[:top_k]
            if verbose:
                print("NLI-Verifikation abgeschlossen.")
        else:
            top_results = [
                SearchResult(text=self.texts[i], index=i, meta=self.chunks[i], score=1.0)
                for i in candidate_indices[:top_k]
            ]

        if self.use_llm:
            if verbose:
                print("Starte LLM-Reranking...")
            top_results = self.llm.rerank(
                query=query,
                candidate_texts=[r.text for r in top_results],
                candidate_meta=[r.meta for r in top_results],
                candidate_indices=[r.index for r in top_results],
                top_k=top_k,
            )
            if verbose:
                print("LLM-Reranking abgeschlossen.")

        return top_results
