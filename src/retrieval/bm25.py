"""
BM25-Retriever als Drop-in-Ersatz für TfidfRetriever.
Verwendet bm25s (C-backed, deutlich schneller als rank_bm25).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import bm25s


@dataclass
class SearchResult:
    index: int
    score: float
    text: str
    meta: object = None


class BM25Retriever:
    """
    BM25-Retriever mit derselben Schnittstelle wie TfidfRetriever.

    :param k1: TF-Sättigungsparameter (Standard: 1.5)
    :param b:  Längen-Normierungsparameter (Standard: 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._retriever: Optional[bm25s.BM25] = None
        self._texts: List[str] = []
        self._meta: Optional[List[Any]] = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        texts: Iterable[str],
        meta: Optional[Iterable[Any]] = None,
    ) -> None:
        """
        Baut den BM25-Index über den gegebenen Texten auf.

        :param texts: Input-Texte bzw. Chunks
        :param meta: Optionale Metadaten
        """
        texts = list(texts)
        self._texts = texts

        if meta is not None:
            meta_list = list(meta)
            if len(meta_list) != len(texts):
                raise ValueError(
                    f"Länge von 'meta' ({len(meta_list)}) "
                    f"stimmt nicht mit Länge von 'texts' ({len(texts)}) überein."
                )
            self._meta = meta_list
        else:
            self._meta = None

        tokenized = bm25s.tokenize(texts, stopwords=None)
        self._retriever = bm25s.BM25(k1=self.k1, b=self.b)
        self._retriever.index(tokenized)
        self._fitted = True

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """
        Sucht die top_k ähnlichsten Texte zur Query.

        :param query: Suchanfrage
        :param top_k: Anzahl der gewünschten Treffer
        :param min_score: Untergrenze für den BM25-Score
        :return: Liste von SearchResult, nach Score absteigend sortiert
        """
        if not self._fitted or self._retriever is None:
            raise RuntimeError("BM25Retriever ist noch nicht fit.")

        if not query.strip():
            return []

        top_k_capped = min(top_k, len(self._texts))
        query_tokens = bm25s.tokenize([query], stopwords=None)
        results, scores = self._retriever.retrieve(query_tokens, k=top_k_capped)

        out: List[SearchResult] = []
        for idx, score in zip(results[0], scores[0]):
            if score < min_score:
                break
            meta = self._meta[idx] if self._meta is not None else None
            out.append(SearchResult(
                index=int(idx),
                score=float(score),
                text=self._texts[idx],
                meta=meta,
            ))
        return out
