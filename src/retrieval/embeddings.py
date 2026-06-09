from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


@dataclass
class SearchResult:
    index: int
    score: float
    text: str
    meta: Any = None


class EmbeddingRetriever:
    """
    Oberklasse für den Embedding-basierten Retriever.

    :param model_name: Name des SentenceTransformer Modells
    :param device: Gerät für die Berechnung ("cpu" oder "cuda")
    :param normalize: Normalisieren der Embeddings für Cosine Similarity
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize

        self.model = SentenceTransformer(model_name, device=self.device)

        self._emb: Optional[np.ndarray] = None
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
        batch_size: int = 64,
        show_progress_bar: bool = True,
    ) -> None:
        """
        Baut den Embedding-Index über den gegebenen Texten auf.
        
        :param texts: Beschreibung der Texte
        :param meta: Beschreibung der Metadaten
        :param batch_size: Beschreibung der Batch-Größe
        :param show_progress_bar: Anzeige des Fortschrittsbalkens
        """
        texts = list(texts)
        self._texts = texts

        if meta is not None:
            meta_list = list(meta)
            if len(meta_list) != len(texts):
                raise ValueError("meta und texts müssen gleich lang sein.")
            self._meta = meta_list
        else:
            self._meta = None

        # E5 Modell erwartet diesen Prefix
        passages = [f"passage: {t}" for t in texts]

        emb = self.model.encode(
            passages,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress_bar,
        )
        self._emb = emb.astype(np.float32, copy=False)
        self._fitted = True

    def load_embeddings(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        meta: Optional[List[Any]] = None,
    ) -> None:
        """
        Lädt vorab berechnete Embeddings.
        
        :param embeddings: Vorab berechnete Embeddings
        :param texts: Liste der zugehörigen Texte
        :param meta: Optionale Metadaten
        """
        self._emb = embeddings.astype(np.float32, copy=False)
        self._texts = texts
        self._meta = meta
        self._fitted = True

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Sucht die top_k ähnlichsten Texte zur Query.
        
        :param query: Suchanfrage
        :param top_k: Anzahl der zurückzugebenden Ergebnisse
        :return: Liste der Suchergebnisse
        """
        if not self._fitted or self._emb is None:
            raise RuntimeError("Retriever nicht fit.")

        q = query.strip()
        if not q:
            return []

        q_emb = self.model.encode(
            [f"query: {q}"],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)

        scores = (self._emb @ q_emb[0]).astype(np.float32)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results: List[SearchResult] = []
        for idx in top_idx:
            meta = self._meta[idx] if self._meta is not None else None
            results.append(SearchResult(index=int(idx), score=float(scores[idx]), text=self._texts[idx], meta=meta))
        return results
