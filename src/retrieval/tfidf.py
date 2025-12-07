from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SearchResult:
    """
    Repräsentiert einen einzelnen Retrieval-Treffer.
    """
    index: int          # Index des Textes im ursprünglichen Korpus
    score: float        # Ähnlichkeits-Score (Cosine Similarity)
    text: str           # Gefundener Text (z.B. Chunk-Content)
    meta: object = None # Zusätzliche Metadaten (z.B. Chunk-Dict)


class TfidfRetriever:
    """
    Einfacher TF-IDF basierter Retriever.

    Typische Verwendung:

    >>> retriever = TfidfRetriever()
    >>> retriever.fit(texts, meta=chunks)
    >>> results = retriever.search("Gregor wacht als Ungeziefer auf", top_k=5)
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: Optional[int] = None,
        stop_words: Optional[str] = None,
    ) -> None:
        """
        :param ngram_range: z.B. (1, 2) für Unigramme + Bigramme
        :param max_features: Optional Begrenzung der Anzahl Features
        :param stop_words: z.B. 'german' (von scikit-learn unterstützt) oder None
        """
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words=stop_words,
        )
        self._X = None  # TF-IDF-Matrix
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
        Baut den TF-IDF-Index über den gegebenen Texten auf.

        :param texts: Iterable von Strings (z.B. Chunk-Contents)
        :param meta: Optionale Metadaten, gleiche Länge wie texts
                     (z.B. komplette Chunk-Dicts aus deiner JSON)
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

        self._X = self.vectorizer.fit_transform(texts)
        self._fitted = True

    def _ensure_fitted(self) -> None:
        if not self._fitted or self._X is None:
            raise RuntimeError(
                "TfidfRetriever ist noch nicht fit(). "
                "Rufe zuerst retriever.fit(texts, ...) auf."
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """
        Sucht die top_k ähnlichsten Texte zur Query.

        :param query: Suchanfrage (z.B. ein Zusammenfassungssatz)
        :param top_k: Anzahl der gewünschten Treffer
        :param min_score: Untergrenze für Cosine Similarity (0.0–1.0)
        :return: Liste von SearchResult, nach Score absteigend sortiert
        """
        self._ensure_fitted()

        if not query.strip():
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._X)[0]

        # Indizes nach Score sortieren (absteigend)
        sorted_indices = sims.argsort()[::-1]

        results: List[SearchResult] = []
        for idx in sorted_indices:
            score = float(sims[idx])
            if score < min_score:
                # Alles darunter ignorieren
                continue

            text = self._texts[idx]
            meta = self._meta[idx] if self._meta is not None else None

            results.append(
                SearchResult(
                    index=idx,
                    score=score,
                    text=text,
                    meta=meta,
                )
            )
            if len(results) >= top_k:
                break

        return results
