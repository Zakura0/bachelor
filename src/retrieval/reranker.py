from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch
from sentence_transformers import CrossEncoder


@dataclass
class RerankResult:
    index: int
    score: float
    text: str
    meta: Any = None


class CrossEncoderReranker:
    """
    Cross-Encoder Reranker:
    - bekommt (query, passage) Paare
    - gibt Scores zurück (höher = besser)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidate_texts: Sequence[str],
        candidate_meta: Optional[Sequence[Any]] = None,
        candidate_indices: Optional[Sequence[int]] = None,
        top_k: int = 10,
        batch_size: int = 32,
    ) -> List[RerankResult]:
        q = query.strip()
        if not q:
            return []

        if candidate_meta is not None and len(candidate_meta) != len(candidate_texts):
            raise ValueError("candidate_meta muss die gleiche Länge haben wie candidate_texts.")
        if candidate_indices is not None and len(candidate_indices) != len(candidate_texts):
            raise ValueError("candidate_indices muss die gleiche Länge haben wie candidate_texts.")

        pairs = [(q, t) for t in candidate_texts]
        scores = self.model.predict(pairs, batch_size=batch_size)

        # build results
        results: List[RerankResult] = []
        for i, (text, score) in enumerate(zip(candidate_texts, scores)):
            meta = candidate_meta[i] if candidate_meta is not None else None
            idx = int(candidate_indices[i]) if candidate_indices is not None else i
            results.append(RerankResult(index=idx, score=float(score), text=text, meta=meta))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
