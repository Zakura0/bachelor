from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from openai import OpenAI


@dataclass
class LLMRerankResult:
    index: int
    score: float  # Rang (1 = bestes), normiert auf [0, 1]
    text: str
    meta: Any = None


class LLMReranker:
    """
    Reranker der ein LLM (OpenAI) verwendet um aus einer Liste von Kandidaten-Chunks
    den relevantesten auszuwählen.

    Der OPENAI_API_KEY wird automatisch aus der Umgebungsvariable gelesen.

    :param model: OpenAI-Modell, z.B. "gpt-4o" oder "gpt-4o-mini"
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.client = OpenAI()
        self.model = model

    def rerank(
        self,
        query: str,
        candidate_texts: Sequence[str],
        candidate_meta: Optional[Sequence[Any]] = None,
        candidate_indices: Optional[Sequence[int]] = None,
        top_k: int = 10,
    ) -> List[LLMRerankResult]:
        """
        Lässt das LLM die Kandidaten nach Relevanz zur Query sortieren.

        :param query: Die Suchanfrage (z.B. Summary-Text)
        :param candidate_texts: Liste der Kandidatentexte (top-k aus der Pipeline)
        :param candidate_meta: Metadaten zu den Kandidaten
        :param candidate_indices: Indizes der Kandidaten im ursprünglichen Chunk-Array
        :param top_k: Wie viele Ergebnisse zurückgegeben werden sollen
        :return: Sortierte Liste von LLMRerankResult
        """
        if not candidate_texts:
            return []

        n = len(candidate_texts)
        indices = list(candidate_indices) if candidate_indices is not None else list(range(n))
        meta = list(candidate_meta) if candidate_meta is not None else [None] * n

        # Kandidaten nummeriert auflisten
        candidates_block = "\n\n".join(
            f"[{i + 1}] {text}" for i, text in enumerate(candidate_texts)
        )

        prompt = f"""Du bekommst eine Zusammenfassung und {n} Textstellen aus einem Buch.
Sortiere die Textstellen nach ihrer inhaltlichen Relevanz zur Zusammenfassung.
Antworte NUR mit den Nummern in sortierter Reihenfolge, getrennt durch Kommas.
Beispiel: 3,1,5,2,4

Zusammenfassung:
\"{query}\"

Textstellen:
{candidates_block}

Sortierte Reihenfolge (nur Nummern):"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()

        # Antwort parsen: "3,1,5,2,4" → [3, 1, 5, 2, 4]
        try:
            order = [int(x.strip()) - 1 for x in raw.split(",") if x.strip().isdigit()]
            # Unbekannte oder doppelte Indizes herausfiltern
            seen = set()
            clean_order = []
            for idx in order:
                if 0 <= idx < n and idx not in seen:
                    clean_order.append(idx)
                    seen.add(idx)
            # Fehlende Indizes ans Ende hängen
            for idx in range(n):
                if idx not in seen:
                    clean_order.append(idx)
        except Exception:
            # Fallback: originale Reihenfolge beibehalten
            clean_order = list(range(n))

        results = []
        for rank, idx in enumerate(clean_order[:top_k], start=1):
            results.append(LLMRerankResult(
                index=indices[idx],
                score=1.0 - (rank - 1) / n,  # normiert: 1.0 = bestes
                text=candidate_texts[idx],
                meta=meta[idx],
            ))

        return results
