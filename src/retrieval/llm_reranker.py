from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from openai import OpenAI

# ---------------------------------------------------------------------------
# Prompt-Templates
# ---------------------------------------------------------------------------

# Fragt das LLM, die relevanteste Textstelle zu wählen (eine Zahl).
PROMPT_PICK_ONE = """\
Du bekommst eine Zusammenfassung und {n} Textstellen aus einem Buch.
Wähle die EINE Textstelle, die am besten zur Zusammenfassung passt.
Antworte NUR mit der Nummer der Textstelle.
Beispiel: 3

Zusammenfassung:
"{query}"

Textstellen:
{candidates_block}

Nummer der relevantesten Textstelle:"""

# Fragt das LLM, alle Textstellen nach Relevanz zu sortieren.
PROMPT_RANK_ALL = """\
Du bekommst eine Zusammenfassung und {n} Textstellen aus einem Buch.
Sortiere die Textstellen nach ihrer inhaltlichen Relevanz zur Zusammenfassung.
Antworte NUR mit den Nummern in sortierter Reihenfolge, getrennt durch Kommas.
Beispiel: 3,1,5,2,4

Zusammenfassung:
"{query}"

Textstellen:
{candidates_block}

Sortierte Reihenfolge (nur Nummern):"""


@dataclass
class LLMRerankResult:
    index: int
    score: float
    text: str
    meta: Any = None


class LLMReranker:
    """
    Reranker der ein LLM (OpenAI) verwendet um aus einer Liste von Kandidaten-Chunks
    den relevantesten auszuwählen.

    Der OPENAI_API_KEY wird automatisch aus der Umgebungsvariable gelesen.

    :param model: OpenAI-Modell, z.B. "gpt-4o" oder "gpt-4o-mini"
    :param prompt_template: Template-String mit den Platzhaltern {n}, {query}
        und {candidates_block}. Standard: PROMPT_PICK_ONE.
    """

    def __init__(self, model: str, prompt_template: str = PROMPT_PICK_ONE) -> None:
        self.client = OpenAI()
        self.model = model
        self.prompt_template = prompt_template

    def rerank(
        self,
        query: str,
        candidate_texts: Sequence[str],
        candidate_meta: Optional[Sequence[Any]] = None,
        candidate_indices: Optional[Sequence[int]] = None,
        top_k: int = 10,
    ) -> List[LLMRerankResult]:
        """
        Lässt das LLM die Kandidaten nach Relevanz zur Query bewerten.

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

        candidates_block = "\n\n".join(
            f"[{i + 1}] {text}" for i, text in enumerate(candidate_texts)
        )

        prompt = self.prompt_template.format(
            n=n,
            query=query,
            candidates_block=candidates_block,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()

        # Alle Zahlen aus der Antwort extrahieren (robust gegenüber beliebigen Formaten)
        parsed = [int(x) - 1 for x in re.findall(r"\b\d+\b", raw)]
        seen = set()
        clean_order = []
        for idx in parsed:
            if 0 <= idx < n and idx not in seen:
                clean_order.append(idx)
                seen.add(idx)
        # Nicht genannte Kandidaten ans Ende hängen
        for idx in range(n):
            if idx not in seen:
                clean_order.append(idx)

        results = []
        for rank, idx in enumerate(clean_order[:top_k], start=1):
            results.append(LLMRerankResult(
                index=indices[idx],
                score=1.0 - (rank - 1) / n,
                text=candidate_texts[idx],
                meta=meta[idx],
            ))

        return results
