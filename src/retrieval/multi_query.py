"""
Multi-Query Generator.

Erzeugt aus einer Zusammenfassung mehrere Paraphrasen, die für eine
breitere Retrieval-Abdeckung verwendet werden.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from openai import OpenAI


PROMPT_MULTI_QUERY = """\
Du bist ein Experte für deutschsprachige Literatur des frühen 20. Jahrhunderts.

Generiere {n} verschiedene Umformulierungen der folgenden Zusammenfassung einer \
literarischen Szene. Die Umformulierungen sollen dieselbe Handlung aus verschiedenen \
Blickwinkeln oder mit anderen Worten beschreiben, um die Suche im Originaltext \
zu verbessern.

Antworte NUR mit den {n} Umformulierungen, eine pro Zeile, ohne Nummerierung \
oder Erklärungen.

Zusammenfassung:
{summary}

Umformulierungen:"""


class MultiQueryGenerator:
    """
    Generiert N Paraphrasen einer Zusammenfassung für Multi-Query-Retrieval.

    :param model: OpenAI-Modell
    :param n: Anzahl der Paraphrasen
    :param prompt_template: Prompt mit Platzhaltern {n} und {summary}
    """

    def __init__(
        self,
        model: str,
        n: int = 3,
        prompt_template: str = PROMPT_MULTI_QUERY,
    ) -> None:
        self.model = model
        self.n = n
        self.prompt_template = prompt_template
        self._client = OpenAI()

    def generate(self, summary: str) -> List[str]:
        """
        Generiert N Paraphrasen für die gegebene Zusammenfassung.
        Gibt immer eine Liste zurück (mindestens die Original-Summary als Fallback).

        :param summary: Originalzusammenfassung
        :return: Liste von N Paraphrasen
        """
        prompt = self.prompt_template.format(n=self.n, summary=summary)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            lines = [
                line.strip()
                for line in response.choices[0].message.content.strip().splitlines()
                if line.strip()
            ]
            return lines[:self.n] if lines else [summary]
        except Exception:
            return [summary]
