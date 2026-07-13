"""
HyDE — Hypothetical Document Embeddings.

Generiert aus einer abstrakten Zusammenfassung eine hypothetische Textpassage
im Stil des Quelltextes, die dann anstelle der Summary für die Embedding-Suche
verwendet wird.
"""
from __future__ import annotations

from openai import OpenAI


PROMPT_HYDE = """\
Du bist ein Literaturexperte. Schreibe einen kurzen Textausschnitt (2-4 Sätze) \
aus einem deutschsprachigen literarischen Werk des frühen 20. Jahrhunderts, \
der die folgende Handlung oder Situation beschreibt. \
Schreibe im Stil der Epoche: präzise, leicht gehoben, keine modernen Ausdrücke. \
Antworte NUR mit dem Textausschnitt selbst, ohne Anführungszeichen oder Erklärungen.

Handlung/Situation:
{summary}

Textausschnitt:"""


class HyDEGenerator:
    """
    Erzeugt hypothetische Dokumentpassagen aus Zusammenfassungen.

    :param model: OpenAI-Modell, z.B. "gpt-4o-mini"
    :param prompt_template: Prompt mit Platzhalter {summary}
    """

    def __init__(self, model: str, prompt_template: str = PROMPT_HYDE) -> None:
        self.model = model
        self.prompt_template = prompt_template
        self._client = OpenAI()

    def generate(self, summary: str) -> str:
        """
        Generiert eine hypothetische Textpassage für die gegebene Zusammenfassung.

        :param summary: Abstrakte Zusammenfassung / Query
        :return: Hypothetische Textpassage
        """
        prompt = self.prompt_template.format(summary=summary)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
