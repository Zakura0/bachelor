from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class NLIResult:
    index: int
    entailment: float
    label: str
    text: str
    meta: Any = None


class NLIVerifier:
    """
    Oberklasse für NLI-Verifikation mittels Cross-Encoder.
    
    :param model_name: Name des Modells
    :param device: Gerät für die Berechnung ("cpu" oder "cuda")
    :param max_length: Maximale Sequenzlänge für das Modell
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_length: int = 384,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Modellspezifische Label-IDs
        self.entailment_id = 2
        self.neutral_id = 1
        self.contradiction_id = 0

    @torch.no_grad()
    def score_entailment(
        self,
        hypothesis: str,
        premises: Sequence[str],
        indices: Optional[Sequence[int]] = None,
        meta: Optional[Sequence[Any]] = None,
        batch_size: int = 16,
    ) -> List[NLIResult]:
        """
        Bewertet die Entailment-Wahrscheinlichkeiten der Prämissen bezüglich der Hypothese.
        
        :param hypothesis: Hypothese, gegen die alle Prämissen geprüft werden (z.B. Summary-Text)
        :param premises: Kandidatentexte, die auf Entailment geprüft werden
        :param indices: Optionale Indizes der Kandidaten im ursprünglichen Chunk-Array
        :param meta: Optionale Metadaten zu den Kandidaten
        :param batch_size: Batch-Größe für die Modell-Inferenz
        :return: Liste der NLI-Ergebnisse, nach Entailment-Score absteigend sortiert
        """
        if indices is not None and len(indices) != len(premises):
            raise ValueError("indices muss die gleiche Länge wie premises haben.")
        if meta is not None and len(meta) != len(premises):
            raise ValueError("meta muss die gleiche Länge wie premises haben.")

        results: List[NLIResult] = []
        hyp = hypothesis.strip()
        if not hyp:
            return results

        for start in range(0, len(premises), batch_size):
            batch = premises[start:start + batch_size]

            enc = self.tokenizer(
                list(batch),
                [hyp] * len(batch),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)

            entail_probs = probs[:, self.entailment_id].detach().cpu().tolist()
            pred_ids = torch.argmax(probs, dim=-1).detach().cpu().tolist()

            for j, (p_text, eprob, pid) in enumerate(zip(batch, entail_probs, pred_ids)):
                i = start + j
                idx = int(indices[i]) if indices is not None else i
                m = meta[i] if meta is not None else None
                label = ["contradiction", "neutral", "entailment"][pid] if pid in (0, 1, 2) else str(pid)

                results.append(NLIResult(index=idx, entailment=float(eprob), label=label, text=p_text, meta=m))

        results.sort(key=lambda r: r.entailment, reverse=True)
        return results
