"""Gemeinsame Hilfsfunktionen für Experiment-Skripte."""
import json
import os
from datetime import datetime


def make_run_dir(base_dir: str, experiment_name: str) -> str:
    """
    Gibt den Pfad base_dir/<experiment_name>/<YYYY-MM-DD_HH-MM-SS> zurück.
    Der Ordner wird erst beim ersten Schreibzugriff angelegt (lazy).
    """
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base_dir, experiment_name, ts)
    return path


def build_misses(per_book_results: dict, got_key: str = "top1_text", span_key: str = "top1_span") -> list:
    """
    Gibt eine Liste von Dicts zurück — einen je Miss.
    got_key:  Schlüssel im Trial-Dict der die System-Antwort enthält
              ("top1_text" für pipeline-Experimente, "answer" für llm_fulltext, "pick_text" für llm_pick).
    span_key: Schlüssel für den Span der System-Antwort
              ("top1_span" für pipeline-Experimente, "pick_span" für llm_pick).
    Texte werden NICHT abgeschnitten.
    """
    misses = []
    for book, bm in per_book_results.items():
        for trial in bm["trials"]:
            if not trial["hit"]:
                misses.append({
                    "book":          book,
                    "query":         trial["query"],
                    "expected_text": trial.get("expected_text", ""),
                    "expected_spans": trial.get("expected_spans", []),
                    "got":           trial.get(got_key) or "",
                    "got_span":      trial.get(span_key) or trial.get("found_at"),
                })
    return misses


def save_misses(misses: list, run_dir: str) -> str:
    path = os.path.join(run_dir, "misses.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(misses, f, ensure_ascii=False, indent=2)
    return path
