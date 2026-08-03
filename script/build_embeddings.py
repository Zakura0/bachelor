import json
import os
import pathlib
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.retrieval.embeddings import EmbeddingRetriever
from config import EMBEDDING_MODEL, DIR_EMBEDDINGS


def build_embeddings(chunks_path: str, output_path: str = None):
    """
    Erstellt Embeddings für Chunks.

    :param chunks_path: Pfad zur Chunks-JSON-Datei
    :param output_path: Pfad zur Ausgabe der Embeddings.npy-Datei
    :return: Pfad zur gespeicherten Embeddings-Datei
    """
    p = pathlib.Path(chunks_path)
    preset = p.stem          # z.B. "medium"
    book   = p.parent.name  # z.B. "verwandlung"
    if output_path is None:
        os.makedirs(os.path.join(DIR_EMBEDDINGS, book), exist_ok=True)
        output_path = os.path.join(DIR_EMBEDDINGS, book, f"{preset}.npy")

    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    texts = [c["content"] for c in chunks]

    retriever = EmbeddingRetriever(model_name=EMBEDDING_MODEL)
    retriever.fit(texts, meta=chunks, batch_size=64)

    # Speichern
    assert retriever._emb is not None
    np.save(output_path, retriever._emb)

    print(f"Embeddings gespeichert: {output_path} | shape={retriever._emb.shape}")
    
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Bitte gib den Pfad zur Chunks-JSON an.")
        print("Beispiel: python build_embeddings.py data/chunks/verwandlung_chunks.json")
        sys.exit(1)
    chunks_path = sys.argv[1]
    build_embeddings(chunks_path)


if __name__ == "__main__":
    main()