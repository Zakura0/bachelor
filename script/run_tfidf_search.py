import json
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.retrieval.tfidf import TfidfRetriever


def main():
    book = "verwandlung"
    filename = f"{book}_chunks.json"

    data_path = os.path.join(project_root, "data", "processed", filename)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Chunk-Datei nicht gefunden: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    texts = [c["content"] for c in chunks]

    retriever = TfidfRetriever()
    retriever.fit(texts, meta=chunks)

    print(f"TF-IDF Retrieval geladen – {len(chunks)} Chunks im Index.")
    print("Tippe eine Query (oder 'exit').\n")

    while True:
        query = input("Query: ").strip()
        if not query or query.lower() in ("exit", "quit", ":q"):
            break

        print("-" * 60)
        results = retriever.search(query, top_k=5)

        for r in results:
            c = r.meta
            print(f"Score={r.score:.3f} | Chunk-ID={c['id']} | Start={c['start_index']}")
            print(r.text.replace("\n", " "))
            print("-" * 60)


if __name__ == "__main__":
    main()
