import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessing.book_chunker import BookChunker


def main():
    book = "judenbuche"
    input_path = os.path.join(project_root, "data", "raw", f"{book}.txt")
    output_path = os.path.join(project_root, "data", "processed", f"{book}_chunks.json")

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunker = BookChunker(
        min_words=10,
        max_words=50,
        sentence_overlap=1,
        use_spacy=False,
    )

    result = chunker.build_chunks(text)

    # Stelle sicher, dass das Output-Verzeichnis existiert
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"{result['total_chunks']} Chunks erstellt und in "
        f"{output_path} gespeichert."
    )


if __name__ == "__main__":
    main()
