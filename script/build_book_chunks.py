import json
import os
import sys
import pathlib

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessing.book_chunker import BookChunker


def build_chunks(book_path: str, output_path: str = None):
    """
    Erstellt Chunks für ein Buch.
    
    :param book_path: Pfad zur Textdatei des Buchs
    :param output_path: Pfad zur Ausgabe der Chunks.
    """    
    book_stem = pathlib.Path(book_path).stem
    if output_path is None:
        output_path = os.path.join(project_root, "data", "processed", f"{book_stem}_chunks.json")

    with open(book_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunker = BookChunker(
        min_words=10,
        max_words=50,
        sentence_overlap=1,
    )

    result = chunker.build_chunks(text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"{result['total_chunks']} Chunks erstellt und in "
        f"{output_path} gespeichert."
    )
    
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Bitte gib den Pfad zum Buch an.")
        sys.exit(1)
    input_path = sys.argv[1]
    build_chunks(input_path)


if __name__ == "__main__":
    main()
