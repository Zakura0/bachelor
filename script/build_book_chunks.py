import json
import os
import sys
import pathlib

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessing.book_chunker import BookChunker


def build_chunks(book_path: str, output_path: str = None, min_words: int = 10, max_words: int = 50, sentence_overlap: int = 1) -> str:
    """
    Erstellt Chunks für ein Buch.
    
    :param book_path: Pfad zur Textdatei des Buchs
    :param output_path: Pfad zur Ausgabe der Chunks.
    """    
    book_stem = pathlib.Path(book_path).stem
    if output_path is None:
        output_path = os.path.join(project_root, "data", "chunks", f"{book_stem}_chunks.json")

    with open(book_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunker = BookChunker(
        min_words,
        max_words,
        sentence_overlap,
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
    if len(sys.argv) < 5:
        print("Bitte gib den Pfad zum Buch, min_words, max_words und sentence_overlap an.")
        sys.exit(1)
    input_path = sys.argv[1]
    min_words = int(sys.argv[2])
    max_words = int(sys.argv[3])
    sentence_overlap = int(sys.argv[4])
    build_chunks(input_path, min_words=min_words, max_words=max_words, sentence_overlap=sentence_overlap)

if __name__ == "__main__":
    main()
