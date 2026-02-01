import os
import sys
import pathlib

from script.build_book_chunks import build_chunks
from script.build_embeddings import build_embeddings
from script.run_search import run_search

def select_book(raw_dir="data/raw"):
    files = [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))]
    if not files:
        print("Keine Bücher im raw-Verzeichnis gefunden.")
        return None
    print("Verfügbare Bücher:")
    for idx, fname in enumerate(files, 1):
        print(f"{idx}: {fname}")
    while True:
        try:
            choice = int(input("Bitte wähle ein Buch durch Eingabe der Zahl: "))
            if 1 <= choice <= len(files):
                selected = files[choice - 1]
                print(f"Du hast ausgewählt: {selected}")
                return os.path.join(raw_dir, selected)
            else:
                print(f"Bitte gib eine Zahl zwischen 1 und {len(files)} ein.")
        except ValueError:
            print("Ungültige Eingabe. Bitte gib eine Zahl ein.")

if __name__ == "__main__":
    selected_book = select_book()
    if not selected_book:
        exit(0)

    book_stem = pathlib.Path(selected_book).stem
    processed_dir = os.path.join("data", "processed")
    chunks_path = os.path.join(processed_dir, f"{book_stem}_chunks.json")
    embeddings_path = os.path.join(processed_dir, f"{book_stem}.embeddings.npy")

    def file_exists(path):
        return os.path.isfile(path)

    def ask_yes_no(prompt):
        while True:
            ans = input(prompt + " (j/n): ").strip().lower()
            if ans in ("j", "ja"): return True
            if ans in ("n", "nein"): return False
            print("Bitte antworte mit 'j' oder 'n'.")

    def create_chunks():
        print("Chunks werden erstellt...")
        build_chunks(selected_book, chunks_path)

    def create_embeddings():
        print("Embeddings werden erstellt...")
        build_embeddings(chunks_path, embeddings_path)

    if not file_exists(chunks_path):
        print(f"Es gibt noch keine Chunks für das Buch ({chunks_path}).")
        if ask_yes_no("Chunks jetzt erstellen?"):
            create_chunks()
        else:
            print("Ohne Chunks kann nicht fortgefahren werden.")
            sys.exit(0)

    if not file_exists(embeddings_path):
        print(f"Es gibt noch keine Embeddings für das Buch ({embeddings_path}).")
        if ask_yes_no("Embeddings jetzt erstellen?"):
            create_embeddings()
        else:
            print("Ohne Embeddings kann nicht fortgefahren werden.")
            sys.exit(0)

    while True:
        print("1: Zusammenfassungssuche durchführen")
        print("2: Chunks neu erstellen")
        print("3: Embeddings neu erstellen")
        print("4: Beenden")
        opt = input("Bitte wähle eine Option: ").strip()
        if opt == "1":
            run_search(chunks_path, embeddings_path)
            break
        elif opt == "2":
            create_chunks()
        elif opt == "3":
            create_embeddings()
        elif opt == "4":
            print("Beende das Programm.")
            sys.exit(0)
        else:
            print("Ungültige Eingabe. Bitte wähle 1, 2, 3 oder 4.")
