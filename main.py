import os
import sys
import pathlib

from script.build_book_chunks import build_chunks
from script.build_embeddings import build_embeddings
from script.run_search import run_search_interactive
from config import DIR_CHUNKS, DIR_EMBEDDINGS


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


def ask_yes_no(prompt):
    while True:
        ans = input(prompt + " (j/n): ").strip().lower()
        if ans in ("j", "ja"):
            return True
        if ans in ("n", "nein"):
            return False
        print("Bitte antworte mit 'j' oder 'n'.")


# Suche

def select_book(raw_dir="data/raw"):
    files = [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f)) and f.endswith(".txt")]
    if not files:
        print("Keine Bücher im raw-Verzeichnis gefunden.")
        return None
    print("Wähle ein Buch:")
    for idx, fname in enumerate(files, 1):
        print(f"  {idx}: {fname}")
    print()
    while True:
        try:
            choice = int(input("Auswahl: "))
            if 1 <= choice <= len(files):
                return os.path.join(raw_dir, files[choice - 1])
            print(f"Bitte gib eine Zahl zwischen 1 und {len(files)} ein.")
        except ValueError:
            print("Ungültige Eingabe.")


def run_suche():
    clear_terminal()
    selected_book = select_book()
    if not selected_book:
        return

    book_stem = pathlib.Path(selected_book).stem

    # Vorhandene Chunk-Dateien für dieses Buch suchen
    book_chunks_dir = os.path.join(DIR_CHUNKS, book_stem)
    existing_chunks = sorted([
        f for f in os.listdir(book_chunks_dir)
        if f.endswith(".json")
    ]) if os.path.isdir(book_chunks_dir) else []

    chunks_path = None

    def create_chunks():
        nonlocal chunks_path
        print("Minimale Wortanzahl pro Chunk? (Standard: 30)")
        min_w = input("Minimum: ").strip()
        min_w = int(min_w) if min_w.isdigit() else 30
        print("Maximale Wortanzahl pro Chunk? (Standard: 100)")
        max_w = input("Maximum: ").strip()
        max_w = int(max_w) if max_w.isdigit() else 100
        print("Satzüberlappung zwischen Chunks? (Standard: 2)")
        overlap = input("Überlappung: ").strip()
        overlap = int(overlap) if overlap.isdigit() else 2
        print("Name für diese Einstellung? (z.B. medium)")
        setting_name = input("Name: ").strip() or "custom"
        chunks_path = os.path.join(DIR_CHUNKS, book_stem, f"{setting_name}.json")
        os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
        clear_terminal()
        print("Chunks werden erstellt...")
        build_chunks(selected_book, chunks_path, min_w, max_w, overlap)
        input("Enter zum Fortfahren...")

    def get_embeddings_path():
        # {book}/{preset}.json  →  embeddings/{book}/{preset}.npy
        preset = pathlib.Path(chunks_path).stem
        book   = pathlib.Path(chunks_path).parent.name
        return os.path.join(DIR_EMBEDDINGS, book, f"{preset}.npy")

    def select_chunks():
        nonlocal chunks_path
        current = sorted([
            f for f in os.listdir(book_chunks_dir)
            if f.endswith(".json")
        ]) if os.path.isdir(book_chunks_dir) else []
        if current:
            print(f"Vorhandene Chunks für '{book_stem}':")
            for idx, f in enumerate(current, 1):
                print(f"  {idx}: {f}")
            print(f"  {len(current)+1}: Neue Chunks erstellen")
            print()
            while True:
                try:
                    choice = int(input("Auswahl: "))
                    if 1 <= choice <= len(current):
                        chunks_path = os.path.join(book_chunks_dir, current[choice - 1])
                        return True
                    elif choice == len(current) + 1:
                        create_chunks()
                        return chunks_path is not None
                except ValueError:
                    print("Ungültige Eingabe.")
        else:
            print(f"Noch keine Chunks für '{book_stem}' vorhanden.")
            if ask_yes_no("Chunks jetzt erstellen?"):
                create_chunks()
                return chunks_path is not None
            return False

    # Erstmalige Chunk-Auswahl
    if not select_chunks():
        print("Ohne Chunks kann nicht fortgefahren werden.")
        return

    embeddings_path = get_embeddings_path()

    def create_embeddings():
        nonlocal embeddings_path
        clear_terminal()
        print("Embeddings werden erstellt...")
        build_embeddings(chunks_path, embeddings_path)
        embeddings_path = get_embeddings_path()
        input("Enter zum Fortfahren...")

    if not os.path.isfile(embeddings_path):
        print(f"Noch keine Embeddings vorhanden ({embeddings_path}).")
        if ask_yes_no("Embeddings jetzt erstellen?"):
            create_embeddings()
        else:
            print("Ohne Embeddings kann nicht fortgefahren werden.")
            return

    while True:
        clear_terminal()
        print("-Suche-")
        print(f"Buch:   {book_stem}")
        print(f"Chunks: {os.path.basename(chunks_path)}")
        print()
        print("  1: Suche starten")
        print("  2: Chunks ändern")
        print("  3: Embeddings neu erstellen")
        print("  4: Zurück")
        print()
        opt = input("Auswahl: ").strip()
        if opt == "1":
            clear_terminal()
            run_search_interactive(chunks_path, embeddings_path)
        elif opt == "2":
            clear_terminal()
            select_chunks()
            embeddings_path = get_embeddings_path()
        elif opt == "3":
            create_embeddings()
        elif opt == "4":
            return
        else:
            print("Ungültige Eingabe.")


# Experimente

def run_experimente():
    from eval.experiment import main as exp_main
    from eval.experiment_llm_pick import main as llm_pick_main
    from eval.llm_fulltext_experiment import main as llm_fulltext_main

    while True:
        clear_terminal()
        print("-Experimente-")
        print()
        print("  1: Recall-Experiment")
        print("  2: LLM-Pick-Experiment")
        print("  3: LLM-Fulltext-Experiment")
        print("  4: Zurück")
        print()
        opt = input("Auswahl: ").strip()
        if opt == "1":
            clear_terminal()
            exp_main()
            input("\nEnter zum Fortfahren...")
        elif opt == "2":
            clear_terminal()
            llm_pick_main()
            input("\nEnter zum Fortfahren...")
        elif opt == "3":
            clear_terminal()
            llm_fulltext_main()
            input("\nEnter zum Fortfahren...")
        elif opt == "4":
            return
        else:
            print("Ungültige Eingabe.")


# Hauptmenü

if __name__ == "__main__":
    while True:
        clear_terminal()
        print("  1: Suche")
        print("  2: Experimente")
        print("  3: Beenden")
        print()
        opt = input("Auswahl: ").strip()
        if opt == "1":
            run_suche()
        elif opt == "2":
            run_experimente()
        elif opt == "3":
            sys.exit(0)
        else:
            print("Ungültige Eingabe.")
