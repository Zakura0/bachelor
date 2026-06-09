import os
import sys
import pathlib

from script.build_book_chunks import build_chunks
from script.build_embeddings import build_embeddings
from script.run_search import run_search_interactive


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


# ---------------------------------------------------------------------------
# Suche
# ---------------------------------------------------------------------------

def select_book(raw_dir="data/raw"):
    files = [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))]
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
    processed_dir = os.path.join("data", "processed")
    chunks_path = os.path.join(processed_dir, f"{book_stem}_chunks.json")
    embeddings_path = os.path.join(processed_dir, f"{book_stem}.embeddings.npy")

    def create_chunks():
        print("Minimale Wortanzahl pro Chunk? (Standard: 10)")
        min_w = input("Minimum: ").strip()
        min_w = int(min_w) if min_w.isdigit() else 10
        print("Maximale Wortanzahl pro Chunk? (Standard: 50)")
        max_w = input("Maximum: ").strip()
        max_w = int(max_w) if max_w.isdigit() else 50
        print("Satzüberlappung zwischen Chunks? (Standard: 1)")
        overlap = input("Überlappung: ").strip()
        overlap = int(overlap) if overlap.isdigit() else 1
        clear_terminal()
        print("Chunks werden erstellt...")
        build_chunks(selected_book, chunks_path, min_w, max_w, overlap)
        input("Enter zum Fortfahren...")

    def create_embeddings():
        clear_terminal()
        print("Embeddings werden erstellt...")
        build_embeddings(chunks_path, embeddings_path)
        input("Enter zum Fortfahren...")

    if not os.path.isfile(chunks_path):
        print(f"Noch keine Chunks vorhanden ({chunks_path}).")
        if ask_yes_no("Chunks jetzt erstellen?"):
            create_chunks()
        else:
            print("Ohne Chunks kann nicht fortgefahren werden.")
            return

    if not os.path.isfile(embeddings_path):
        print(f"Noch keine Embeddings vorhanden ({embeddings_path}).")
        if ask_yes_no("Embeddings jetzt erstellen?"):
            create_embeddings()
        else:
            print("Ohne Embeddings kann nicht fortgefahren werden.")
            return

    while True:
        clear_terminal()
        print("=== SUCHE ===")
        print(f"Buch: {book_stem}")
        print()
        print("  1: Suche starten")
        print("  2: Chunks neu erstellen")
        print("  3: Embeddings neu erstellen")
        print("  4: Zurück")
        print()
        opt = input("Auswahl: ").strip()
        if opt == "1":
            clear_terminal()
            run_search_interactive(chunks_path, embeddings_path)
        elif opt == "2":
            create_chunks()
        elif opt == "3":
            create_embeddings()
        elif opt == "4":
            return
        else:
            print("Ungültige Eingabe.")


# ---------------------------------------------------------------------------
# Experimente
# ---------------------------------------------------------------------------

def run_experimente():
    from eval.experiment import main as exp_main

    while True:
        clear_terminal()
        print("=== EXPERIMENTE ===")
        print()
        print("  1: Experiment starten  (konfigurierbar in eval/top1_experiment.py)")
        print("  2: Zurück")
        print()
        opt = input("Auswahl: ").strip()
        if opt == "1":
            clear_terminal()
            exp_main()
            input("\nEnter zum Fortfahren...")
        elif opt == "2":
            return
        else:
            print("Ungültige Eingabe.")


# ---------------------------------------------------------------------------
# Hauptmenü
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    while True:
        clear_terminal()
        print("=== LITERATUR-RETRIEVAL ===")
        print()
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
            print("Auf Wiedersehen.")
            sys.exit(0)
        else:
            print("Ungültige Eingabe.")
