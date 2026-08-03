"""
Migrations-Skript: Befüllt library.db aus den vorhandenen Datei-Artefakten.

Liest:
  data/raw/{book}.txt                      → books
  data/chunks/{book}/{preset}.json         → chunks
  data/embeddings/e5-large/{book}/{preset}.npy → embeddings
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECT_ROOT, DIR_CHUNKS, DIR_EMBEDDINGS, EMBEDDING_MODEL
from db.database import init_db, get_connection, vector_to_blob

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def migrate_books(conn) -> dict[str, int]:
    """Rohtexte einlesen, in DB schreiben. Gibt {name: id} zurück."""
    book_ids: dict[str, int] = {}
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".txt"):
            continue
        name = fname[:-4]
        raw_text = open(os.path.join(RAW_DIR, fname), encoding="utf-8").read()
        cur = conn.execute(
            "INSERT OR IGNORE INTO books (name, raw_text) VALUES (?, ?)",
            (name, raw_text),
        )
        if cur.lastrowid:
            book_id = cur.lastrowid
            print(f"  Buch '{name}' eingefügt (id={book_id})")
        else:
            book_id = conn.execute(
                "SELECT id FROM books WHERE name = ?", (name,)
            ).fetchone()["id"]
            print(f"  Buch '{name}' bereits vorhanden (id={book_id})")
        book_ids[name] = book_id
    return book_ids


def migrate_chunks(conn, book_ids: dict[str, int]) -> dict[tuple, dict[int, int]]:
    """
    Chunks einlesen und in DB schreiben.
    Gibt {(book, preset): {chunk_file_index: chunk_db_id}} zurück.
    """
    index_map: dict[tuple, dict[int, int]] = {}

    for book, book_id in book_ids.items():
        book_dir = os.path.join(DIR_CHUNKS, book)
        if not os.path.isdir(book_dir):
            continue
        for fname in sorted(os.listdir(book_dir)):
            if not fname.endswith(".json"):
                continue
            preset = fname[:-5]
            path = os.path.join(book_dir, fname)

            existing = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE book_id = ? AND preset_name = ?",
                (book_id, preset),
            ).fetchone()[0]
            if existing:
                print(f"  Chunks {book}/{preset} bereits vorhanden ({existing} Stück)")
                rows = conn.execute(
                    "SELECT id FROM chunks WHERE book_id = ? AND preset_name = ? ORDER BY start_index",
                    (book_id, preset),
                ).fetchall()
                index_map[(book, preset)] = {i: r["id"] for i, r in enumerate(rows)}
                continue

            data = json.load(open(path, encoding="utf-8"))
            chunks = data["chunks"]
            id_map: dict[int, int] = {}
            for i, c in enumerate(chunks):
                cur = conn.execute(
                    "INSERT INTO chunks (book_id, preset_name, start_index, end_index, content, word_count) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (book_id, preset, c["start_index"], c["end_index"], c["content"], c["word_count"]),
                )
                id_map[i] = cur.lastrowid
            index_map[(book, preset)] = id_map
            print(f"  Chunks {book}/{preset}: {len(chunks)} eingefügt")

    return index_map


def migrate_embeddings(conn, index_map: dict[tuple, dict[int, int]]) -> None:
    """Embedding-Vektoren einlesen und in DB schreiben."""
    for (book, preset), id_map in index_map.items():
        emb_path = os.path.join(DIR_EMBEDDINGS, book, f"{preset}.npy")
        if not os.path.exists(emb_path):
            print(f"  Embeddings {book}/{preset}: Datei nicht gefunden, übersprungen")
            continue

        first_chunk_id = id_map.get(0)
        if first_chunk_id:
            existing = conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE chunk_id = ? AND model_name = ?",
                (first_chunk_id, EMBEDDING_MODEL),
            ).fetchone()[0]
            if existing:
                print(f"  Embeddings {book}/{preset} bereits vorhanden")
                continue

        vectors = np.load(emb_path)
        for i, vec in enumerate(vectors):
            chunk_id = id_map.get(i)
            if chunk_id is None:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO embeddings (chunk_id, model_name, vector) VALUES (?, ?, ?)",
                (chunk_id, EMBEDDING_MODEL, vector_to_blob(vec)),
            )
        print(f"  Embeddings {book}/{preset}: {len(vectors)} eingefügt ({EMBEDDING_MODEL})")


def main():
    init_db()
    print("Migration gestartet...\n")
    with get_connection() as conn:
        print("=== Bücher ===")
        book_ids = migrate_books(conn)
        print(f"\n=== Chunks ===")
        index_map = migrate_chunks(conn, book_ids)
        print(f"\n=== Embeddings ===")
        migrate_embeddings(conn, index_map)
    print("\nMigration abgeschlossen.")


if __name__ == "__main__":
    main()
