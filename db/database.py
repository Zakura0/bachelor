"""
SQLite-Datenbankschicht.

Tabellen:
  books      — Bücher mit Rohtext
  chunks     — Textchunks pro Buch und Chunk-Preset
  embeddings — Embedding-Vektoren pro Chunk und Modell (BLOB)
"""
import sqlite3
import os
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECT_ROOT

DB_PATH = os.path.join(PROJECT_ROOT, "db", "library.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Zeilen als dict-ähnliche Objekte
    return conn


_KNOWN_TITLES: dict[str, str] = {
    "erdbeben":    "Das Erdbeben in Chili",
    "harrypotter": "Harry Potter",
    "judenbuche":  "Die Judenbuche",
    "krambambuli": "Krambambuli",
    "verwandlung":  "Die Verwandlung",
}


def migrate_add_title() -> None:
    """Adds title column to books if missing and back-fills known titles."""
    with get_connection() as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(books)").fetchall()]
        if "title" not in cols:
            conn.execute("ALTER TABLE books ADD COLUMN title TEXT NOT NULL DEFAULT ''")
        for name, title in _KNOWN_TITLES.items():
            conn.execute(
                "UPDATE books SET title = ? WHERE name = ? AND title = ''",
                (title, name),
            )
        conn.commit()


def init_db() -> None:
    """Schema anlegen falls noch nicht vorhanden."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS books (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                name      TEXT    NOT NULL UNIQUE,
                title     TEXT    NOT NULL DEFAULT '',
                raw_text  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id     INTEGER NOT NULL REFERENCES books(id),
                preset_name TEXT    NOT NULL,
                start_index INTEGER NOT NULL,
                end_index   INTEGER NOT NULL,
                content     TEXT    NOT NULL,
                word_count  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_book_preset
                ON chunks(book_id, preset_name);

            CREATE TABLE IF NOT EXISTS embeddings (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id   INTEGER NOT NULL REFERENCES chunks(id),
                model_name TEXT    NOT NULL,
                vector     BLOB    NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_chunk_model
                ON embeddings(chunk_id, model_name);
        """)


# --- Hilfsfunktionen ---

def vector_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def migrate_add_cover() -> None:
    """Adds cover_image and cover_mime columns to books if missing."""
    with get_connection() as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(books)").fetchall()]
        if "cover_image" not in cols:
            conn.execute("ALTER TABLE books ADD COLUMN cover_image BLOB")
            conn.execute("ALTER TABLE books ADD COLUMN cover_mime TEXT")
            conn.commit()


def get_book_by_name(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM books WHERE name = ?", (name,)).fetchone()


def get_all_books(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Only returns books that have at least one indexed embedding preset."""
    return conn.execute(
        """
        SELECT DISTINCT b.id, b.name, b.title,
               (b.cover_image IS NOT NULL) AS has_cover
        FROM books b
        JOIN chunks c ON c.book_id = b.id
        JOIN embeddings e ON e.chunk_id = c.id
        ORDER BY b.name
        """
    ).fetchall()


def get_chunks(conn: sqlite3.Connection, book_id: int, preset_name: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM chunks WHERE book_id = ? AND preset_name = ? ORDER BY start_index",
        (book_id, preset_name),
    ).fetchall()


def get_chunk_presets(conn: sqlite3.Connection, book_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT preset_name FROM chunks WHERE book_id = ? ORDER BY preset_name",
        (book_id,),
    ).fetchall()
    return [r["preset_name"] for r in rows]


def get_valid_presets(conn: sqlite3.Connection, book_id: int, model_name: str) -> list[str]:
    """Only presets that have embeddings in the DB for the given model."""
    rows = conn.execute(
        """
        SELECT DISTINCT c.preset_name
        FROM chunks c
        JOIN embeddings e ON e.chunk_id = c.id AND e.model_name = ?
        WHERE c.book_id = ?
        ORDER BY c.preset_name
        """,
        (model_name, book_id),
    ).fetchall()
    return [r["preset_name"] for r in rows]


def get_chunks_with_embeddings(
    conn: sqlite3.Connection, book_id: int, preset_name: str, model_name: str
) -> tuple[list[dict], np.ndarray] | tuple[None, None]:
    """Returns (chunks_list, embedding_matrix) loaded from DB, or (None, None) if not indexed."""
    rows = conn.execute(
        """
        SELECT c.start_index, c.end_index, c.content, c.word_count, e.vector
        FROM chunks c
        JOIN embeddings e ON e.chunk_id = c.id AND e.model_name = ?
        WHERE c.book_id = ? AND c.preset_name = ?
        ORDER BY c.start_index
        """,
        (model_name, book_id, preset_name),
    ).fetchall()
    if not rows:
        return None, None
    chunks = [
        {"start_index": r["start_index"], "end_index": r["end_index"],
         "content": r["content"], "word_count": r["word_count"]}
        for r in rows
    ]
    emb = np.stack([blob_to_vector(r["vector"]) for r in rows])
    return chunks, emb
