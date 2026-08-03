from fastapi import APIRouter, HTTPException
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from db.database import get_connection, get_all_books, get_chunk_presets

router = APIRouter(prefix="/api/books", tags=["books"])


@router.get("/")
def list_books():
    with get_connection() as conn:
        books = get_all_books(conn)
    return [dict(b) for b in books]


@router.get("/{book_id}/presets")
def list_presets(book_id: int):
    with get_connection() as conn:
        book = conn.execute("SELECT id, name FROM books WHERE id = ?", (book_id,)).fetchone()
        if not book:
            raise HTTPException(status_code=404, detail="Buch nicht gefunden")
        presets = get_chunk_presets(conn, book_id)
    return {"book_id": book_id, "book_name": book["name"], "presets": presets}
