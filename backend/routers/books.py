from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json, os, queue, re, sys, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import EMBEDDING_MODEL
from db.database import get_connection, get_all_books, get_valid_presets, vector_to_blob

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
        presets = get_valid_presets(conn, book_id, EMBEDDING_MODEL)
    return {"book_id": book_id, "book_name": book["name"], "presets": presets}


@router.post("/")
async def create_book(
    title: str = Form(...),
    name: str = Form(None),
    file: UploadFile = File(...),
):
    raw_bytes = await file.read()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin-1")

    if not name:
        name = re.sub(r"[^a-z0-9]", "", (file.filename or "buch").lower().replace(".txt", ""))
    if not name:
        raise HTTPException(status_code=422, detail="Konnte keinen Namen ableiten")

    with get_connection() as conn:
        if conn.execute("SELECT id FROM books WHERE name = ?", (name,)).fetchone():
            raise HTTPException(status_code=409, detail=f"Buch '{name}' existiert bereits")
        cur = conn.execute(
            "INSERT INTO books (name, title, raw_text) VALUES (?, ?, ?)",
            (name, title, raw_text),
        )
        book_id = cur.lastrowid
        conn.commit()

    return {"id": book_id, "name": name, "title": title}


class CreateAndIndexRequest(BaseModel):
    title: str
    name: str
    raw_text: str
    preset_name: str = "large"
    min_words: int = 50
    max_words: int = 150
    overlap: int = 2


@router.post("/create-and-index")
def create_and_index(req: CreateAndIndexRequest):
    """Atomically create a book and index its first preset. Rolls back on error."""
    with get_connection() as conn:
        if conn.execute("SELECT id FROM books WHERE name = ?", (req.name,)).fetchone():
            raise HTTPException(status_code=409, detail=f"Buch '{req.name}' existiert bereits")

    def generate():
        book_id = None
        try:
            from src.preprocessing.book_chunker import BookChunker
            from src.retrieval.embeddings import EmbeddingRetriever

            with get_connection() as conn:
                cur = conn.execute(
                    "INSERT INTO books (name, title, raw_text) VALUES (?, ?, ?)",
                    (req.name, req.title, req.raw_text),
                )
                book_id = cur.lastrowid
                conn.commit()

            yield _sse({"type": "book_created", "id": book_id, "name": req.name, "title": req.title})
            yield _sse({"type": "progress", "message": f"Chunks erstellen ({req.preset_name})…"})

            chunker = BookChunker(req.min_words, req.max_words, req.overlap)
            result = chunker.build_chunks(req.raw_text)
            chunks = result["chunks"]

            with get_connection() as conn:
                chunk_ids = []
                for c in chunks:
                    cur = conn.execute(
                        "INSERT INTO chunks (book_id, preset_name, start_index, end_index, content, word_count) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (book_id, req.preset_name, c["start_index"], c["end_index"], c["content"], c["word_count"]),
                    )
                    chunk_ids.append(cur.lastrowid)
                conn.commit()

            yield _sse({"type": "progress", "message": f"{len(chunks)} Chunks erstellt. Lade Embedding-Modell…"})
            retriever = EmbeddingRetriever(model_name=EMBEDDING_MODEL)
            texts = [c["content"] for c in chunks]

            yield _sse({"type": "progress", "message": f"Berechne Embeddings für {len(texts)} Chunks…"})
            retriever.fit(texts, batch_size=64, show_progress_bar=False)

            yield _sse({"type": "progress", "message": "Embeddings in DB speichern…"})
            with get_connection() as conn:
                for chunk_id, vec in zip(chunk_ids, retriever._emb):
                    conn.execute(
                        "INSERT OR IGNORE INTO embeddings (chunk_id, model_name, vector) VALUES (?, ?, ?)",
                        (chunk_id, EMBEDDING_MODEL, vector_to_blob(vec)),
                    )
                conn.commit()

            yield _sse({"type": "done", "chunk_count": len(chunks)})

        except Exception as e:
            # Roll back: remove book and any partial chunks/embeddings
            if book_id:
                with get_connection() as conn:
                    conn.execute(
                        "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE book_id = ?)",
                        (book_id,),
                    )
                    conn.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
                    conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
                    conn.commit()
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream")


class IndexRequest(BaseModel):
    preset_name: str
    min_words: int
    max_words: int
    overlap: int


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/{book_id}/cover")
def get_cover(book_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT cover_image, cover_mime FROM books WHERE id = ?", (book_id,)).fetchone()
    if not row or not row["cover_image"]:
        raise HTTPException(status_code=404, detail="Kein Cover")
    return Response(content=row["cover_image"], media_type=row["cover_mime"] or "image/jpeg")


@router.post("/{book_id}/cover")
async def upload_cover(book_id: int, file: UploadFile = File(...)):
    data = await file.read()
    mime = file.content_type or "image/jpeg"
    with get_connection() as conn:
        if not conn.execute("SELECT id FROM books WHERE id = ?", (book_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Buch nicht gefunden")
        conn.execute("UPDATE books SET cover_image = ?, cover_mime = ? WHERE id = ?", (data, mime, book_id))
        conn.commit()
    return {"ok": True}


@router.get("/{book_id}/text")
def get_book_text(book_id: int):
    with get_connection() as conn:
        book = conn.execute("SELECT raw_text FROM books WHERE id = ?", (book_id,)).fetchone()
        if not book:
            raise HTTPException(status_code=404, detail="Buch nicht gefunden")
    return {"raw_text": book["raw_text"]}


@router.delete("/{book_id}")
def delete_book(book_id: int):
    with get_connection() as conn:
        book = conn.execute("SELECT id, name, title FROM books WHERE id = ?", (book_id,)).fetchone()
        if not book:
            raise HTTPException(status_code=404, detail="Buch nicht gefunden")
        conn.execute(
            "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE book_id = ?)",
            (book_id,),
        )
        conn.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
        conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
        conn.commit()
    return {"deleted": book_id, "name": book["name"]}


@router.post("/{book_id}/index")
def index_preset(book_id: int, req: IndexRequest):
    with get_connection() as conn:
        book = conn.execute("SELECT id, name, raw_text FROM books WHERE id = ?", (book_id,)).fetchone()
        if not book:
            raise HTTPException(status_code=404, detail="Buch nicht gefunden")
        already = conn.execute(
            """SELECT COUNT(*) FROM embeddings e
               JOIN chunks c ON c.id = e.chunk_id
               WHERE c.book_id = ? AND c.preset_name = ? AND e.model_name = ?""",
            (book_id, req.preset_name, EMBEDDING_MODEL),
        ).fetchone()[0]
    if already:
        raise HTTPException(status_code=409, detail=f"Preset '{req.preset_name}' ist bereits indexiert")

    raw_text = book["raw_text"]

    def generate():
        q: queue.Queue = queue.Queue()

        def run():
            try:
                from src.preprocessing.book_chunker import BookChunker
                from src.retrieval.embeddings import EmbeddingRetriever
                import numpy as np

                q.put({"type": "progress", "message": f"Chunks erstellen ({req.preset_name})…"})
                chunker = BookChunker(req.min_words, req.max_words, req.overlap)
                result = chunker.build_chunks(raw_text)
                chunks = result["chunks"]

                with get_connection() as conn:
                    chunk_ids = []
                    for c in chunks:
                        cur = conn.execute(
                            "INSERT INTO chunks (book_id, preset_name, start_index, end_index, content, word_count) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (book_id, req.preset_name, c["start_index"], c["end_index"], c["content"], c["word_count"]),
                        )
                        chunk_ids.append(cur.lastrowid)
                    conn.commit()

                q.put({"type": "progress", "message": f"{len(chunks)} Chunks erstellt. Lade Embedding-Modell…"})
                retriever = EmbeddingRetriever(model_name=EMBEDDING_MODEL)
                texts = [c["content"] for c in chunks]

                q.put({"type": "progress", "message": f"Berechne Embeddings für {len(texts)} Chunks…"})
                retriever.fit(texts, batch_size=64, show_progress_bar=False)
                vectors = retriever._emb

                q.put({"type": "progress", "message": "Embeddings in DB speichern…"})
                with get_connection() as conn:
                    for chunk_id, vec in zip(chunk_ids, vectors):
                        conn.execute(
                            "INSERT OR IGNORE INTO embeddings (chunk_id, model_name, vector) VALUES (?, ?, ?)",
                            (chunk_id, EMBEDDING_MODEL, vector_to_blob(vec)),
                        )
                    conn.commit()

                q.put({"type": "done", "chunk_count": len(chunks)})
            except Exception as e:
                q.put({"type": "error", "message": str(e)})

        t = threading.Thread(target=run, daemon=True)
        t.start()
        while True:
            event = q.get()
            yield _sse(event)
            if event["type"] in ("done", "error"):
                break
        t.join()

    return StreamingResponse(generate(), media_type="text/event-stream")
