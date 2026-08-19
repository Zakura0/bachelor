import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import DIR_CHUNKS, DIR_EMBEDDINGS
from db.database import get_connection
from src.retrieval.pipeline import SearchPipeline

router = APIRouter(prefix="/api/search", tags=["search"])

# Geladene Pipeline-Instanzen cachen — Modelle nur einmal laden
_pipeline_cache: dict[tuple, SearchPipeline] = {}


class SearchRequest(BaseModel):
    book_id: int
    preset_name: str
    pipeline: int
    query: str
    top_k: int = 5


class SearchResultItem(BaseModel):
    rank: int
    score: float
    content: str
    start_index: int
    end_index: int


@router.post("/", response_model=list[SearchResultItem])
def search(req: SearchRequest):
    with get_connection() as conn:
        book = conn.execute("SELECT name FROM books WHERE id = ?", (req.book_id,)).fetchone()
    if not book:
        raise HTTPException(status_code=404, detail="Buch nicht gefunden")

    book_name = book["name"]
    cache_key = (book_name, req.preset_name, req.pipeline)

    if cache_key not in _pipeline_cache:
        chunks_path = os.path.join(DIR_CHUNKS, book_name, f"{req.preset_name}.json")
        emb_path = os.path.join(DIR_EMBEDDINGS, book_name, f"{req.preset_name}.npy")

        if not os.path.exists(chunks_path):
            raise HTTPException(status_code=422, detail=f"Chunks nicht gefunden für {book_name}/{req.preset_name}")
        if not os.path.exists(emb_path):
            raise HTTPException(status_code=422, detail=f"Embeddings nicht gefunden für {book_name}/{req.preset_name} — bitte zuerst Embeddings generieren")

        try:
            _pipeline_cache[cache_key] = SearchPipeline(chunks_path, emb_path, req.pipeline)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    pipeline = _pipeline_cache[cache_key]
    results = pipeline.search(req.query, top_k=req.top_k)

    return [
        SearchResultItem(
            rank=i + 1,
            score=float(r.score),
            content=r.text,
            start_index=r.meta["start_index"],
            end_index=r.meta["end_index"],
        )
        for i, r in enumerate(results)
    ]
