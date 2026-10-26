import json
import logging
import os
import queue
import sys
import threading
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import DIR_CHUNKS, DIR_EMBEDDINGS, EMBEDDING_MODEL
from db.database import get_connection, get_chunks_with_embeddings
from src.retrieval.pipeline import SearchPipeline
from src.retrieval.llm_reranker import PROMPT_RANK_ALL, PROMPT_RANK_ALL_NORMAL

router = APIRouter(prefix="/api/search", tags=["search"])
log = logging.getLogger(__name__)

# Modelle sind prozessweit geteilt, ein Eintrag hält nur Chunks, TF-IDF-Index und Embedding-Matrix
_PIPELINE_CACHE_MAX = 32
_pipeline_cache: OrderedDict[tuple, SearchPipeline] = OrderedDict()
_cache_lock = threading.Lock()
_build_locks: dict[tuple, threading.Lock] = {}


class SearchRequest(BaseModel):
    book_id: int
    preset_name: str
    pipeline: int
    query: str
    top_k: int = 5
    llm_prompt: str = 'expert'  # 'expert' | 'normal'


_PROMPT_MAP: dict[str, str] = {
    'expert': PROMPT_RANK_ALL,
    'normal': PROMPT_RANK_ALL_NORMAL,
}


class SearchResultItem(BaseModel):
    rank: int
    score: float
    content: str
    start_index: int
    end_index: int


def _build_pipeline(book_id: int, book_name: str, preset_name: str, pipeline_preset: int, llm_prompt: str) -> SearchPipeline:
    prompt_template = _PROMPT_MAP.get(llm_prompt, PROMPT_RANK_ALL)
    # DB-first: Chunks + Embeddings aus der Datenbank laden
    with get_connection() as conn:
        chunks, emb = get_chunks_with_embeddings(conn, book_id, preset_name, EMBEDDING_MODEL)
    if chunks is not None:
        return SearchPipeline.from_data(chunks, emb, pipeline_preset, llm_prompt=prompt_template)
    # Fallback: aus Dateien laden (Legacy-Bücher)
    chunks_path = os.path.join(DIR_CHUNKS, book_name, f"{preset_name}.json")
    emb_path = os.path.join(DIR_EMBEDDINGS, book_name, f"{preset_name}.npy")
    if not os.path.exists(chunks_path) or not os.path.exists(emb_path):
        raise FileNotFoundError(f"Keine Daten für {book_name}/{preset_name}")
    return SearchPipeline(chunks_path, emb_path, pipeline_preset, llm_prompt=prompt_template)


def _get_pipeline(book_id: int, book_name: str, preset_name: str, pipeline_preset: int, llm_prompt: str = 'expert') -> SearchPipeline:
    cache_key = (book_id, preset_name, pipeline_preset, llm_prompt)
    with _cache_lock:
        if cache_key in _pipeline_cache:
            _pipeline_cache.move_to_end(cache_key)
            return _pipeline_cache[cache_key]
        build_lock = _build_locks.setdefault(cache_key, threading.Lock())

    # Pro Key nur ein Aufbau gleichzeitig, andere Keys werden nicht blockiert
    with build_lock:
        with _cache_lock:
            if cache_key in _pipeline_cache:
                return _pipeline_cache[cache_key]
        try:
            pipeline = _build_pipeline(book_id, book_name, preset_name, pipeline_preset, llm_prompt)
        finally:
            with _cache_lock:
                _build_locks.pop(cache_key, None)
        with _cache_lock:
            _pipeline_cache[cache_key] = pipeline
            while len(_pipeline_cache) > _PIPELINE_CACHE_MAX:
                _pipeline_cache.popitem(last=False)
    return pipeline


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/stream")
def search_stream(req: SearchRequest):
    with get_connection() as conn:
        book = conn.execute("SELECT name FROM books WHERE id = ?", (req.book_id,)).fetchone()
    if not book:
        raise HTTPException(status_code=404, detail="Buch nicht gefunden")

    book_name = book["name"]
    log.info("search | book=%s preset=%s pipeline=%d prompt=%s query=%r",
             book_name, req.preset_name, req.pipeline, req.llm_prompt, req.query[:120])

    try:
        pipeline = _get_pipeline(req.book_id, book_name, req.preset_name, req.pipeline, req.llm_prompt)
    except FileNotFoundError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    q: queue.Queue = queue.Queue()

    def run():
        try:
            results = pipeline.search(
                req.query,
                top_k=req.top_k,
                on_progress=lambda msg: q.put({"type": "progress", "message": msg}),
            )
            q.put({"type": "result", "data": [
                SearchResultItem(
                    rank=i + 1,
                    score=float(r.score),
                    content=r.text,
                    start_index=r.meta["start_index"],
                    end_index=r.meta["end_index"],
                ).model_dump()
                for i, r in enumerate(results)
            ]})
        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(None)

    def generate():
        threading.Thread(target=run, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            yield _sse(item)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/", response_model=list[SearchResultItem])
def search(req: SearchRequest):
    with get_connection() as conn:
        book = conn.execute("SELECT name FROM books WHERE id = ?", (req.book_id,)).fetchone()
    if not book:
        raise HTTPException(status_code=404, detail="Buch nicht gefunden")
    book_name = book["name"]
    try:
        pipeline = _get_pipeline(req.book_id, book_name, req.preset_name, req.pipeline)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    results = pipeline.search(req.query, top_k=req.top_k)
    return [
        SearchResultItem(
            rank=i + 1, score=float(r.score), content=r.text,
            start_index=r.meta["start_index"], end_index=r.meta["end_index"],
        )
        for i, r in enumerate(results)
    ]
