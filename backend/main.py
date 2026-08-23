import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend.log")),
        logging.StreamHandler(),
    ],
)
# Silence noisy third-party loggers
for _noisy in ("sentence_transformers", "httpx", "httpcore", "transformers", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from db.database import init_db, migrate_add_title, migrate_add_cover
from backend.routers import books, search

app = FastAPI(title="Bachelor IR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    migrate_add_title()
    migrate_add_cover()

app.include_router(books.router)
app.include_router(search.router)

# Serve built frontend — only active when dist/ exists
_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.isdir(_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(_dist, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        return FileResponse(os.path.join(_dist, "index.html"))


@app.get("/")
def root():
    return {"status": "ok"}
