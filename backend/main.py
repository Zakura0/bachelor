import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.database import init_db
from backend.routers import books, search

app = FastAPI(title="Bachelor IR API")

# React Dev-Server auf Port 5173 darf Requests schicken
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()

app.include_router(books.router)
app.include_router(search.router)


@app.get("/")
def root():
    return {"status": "ok"}
