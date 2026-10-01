# --- Stage 1: Frontend bauen ---
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Backend + Runtime ---
FROM python:3.12-slim
WORKDIR /app

# Für sentencepiece/torch-Build sowie sqlite3-CLI (Debugging)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py main.py ./
COPY backend/ ./backend/
COPY src/ ./src/
COPY script/ ./script/
COPY db/ ./db/
COPY data/ ./data/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

ENV PYTHONUNBUFFERED=1
EXPOSE 8100

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8100"]
