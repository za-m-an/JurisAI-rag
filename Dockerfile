FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EMBEDDINGS_BACKEND=hash \
    MOCK_LLM=1 \
    PORT=8001

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY data /app/data
COPY scripts /app/scripts
COPY tests /app/tests

# Bake the ChromaDB vector index, BM25 index, and citation_index.json into the image at build time
RUN python -m app.ingest

EXPOSE 8001

CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8001}"]
