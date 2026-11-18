# Juris AI — RAG & Citation Verification Service (`rag/`)

The **Juris AI RAG Service** is a Python 3.11 + FastAPI microservice that indexes legal authorities across **Bangladesh (`BD`)**, **United States (`US`)**, and **European Union (`EU`)** jurisdictions, performs hybrid vector (`ChromaDB`) + lexical (`BM25`) retrieval with Reciprocal Rank Fusion (`RRF`), analyzes commercial contracts for non-obvious legal risks, deterministically verifies every citation against the indexed corpus, and generates legal memoranda free of hallucinated citations.

## 1. Local Development & Execution

```bash
cd rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Build ChromaDB vector index, BM25 index, and citation_index.json
python -m app.ingest

# Run the FastAPI server on port 8001
uvicorn app.server:app --host 0.0.0.0 --port 8001
```

### Running the Test Suite

Everything runs offline without API keys (`MOCK_LLM=1`, `EMBEDDINGS_BACKEND=hash`):

```bash
cd rag
pytest -q
```

## 2. Docker Execution

### Standalone Docker Build & Run

```bash
cd rag
docker build -t juris-ai-rag:latest .
docker run --rm -p 8001:8001 --env-file .env.example juris-ai-rag:latest
```

### Docker Compose (from repository root)

```bash
cp rag/.env.example rag/.env
docker compose up --build
```

## 3. Cloud Deployment Instructions

### Deploy to Render

1. Create a new **Web Service** in Render connected to this repository.
2. Set **Root Directory** to `rag` and **Environment** to `Docker` (uses `rag/Dockerfile`, which runs `python -m app.ingest` at image build time so no persistent disk is required).
3. Configure the following Environment Variables in Render:
   - `ENV=production`
   - `RAG_API_KEY=<your-strong-production-secret>`
   - `ANTHROPIC_API_KEY=<your-anthropic-api-key>`
   - `OPENAI_API_KEY=<optional-openai-api-key>`
   - `EMBEDDINGS_BACKEND=hash` (or `openai` if `OPENAI_API_KEY` is set at build & runtime)
   - `CORS_ORIGINS=https://your-web-frontend.onrender.com`
4. Render automatically injects `PORT`, which `Dockerfile` honours via `uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8001}`.

### Deploy to Google Cloud Run

```bash
cd rag

# Build and push container image to Google Artifact Registry / Container Registry
gcloud builds submit --tag gcr.io/<PROJECT_ID>/juris-ai-rag

# Deploy to Cloud Run
gcloud run deploy juris-ai-rag \
  --image gcr.io/<PROJECT_ID>/juris-ai-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="ENV=production,RAG_API_KEY=<YOUR_RAG_API_KEY>,ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>,EMBEDDINGS_BACKEND=hash"
```

## 4. Refreshing the Legal Corpus

1. Add or update entries in `rag/data/`:
   - `laws_registry.csv`
   - `key_provisions.csv`
   - `landmark_cases.csv`
   - `corpus_chunks.jsonl` (optional full-text chunks generated via `python scripts/fetch_fulltext.py`)
2. Rebuild the local index (idempotent upsert):
   ```bash
   cd rag
   python -m app.ingest
   ```
3. If deployed via Docker, rebuild the image (`docker build -t juris-ai-rag:latest .`) so the updated `chroma_db/` index is baked into the container.

## 5. API Endpoints Summary

All endpoints except `GET /health` require header `X-API-Key: <RAG_API_KEY>`.

- `GET /health` — Service status, active embeddings backend, and corpus counts by jurisdiction (`BD`, `US`, `EU`).
- `POST /rag/search` — Hybrid vector + BM25 RRF retrieval with jurisdiction and risk-tag filtering.
- `POST /analyze` — Multi-stage contract risk analysis, citation verification, and Markdown memorandum generation.
- `POST /chat` — Legal follow-up Q&A grounded in verified corpus citations.
- `GET /citation/{citation_string}` — URL-encoded deterministic citation status check (`verified`, `found_in_corpus`, `unverified`).
