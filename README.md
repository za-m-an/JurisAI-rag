# Juris AI — Legal Knowledge Graph RAG Service (`rag/`)

The **Juris AI RAG Service** is a Python + FastAPI microservice powered by a **zero-vector Legal Knowledge Graph** across **Bangladesh (`BD`)**, **United States (`US`)**, and **European Union (`EU`)** jurisdictions. It models statutory authorities using structured object-oriented legal hierarchies (`Law`, `SectionCode`, `ArticleNumber`, `ArticleSummary`, `PrecedentCase`, and `LawEdge` difficulty-escalation paths) to perform deterministic citation verification, multi-hop statutory risk analysis, and grounded legal consultation.

## 1. Architecture Overview

- **Zero Vector Database**: Replaces vector embeddings and ChromaDB with an in-memory relational knowledge graph (`rag/data/legal_graph.json`, 1,120 nodes across 53 statutes and 56 landmark precedents).
- **Object-Oriented Legal Data Structures (`app/legal_structures.py`)**:
  - `Law`: Statute metadata, jurisdiction (`BD`, `US`, `EU`), enactment year, and penal code references.
  - `SectionCode` & `ArticleNumber`: Granular statutory identifiers.
  - `ArticleSummary`: Executive legal rule and risk tier classification.
  - `PrecedentCase`: Landmark judicial holdings linked to statutory sections.
  - `LawEdge`: Relational difficulty-escalation edges (e.g., `penalty_clause`, `restraint_of_trade`, `fraud_in_contract`, `dispute_resolution`, `data_protection`).
- **Open Statutory Datasets (`data/open_datasets/`)**:
  - **Bangladesh (`BD`)**: Penal Code 1860, Constitution of Bangladesh, Transfer of Property Act 1882, Registration Act 1908, Code of Criminal Procedure 1898, Contract Act 1872, Competition Act 2012, Specific Relief Act 1877, Cyber Security Act 2023, Labour Act 2006.
  - **United States (`US`)**: Title 9 U.S.C. (Federal Arbitration Act), Title 15 U.S.C. (Commerce & Antitrust), Title 18 U.S.C. (CFAA & Wire Fraud).
  - **European Union (`EU`)**: Regulation (EU) 2016/679 (GDPR Articles 1–99), TFEU, Rome I.

## 2. Local Development & Execution

```bash
cd rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Build or refresh the Legal Knowledge Graph (rag/data/legal_graph.json)
python3 -m app.graph_builder

# Run the FastAPI server on port 8001
uvicorn app.server:app --host 0.0.0.0 --port 8001
```

### Running the Test Suite

```bash
cd rag
pytest tests/
```

## 3. Docker & Render Deployment

### Standalone Docker Build & Run

```bash
cd rag
docker build -t juris-ai-rag:latest .
docker run --rm -p 8001:8001 --env-file .env.example juris-ai-rag:latest
```

### Deploy to Render

1. Create a **Web Service** in Render connected to this repository with **Environment** set to `Docker`.
2. Configure the required Environment Variables in Render:
   - `ENV=production`
   - `RAG_API_KEY=<your-production-api-key>`
   - `MOCK_LLM=0` (when using live inference keys)
   - `CORS_ORIGINS=https://your-web-domain.vercel.app`
3. Render injects `PORT` automatically, honoured by `Dockerfile` via `uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8001}`.

## 4. API Endpoints Summary

All endpoints except `GET /health` require the `X-API-Key` header.

- `GET /health` — Service status and corpus counts by jurisdiction (`BD`, `US`, `EU`).
- `POST /rag/search` — Graph keyword, section code, and difficulty-trigger retrieval.
- `POST /analyze` — Multi-stage contract risk analysis, difficulty traversal, and citation verification.
- `POST /chat` — Multi-turn legal consultation grounded in the Legal Knowledge Graph.
- `GET /citation/{citation_string}` — Deterministic statutory and precedent lookup.
