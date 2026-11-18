"""FastAPI application implementing the Section 3 Interface Contract."""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import unquote

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .analyze import DocumentTooLargeError, analyze_document, chat_answer
from .config import load_settings
from .ingest import run_ingest
from .llm import LLMUnavailableError
from .models import (
    AnalyzeRequest,
    ChatRequest,
    CitationLookupResponse,
    CorpusStats,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    model_to_dict,
)
from .retrieve import get_corpus_health_stats, is_index_ready, search
from .verify import verify_citation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("juris_rag.server")

MAX_BODY_BYTES = 1_500_000  # 1.5 MB request body limit


class ServiceAPIError(Exception):
    def __init__(self, status_code: int, error: str, message: str):
        self.status_code = status_code
        self.error = error
        self.message = message
        super().__init__(message)


class BodySizeTooLargeError(Exception):
    pass


class ASGIGuardAndLoggingMiddleware:
    """
    Pure ASGI middleware that enforces the 1.5 MB request body limit,
    attaches X-Request-ID header, and emits structured logs without reading document content.
    Avoids Starlette BaseHTTPMiddleware stream deadlocks across all FastAPI/Starlette versions.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        t0 = time.perf_counter()
        headers = dict(scope.get("headers") or [])
        req_id_bytes = headers.get(b"x-request-id")
        req_id = req_id_bytes.decode("latin1") if req_id_bytes else str(uuid.uuid4())
        method = scope.get("method", "GET")
        path = scope.get("path", "/")

        # Check Content-Length upfront if present
        content_length_bytes = headers.get(b"content-length")
        if content_length_bytes:
            try:
                if int(content_length_bytes.decode("latin1")) > MAX_BODY_BYTES:
                    resp = JSONResponse(
                        status_code=413,
                        content={
                            "error": "payload_too_large",
                            "message": f"Request body exceeds {MAX_BODY_BYTES} bytes limit.",
                        },
                        headers={"X-Request-ID": req_id},
                    )
                    await resp(scope, receive, send)
                    return
            except ValueError:
                pass

        received_bytes = 0
        status_code_holder = [200]

        async def wrapped_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                received_bytes += len(body)
                if received_bytes > MAX_BODY_BYTES:
                    raise BodySizeTooLargeError()
            return message

        async def wrapped_send(message):
            if message["type"] == "http.response.start":
                status_code_holder[0] = message.get("status", 200)
                raw_headers = list(message.get("headers") or [])
                raw_headers.append((b"x-request-id", req_id.encode("latin1")))
                message = {**message, "headers": raw_headers}
            await send(message)

        try:
            await self.app(scope, wrapped_receive, wrapped_send)
        except BodySizeTooLargeError:
            status_code_holder[0] = 413
            resp = JSONResponse(
                status_code=413,
                content={
                    "error": "payload_too_large",
                    "message": f"Request body exceeds {MAX_BODY_BYTES} bytes limit.",
                },
                headers={"X-Request-ID": req_id},
            )
            await resp(scope, receive, send)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": req_id,
                    "method": method,
                    "path": path,
                    "status_code": status_code_holder[0],
                    "duration_ms": elapsed_ms,
                }
            )
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    if not is_index_ready(settings.chroma_dir):
        logger.info(json.dumps({"event": "startup_auto_ingest", "chroma_dir": str(settings.chroma_dir)}))
        run_ingest(data_dir=settings.data_dir, chroma_dir=settings.chroma_dir, settings=settings)
    yield


def create_app() -> FastAPI:
    cfg = load_settings()
    app = FastAPI(
        title="Juris AI RAG Service",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(ASGIGuardAndLoggingMiddleware)

    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.exception_handler(ServiceAPIError)
    async def _handle_service_error(request: Request, exc: ServiceAPIError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error, "message": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": str(exc.errors()),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(request: Request, exc: StarletteHTTPException):
        code_map = {
            400: "bad_request",
            401: "unauthorized",
            404: "not_found",
            413: "payload_too_large",
            422: "validation_error",
            500: "internal_error",
            503: "service_unavailable",
        }
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": code_map.get(exc.status_code, "http_error"),
                "message": str(exc.detail),
            },
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, exc: Exception):
        logger.error(json.dumps({"event": "unhandled_error", "type": type(exc).__name__, "detail": str(exc)}))
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "An unexpected internal server error occurred."},
        )

    def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> str:
        settings = load_settings()
        if not x_api_key or x_api_key != settings.rag_api_key:
            raise ServiceAPIError(status_code=401, error="unauthorized", message="Missing or invalid X-API-Key")
        return x_api_key

    def ensure_index_available() -> None:
        settings = load_settings()
        if not is_index_ready(settings.chroma_dir):
            raise ServiceAPIError(
                status_code=503,
                error="index_missing",
                message="Corpus index is missing; run python -m app.ingest",
            )

    @app.get("/health", response_model=HealthResponse)
    def get_health():
        settings = load_settings()
        ensure_index_available()
        stats = get_corpus_health_stats(chroma_dir=settings.chroma_dir)
        resp = HealthResponse(
            status="ok",
            version=__version__,
            mock=settings.mock_llm,
            embeddings=settings.embeddings_backend,  # type: ignore[arg-type]
            corpus=CorpusStats(**stats),
        )
        return model_to_dict(resp)

    @app.post("/rag/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
    def post_rag_search(payload: SearchRequest):
        settings = load_settings()
        ensure_index_available()
        try:
            items = search(
                query=payload.query,
                jurisdictions=list(payload.jurisdictions) if payload.jurisdictions else None,
                risk_tags=payload.risk_tags,
                k=payload.k,
                chroma_dir=settings.chroma_dir,
            )
            return model_to_dict(SearchResponse(context=items))
        except LLMUnavailableError as exc:
            raise ServiceAPIError(status_code=503, error="llm_unavailable", message=str(exc)) from exc

    @app.post("/analyze", dependencies=[Depends(require_api_key)])
    def post_analyze(payload: AnalyzeRequest):
        settings = load_settings()
        ensure_index_available()
        try:
            result = analyze_document(
                document_text=payload.document_text,
                jurisdictions=list(payload.jurisdictions) if payload.jurisdictions else None,
                doc_name=payload.doc_name,
                chroma_dir=settings.chroma_dir,
            )
            return model_to_dict(result)
        except DocumentTooLargeError as exc:
            raise ServiceAPIError(status_code=413, error="document_too_large", message=str(exc)) from exc
        except LLMUnavailableError as exc:
            raise ServiceAPIError(status_code=503, error="llm_unavailable", message=str(exc)) from exc

    @app.post("/chat", dependencies=[Depends(require_api_key)])
    def post_chat(payload: ChatRequest):
        settings = load_settings()
        ensure_index_available()
        try:
            result = chat_answer(
                messages=payload.messages,
                jurisdictions=list(payload.jurisdictions) if payload.jurisdictions else None,
                context_note=payload.context_note,
                chroma_dir=settings.chroma_dir,
            )
            return model_to_dict(result)
        except LLMUnavailableError as exc:
            raise ServiceAPIError(status_code=503, error="llm_unavailable", message=str(exc)) from exc

    @app.get("/citation/{citation_string:path}", dependencies=[Depends(require_api_key)])
    def get_citation(citation_string: str):
        settings = load_settings()
        ensure_index_available()
        decoded = unquote(citation_string).strip()
        if not decoded:
            raise ServiceAPIError(status_code=400, error="bad_request", message="citation_string cannot be empty")
        res = verify_citation(decoded, quote=None, chroma_dir=settings.chroma_dir)
        out = CitationLookupResponse(
            citation_string=decoded,
            status=res["status"],  # type: ignore[arg-type]
            reason=res["reason"] or "",
            source_url=res.get("source_url"),
        )
        return model_to_dict(out)

    return app


app = create_app()
