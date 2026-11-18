"""Swappable embeddings backend ('openai', 'local', 'hash') for Juris AI RAG Service."""

import hashlib
import math
from typing import List, Optional

import httpx

from .config import Settings, load_settings
from .text_utils import tokenize_bm25


class Embedder:
    """
    Unified embedding wrapper supporting:
    - 'hash': deterministic 384-dim hashing-trick bag-of-words/n-gram embedder (no downloads)
    - 'local': sentence-transformers BAAI/bge-small-en-v1.5 (optional import)
    - 'openai': OpenAI text-embedding-3-small
    """

    DIM = 384

    def __init__(self, backend: Optional[str] = None, settings: Optional[Settings] = None):
        self.settings = settings or load_settings()
        self.backend = (backend or self.settings.embeddings_backend).lower()
        if self.backend not in ("openai", "local", "hash"):
            self.backend = "hash"
        self._local_model = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.backend == "openai":
            return self._embed_openai(texts)
        if self.backend == "local":
            return self._embed_local(texts)
        return [self._embed_hash(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        res = self.embed_documents([text])
        return res[0] if res else [0.0] * self.DIM

    def _embed_hash(self, text: str) -> List[float]:
        vec = [0.0] * self.DIM
        tokens = tokenize_bm25(text)
        if not tokens:
            return vec

        # Include unigrams and bigrams for stronger legal phrase sensitivity
        features: List[tuple[str, float]] = [(tok, 1.0) for tok in tokens]
        for i in range(len(tokens) - 1):
            features.append((f"{tokens[i]}_{tokens[i + 1]}", 1.25))

        for feat, weight in features:
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.DIM
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign * weight

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [round(v / norm, 6) for v in vec]
        return vec

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed; use EMBEDDINGS_BACKEND=hash or openai"
                ) from exc
            self._local_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        encoded = self._local_model.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in encoded]

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        api_key = self.settings.openai_api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when EMBEDDINGS_BACKEND=openai")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "text-embedding-3-small",
            "input": texts,
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        return [list(map(float, item["embedding"])) for item in items]
