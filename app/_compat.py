"""Offline fallback implementations used only when chromadb, rank_bm25, tenacity, or anthropic are not installed."""

import functools
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class BM25OkapiFallback:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.corpus_size = len(corpus)
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []
        nd: Dict[str, int] = {}
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)
            frequencies: Dict[str, int] = {}
            for word in document:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1
        self.avgdl = (num_doc / self.corpus_size) if self.corpus_size > 0 else 0.0
        idf_sum = 0.0
        negative_idfs: List[str] = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = (idf_sum / len(self.idf)) if self.idf else 0.0
        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for q in query:
            if q not in self.idf:
                continue
            idf = self.idf[q]
            for idx, doc_freq in enumerate(self.doc_freqs):
                q_freq = doc_freq.get(q, 0)
                if q_freq == 0:
                    continue
                dl = self.doc_len[idx]
                denom = q_freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[idx] += idf * (q_freq * (self.k1 + 1)) / denom
        return scores


def _cosine_distance(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - (dot / (na * nb))


def _match_where(metadata: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
    if not where:
        return True
    for k, v in where.items():
        if k == "$and":
            if not all(_match_where(metadata, clause) for clause in v):
                return False
        elif k == "$or":
            if not any(_match_where(metadata, clause) for clause in v):
                return False
        elif isinstance(v, dict):
            val = metadata.get(k)
            if "$in" in v and val not in v["$in"]:
                return False
            if "$eq" in v and val != v["$eq"]:
                return False
            if "$ne" in v and val == v["$ne"]:
                return False
        else:
            if metadata.get(k) != v:
                return False
    return True


class _CollectionFallback:
    def __init__(self, db_path: Path, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.db_path = db_path
        self.name = name
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_tables(metadata)

    def _init_tables(self, metadata: Optional[Dict[str, Any]]):
        with self._conn:
            self._conn.execute("CREATE TABLE IF NOT EXISTS collections (name TEXT PRIMARY KEY, metadata TEXT)")
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS embeddings (
                    collection TEXT,
                    id TEXT,
                    document TEXT,
                    metadata TEXT,
                    embedding TEXT,
                    PRIMARY KEY (collection, id)
                )"""
            )
            cur = self._conn.execute("SELECT metadata FROM collections WHERE name = ?", (self.name,))
            row = cur.fetchone()
            if row is None:
                meta_json = json.dumps(metadata or {})
                self._conn.execute(
                    "INSERT INTO collections (name, metadata) VALUES (?, ?)", (self.name, meta_json)
                )
                self.metadata = metadata or {}
            else:
                existing = json.loads(row[0]) if row[0] else {}
                if metadata:
                    existing.update(metadata)
                    self._conn.execute(
                        "UPDATE collections SET metadata = ? WHERE name = ?",
                        (json.dumps(existing), self.name),
                    )
                self.metadata = existing

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        documents = documents or ["" for _ in ids]
        metadatas = metadatas or [{} for _ in ids]
        for meta in metadatas:
            for mk, mv in meta.items():
                if not isinstance(mv, (str, int, float, bool)):
                    raise ValueError(f"Chroma metadata value for {mk} must be str/int/float/bool")
        with self._conn:
            for doc_id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
                self._conn.execute(
                    """INSERT OR REPLACE INTO embeddings (collection, id, document, metadata, embedding)
                       VALUES (?, ?, ?, ?, ?)""",
                    (self.name, str(doc_id), str(doc), json.dumps(meta), json.dumps(list(emb))),
                )

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        cur = self._conn.execute(
            "SELECT id, document, metadata, embedding FROM embeddings WHERE collection = ?",
            (self.name,),
        )
        rows = cur.fetchall()
        candidates = []
        for r_id, r_doc, r_meta, r_emb in rows:
            meta = json.loads(r_meta)
            if _match_where(meta, where):
                candidates.append((r_id, r_doc, meta, json.loads(r_emb)))

        all_ids, all_docs, all_metas, all_dists = [], [], [], []
        for q_emb in query_embeddings:
            scored = []
            for r_id, r_doc, meta, emb in candidates:
                dist = _cosine_distance(q_emb, emb)
                scored.append((dist, r_id, r_doc, meta))
            scored.sort(key=lambda x: (x[0], x[1]))
            top = scored[:n_results]
            all_ids.append([x[1] for x in top])
            all_docs.append([x[2] for x in top])
            all_metas.append([x[3] for x in top])
            all_dists.append([x[0] for x in top])
        return {
            "ids": all_ids,
            "documents": all_docs,
            "metadatas": all_metas,
            "distances": all_dists,
        }


class PersistentClientFallback:
    def __init__(self, path: str = "./chroma_db", **kwargs):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_file = self.path / "chroma.sqlite3"

    def get_or_create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None, embedding_function: Any = None
    ) -> _CollectionFallback:
        return _CollectionFallback(self.db_file, name, metadata=metadata)

    def get_collection(self, name: str, embedding_function: Any = None) -> _CollectionFallback:
        conn = sqlite3.connect(str(self.db_file))
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='collections'")
        if not cur.fetchone():
            conn.close()
            raise ValueError(f"Collection {name} does not exist")
        cur2 = conn.execute("SELECT metadata FROM collections WHERE name = ?", (name,))
        row = cur2.fetchone()
        conn.close()
        if row is None:
            raise ValueError(f"Collection {name} does not exist")
        return _CollectionFallback(self.db_file, name)

    def delete_collection(self, name: str):
        if not self.db_file.exists():
            return
        conn = sqlite3.connect(str(self.db_file))
        with conn:
            conn.execute("DELETE FROM collections WHERE name = ?", (name,))
            conn.execute("DELETE FROM embeddings WHERE collection = ?", (name,))
        conn.close()


def stop_after_attempt_fallback(max_attempt_number: int):
    return max_attempt_number


def wait_exponential_fallback(multiplier: float = 1, min: float = 0, max: float = 10):
    return (multiplier, min, max)


def retry_fallback(stop=None, wait=None, reraise: bool = True, **kwargs):
    max_attempts = stop if isinstance(stop, int) else 3

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kw):
            last_err = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kw)
                except Exception as exc:
                    last_err = exc
                    if attempt + 1 < max_attempts:
                        time.sleep(0.01)
            if reraise and last_err is not None:
                raise last_err

        return wrapper

    return decorator

PersistentClient = PersistentClientFallback
BM25Okapi = BM25OkapiFallback
