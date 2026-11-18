"""Hybrid vector + BM25 retrieval with query rewriting, RRF fusion, and jurisdiction guarantees."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import chromadb
except ImportError:
    from . import _compat as chromadb  # type: ignore

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    from ._compat import BM25OkapiFallback as BM25Okapi  # type: ignore

from .config import Settings, load_settings
from .embeddings import Embedder
from .ingest import COLLECTION_NAME, run_ingest
from .llm import llm
from .models import ContextItem
from .text_utils import tokenize_bm25


_CACHE: Dict[str, Any] = {
    "chroma_dir": None,
    "collection": None,
    "bm25_data": None,
    "citation_index": None,
    "embedder": None,
}


def reset_retrieval_cache() -> None:
    """Clear cached Chroma/BM25 objects (used when switching directories in tests)."""
    for k in list(_CACHE.keys()):
        _CACHE[k] = None


def is_index_ready(chroma_dir: Optional[Path] = None) -> bool:
    """Check whether both ChromaDB and BM25/citation index files exist."""
    cfg = load_settings()
    c_dir = Path(chroma_dir) if chroma_dir else cfg.chroma_dir
    return (
        c_dir.exists()
        and (c_dir / "bm25.pkl").exists()
        and (c_dir / "citation_index.json").exists()
    )


def get_cached_resources(
    chroma_dir: Optional[Path] = None,
    auto_ingest: bool = False,
) -> Tuple[Any, Dict[str, Any], Dict[str, List[str]], Embedder]:
    """Load and cache Chroma collection, BM25 payload, citation index, and embedder."""
    cfg = load_settings()
    c_dir = Path(chroma_dir) if chroma_dir else cfg.chroma_dir

    if not is_index_ready(c_dir):
        if auto_ingest:
            run_ingest(chroma_dir=c_dir, settings=cfg)
        else:
            raise FileNotFoundError(f"Index missing in {c_dir}; run python -m app.ingest")

    if (
        _CACHE["chroma_dir"] == str(c_dir.resolve())
        and _CACHE["collection"] is not None
        and _CACHE["bm25_data"] is not None
        and _CACHE["citation_index"] is not None
        and _CACHE["embedder"] is not None
    ):
        return (
            _CACHE["collection"],
            _CACHE["bm25_data"],
            _CACHE["citation_index"],
            _CACHE["embedder"],
        )

    client = chromadb.PersistentClient(path=str(c_dir))
    collection = client.get_collection(name=COLLECTION_NAME)
    with open(c_dir / "bm25.pkl", "rb") as f:
        try:
            bm25_data = pickle.load(f)
        except Exception:
            bm25_data = {}
    if "bm25" not in bm25_data and "tokenized_corpus" in bm25_data:
        bm25_data["bm25"] = BM25Okapi(bm25_data["tokenized_corpus"])
    with open(c_dir / "citation_index.json", "r", encoding="utf-8") as f:
        citation_index = json.load(f)
    embedder = Embedder(settings=cfg)

    _CACHE["chroma_dir"] = str(c_dir.resolve())
    _CACHE["collection"] = collection
    _CACHE["bm25_data"] = bm25_data
    _CACHE["citation_index"] = citation_index
    _CACHE["embedder"] = embedder
    return collection, bm25_data, citation_index, embedder


def get_corpus_health_stats(chroma_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Return document counts by jurisdiction and full_text count for GET /health."""
    _, bm25_data, _, _ = get_cached_resources(chroma_dir=chroma_dir, auto_ingest=False)
    docs_by_id: Dict[str, Dict[str, Any]] = bm25_data.get("docs_by_id", {})
    by_j = {"BD": 0, "US": 0, "EU": 0}
    full_text_count = 0
    for d in docs_by_id.values():
        meta = d.get("metadata", {})
        j = meta.get("jurisdiction", "BD")
        if j in by_j:
            by_j[j] += 1
        if meta.get("source_type") == "full_text":
            full_text_count += 1
    return {
        "documents": len(docs_by_id),
        "by_jurisdiction": by_j,
        "full_text_documents": full_text_count,
    }


def _doc_matches_risk_tags(doc: Dict[str, Any], risk_tags: List[str]) -> bool:
    if not risk_tags:
        return True
    stored = (doc.get("metadata", {}).get("risk_tags") or "").lower()
    if not stored:
        return False
    return any(f"|{tag.strip().lower()}|" in stored for tag in risk_tags if tag.strip())


def _to_context_item(doc: Dict[str, Any], max_item_chars: int = 1500) -> ContextItem:
    meta = doc.get("metadata", {})
    raw_text = doc.get("text", "")
    truncated_text = raw_text[:max_item_chars] if len(raw_text) > max_item_chars else raw_text
    return ContextItem(
        id=str(meta.get("id", doc.get("id", ""))),
        citation_string=str(meta.get("citation_string", "")),
        jurisdiction=meta.get("jurisdiction", "BD"),
        title=str(meta.get("title", "")),
        text=truncated_text,
        source_url=str(meta.get("source_url", "")),
        source_type=meta.get("source_type", "provision_summary"),
        verified=bool(meta.get("verified", False)),
        confidence=meta.get("confidence", "medium"),
    )


def search(
    query: str,
    jurisdictions: Optional[List[str]] = None,
    risk_tags: Optional[List[str]] = None,
    k: int = 8,
    chroma_dir: Optional[Path] = None,
) -> List[ContextItem]:
    """
    Execute hybrid RAG search:
    1. Rewrite query into 3 legal search queries via llm.small_json (+ original query).
    2. Run Chroma vector search + BM25 (n=20 each; over-fetch 4x if risk_tags present) with jurisdiction filter.
    3. Merge with Reciprocal Rank Fusion (RRF, k=60), dedupe by id, apply soft risk-tag filter.
    4. Use llm.small_json to select best k from top 20 fused candidates (fallback to fusion order).
    5. Guarantee >= 1 item per requested jurisdiction (if present in fused list) and cap total text at 12,000 chars.
    """
    k = max(1, min(int(k), 20))
    req_jurisdictions = [j.upper() for j in (jurisdictions or ["BD", "US", "EU"]) if j.upper() in ("BD", "US", "EU")]
    if not req_jurisdictions:
        req_jurisdictions = ["BD", "US", "EU"]

    collection, bm25_data, _, embedder = get_cached_resources(chroma_dir=chroma_dir, auto_ingest=False)
    docs_by_id: Dict[str, Dict[str, Any]] = bm25_data["docs_by_id"]
    bm25_ids: List[str] = bm25_data["ids"]
    bm25_obj = bm25_data["bm25"]

    # 1. Query rewrite (3 queries + original query)
    rewrite_system = (
        "You are a legal search query planner for BD, US, and EU competition and commercial contract law. "
        'Rewrite the user query into exactly 3 short legal search queries as JSON: {"queries": ["q1", "q2", "q3"]}.'
    )
    rewritten: List[str] = []
    try:
        rw_resp = llm.small_json(rewrite_system, query)
        if isinstance(rw_resp, dict) and isinstance(rw_resp.get("queries"), list):
            rewritten = [str(q).strip() for q in rw_resp["queries"] if str(q).strip()][:3]
    except Exception:
        # In real mode, if LLM is unavailable, propagate LLMUnavailableError
        raise

    all_queries = [query.strip()] + [q for q in rewritten if q.lower() != query.strip().lower()]

    # Over-fetch 4x when filtering by risk_tags in Python after retrieval
    fetch_n = 80 if (risk_tags and len(risk_tags) > 0) else 20
    where_filter: Optional[Dict[str, Any]] = None
    if len(req_jurisdictions) == 1:
        where_filter = {"jurisdiction": req_jurisdictions[0]}
    elif len(req_jurisdictions) < 3:
        where_filter = {"jurisdiction": {"$in": req_jurisdictions}}

    rrf_scores: Dict[str, float] = {}
    rrf_k = 60

    # 2. Vector + BM25 search across all queries
    q_embeddings = embedder.embed_documents(all_queries)
    vec_results = collection.query(
        query_embeddings=q_embeddings,
        n_results=fetch_n,
        where=where_filter,
    )

    for q_idx, q_text in enumerate(all_queries):
        # Vector ranks
        v_ids = (vec_results.get("ids") or [[]])[q_idx] if q_idx < len(vec_results.get("ids", [])) else []
        for rank_idx, doc_id in enumerate(v_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank_idx))

        # BM25 ranks with jurisdiction filter
        q_tokens = tokenize_bm25(q_text)
        bm25_scores = bm25_obj.get_scores(q_tokens)
        scored_bm25: List[Tuple[float, str]] = []
        for idx, score in enumerate(bm25_scores):
            doc_id = bm25_ids[idx]
            doc = docs_by_id.get(doc_id)
            if not doc:
                continue
            if doc["metadata"].get("jurisdiction") not in req_jurisdictions:
                continue
            scored_bm25.append((float(score), doc_id))

        scored_bm25.sort(key=lambda x: (-x[0], x[1]))
        for rank_idx, (_, doc_id) in enumerate(scored_bm25[:fetch_n], start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank_idx))

    # Sort all fused candidates deterministically by (-score, doc_id)
    fused_ids = [doc_id for doc_id, _ in sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0])) if doc_id in docs_by_id]

    # Soft risk-tag filter in Python after retrieval
    if risk_tags:
        filtered_ids = [did for did in fused_ids if _doc_matches_risk_tags(docs_by_id[did], risk_tags)]
        if len(filtered_ids) < k:
            for did in fused_ids:
                if did not in filtered_ids:
                    filtered_ids.append(did)
        candidate_pool_ids = filtered_ids
    else:
        candidate_pool_ids = fused_ids

    top_20_ids = candidate_pool_ids[:20]

    # 3. LLM candidate selection from top 20 fused candidates
    selected_ids: List[str] = []
    if top_20_ids:
        select_system = (
            'From the top fused legal candidates, choose the best k IDs for the query as JSON: {"ids": ["id1", ...]}.'
        )
        summary_lines = [f"k={k}", f"Query: {query}"]
        for did in top_20_ids:
            d = docs_by_id[did]
            m = d["metadata"]
            summary_lines.append(f"ID: {did} | [{m['jurisdiction']}] {m['citation_string']} - {d['text'][:160]}")
        try:
            sel_resp = llm.small_json(select_system, "\n".join(summary_lines))
            if isinstance(sel_resp, dict) and isinstance(sel_resp.get("ids"), list):
                valid_set = set(top_20_ids)
                for sid in sel_resp["ids"]:
                    sid_str = str(sid).strip()
                    if sid_str in valid_set and sid_str not in selected_ids:
                        selected_ids.append(sid_str)
        except Exception:
            # On selection failure fall back to top-k fusion order
            selected_ids = []

    for did in candidate_pool_ids:
        if len(selected_ids) >= k:
            break
        if did not in selected_ids:
            selected_ids.append(did)
    selected_ids = selected_ids[:k]

    # 4. Guarantee at least 1 item from each requested jurisdiction if present in fused list
    for j in req_jurisdictions:
        if any(docs_by_id[did]["metadata"].get("jurisdiction") == j for did in selected_ids):
            continue
        rep_id = next((did for did in fused_ids if docs_by_id[did]["metadata"].get("jurisdiction") == j), None)
        if rep_id is not None:
            if len(selected_ids) < k:
                selected_ids.append(rep_id)
            else:
                # Replace the lowest-ranked item from a jurisdiction that has > 1 items
                j_counts: Dict[str, int] = {}
                for did in selected_ids:
                    dj = docs_by_id[did]["metadata"].get("jurisdiction", "")
                    j_counts[dj] = j_counts.get(dj, 0) + 1
                replace_idx = None
                for idx in range(len(selected_ids) - 1, -1, -1):
                    dj = docs_by_id[selected_ids[idx]]["metadata"].get("jurisdiction", "")
                    if j_counts.get(dj, 0) > 1:
                        replace_idx = idx
                        break
                if replace_idx is not None:
                    selected_ids[replace_idx] = rep_id
                else:
                    selected_ids[-1] = rep_id

    # 5. Cap प्रत्येक item at 1,500 chars and total returned text at 12,000 chars
    results: List[ContextItem] = []
    total_chars = 0
    for did in selected_ids:
        item = _to_context_item(docs_by_id[did], max_item_chars=1500)
        if total_chars + len(item.text) > 12000:
            break
        results.append(item)
        total_chars += len(item.text)

    return results
