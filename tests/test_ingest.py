"""Unit tests for corpus ingestion, idempotency, and alias indexing (Phase 2)."""

import json
from pathlib import Path

from app.ingest import run_ingest
from app.text_utils import normalize_citation


def test_ingest_counts_and_idempotency(tmp_path: Path):
    chroma_dir = tmp_path / "chroma_db"
    stats1 = run_ingest(chroma_dir=chroma_dir)
    assert stats1["total_documents"] > 0
    assert stats1["by_jurisdiction"]["BD"] > 0
    assert stats1["by_jurisdiction"]["US"] > 0
    assert stats1["by_jurisdiction"]["EU"] > 0
    assert stats1["low_confidence_flagged"] > 0

    cit_index_path = chroma_dir / "citation_index.json"
    bm25_path = chroma_dir / "bm25.pkl"
    assert cit_index_path.exists()
    assert bm25_path.exists()

    cit_index = json.loads(cit_index_path.read_text(encoding="utf-8"))
    assert len(cit_index) > 0
    assert normalize_citation("Competition Act 2012 (BD), s.15(1)") in cit_index
    assert normalize_citation("15 U.S.C. § 1") in cit_index
    assert normalize_citation("TFEU Art. 101(1)") in cit_index

    # Second run must be idempotent (identical document count and citation index size)
    stats2 = run_ingest(chroma_dir=chroma_dir)
    assert stats2["total_documents"] == stats1["total_documents"]
    assert stats2["by_jurisdiction"] == stats1["by_jurisdiction"]
    assert stats2["citation_index_keys"] == stats1["citation_index_keys"]
