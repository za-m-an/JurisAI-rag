"""Unit tests for citation verification semantics (Phase 4)."""

import json
import shutil
from pathlib import Path

from app.config import DEFAULT_DATA_DIR
from app.ingest import run_ingest
from app.retrieve import reset_retrieval_cache
from app.verify import verify_all, verify_citation


def test_verify_summaries_and_unknown(tmp_path: Path):
    chroma_dir = tmp_path / "chroma_db"
    reset_retrieval_cache()
    run_ingest(data_dir=DEFAULT_DATA_DIR, chroma_dir=chroma_dir)

    # Summary-only entry in CSVs -> found_in_corpus
    res_summary = verify_citation("Contract Act 1872 (BD), s.10", chroma_dir=chroma_dir)
    assert res_summary["status"] == "found_in_corpus"
    assert "summary" in res_summary["reason"].lower()

    # Case reference in CSVs -> found_in_corpus
    res_case = verify_citation("Leegin Creative Leather Products v. PSKS, 551 U.S. 877", chroma_dir=chroma_dir)
    assert res_case["status"] == "found_in_corpus"

    # Unknown citation -> unverified
    res_unknown = verify_citation("Bangladesh Digital Monopoly Act 2031, s.99", chroma_dir=chroma_dir)
    assert res_unknown["status"] == "unverified"
    assert res_unknown["source_url"] is None


def test_verify_full_text_and_quote_downgrade(tmp_path: Path):
    data_dir = tmp_path / "data"
    chroma_dir = tmp_path / "chroma_db"
    data_dir.mkdir(parents=True, exist_ok=True)
    for csv_file in DEFAULT_DATA_DIR.glob("*.csv"):
        shutil.copy2(csv_file, data_dir / csv_file.name)

    # Add a full_text chunk in corpus_chunks.jsonl matching EU-TFEU-101 Art.101(1)
    chunk_row = {
        "law_id": "EU-TFEU-101",
        "jurisdiction": "EU",
        "title": "Treaty on the Functioning of the EU, Article 101",
        "chunk_id": "EU-TFEU-101#0001",
        "provision_label": "Article 101(1)",
        "text": (
            "The following shall be prohibited as incompatible with the internal market: "
            "all agreements between undertakings, decisions by associations of undertakings "
            "and concerted practices which may affect trade between Member States."
        ),
        "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:12016E101",
        "retrieved": "2026-10-20",
        "status": "in_force",
    }
    (data_dir / "corpus_chunks.jsonl").write_text(json.dumps(chunk_row) + "\n", encoding="utf-8")

    reset_retrieval_cache()
    run_ingest(data_dir=data_dir, chroma_dir=chroma_dir)

    # 1. Full text with matching quote (case/whitespace-insensitive) -> verified
    res_verified = verify_citation(
        "TFEU Art. 101(1)",
        quote="all agreements between undertakings,   decisions by associations of undertakings",
        chroma_dir=chroma_dir,
    )
    assert res_verified["status"] == "verified"
    assert res_verified["reason"] == "Text matched in corpus"

    # 2. Full text with mismatched quote -> downgraded to found_in_corpus
    res_mismatch = verify_citation(
        "TFEU Art. 101(1)",
        quote="this fabricated quote does not appear anywhere in the treaty text",
        chroma_dir=chroma_dir,
    )
    assert res_mismatch["status"] == "found_in_corpus"

    # 3. verify_all overrides any fake model-supplied status
    batch = verify_all(
        [
            {"citation_string": "TFEU Art. 101(1)", "quote": "prohibited as incompatible with the internal market", "status": "unverified"},
            {"citation_string": "Contract Act 1872 (BD), s.27", "quote": "void", "status": "verified"},
            {"citation_string": "Nonexistent Statute 1999, s.4", "quote": None, "status": "verified"},
        ],
        chroma_dir=chroma_dir,
    )
    assert batch[0]["status"] == "verified"
    assert batch[1]["status"] == "found_in_corpus"
    assert batch[2]["status"] == "unverified"
    reset_retrieval_cache()
