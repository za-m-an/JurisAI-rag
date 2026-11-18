"""Corpus ingestion script and indexer (`python -m app.ingest`)."""

import csv
import hashlib
import json
import pickle
import re
import shutil
import sys
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
from .text_utils import normalize_citation, tokenize_bm25


COLLECTION_NAME = "juris_legal_corpus"

FIXTURE_PROVISIONS = [
    ("BD-COMP-2012", "s.15(1)", "Competition Act 2012 (BD), s.15(1)", "anti-competitive agreements", "General prohibition on agreements or collusion about production, supply, distribution, storage or acquisition of goods/services that harm competition.", "cartel", "high", "False"),
    ("BD-COMP-2012", "s.15(3)", "Competition Act 2012 (BD), s.15(3)", "vertical restraints", "Addresses tie-in, exclusive supply, exclusive distribution, refusal to deal and resale price maintenance arrangements.", "exclusivity|tying|rpm", "high", "False"),
    ("BD-COMP-2012", "s.16(1)-(2)", "Competition Act 2012 (BD), s.16", "abuse of dominant position", "Prohibits abuse of dominance; s.16(2) lists presumed abuses.", "abuse_dominance", "high", "False"),
    ("BD-CONTRACT-1872", "s.27", "Contract Act 1872 (BD), s.27", "restraint of trade", "Agreements restraining a lawful profession, trade or business are void (limited exception for sale of goodwill).", "restraint_of_trade", "high", "False"),
    ("BD-CONTRACT-1872", "s.74", "Contract Act 1872 (BD), s.74", "liquidated damages and penalties", "Where a sum is named for breach, courts award reasonable compensation not exceeding the named amount.", "penalty_clause", "high", "False"),
    ("US-SHERMAN-1890", "s.1 / 15 USC 1", "15 U.S.C. § 1", "restraint of trade", "Prohibits contracts, combinations and conspiracies in restraint of trade.", "cartel|rpm|exclusivity", "high", "False"),
    ("US-SHERMAN-1890", "s.2 / 15 USC 2", "15 U.S.C. § 2", "monopolization", "Prohibits monopolization, attempts and conspiracies to monopolize.", "abuse_dominance", "high", "False"),
    ("US-CLAYTON-1914", "s.3 / 15 USC 14", "15 U.S.C. § 14", "tying and exclusive dealing (goods)", "Unlawful to sell/lease goods on condition of not dealing with competitors where effect may substantially lessen competition.", "tying|exclusivity", "high", "False"),
    ("US-UCC-ART2", "UCC 2-718", "UCC § 2-718", "liquidated damages", "Liquidated damages valid only if reasonable in light of anticipated or actual harm.", "penalty_clause", "high", "False"),
    ("EU-TFEU-101", "Art.101(1)", "TFEU Art. 101(1)", "prohibited agreements", "Prohibits agreements, decisions and concerted practices that restrict competition and affect trade between Member States.", "cartel|exclusivity|rpm", "high", "False"),
    ("EU-TFEU-102", "Art.102", "TFEU Art. 102", "abuse of dominance", "Prohibits abuse of a dominant position (unfair prices, limiting production, discrimination, tying).", "abuse_dominance|tying", "high", "False"),
    ("EU-VBER-2022", "Art.4", "VBER 2022/720, Art. 4", "hardcore restrictions", "Resale price maintenance and certain territorial/customer restrictions remove block-exemption benefit.", "rpm", "high", "False"),
    ("EU-VBER-2022", "Art.5", "VBER 2022/720, Art. 5", "excluded restrictions", "Non-compete obligations exceeding five years or post-term non-competes are not exempted.", "exclusivity|non_compete", "high", "False"),
]


def _make_stable_id(source_type: str, key: str) -> str:
    return hashlib.sha256(f"{source_type}:{key}".encode("utf-8")).hexdigest()[:16]


def _parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


def _format_risk_tags(raw: str) -> str:
    if not raw or not raw.strip():
        return ""
    tags = [t.strip() for t in raw.split("|") if t.strip()]
    return f"|{'|'.join(tags)}|" if tags else ""


def _find_and_copy_dataset(data_dir: Path) -> bool:
    """
    Ensure laws_registry.csv, key_provisions.csv, landmark_cases.csv exist in data_dir.
    If absent, search working directory and parents for 'legal_dataset' and copy.
    Returns True if real dataset found, False if fallback fixture dataset had to be generated.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "laws_registry.csv").exists() and (data_dir / "key_provisions.csv").exists():
        return True

    search_roots = [Path.cwd(), data_dir.resolve().parent, data_dir.resolve().parent.parent]
    for base in list(search_roots):
        for parent in [base] + list(base.parents):
            candidate = parent / "legal_dataset"
            if candidate.is_dir() and (candidate / "laws_registry.csv").exists():
                for fname in [
                    "laws_registry.csv",
                    "key_provisions.csv",
                    "landmark_cases.csv",
                    "corpus_chunks.jsonl",
                    "legal_corpus_seed.json",
                ]:
                    src = candidate / fname
                    if src.exists():
                        shutil.copy2(src, data_dir / fname)
                return True

    # Create clearly-labelled FIXTURE dataset if not found
    print("WARNING: 'legal_dataset' folder not found! Generating clearly-labelled FIXTURE dataset in rag/data/.")
    with open(data_dir / "laws_registry.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "jurisdiction", "title", "short_name", "year", "instrument_no", "doc_type",
            "status", "domain_tags", "mvp_priority", "key_risk_topics", "source_name",
            "source_url", "celex", "text_acquisition", "confidence", "verified", "notes",
        ])
        fixture_laws = [
            ("BD-COMP-2012", "BD", "The Competition Act, 2012", "Competition Act 2012", "2012", "Act No. 30 of 2012", "statute", "in_force", "competition", "1", "cartel|exclusivity|rpm|abuse_dominance", "bdlaws", "http://bdlaws.minlaw.gov.bd/", "", "manual", "high", "False", "FIXTURE"),
            ("BD-CONTRACT-1872", "BD", "The Contract Act, 1872", "Contract Act 1872", "1872", "Act No. IX of 1872", "statute", "in_force", "contract", "1", "restraint_of_trade|penalty_clause", "bdlaws", "http://bdlaws.minlaw.gov.bd/", "", "manual", "high", "False", "FIXTURE"),
            ("US-SHERMAN-1890", "US", "Sherman Antitrust Act, 15 U.S.C. §§ 1-7", "Sherman Act", "1890", "15 U.S.C. §§ 1-7", "statute", "in_force", "antitrust", "1", "cartel|rpm|exclusivity|abuse_dominance", "Cornell LII", "https://www.law.cornell.edu/uscode/text/15/chapter-1", "", "fetch_uscode", "high", "False", "FIXTURE"),
            ("US-CLAYTON-1914", "US", "Clayton Antitrust Act, 15 U.S.C. §§ 12-27", "Clayton Act", "1914", "15 U.S.C. §§ 12-27", "statute", "in_force", "antitrust", "1", "tying|exclusivity|merger", "Cornell LII", "https://www.law.cornell.edu/uscode/text/15/chapter-1", "", "fetch_uscode", "high", "False", "FIXTURE"),
            ("US-UCC-ART2", "US", "Uniform Commercial Code Article 2 (Sales)", "UCC Art. 2", "", "UCC Art. 2", "model_law", "in_force_state_variation", "contract", "1", "penalty_clause|liability_limitation", "Cornell LII", "https://www.law.cornell.edu/ucc/2", "", "fetch_html", "high", "False", "FIXTURE"),
            ("EU-TFEU-101", "EU", "Treaty on the Functioning of the EU, Article 101", "TFEU Art. 101", "2016", "CELEX 12016E101", "treaty", "in_force", "competition", "1", "cartel|exclusivity|rpm", "EUR-Lex", "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:12016E101", "12016E101", "fetch_eurlex", "high", "False", "FIXTURE"),
            ("EU-TFEU-102", "EU", "Treaty on the Functioning of the EU, Article 102", "TFEU Art. 102", "2016", "CELEX 12016E102", "treaty", "in_force", "competition", "1", "abuse_dominance|tying", "EUR-Lex", "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:12016E102", "12016E102", "fetch_eurlex", "high", "False", "FIXTURE"),
            ("EU-VBER-2022", "EU", "Commission Regulation (EU) 2022/720 (Vertical Block Exemption)", "VBER 2022/720", "2022", "CELEX 32022R0720", "regulation", "in_force", "competition", "1", "rpm|exclusivity|non_compete", "EUR-Lex", "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R0720", "32022R0720", "fetch_eurlex", "high", "False", "FIXTURE"),
        ]
        writer.writerows(fixture_laws)

    with open(data_dir / "key_provisions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["law_id", "provision_ref", "citation_string", "topic", "summary_own_words", "risk_tag", "confidence", "verified"])
        writer.writerows(FIXTURE_PROVISIONS)

    with open(data_dir / "landmark_cases.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "jurisdiction", "court", "case_name", "citation", "year", "topic", "risk_tag", "lookup_source", "confidence", "verified"])
        writer.writerow(["US-CASE-03", "US", "SCOTUS/Fed.", "Leegin Creative Leather Products v. PSKS", "551 U.S. 877", "2007", "RPM judged under rule of reason", "rpm", "CourtListener", "medium", "False"])
    return False


def _provision_ref_matches_label(provision_ref: str, provision_label: str) -> bool:
    """
    Check case-insensitive containment of the article/section number from provision_ref
    inside provision_label (e.g., provision_ref='s.15(1)' or 'Art.101(1)' vs provision_label='Section 15' or 'Article 101(1)').
    """
    if not provision_ref or not provision_label:
        return False
    ref_low = provision_ref.strip().lower()
    lbl_low = provision_label.strip().lower()
    if ref_low in lbl_low:
        return True
    norm_ref = normalize_citation(provision_ref)
    norm_lbl = normalize_citation(provision_label)
    if norm_ref and norm_ref in norm_lbl:
        return True
    # Extract base section/article number (e.g. '15' from 's.15(1)' or '101' from 'Art.101(1)')
    m_ref = re.search(r"\b(?:s|ss|art|arts|ucc)\s+(\d+[a-z]?(?:\(\d+\))?)", norm_ref)
    if m_ref:
        sec_token = m_ref.group(1)
        if re.search(rf"\b{re.escape(sec_token)}\b", norm_lbl):
            return True
        base_num = re.match(r"^(\d+[a-z]?)", sec_token)
        if base_num and re.search(rf"\b{re.escape(base_num.group(1))}\b", norm_lbl):
            return True
    return False


def run_ingest(
    data_dir: Optional[Path] = None,
    chroma_dir: Optional[Path] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """
    Ingest CSVs and optional corpus_chunks.jsonl into ChromaDB, BM25, and citation_index.json.
    Returns summary statistics dictionary.
    """
    cfg = settings or load_settings()
    d_dir = Path(data_dir) if data_dir else cfg.data_dir
    c_dir = Path(chroma_dir) if chroma_dir else cfg.chroma_dir

    _find_and_copy_dataset(d_dir)
    c_dir.mkdir(parents=True, exist_ok=True)

    laws_by_id: Dict[str, Dict[str, str]] = {}
    documents: List[Dict[str, Any]] = []
    citation_index: Dict[str, List[str]] = {}

    def register_alias(cit_str: str, doc_id: str) -> None:
        norm = normalize_citation(cit_str)
        if not norm:
            return
        bucket = citation_index.setdefault(norm, [])
        if doc_id not in bucket:
            bucket.append(doc_id)

    # 1. Load laws_registry.csv -> law_summary docs
    laws_csv = d_dir / "laws_registry.csv"
    if laws_csv.exists():
        with open(laws_csv, mode="r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                law_id = (row.get("id") or "").strip()
                if not law_id:
                    continue
                laws_by_id[law_id] = row
                jurisdiction = (row.get("jurisdiction") or "BD").strip()
                title = (row.get("title") or "").strip()
                short_name = (row.get("short_name") or "").strip()
                instrument_no = (row.get("instrument_no") or "").strip()
                status = (row.get("status") or "in_force").strip()
                key_risk_topics = (row.get("key_risk_topics") or "").strip()
                notes = (row.get("notes") or "").strip()
                source_url = (row.get("source_url") or "").strip()
                confidence = (row.get("confidence") or "medium").strip().lower()
                if confidence not in ("high", "medium", "low"):
                    confidence = "medium"
                verified = _parse_bool(row.get("verified", False))

                doc_id = _make_stable_id("law_summary", law_id)
                flag_prefix = "[LOW CONFIDENCE - VERIFY PRIMARY TEXT] " if confidence == "low" else ""
                text_parts = [f"{flag_prefix}{title} ({jurisdiction})"]
                if short_name:
                    text_parts.append(f"Short name: {short_name}.")
                if status:
                    text_parts.append(f"Status: {status}.")
                if key_risk_topics:
                    text_parts.append(f"Key risk topics: {key_risk_topics.replace('|', ', ')}.")
                if notes:
                    text_parts.append(f"Notes: {notes}")
                text = " ".join(text_parts).strip()

                doc = {
                    "id": doc_id,
                    "text": text,
                    "metadata": {
                        "id": doc_id,
                        "jurisdiction": jurisdiction,
                        "law_id": law_id,
                        "citation_string": title,
                        "title": title,
                        "source_type": "law_summary",
                        "risk_tags": _format_risk_tags(key_risk_topics),
                        "confidence": confidence,
                        "verified": verified,
                        "source_url": source_url,
                        "status": status,
                        "flagged_low_confidence": confidence == "low",
                    },
                }
                documents.append(doc)
                register_alias(title, doc_id)
                if short_name:
                    register_alias(short_name, doc_id)
                    register_alias(f"{short_name} ({jurisdiction})", doc_id)
                if instrument_no:
                    register_alias(instrument_no, doc_id)

    # 2. Load key_provisions.csv -> provision_summary docs
    provisions_rows: List[Dict[str, str]] = []
    provisions_csv = d_dir / "key_provisions.csv"
    if provisions_csv.exists():
        with open(provisions_csv, mode="r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                law_id = (row.get("law_id") or "").strip()
                prov_ref = (row.get("provision_ref") or "").strip()
                cit_str = (row.get("citation_string") or "").strip()
                if not law_id or not cit_str:
                    continue
                provisions_rows.append(row)
                parent_law = laws_by_id.get(law_id, {})
                jurisdiction = (parent_law.get("jurisdiction") or law_id.split("-")[0] or "BD").strip()
                title = (parent_law.get("title") or cit_str).strip()
                short_name = (parent_law.get("short_name") or "").strip()
                source_url = (parent_law.get("source_url") or "").strip()
                status = (parent_law.get("status") or "in_force").strip()

                topic = (row.get("topic") or "").strip()
                summary = (row.get("summary_own_words") or "").strip()
                risk_tag = (row.get("risk_tag") or "").strip()
                confidence = (row.get("confidence") or "medium").strip().lower()
                if confidence not in ("high", "medium", "low"):
                    confidence = "medium"
                verified = _parse_bool(row.get("verified", False))

                doc_id = _make_stable_id("provision_summary", f"{law_id}:{prov_ref}:{cit_str}")
                base_text = f"{cit_str}: {topic}. {summary}"
                if confidence == "low":
                    base_text = f"[LOW CONFIDENCE - VERIFY PRIMARY TEXT] {base_text}"

                doc = {
                    "id": doc_id,
                    "text": base_text,
                    "metadata": {
                        "id": doc_id,
                        "jurisdiction": jurisdiction,
                        "law_id": law_id,
                        "citation_string": cit_str,
                        "title": title,
                        "source_type": "provision_summary",
                        "risk_tags": _format_risk_tags(risk_tag),
                        "confidence": confidence,
                        "verified": verified,
                        "source_url": source_url,
                        "status": status,
                        "flagged_low_confidence": confidence == "low",
                    },
                }
                documents.append(doc)
                register_alias(cit_str, doc_id)
                if short_name:
                    register_alias(f"{short_name}, {prov_ref}", doc_id)
                    register_alias(f"{short_name} {prov_ref}", doc_id)
                if title:
                    register_alias(f"{title}, {prov_ref}", doc_id)

    # 3. Load landmark_cases.csv -> case_reference docs
    cases_csv = d_dir / "landmark_cases.csv"
    if cases_csv.exists():
        with open(cases_csv, mode="r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                case_id = (row.get("id") or "").strip()
                case_name = (row.get("case_name") or "").strip()
                citation = (row.get("citation") or "").strip()
                if not case_id or not case_name:
                    continue
                jurisdiction = (row.get("jurisdiction") or "US").strip()
                year = (row.get("year") or "").strip()
                topic = (row.get("topic") or "").strip()
                risk_tag = (row.get("risk_tag") or "").strip()
                lookup_source = (row.get("lookup_source") or "").strip()
                confidence = (row.get("confidence") or "medium").strip().lower()
                if confidence not in ("high", "medium", "low"):
                    confidence = "medium"
                verified = _parse_bool(row.get("verified", False))

                cit_str = f"{case_name}, {citation}" if citation else case_name
                text = f"{case_name} ({year}) {citation}. Topic: {topic}."
                if confidence == "low":
                    text = f"[LOW CONFIDENCE - VERIFY PRIMARY TEXT] {text}"

                source_url = lookup_source if lookup_source.startswith("http") else ""
                doc_id = _make_stable_id("case_reference", case_id)
                doc = {
                    "id": doc_id,
                    "text": text,
                    "metadata": {
                        "id": doc_id,
                        "jurisdiction": jurisdiction,
                        "law_id": case_id,
                        "citation_string": cit_str,
                        "title": case_name,
                        "source_type": "case_reference",
                        "risk_tags": _format_risk_tags(risk_tag),
                        "confidence": confidence,
                        "verified": verified,
                        "source_url": source_url,
                        "status": "in_force",
                        "flagged_low_confidence": confidence == "low",
                    },
                }
                documents.append(doc)
                register_alias(cit_str, doc_id)
                register_alias(case_name, doc_id)
                if citation:
                    register_alias(citation, doc_id)

    # 4. Load optional corpus_chunks.jsonl -> full_text docs
    chunks_jsonl = d_dir / "corpus_chunks.jsonl"
    if chunks_jsonl.exists():
        with open(chunks_jsonl, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                law_id = (row.get("law_id") or "").strip()
                chunk_id = (row.get("chunk_id") or "").strip()
                jurisdiction = (row.get("jurisdiction") or "BD").strip()
                title = (row.get("title") or "").strip()
                prov_label = (row.get("provision_label") or "").strip()
                text = (row.get("text") or "").strip()
                source_url = (row.get("source_url") or "").strip()
                status = (row.get("status") or "in_force").strip()

                if not text:
                    continue
                cit_str = f"{title}, {prov_label}" if prov_label else title
                parent_law = laws_by_id.get(law_id, {})
                risk_topics = parent_law.get("key_risk_topics", "")
                confidence = (parent_law.get("confidence") or "high").strip().lower()
                if confidence not in ("high", "medium", "low"):
                    confidence = "high"

                doc_id = _make_stable_id("full_text", chunk_id or f"{law_id}:{prov_label}:{text[:40]}")
                doc = {
                    "id": doc_id,
                    "text": text,
                    "metadata": {
                        "id": doc_id,
                        "jurisdiction": jurisdiction,
                        "law_id": law_id,
                        "citation_string": cit_str,
                        "title": title,
                        "source_type": "full_text",
                        "risk_tags": _format_risk_tags(risk_topics),
                        "confidence": confidence,
                        "verified": True,
                        "source_url": source_url,
                        "status": status,
                        "flagged_low_confidence": confidence == "low",
                    },
                }
                documents.append(doc)
                register_alias(cit_str, doc_id)

                # Register alias for matching key_provisions.citation_string
                for prov in provisions_rows:
                    if prov.get("law_id", "").strip() == law_id:
                        p_ref = prov.get("provision_ref", "").strip()
                        p_cit = prov.get("citation_string", "").strip()
                        if p_cit and _provision_ref_matches_label(p_ref, prov_label):
                            register_alias(p_cit, doc_id)

    if not documents:
        print("ERROR: Zero documents loaded during ingestion.", file=sys.stderr)
        sys.exit(1)

    # Deduplicate documents by id (last wins, preserving idempotency)
    deduped_map: Dict[str, Dict[str, Any]] = {}
    for d in documents:
        deduped_map[d["id"]] = d
    unique_docs = list(deduped_map.values())

    # Compute embeddings explicitly
    embedder = Embedder(settings=cfg)
    ids = [d["id"] for d in unique_docs]
    texts = [d["text"] for d in unique_docs]
    metadatas = [d["metadata"] for d in unique_docs]
    embeddings = embedder.embed_documents(texts)

    # Persist in ChromaDB; rebuild collection if backend changed
    client = chromadb.PersistentClient(path=str(c_dir))
    try:
        existing_col = client.get_collection(name=COLLECTION_NAME)
        existing_backend = (existing_col.metadata or {}).get("embeddings_backend")
        if existing_backend and existing_backend != embedder.backend:
            client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"embeddings_backend": embedder.backend, "hnsw:space": "cosine"},
    )
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    # Build and save BM25 index
    tokenized_corpus = [tokenize_bm25(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    docs_by_id = {d["id"]: d for d in unique_docs}
    bm25_payload = {
        "ids": ids,
        "tokenized_corpus": tokenized_corpus,
        "bm25": bm25,
        "docs_by_id": docs_by_id,
        "embeddings_backend": embedder.backend,
    }
    with open(c_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25_payload, f)

    # Write citation_index.json
    with open(c_dir / "citation_index.json", "w", encoding="utf-8") as f:
        json.dump(citation_index, f, indent=2, ensure_ascii=False)

    # Compute summary counts
    by_jurisdiction: Dict[str, int] = {"BD": 0, "US": 0, "EU": 0}
    by_source_type: Dict[str, int] = {
        "full_text": 0,
        "provision_summary": 0,
        "case_reference": 0,
        "law_summary": 0,
    }
    low_conf_count = 0
    for d in unique_docs:
        m = d["metadata"]
        j = m["jurisdiction"]
        st = m["source_type"]
        by_jurisdiction[j] = by_jurisdiction.get(j, 0) + 1
        by_source_type[st] = by_source_type.get(st, 0) + 1
        if m.get("confidence") == "low":
            low_conf_count += 1

    stats = {
        "total_documents": len(unique_docs),
        "by_jurisdiction": by_jurisdiction,
        "by_source_type": by_source_type,
        "low_confidence_flagged": low_conf_count,
        "citation_index_keys": len(citation_index),
        "embeddings_backend": embedder.backend,
    }
    print(f"Ingestion complete: {stats['total_documents']} documents indexed.")
    print(f"  By jurisdiction: {by_jurisdiction}")
    print(f"  By source_type:  {by_source_type}")
    print(f"  Low-confidence flagged: {low_conf_count}")
    print(f"  Citation index entries: {len(citation_index)}")
    return stats


if __name__ == "__main__":
    run_ingest()
