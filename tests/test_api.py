"""End-to-end API contract tests using FastAPI TestClient (Phase 7)."""

import os
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from app.config import DEFAULT_CHROMA_DIR
from app.retrieve import get_cached_resources
from app.server import app
from app.text_utils import normalize_citation


API_KEY = "dev-rag-key"
HEADERS = {"X-API-Key": API_KEY}
SAMPLE_CONTRACT_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_contract.txt"


@pytest.fixture
def client():
    return TestClient(app)


def test_auth_required_on_protected_endpoints(client: TestClient):
    # Missing X-API-Key -> 401 {"error": "unauthorized", ...}
    r1 = client.post("/rag/search", json={"query": "resale price maintenance"})
    assert r1.status_code == 401
    assert r1.json()["error"] == "unauthorized"

    # Wrong X-API-Key -> 401 {"error": "unauthorized", ...}
    r2 = client.post(
        "/rag/search",
        headers={"X-API-Key": "wrong-key"},
        json={"query": "resale price maintenance"},
    )
    assert r2.status_code == 401
    assert r2.json()["error"] == "unauthorized"

    r3 = client.get("/citation/15%20U.S.C.%20%C2%A7%201")
    assert r3.status_code == 401
    assert r3.json()["error"] == "unauthorized"


def test_health_endpoint_no_auth_and_contract_shape(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"status", "version", "mock", "embeddings", "corpus"}
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert isinstance(data["mock"], bool)
    assert data["embeddings"] in ("openai", "local", "hash")
    corpus = data["corpus"]
    assert set(corpus.keys()) == {"documents", "by_jurisdiction", "full_text_documents"}
    assert corpus["documents"] > 0
    assert set(corpus["by_jurisdiction"].keys()) == {"BD", "US", "EU"}
    assert corpus["by_jurisdiction"]["BD"] > 0
    assert corpus["by_jurisdiction"]["US"] > 0
    assert corpus["by_jurisdiction"]["EU"] > 0
    assert isinstance(corpus["full_text_documents"], int)


def test_rag_search_endpoint_contract(client: TestClient):
    r = client.post(
        "/rag/search",
        headers=HEADERS,
        json={
            "query": "resale price maintenance and exclusive distribution",
            "jurisdictions": ["BD", "US", "EU"],
            "risk_tags": ["rpm", "exclusivity"],
            "k": 6,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"context"}
    ctx = data["context"]
    assert 1 <= len(ctx) <= 6

    expected_item_keys = {
        "id",
        "citation_string",
        "jurisdiction",
        "title",
        "text",
        "source_url",
        "source_type",
        "verified",
        "confidence",
    }
    returned_jurisdictions = set()
    for item in ctx:
        assert set(item.keys()) == expected_item_keys
        assert item["jurisdiction"] in ("BD", "US", "EU")
        assert item["source_type"] in ("full_text", "provision_summary", "case_reference", "law_summary")
        assert item["confidence"] in ("high", "medium", "low")
        assert isinstance(item["verified"], bool)
        returned_jurisdictions.add(item["jurisdiction"])

    # Guarantee at least 1 item from each requested jurisdiction
    assert {"BD", "US", "EU"}.issubset(returned_jurisdictions)


def test_analyze_sample_contract_end_to_end(client: TestClient):
    contract_text = SAMPLE_CONTRACT_PATH.read_text(encoding="utf-8")
    r = client.post(
        "/analyze",
        headers=HEADERS,
        json={
            "document_text": contract_text,
            "jurisdictions": ["BD", "US", "EU"],
            "doc_name": "sample_contract.txt",
        },
    )
    assert r.status_code == 200
    data = r.json()

    expected_top_keys = {
        "summary",
        "document_type",
        "governing_law",
        "risks",
        "verification",
        "memo_markdown",
        "disclaimer",
        "model_info",
        "timings_ms",
    }
    assert set(data.keys()) == expected_top_keys
    assert len(data["risks"]) >= 3

    _, _, citation_index, _ = get_cached_resources(chroma_dir=DEFAULT_CHROMA_DIR)

    expected_risk_keys = {
        "title",
        "clause_refs",
        "why_non_obvious",
        "severity",
        "jurisdiction_notes",
        "citations",
        "mitigation_clause_text",
    }
    for risk in data["risks"]:
        assert set(risk.keys()) == expected_risk_keys
        assert risk["severity"] in ("high", "medium", "low")
        assert set(risk["jurisdiction_notes"].keys()) == {"BD", "US", "EU"}
        assert len(risk["clause_refs"]) >= 1
        for cit in risk["citations"]:
            assert set(cit.keys()) == {"citation_string", "quote", "status", "reason"}
            assert cit["status"] in ("verified", "found_in_corpus", "unverified")
            if cit["status"] == "verified":
                assert normalize_citation(cit["citation_string"]) in citation_index

    ver = data["verification"]
    assert set(ver.keys()) == {"total", "verified", "found_in_corpus", "unverified"}
    assert ver["total"] == ver["verified"] + ver["found_in_corpus"] + ver["unverified"]

    timings = data["timings_ms"]
    assert set(timings.keys()) == {"rewrite", "retrieval", "analysis", "verification", "memo", "total"}
    assert all(isinstance(v, int) and v >= 0 for v in timings.values())

    model_info = data["model_info"]
    assert set(model_info.keys()) == {"main", "small", "mock"}

    memo = data["memo_markdown"]
    assert "Executive summary" in memo
    assert "Risk table" in memo
    assert "Cross-jurisdiction comparison table" in memo
    assert "Recommended mitigation clauses" in memo
    assert "Limitations" in memo
    assert memo.strip().endswith(data["disclaimer"])


def test_chat_endpoint_and_validation(client: TestClient):
    r = client.post(
        "/chat",
        headers=HEADERS,
        json={
            "messages": [
                {"role": "user", "content": "Is a 6-year post-termination non-compete valid in Bangladesh and the EU?"}
            ],
            "jurisdictions": ["BD", "EU"],
            "context_note": "Distribution agreement for industrial sensors.",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"answer_markdown", "citations", "disclaimer", "mock", "thinking"}
    assert data["thinking"] is not None
    assert data["disclaimer"] in data["answer_markdown"]
    for c in data["citations"]:
        assert set(c.keys()) == {"citation_string", "status", "reason"}
        assert c["status"] in ("verified", "found_in_corpus", "unverified")

    # Last message must be role 'user' -> 422
    r_bad = client.post(
        "/chat",
        headers=HEADERS,
        json={
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        },
    )
    assert r_bad.status_code == 422
    assert "error" in r_bad.json()


def test_citation_lookup_endpoint(client: TestClient):
    # Known provision in corpus
    encoded_known = quote("Competition Act 2012 (BD), s.15(1)", safe="")
    r1 = client.get(f"/citation/{encoded_known}", headers=HEADERS)
    assert r1.status_code == 200
    d1 = r1.json()
    assert set(d1.keys()) == {"citation_string", "status", "reason", "source_url"}
    assert d1["status"] in ("found_in_corpus", "verified")

    # Known citation with slash (VBER 2022/720, Art. 4)
    encoded_slash = quote("VBER 2022/720, Art. 4", safe="")
    r_slash = client.get(f"/citation/{encoded_slash}", headers=HEADERS)
    assert r_slash.status_code == 200
    assert r_slash.json()["status"] in ("found_in_corpus", "verified")

    # Unknown citation
    encoded_unknown = quote("Imaginary Space Treaty 2099, Art. 42", safe="")
    r2 = client.get(f"/citation/{encoded_unknown}", headers=HEADERS)
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["status"] == "unverified"
    assert d2["source_url"] is None


def test_oversized_document_returns_413(client: TestClient):
    oversized_text = "A" * 150005
    r = client.post(
        "/analyze",
        headers=HEADERS,
        json={"document_text": oversized_text, "doc_name": "huge.txt"},
    )
    assert r.status_code == 413
    data = r.json()
    assert "error" in data
    assert "message" in data


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Real LLM test requires ANTHROPIC_API_KEY")
def test_real_llm_integration_when_key_present(client: TestClient):
    r = client.post(
        "/rag/search",
        headers=HEADERS,
        json={"query": "Sherman Act section 1 restraint of trade", "k": 3},
    )
    assert r.status_code == 200
