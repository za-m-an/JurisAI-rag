"""Hallucination prevention and unverified citation stripping tests (Phase 7)."""

from typing import Any, Dict

from app.analyze import analyze_document
from app.llm import llm
from app.verify import verify_citation


def test_hallucinated_statute_is_unverified_and_excluded_from_memo(monkeypatch):
    fake_cit = "Bangladesh Digital Monopoly Act 2031"

    # 1. Direct verification check
    v_res = verify_citation(fake_cit)
    assert v_res["status"] == "unverified"
    assert v_res["source_url"] is None

    # 2. Inject fake citation into main_json output and verify analyze_document:
    #    - keeps it in risks[].citations with status='unverified' (override model's fake 'verified' status)
    #    - increments verification.unverified
    #    - NEVER includes "Bangladesh Digital Monopoly Act 2031" anywhere in memo_markdown
    orig_main_json = llm.main_json

    def _patched_main_json(system: str, user: str) -> Dict[str, Any]:
        res = orig_main_json(system, user)
        res["risks"][0]["citations"].append(
            {
                "citation_string": fake_cit,
                "quote": "Fabricated statutory quote that does not exist",
                "status": "verified",  # Intentionally lie about status to test verify_all override
                "reason": "Model hallucinated verified status",
            }
        )
        return res

    monkeypatch.setattr(llm, "main_json", _patched_main_json)

    sample_text = (
        "1. Exclusivity\nDistributor shall buy 100% exclusively from Supplier.\n\n"
        "2. Resale Price\nDistributor must sell at the Recommended Resale Price or pay a 35% penalty.\n\n"
        "3. Non-Compete\nDistributor shall not compete for 6 years after termination."
    )
    analysis = analyze_document(sample_text, jurisdictions=["BD", "US", "EU"], doc_name="hallucination_test.txt")

    # Verify it is kept in risks[].citations with status == 'unverified'
    first_risk_cits = analysis.risks[0].citations
    hallucinated_items = [c for c in first_risk_cits if c.citation_string == fake_cit]
    assert len(hallucinated_items) == 1
    assert hallucinated_items[0].status == "unverified"
    assert analysis.verification.unverified >= 1

    # Verify it NEVER appears in memo_markdown
    assert fake_cit not in analysis.memo_markdown
