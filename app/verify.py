"""Pure-Python deterministic citation verification against the indexed legal corpus."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .retrieve import get_cached_resources
from .text_utils import normalize_citation, normalize_for_quote_match


def verify_citation(
    citation_string: str,
    quote: Optional[str] = None,
    chroma_dir: Optional[Path] = None,
) -> Dict[str, Optional[str]]:
    """
    Verify a citation string and optional verbatim quote against the corpus index.
    Status semantics (Section 3):
    - 'verified': normalised citation exists in citation_index AND matching doc has source_type='full_text'
                  AND, if a quote was supplied, the quote (case/whitespace-insensitive) appears in that text.
    - 'found_in_corpus': citation exists in the index only as 'provision_summary', 'case_reference', or
                         'law_summary', or full_text exists but no quote supplied/matched.
    - 'unverified': not in the index.
    """
    norm_cit = normalize_citation(citation_string)
    if not norm_cit:
        return {
            "citation_string": citation_string,
            "status": "unverified",
            "reason": "Not found in corpus",
            "source_url": None,
        }

    _, bm25_data, citation_index, _ = get_cached_resources(chroma_dir=chroma_dir, auto_ingest=False)
    docs_by_id: Dict[str, Dict[str, Any]] = bm25_data.get("docs_by_id", {})

    doc_ids = citation_index.get(norm_cit, [])
    if not doc_ids:
        return {
            "citation_string": citation_string,
            "status": "unverified",
            "reason": "Not found in corpus",
            "source_url": None,
        }

    matched_docs = [docs_by_id[did] for did in doc_ids if did in docs_by_id]
    if not matched_docs:
        return {
            "citation_string": citation_string,
            "status": "unverified",
            "reason": "Not found in corpus",
            "source_url": None,
        }

    full_text_docs = [d for d in matched_docs if d.get("metadata", {}).get("source_type") == "full_text"]
    best_url = next(
        (d.get("metadata", {}).get("source_url") for d in matched_docs if d.get("metadata", {}).get("source_url")),
        None,
    )

    if full_text_docs:
        ft_url = next(
            (d.get("metadata", {}).get("source_url") for d in full_text_docs if d.get("metadata", {}).get("source_url")),
            best_url,
        )
        if quote is None:
            return {
                "citation_string": citation_string,
                "status": "verified",
                "reason": "Text matched in corpus",
                "source_url": ft_url,
            }
        norm_quote = normalize_for_quote_match(quote)
        if not norm_quote:
            return {
                "citation_string": citation_string,
                "status": "found_in_corpus",
                "reason": "Listed in corpus as full text, but no quote supplied to match; confirm against primary source",
                "source_url": ft_url,
            }
        for ft_doc in full_text_docs:
            norm_doc_text = normalize_for_quote_match(ft_doc.get("text", ""))
            if norm_quote in norm_doc_text:
                return {
                    "citation_string": citation_string,
                    "status": "verified",
                    "reason": "Text matched in corpus",
                    "source_url": ft_doc.get("metadata", {}).get("source_url") or ft_url,
                }
        return {
            "citation_string": citation_string,
            "status": "found_in_corpus",
            "reason": "Full text in corpus, but supplied quote did not match; confirm against primary source",
            "source_url": ft_url,
        }

    return {
        "citation_string": citation_string,
        "status": "found_in_corpus",
        "reason": "Listed in corpus as a summary only; confirm against primary source",
        "source_url": best_url,
    }


def verify_all(
    citations: List[Dict[str, Any]],
    chroma_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Verify a list of citation dicts, overriding any model-supplied status/reason."""
    verified_list: List[Dict[str, Any]] = []
    for item in citations:
        cit_str = str(item.get("citation_string", "")).strip()
        quote = item.get("quote")
        res = verify_citation(cit_str, quote=quote, chroma_dir=chroma_dir)
        out_item: Dict[str, Any] = {
            "citation_string": cit_str,
            "status": res["status"],
            "reason": res["reason"],
        }
        if "quote" in item or quote is not None:
            out_item["quote"] = quote
        if res.get("source_url") is not None:
            out_item["source_url"] = res["source_url"]
        verified_list.append(out_item)
    return verified_list
