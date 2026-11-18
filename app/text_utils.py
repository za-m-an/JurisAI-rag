"""Text normalisation, citation normalisation, and clause splitting utilities."""

import re
from typing import List, Tuple


def normalize_citation(citation: str) -> str:
    """
    Normalise a legal citation string for exact index lookup:
    1. Lowercase
    2. Replace '§' with 's.' (and '§§' with 'ss.')
    3. Unify 'section'/'sec.'/'s.' to 's.' and 'article'/'art.' to 'art.'
    4. Remove all characters except letters, digits, parentheses, and whitespace
    5. Collapse whitespace
    """
    if not citation:
        return ""
    s = citation.lower()

    # Replace section symbols
    s = s.replace("§§", " ss. ")
    s = s.replace("§", " s. ")

    # Unify plural sections / articles before singular
    s = re.sub(r"\b(?:sections|secs\.?|ss\.)\s*(?=[0-9(])", "ss. ", s)
    s = re.sub(r"\b(?:section|sec\.?|s\.)\s*(?=[0-9(])", "s. ", s)
    s = re.sub(r"\b(?:articles|arts\.?)\s*(?=[0-9(])", "arts. ", s)
    s = re.sub(r"\b(?:article|art\.?)\s*(?=[0-9(])", "art. ", s)

    # Also unify standalone word tokens 'section'/'sec.' and 'article'/'art.'
    s = re.sub(r"\b(?:section|sec)\b\.?", "s.", s)
    s = re.sub(r"\b(?:article|art)\b\.?", "art.", s)

    # Remove all characters except letters, digits, parentheses, and whitespace
    s = re.sub(r"[^a-z0-9()\s]", "", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_whitespace(text: str) -> str:
    """Normalise line endings and collapse excessive blank lines and horizontal spaces."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_quote_match(text: str) -> str:
    """Case- and whitespace-insensitive normalisation for verbatim quote checking."""
    if not text:
        return ""
    s = text.lower()
    # Normalise smart quotes and dashes to ASCII equivalents
    s = (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("‘", "'")
        .replace("—", "-")
        .replace("–", "-")
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


_CLAUSE_HEADING_RE = re.compile(
    r"(?m)^[ \t]*("
    r"(?:Article|Section|Clause)\s+\d+(?:\.\d+)*[A-Za-z]?"
    r"|\d+(?:\.\d+)+\.?"
    r"|\d+\."
    r")(?=[ \t]+\S|\s*$)"
)


def split_into_clauses(text: str) -> List[Tuple[str, str]]:
    """
    Split a contract/document into (clause_ref, clause_text) pairs.
    Recognises numbered headings ('1.', '1.1', 'Article 3', 'Section 4', 'Clause 5').
    Falls back to blank-line paragraphs numbered P1..Pn if no structure is found.
    """
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    matches = list(_CLAUSE_HEADING_RE.finditer(cleaned))
    if matches:
        clauses: List[Tuple[str, str]] = []
        for idx, m in enumerate(matches):
            raw_ref = m.group(1).strip()
            ref = raw_ref.rstrip(".") if re.match(r"^\d+\.$", raw_ref) is None else raw_ref
            start = m.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
            body = cleaned[start:end].strip()
            if body:
                clauses.append((ref, body))
        if clauses:
            return clauses

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    return [(f"P{i + 1}", p) for i, p in enumerate(paragraphs)]


def tokenize_bm25(text: str) -> List[str]:
    """Simple alphanumeric legal tokenizer for BM25 and hashing embeddings."""
    if not text:
        return []
    norm = normalize_citation(text)
    return re.findall(r"[a-z0-9]+", norm)
