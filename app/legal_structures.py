"""
Object-Oriented Legal Data Structures for Graph RAG.

Modeled as a hierarchical class structure (analogous to a Java class system):
- Law (Main Class representing an enactment / statutory code)
  - SectionCode (subclass for section numbering, clauses, penal code numbers)
  - ArticleNumber (subclass for article numbering in constitutions, treaties, EU regulations)
  - ArticleSummary (subclass for the whole article one-week summary, operative rules, and risk tags)
  - PrecedentCase (subclass for precedents or statutory cases for that exact law)
  - LawEdge (relational edge linking laws to other related companion laws when difficulties arise)
- LegalProvision (provision-level node holding section/article, text, precedents, and edges)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class PrecedentCase:
    """Precedent or statutory case interpreting an exact law/section."""
    case_name: str
    citation: str
    court: str
    year: int
    jurisdiction: str  # "BD" | "US" | "EU"
    holding_summary: str
    relevance_to_section: str
    precedent_type: str = "landmark"  # "landmark" | "statutory_interpretation" | "constitutional"
    source_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_name": self.case_name,
            "citation": self.citation,
            "court": self.court,
            "year": self.year,
            "jurisdiction": self.jurisdiction,
            "holding_summary": self.holding_summary,
            "relevance_to_section": self.relevance_to_section,
            "precedent_type": self.precedent_type,
            "source_url": self.source_url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrecedentCase":
        return cls(
            case_name=data.get("case_name", ""),
            citation=data.get("citation", ""),
            court=data.get("court", ""),
            year=int(data.get("year", 0)),
            jurisdiction=data.get("jurisdiction", "BD"),
            holding_summary=data.get("holding_summary", ""),
            relevance_to_section=data.get("relevance_to_section", ""),
            precedent_type=data.get("precedent_type", "landmark"),
            source_url=data.get("source_url"),
        )


@dataclass
class SectionCode:
    """Section code structure: section number, clause, subsection, and penal code number."""
    code: str  # e.g. "s.10", "s.27", "s.74", "s.420"
    number: str  # e.g. "10", "27", "74", "420"
    chapter: Optional[str] = None
    is_penal: bool = False
    penal_code_number: Optional[str] = None  # e.g. "Act No. XLV of 1860, s.420"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "number": self.number,
            "chapter": self.chapter,
            "is_penal": self.is_penal,
            "penal_code_number": self.penal_code_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SectionCode":
        return cls(
            code=data.get("code", ""),
            number=data.get("number", ""),
            chapter=data.get("chapter"),
            is_penal=data.get("is_penal", False),
            penal_code_number=data.get("penal_code_number"),
        )


@dataclass
class ArticleNumber:
    """Article number structure: for constitutional, EU, treaty, or regulatory articles."""
    number: str  # e.g. "Art. 6", "Art. 28", "Art. 101"
    paragraph: Optional[str] = None
    subparagraph: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "paragraph": self.paragraph,
            "subparagraph": self.subparagraph,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArticleNumber":
        return cls(
            number=data.get("number", ""),
            paragraph=data.get("paragraph"),
            subparagraph=data.get("subparagraph"),
        )


@dataclass
class ArticleSummary:
    """
    Comprehensive summary of the article or section.
    Contains the whole article one-week summary (digest for fast review),
    operative rules, statutory exceptions, and risk topics.
    """
    one_week_summary: str  # Fast executive digest / one-week executive summary
    operative_rule: str
    exceptions: List[str] = field(default_factory=list)
    risk_topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "one_week_summary": self.one_week_summary,
            "operative_rule": self.operative_rule,
            "exceptions": self.exceptions,
            "risk_topics": self.risk_topics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArticleSummary":
        return cls(
            one_week_summary=data.get("one_week_summary", ""),
            operative_rule=data.get("operative_rule", ""),
            exceptions=data.get("exceptions", []),
            risk_topics=data.get("risk_topics", []),
        )


@dataclass
class LawEdge:
    """
    Relational edge connecting this law/section to another related law/section.
    Used during graph traversal when difficulties arise (ambiguity, conflict of law,
    impossibility of performance, excessive penalty, non-compete voidness, data breach).
    """
    target_law_id: str
    target_citation: str
    relation_type: str  # e.g. "DIFFICULTY_ESCALATION", "STATUTORY_EXCEPTION", "PENAL_CONSEQUENCE", "PROCEDURAL_REMEDY", "CROSS_REFERENCE"
    reason: str  # Reason why laws are closely related
    difficulty_trigger: str  # Condition/context when difficulties arise
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_law_id": self.target_law_id,
            "target_citation": self.target_citation,
            "relation_type": self.relation_type,
            "reason": self.reason,
            "difficulty_trigger": self.difficulty_trigger,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LawEdge":
        return cls(
            target_law_id=data.get("target_law_id", ""),
            target_citation=data.get("target_citation", ""),
            relation_type=data.get("relation_type", "CROSS_REFERENCE"),
            reason=data.get("reason", ""),
            difficulty_trigger=data.get("difficulty_trigger", ""),
            weight=float(data.get("weight", 1.0)),
        )


@dataclass
class LegalProvision:
    """A specific provision (section or article) within a law."""
    provision_id: str  # e.g. "BD-CONTRACT-1872:s.27"
    law_id: str  # Parent law ID
    jurisdiction: str  # "BD" | "US" | "EU"
    citation_string: str  # e.g. "Contract Act 1872 (BD), s.27"
    title: str
    section_code: Optional[SectionCode] = None
    article_number: Optional[ArticleNumber] = None
    summary: ArticleSummary = field(default_factory=lambda: ArticleSummary(one_week_summary="", operative_rule=""))
    full_text: str = ""
    source_url: str = ""
    precedents: List[PrecedentCase] = field(default_factory=list)
    edges: List[LawEdge] = field(default_factory=list)
    confidence: str = "high"  # "high" | "medium" | "low"
    verified: bool = True

    def add_precedent(self, precedent: PrecedentCase) -> None:
        self.precedents.append(precedent)

    def add_edge(self, edge: LawEdge) -> None:
        self.edges.append(edge)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provision_id": self.provision_id,
            "law_id": self.law_id,
            "jurisdiction": self.jurisdiction,
            "citation_string": self.citation_string,
            "title": self.title,
            "section_code": self.section_code.to_dict() if self.section_code else None,
            "article_number": self.article_number.to_dict() if self.article_number else None,
            "summary": self.summary.to_dict() if self.summary else None,
            "full_text": self.full_text,
            "source_url": self.source_url,
            "precedents": [p.to_dict() for p in self.precedents],
            "edges": [e.to_dict() for e in self.edges],
            "confidence": self.confidence,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LegalProvision":
        sec_code = SectionCode.from_dict(data["section_code"]) if data.get("section_code") else None
        art_no = ArticleNumber.from_dict(data["article_number"]) if data.get("article_number") else None
        summary = ArticleSummary.from_dict(data["summary"]) if data.get("summary") else ArticleSummary(one_week_summary="", operative_rule="")
        precedents = [PrecedentCase.from_dict(p) for p in data.get("precedents", [])]
        edges = [LawEdge.from_dict(e) for e in data.get("edges", [])]
        return cls(
            provision_id=data.get("provision_id", ""),
            law_id=data.get("law_id", ""),
            jurisdiction=data.get("jurisdiction", "BD"),
            citation_string=data.get("citation_string", ""),
            title=data.get("title", ""),
            section_code=sec_code,
            article_number=art_no,
            summary=summary,
            full_text=data.get("full_text", ""),
            source_url=data.get("source_url", ""),
            precedents=precedents,
            edges=edges,
            confidence=data.get("confidence", "high"),
            verified=data.get("verified", True),
        )


@dataclass
class Law:
    """
    Main Law Class (analogous to the main class in Java).
    Encapsulates the statutory instrument, its section codes, penal code numbers,
    article numbers, operative summaries, precedents, and relational edges.
    """
    law_id: str  # e.g. "BD-CONTRACT-1872", "BD-PENAL-1860", "US-SHERMAN-1890", "EU-GDPR-2016"
    jurisdiction: str  # "BD" | "US" | "EU"
    title: str  # e.g. "The Contract Act, 1872"
    short_title: str  # e.g. "Contract Act 1872"
    year: int
    act_number: Optional[str] = None  # e.g. "Act No. IX of 1872"
    penal_code_number: Optional[str] = None  # e.g. "Act No. XLV of 1860" for penal codes
    legal_domain: str = "commercial"  # "commercial" | "penal" | "competition" | "constitutional" | "privacy"
    status: str = "in_force"
    source_url: str = ""
    provisions: Dict[str, LegalProvision] = field(default_factory=dict)
    precedents: List[PrecedentCase] = field(default_factory=list)
    law_level_edges: List[LawEdge] = field(default_factory=list)

    def add_provision(self, provision: LegalProvision) -> None:
        self.provisions[provision.provision_id] = provision

    def get_provision(self, provision_id: str) -> Optional[LegalProvision]:
        return self.provisions.get(provision_id)

    def add_precedent(self, precedent: PrecedentCase) -> None:
        self.precedents.append(precedent)

    def add_edge(self, edge: LawEdge) -> None:
        self.law_level_edges.append(edge)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "law_id": self.law_id,
            "jurisdiction": self.jurisdiction,
            "title": self.title,
            "short_title": self.short_title,
            "year": self.year,
            "act_number": self.act_number,
            "penal_code_number": self.penal_code_number,
            "legal_domain": self.legal_domain,
            "status": self.status,
            "source_url": self.source_url,
            "provisions": {k: v.to_dict() for k, v in self.provisions.items()},
            "precedents": [p.to_dict() for p in self.precedents],
            "law_level_edges": [e.to_dict() for e in self.law_level_edges],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Law":
        provisions = {
            k: LegalProvision.from_dict(v) for k, v in data.get("provisions", {}).items()
        }
        precedents = [PrecedentCase.from_dict(p) for p in data.get("precedents", [])]
        law_edges = [LawEdge.from_dict(e) for e in data.get("law_level_edges", [])]
        return cls(
            law_id=data.get("law_id", ""),
            jurisdiction=data.get("jurisdiction", "BD"),
            title=data.get("title", ""),
            short_title=data.get("short_title", ""),
            year=int(data.get("year", 0)),
            act_number=data.get("act_number"),
            penal_code_number=data.get("penal_code_number"),
            legal_domain=data.get("legal_domain", "commercial"),
            status=data.get("status", "in_force"),
            source_url=data.get("source_url", ""),
            provisions=provisions,
            precedents=precedents,
            law_level_edges=law_edges,
        )
