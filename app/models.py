"""Pydantic schemas implementing the Section 3 Interface Contract."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

try:
    from pydantic import field_validator as _field_validator

    PYDANTIC_V2 = True
except ImportError:
    from pydantic import validator as _validator

    PYDANTIC_V2 = False


JurisdictionCode = Literal["BD", "US", "EU"]
SourceType = Literal["full_text", "provision_summary", "case_reference", "law_summary"]
ConfidenceLevel = Literal["high", "medium", "low"]
SeverityLevel = Literal["high", "medium", "low"]
VerificationStatus = Literal["verified", "found_in_corpus", "unverified"]


class ErrorResponse(BaseModel):
    error: str
    message: str


class CorpusStats(BaseModel):
    documents: int
    by_jurisdiction: Dict[str, int]
    full_text_documents: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str = "0.1.0"
    mock: bool
    embeddings: Literal["openai", "local", "hash"]
    corpus: CorpusStats


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    jurisdictions: Optional[List[JurisdictionCode]] = None
    risk_tags: Optional[List[str]] = None
    k: int = Field(default=8, ge=1, le=20)


class ContextItem(BaseModel):
    id: str
    citation_string: str
    jurisdiction: JurisdictionCode
    title: str
    text: str
    source_url: str
    source_type: SourceType
    verified: bool
    confidence: ConfidenceLevel


class SearchResponse(BaseModel):
    context: List[ContextItem]


class AnalyzeRequest(BaseModel):
    document_text: str = Field(..., min_length=1)
    jurisdictions: Optional[List[JurisdictionCode]] = None
    doc_name: Optional[str] = None


class RiskCitation(BaseModel):
    citation_string: str
    quote: Optional[str] = None
    status: VerificationStatus
    reason: str


class RiskItem(BaseModel):
    title: str
    clause_refs: List[str]
    why_non_obvious: str
    severity: SeverityLevel
    jurisdiction_notes: Dict[str, str]
    citations: List[RiskCitation]
    mitigation_clause_text: str


class VerificationCounts(BaseModel):
    total: int
    verified: int
    found_in_corpus: int
    unverified: int


class ModelInfo(BaseModel):
    main: str
    small: str
    mock: bool


class TimingsMs(BaseModel):
    rewrite: int
    retrieval: int
    analysis: int
    verification: int
    memo: int
    total: int


class AnalyzeResponse(BaseModel):
    summary: str
    document_type: str
    governing_law: str
    risks: List[RiskItem]
    verification: VerificationCounts
    memo_markdown: str
    disclaimer: str
    model_info: ModelInfo
    timings_ms: TimingsMs


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1, max_length=12)
    jurisdictions: Optional[List[JurisdictionCode]] = None
    context_note: Optional[str] = Field(default=None, max_length=8000)
    mode: Optional[str] = Field(default="conversational")

    if PYDANTIC_V2:
        @_field_validator("messages")
        @classmethod
        def _validate_messages(cls, v: List[ChatMessage]) -> List[ChatMessage]:
            if not v or len(v) > 12:
                raise ValueError("messages must have between 1 and 12 items")
            if v[-1].role != "user":
                raise ValueError("last message must have role 'user'")
            return v
    else:
        @_validator("messages")
        def _validate_messages(cls, v: List[ChatMessage]) -> List[ChatMessage]:
            if not v or len(v) > 12:
                raise ValueError("messages must have between 1 and 12 items")
            if v[-1].role != "user":
                raise ValueError("last message must have role 'user'")
            return v


class ChatCitation(BaseModel):
    citation_string: str
    status: VerificationStatus
    reason: str


class ChatResponse(BaseModel):
    answer_markdown: str
    citations: List[ChatCitation]
    disclaimer: str
    mock: bool


class CitationLookupResponse(BaseModel):
    citation_string: str
    status: VerificationStatus
    reason: str
    source_url: Optional[str] = None


def model_to_dict(instance: BaseModel) -> dict:
    """Serialize Pydantic model across v1 and v2."""
    if hasattr(instance, "model_dump"):
        return instance.model_dump()
    return instance.dict()
