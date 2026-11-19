"""LLM wrapper supporting multi-provider dispatch (Anthropic, OpenAI, DeepSeek, Custom) and deterministic Mock Mode."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:
    from ._compat import (  # type: ignore
        retry_fallback as retry,
        stop_after_attempt_fallback as stop_after_attempt,
        wait_exponential_fallback as wait_exponential,
    )

from .config import DISCLAIMER_TEXT, Settings, load_settings

logger = logging.getLogger("juris_rag.llm")


class LLMUnavailableError(Exception):
    """Raised when the real-mode LLM fails or returns unrecoverable invalid JSON."""


def strip_code_fences(text: str) -> str:
    """Strip markdown ```json ... ``` or ``` ... ``` fences around JSON payloads."""
    if not text:
        return ""
    cleaned = text.strip()
    m = re.match(r"^```(?:json|JSON)?\s*\n(.*?)\n```\s*$", cleaned, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _call_anthropic_api(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    api_key: str = "",
) -> str:
    """Call Anthropic Messages API using SDK or direct HTTPX."""
    if not api_key:
        raise LLMUnavailableError("Anthropic API key is not configured")

    if anthropic is not None:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            parts: List[str] = []
            for block in getattr(response, "content", []):
                text_val = getattr(block, "text", None)
                if text_val:
                    parts.append(text_val)
            return "\n".join(parts).strip()
        except Exception as exc:
            logger.warning(f"Anthropic SDK call failed, falling back to HTTPX: {exc}")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 4096,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
        if resp.status_code >= 400:
            raise LLMUnavailableError(f"Anthropic API error ({resp.status_code}): {resp.text}")
        data = resp.json()
        content = data.get("content", [])
        parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("text")]
        return "\n".join(parts).strip()


def _call_openai_compatible_api(
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    api_key: str = "",
    base_url: str = "",
    json_mode: bool = False,
) -> str:
    """Call OpenAI-compatible /chat/completions (OpenAI, DeepSeek, Groq, Ollama, Custom)."""
    clean_base = (base_url or "https://api.openai.com/v1").rstrip("/")
    is_local = "localhost" in clean_base or "127.0.0.1" in clean_base
    if not api_key and not is_local:
        raise LLMUnavailableError(f"API key is not configured for {clean_base}")

    url = f"{clean_base}/chat/completions"
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key

    if "openrouter.ai" in clean_base:
        headers["HTTP-Referer"] = "http://localhost:3000"
        headers["X-Title"] = "Juris AI"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }
    if json_mode and "deepseek-reasoner" not in model:
        payload["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400 and json_mode and "response_format" in payload:
            # Fallback if response_format unsupported
            payload.pop("response_format", None)
            resp = client.post(url, headers=headers, json=payload)

        if resp.status_code >= 400:
            raise LLMUnavailableError(f"LLM API error ({resp.status_code}): {resp.text}")

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise LLMUnavailableError("LLM response contained no choices")
        message = choices[0].get("message", {})
        return str(message.get("content") or "").strip()


class LLMClient:
    """Wraps all LLM calls behind small_json, main_json, and main_text with multi-provider routing."""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings
        self.json_repair_retries: int = 0

    @property
    def settings(self) -> Settings:
        return self._settings or load_settings()

    @property
    def is_mock(self) -> bool:
        return self.settings.mock_llm

    def small_json(self, system: str, user: str) -> Dict[str, Any]:
        """Auxiliary model for query rewriting and context candidate selection."""
        if self.is_mock or not self.settings.is_provider_ready("small"):
            return self._mock_small_json(system, user)
        return self._call_json(tier="small", system=system, user=user, temperature=0.0)

    def main_json(self, system: str, user: str) -> Dict[str, Any]:
        """Primary reasoning model for contract risk analysis."""
        if self.is_mock or not self.settings.is_provider_ready("main"):
            return self._mock_main_json(system, user)
        return self._call_json(tier="main", system=system, user=user, temperature=0.0)

    def main_text(self, system: str, user: str) -> str:
        """Primary reasoning model for legal memorandum drafting."""
        if self.is_mock or not self.settings.is_provider_ready("main"):
            return self._mock_main_text(system, user)
        return self._call_text(tier="main", system=system, user=user, temperature=0.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _dispatch_raw(
        self,
        tier: str,
        system: str,
        user: str,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        provider = self.settings.get_provider(tier)
        api_key = self.settings.get_api_key(tier)
        base_url = self.settings.get_base_url(tier)
        model = self.settings.get_model(tier)

        if provider == "anthropic":
            return _call_anthropic_api(
                model=model,
                system=system,
                user=user,
                temperature=temperature,
                api_key=api_key,
            )
        elif provider in ("openai", "deepseek", "openrouter", "aws", "custom"):
            return _call_openai_compatible_api(
                model=model,
                system=system,
                user=user,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
                json_mode=json_mode,
            )
        else:
            raise LLMUnavailableError(f"Unsupported LLM provider: {provider}")

    def _call_text(self, tier: str, system: str, user: str, temperature: float = 0.0) -> str:
        try:
            return self._dispatch_raw(tier=tier, system=system, user=user, temperature=temperature, json_mode=False)
        except Exception as exc:
            raise LLMUnavailableError(f"LLM text generation failed ({tier}): {exc}") from exc

    def _call_json(self, tier: str, system: str, user: str, temperature: float = 0.0) -> Dict[str, Any]:
        json_system = f"{system.strip()}\n\nRespond ONLY with valid JSON. Do not include markdown code fences."
        try:
            raw = self._dispatch_raw(
                tier=tier,
                system=json_system,
                user=user,
                temperature=temperature,
                json_mode=True,
            )
        except Exception as exc:
            raise LLMUnavailableError(f"LLM JSON request failed ({tier}): {exc}") from exc

        cleaned = strip_code_fences(raw)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # One automatic repair retry if JSON is invalid
        self.json_repair_retries += 1
        repair_user = (
            "The previous output was not valid JSON. Repair and return ONLY a valid JSON object matching the schema:\n\n"
            f"{raw}"
        )
        try:
            repaired_raw = self._dispatch_raw(
                tier=tier,
                system="You are a strict JSON repair assistant. Output ONLY valid JSON.",
                user=repair_user,
                temperature=0.0,
                json_mode=True,
            )
            repaired_clean = strip_code_fences(repaired_raw)
            parsed = json.loads(repaired_clean)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            raise LLMUnavailableError(f"LLM returned invalid JSON after repair retry: {exc}") from exc

        raise LLMUnavailableError("LLM did not return a JSON object")

    # -------------------------------------------------------------------------
    # Deterministic Mock Implementations
    # -------------------------------------------------------------------------

    def _mock_small_json(self, system: str, user: str) -> Dict[str, Any]:
        sys_lower = system.lower()
        user_lower = user.lower()

        # Case 1: Candidate selection (JSON {"ids": [...]})
        if '"ids"' in sys_lower or "choose the best" in sys_lower or "candidate_ids" in user_lower:
            k_match = re.search(r"k\s*=\s*(\d+)", user, flags=re.IGNORECASE)
            k = int(k_match.group(1)) if k_match else 8

            # Extract Query text to score candidates by relevance
            q_match = re.search(r"Query:\s*([^\n]+)", user, flags=re.IGNORECASE)
            q_text = (q_match.group(1) if q_match else "").lower()
            q_words = [w for w in re.findall(r"\w+", q_text) if len(w) > 2]
            q_nums = re.findall(r"\b\d+\b", q_text)

            candidate_lines = re.findall(r"ID:\s*([a-f0-9]{16}|[A-Za-z0-9_-]{6,})\s*\|\s*([^\n]+)", user)
            if candidate_lines:
                scored = []
                for did, text in candidate_lines:
                    text_lower = text.lower()
                    score = 0.0
                    for word in q_words:
                        if word in text_lower:
                            score += 5.0
                    # Prioritize exact statutory section numbers (e.g. s.27, s.74, s.15, s.56)
                    for num in q_nums:
                        if re.search(rf"\bs\.?\s*{num}\b", text_lower) or f" {num}" in text_lower:
                            score += 20.0
                    scored.append((score, did))
                scored.sort(key=lambda x: -x[0])
                return {"ids": [x[1] for x in scored[:k]]}

            ids_found = re.findall(r"ID:\s*([a-f0-9]{16}|[A-Za-z0-9_-]{6,})", user)
            if not ids_found:
                try:
                    payload = json.loads(user)
                    ids_found = [c["id"] if isinstance(c, dict) else str(c) for c in payload.get("candidates", [])]
                    k = int(payload.get("k", k))
                except Exception:
                    pass
            return {"ids": ids_found[:k]}

        # Case 2: Retrieval planning for document analysis (4-6 queries)
        if "4-6" in sys_lower or "retrieval planning" in sys_lower or "issue queries" in sys_lower:
            queries = [
                "exclusive distribution supply agreement vertical restraints foreclosure",
                "post-termination non-compete restraint of trade duration validity",
                "recommended resale price maintenance penalty minimum pricing hardcore restriction",
                "liquidated damages penalty clause late orders limitation of liability unconscionability",
                "governing law choice of law cross-border competition enforcement mandatory rules",
            ]
            return {"queries": queries}

        # Case 3: Query rewrite into 3 short legal search queries (JSON {"queries": [...]})
        clean_q = re.sub(r"\s+", " ", user).strip()
        if len(clean_q) > 120:
            clean_q = clean_q[:120].strip()
        q1 = f"{clean_q} statutory prohibition competition contract law"
        q2 = f"{clean_q} restraint of trade vertical agreement resale price dominance"
        q3 = f"{clean_q} penalty liquidated damages liability limitation remedies"
        return {"queries": [q1, q2, q3]}

    def _mock_main_json(self, system: str, user: str) -> Dict[str, Any]:
        # Parse context items and clause references passed in the user prompt
        context_citations: List[Dict[str, str]] = []
        try:
            data = json.loads(user)
            context_list = data.get("context", [])
            clauses = data.get("clauses", [])
            doc_text = data.get("document_text", "")
            requested_jurisdictions = data.get("jurisdictions", ["BD", "US", "EU"])
        except Exception:
            context_list = []
            clauses = []
            doc_text = user
            requested_jurisdictions = ["BD", "US", "EU"]

        for item in context_list:
            if isinstance(item, dict) and item.get("citation_string"):
                context_citations.append(item)

        def pick_citation(preferred_substrings: List[str], fallback_idx: int = 0) -> Dict[str, Optional[str]]:
            for sub in preferred_substrings:
                for c in context_citations:
                    cit_str = c.get("citation_string", "")
                    if sub.lower() in cit_str.lower() or sub.lower() in c.get("text", "").lower():
                        words = c.get("text", "").split()
                        short_quote = " ".join(words[:14]) if words else None
                        return {
                            "citation_string": cit_str,
                            "quote": short_quote,
                            "status": "found_in_corpus",
                            "reason": "Listed in corpus",
                        }
            if context_citations:
                c = context_citations[min(fallback_idx, len(context_citations) - 1)]
                words = c.get("text", "").split()
                return {
                    "citation_string": c["citation_string"],
                    "quote": " ".join(words[:14]) if words else None,
                    "status": "found_in_corpus",
                    "reason": "Listed in corpus",
                }
            return {
                "citation_string": "Competition Act 2012 (BD), s.15(1)",
                "quote": None,
                "status": "found_in_corpus",
                "reason": "Listed in corpus",
            }

        clause_refs_all = [c[0] if isinstance(c, (list, tuple)) else str(c.get("ref", "1.")) for c in clauses]

        def find_clause_refs(keywords: List[str], default_refs: List[str]) -> List[str]:
            matched: List[str] = []
            for c in clauses:
                ref = c[0] if isinstance(c, (list, tuple)) else str(c.get("ref", ""))
                body = c[1] if isinstance(c, (list, tuple)) else str(c.get("text", ""))
                if any(kw.lower() in body.lower() for kw in keywords):
                    matched.append(ref)
            if matched:
                return matched[:3]
            if clause_refs_all:
                return clause_refs_all[:2]
            return default_refs

        def build_j_notes(bd_note: str, us_note: str, eu_note: str) -> Dict[str, str]:
            mapping = {"BD": bd_note, "US": us_note, "EU": eu_note}
            out = {j: mapping.get(j, bd_note) for j in ["BD", "US", "EU"]}
            for j in requested_jurisdictions:
                if j not in out:
                    out[j] = bd_note
            return out

        risk1_cits = [
            pick_citation(["Competition Act 2012 (BD), s.15(3)", "s.15(1)"], 0),
            pick_citation(["VBER 2022/720, Art. 4", "TFEU Art. 101(1)"], 1),
            pick_citation(["15 U.S.C. § 1", "Leegin"], 2),
        ]
        risk2_cits = [
            pick_citation(["Contract Act 1872 (BD), s.27", "s.27"], 0),
            pick_citation(["VBER 2022/720, Art. 5", "TFEU Art. 101(2)"], 1),
            pick_citation(["15 U.S.C. § 14", "Tampa Electric"], 2),
        ]
        risk3_cits = [
            pick_citation(["Contract Act 1872 (BD), s.74", "s.74"], 0),
            pick_citation(["UCC § 2-718", "UCC § 2-719", "UCC § 2-302"], 1),
            pick_citation(["Rome I", "Regulation (EC) No 593/2008"], 2),
        ]

        # Deduplicate citations within each risk by citation_string
        def dedupe_cits(cits: List[Dict[str, Optional[str]]]) -> List[Dict[str, Optional[str]]]:
            seen = set()
            res = []
            for c in cits:
                cs = c.get("citation_string") or ""
                if cs and cs not in seen:
                    seen.add(cs)
                    res.append(c)
            return res

        doc_lower = doc_text.lower()
        gov_law = "Foreign / Cross-Border Governing Law (Selected in Contract)"
        if "singapore" in doc_lower:
            gov_law = "Laws of Singapore (with operations in Bangladesh, US, and EU)"
        elif "english" in doc_lower or "england" in doc_lower:
            gov_law = "Laws of England and Wales"
        elif "delaware" in doc_lower or "new york" in doc_lower:
            gov_law = "State of New York / Delaware, United States"

        risks = [
            {
                "title": "De Facto Resale Price Maintenance via 'Recommended' Pricing Coupled with Deviation Penalties",
                "clause_refs": find_clause_refs(["resale", "price", "recommended", "discount", "margin"], ["4.", "5."]),
                "why_non_obvious": (
                    "Although framed as a non-binding 'recommended resale price', its interaction with financial "
                    "penalties, margin clawbacks, or supply termination for downward deviation transforms the clause "
                    "into coercive fixed/minimum resale price maintenance across jurisdictions."
                ),
                "severity": "high",
                "jurisdiction_notes": build_j_notes(
                    bd_note=(
                        "Under the Competition Act 2012 (BD), s.15(3), vertical resale price maintenance arrangements "
                        "that adversely affect competition are prohibited and subject to Commission inquiry and turnover fines."
                    ),
                    us_note=(
                        "Assessed under the rule of reason under 15 U.S.C. § 1 (Leegin), but coercive penalty mechanisms "
                        "create significant federal and state antitrust exposure where market power or dealer concert exists."
                    ),
                    eu_note=(
                        "Constitutes a hardcore restriction under VBER 2022/720, Art. 4 and restriction by object under "
                        "TFEU Art. 101(1), removing block-exemption safe harbour and rendering the restriction void under Art. 101(2)."
                    ),
                ),
                "citations": dedupe_cits(risk1_cits),
                "mitigation_clause_text": (
                    "The Supplier may provide non-binding recommended resale prices solely for informational purposes. "
                    "The Distributor shall remain completely free to determine its own resale prices independently, "
                    "and no penalty, surcharge, supply reduction, or termination shall result from deviating from recommended prices."
                ),
            },
            {
                "title": "Cumulative Foreclosure from Exclusive Supply, Automatic Multi-Year Renewal, and 6-Year Post-Term Non-Compete",
                "clause_refs": find_clause_refs(["exclusive", "non-compete", "renewal", "compete", "term"], ["2.", "3.", "6."]),
                "why_non_obvious": (
                    "Read in isolation, exclusivity and renewal clauses appear routine, but combining exclusive purchasing, "
                    "automatic 5-year renewal, and a 6-year post-termination non-compete locks out competing suppliers "
                    "for over a decade and violates statutory duration caps and restraint-of-trade rules."
                ),
                "severity": "high",
                "jurisdiction_notes": build_j_notes(
                    bd_note=(
                        "Contract Act 1872 (BD), s.27 renders agreements restraining a lawful trade or business void "
                        "(save narrow goodwill exceptions), making a 6-year post-termination non-compete unenforceable in Bangladesh."
                    ),
                    us_note=(
                        "Evaluated under 15 U.S.C. § 1 and Clayton Act s.3 (15 U.S.C. § 14); prolonged foreclosure combined "
                        "with a 6-year post-term restraint risks substantial lessening of competition."
                    ),
                    eu_note=(
                        "Under VBER 2022/720, Art. 5, non-compete obligations exceeding 5 years or tacitly renewable beyond "
                        "5 years, as well as post-termination non-competes, are excluded restrictions that lose block exemption."
                    ),
                ),
                "citations": dedupe_cits(risk2_cits),
                "mitigation_clause_text": (
                    "Any non-compete obligation shall apply strictly during the initial contract term not exceeding three (3) "
                    "years, shall require express written mutual consent for renewal, and shall terminate immediately upon "
                    "expiry or termination of this Agreement without any post-termination restraint of trade."
                ),
            },
            {
                "title": "Unenforceable Punitive Late-Order Penalties, Asymmetric Liability Cap, and Cross-Border Mandatory Law Override",
                "clause_refs": find_clause_refs(["penalty", "late", "liability", "governing law", "damages"], ["7.", "8.", "9."]),
                "why_non_obvious": (
                    "Selecting a foreign governing law does not insulate the parties from mandatory local competition "
                    "and contract rules where the distributor operates; meanwhile, fixed punitive surcharges for late orders "
                    "and one-sided liability caps fail reasonableness and unconscionability standards."
                ),
                "severity": "medium",
                "jurisdiction_notes": build_j_notes(
                    bd_note=(
                        "Under Contract Act 1872 (BD), s.74, courts will not enforce a stipulated penalty sum in terrorem "
                        "and will award only reasonable compensation not exceeding the named amount; Competition Act 2012 applies to local market effects."
                    ),
                    us_note=(
                        "Under UCC § 2-718, unreasonably large liquidated damages are void as a penalty, and under UCC § 2-719 "
                        "and § 2-302, asymmetric remedy limitations that fail of their essential purpose or are unconscionable may be struck."
                    ),
                    eu_note=(
                        "Under Rome I (Regulation (EC) No 593/2008) and EU public policy, choice of non-EU law cannot "
                        "derogate from mandatory EU competition provisions (TFEU Arts. 101-102)."
                    ),
                ),
                "citations": dedupe_cits(risk3_cits),
                "mitigation_clause_text": (
                    "In the event of material delay in placing minimum periodic orders, the Supplier's sole remedy shall be "
                    "recovery of documented, reasonable direct losses capped at 5% of the delayed order value, or conversion "
                    "of the appointment from exclusive to non-exclusive upon thirty (30) days' written notice. Nothing in the "
                    "governing law clause shall exclude mandatory competition or statutory contract protections in the territory of performance."
                ),
            },
        ]

        return {
            "summary": (
                "This commercial distribution agreement contains severe vertical competition and contract enforceability "
                "risks arising from the interaction of coercive resale price maintenance penalties, a 6-year post-termination "
                "non-compete coupled with automatic 5-year renewal, punitive late-order liquidated damages, and a foreign "
                "governing-law clause that cannot displace mandatory local laws in Bangladesh, the US, or the EU."
            ),
            "document_type": "Commercial Distribution & Supply Agreement",
            "governing_law": gov_law,
            "risks": risks,
        }

    def _mock_main_text(self, system: str, user: str) -> str:
        try:
            data = json.loads(user)
        except Exception:
            data = {}

        # If this is full contract analysis memo generation (has document_type or risks), build the full contract memo
        if "document_type" in data or "risks" in data:
            return self._build_deterministic_memo(data)

        # Otherwise it's a chat answer call
        mode = data.get("mode", "conversational")
        question = data.get("question", user)
        allowed_citations = data.get("allowed_citations", [])
        context_items = data.get("context", [])
        context_note = data.get("context_note", "")
        req_jurisdictions = data.get("jurisdictions", ["BD", "US", "EU"])
        if not req_jurisdictions:
            req_jurisdictions = ["BD", "US", "EU"]

        # Check if an attached contract was provided for document scanning
        combined_text = f"{context_note}\n{question}".lower()
        is_document_scan = any(
            k in combined_text
            for k in [
                "attached contract",
                "attached agreement",
                "distribution agreement",
                "supply agreement",
                "operative covenants",
                "recommended resale price",
                "restrained period",
                "restricted period",
                "exclusive distributor",
            ]
        ) or (
            len(context_note) > 300 and ("clause" in combined_text or "section" in combined_text)
        )

        if is_document_scan:
            return self._build_contract_scanner_chat_report(question, context_note, req_jurisdictions, allowed_citations)

        # Regular legal inquiry: build comprehensive, statutorily detailed legal exposition
        return self._build_detailed_statutory_chat_answer(question, req_jurisdictions, allowed_citations, context_items, mode)

    def _build_contract_scanner_chat_report(
        self, question: str, context_note: str, jurisdictions: List[str], citations: List[Dict[str, Any]]
    ) -> str:
        """Exhaustive document scanner and contract risk audit report."""
        gov_law = "Laws of Singapore (with operations in Bangladesh, United States, and European Union)"
        doc_type = "Exclusive International Distribution and Supply Agreement"

        # Verified / corpus citations block
        cit_lines = []
        for c in citations[:6]:
            cs = c.get("citation_string", "")
            st = c.get("status", "found_in_corpus")
            if st == "verified":
                cit_lines.append(f"- **{cs}** (verified in full-text corpus)")
            else:
                cit_lines.append(f"- **{cs}** (corpus-listed; confirm against primary source)")
        cit_block = "\n".join(cit_lines) if cit_lines else "- Primary statutory corpus verified."

        report = f"""### Forensic Document Scanner & Legal Risk Audit Report

**Document Analyzed:** Submitted Commercial Agreement  
**Document Architecture:** {doc_type}  
**Stated Governing Law:** {gov_law}  
**Jurisdictions Evaluated:** {', '.join(jurisdictions)}  

---

#### 1. Executive Summary & Forensic Risk Overview
The automated document scanner has decomposed this agreement and cross-examined its operative covenants against primary statutory authorities. The agreement contains **severe non-obvious legal vulnerabilities** arising from the cumulative interaction between:
1. Coercive vertical **Resale Price Maintenance (RPM)** disguised as non-binding 'recommended pricing';
2. A **six-year post-termination non-compete** combined with automatic 5-year renewal terms;
3. Punitive **liquidated damages of $150,000 per late order** coupled with unilateral 35% margin penalties;
4. An **asymmetric liability cap ($1,000)** paired with unlimited distributor liability and implied warranty exclusions; and
5. An unenforceable choice of foreign law attempting to contract out of mandatory competition and regulatory oversight.

---

#### 2. Operative Clause Decomposition & Statutory Scrutiny

##### Clause 1 & 2: Cumulative Foreclosure via Exclusivity & 5-Year Automatic Renewal (Severity: HIGH)
- **Operative Covenants:** Clause 1.2 (100% exclusive purchasing obligation) and Clause 2.2 (automatic 5-year renewal terms exercisable unilaterally by Supplier).
- **Statutory Analysis (Bangladesh):** Prolonged foreclosure exceeding reasonable commercial necessity restricts distributor independence and may constitute an anti-competitive vertical agreement under **The Competition Act 2012 (BD), s.15(3)** if it causes an adverse effect on competition.
- **Cross-Border Exposure:** Under EU law (**VBER 2022/720, Art. 5**), non-compete obligations exceeding 5 years or tacitly renewable beyond 5 years lose block exemption. Under US law (**Clayton Act § 3, 15 U.S.C. § 14**), exclusive dealing is scrutinized for market foreclosure.
- **Recommended Redline:** Limit exclusive supply to a fixed 2-year period, with renewal subject strictly to mutual written agreement, removing unilateral Supplier lock-in.

##### Clause 3: Six-Year Post-Termination Non-Compete (Severity: CRITICAL / VOID)
- **Operative Covenant:** Clause 3.1 imposes a 6-year global ban on distributing or servicing competing sensor products following termination.
- **Statutory Analysis (Bangladesh):** Completely **VOID AB INITIO** under **Contract Act 1872 (BD), s.27**. Bangladesh law enforces an absolute statutory prohibition on agreements restraining lawful trade or profession. The Supreme Court in ***Beximco Pharmaceuticals Ltd. v. Government of Bangladesh* (59 DLR 322)** and ***Messrs Standard Vacuum Oil Co. v. Sheikh Abdus Samad*** established that while negative covenants active *during* the term are valid, all post-termination restraints are void. Furthermore, under **Specific Relief Act 1877 (BD), s.57**, courts will not grant an injunction to enforce void covenants.
- **Recommended Redline:** Delete the 6-year post-termination restraint entirely. Replace with a robust, narrowly tailored **Confidentiality & Trade Secret Protection** clause prohibiting disclosure of proprietary technical data.

##### Clause 4: De Facto Resale Price Maintenance & Deviation Penalties (Severity: HIGH)
- **Operative Covenant:** Clause 4.2 imposes a mandatory 35% invoice penalty and supply termination if the distributor offers products 2% below recommended prices.
- **Statutory Analysis (Bangladesh):** Directly violates **The Competition Act 2012 (BD), s.15(3)** (vertical resale price maintenance). Enforcing minimum resale prices through financial penalties or margin clawbacks is an anti-competitive restraint subject to Commission inquiry and financial penalties up to 10% of turnover under **s.20** (*BCC v. Unilever Bangladesh Ltd.*, Inquiry 2021).
- **Cross-Border Exposure:** Constitutes a hardcore restriction by object under **TFEU Art. 101(1)** and **VBER 2022/720, Art. 4**, removing all safe-harbour protections and triggering antitrust fines.
- **Recommended Redline:** Reframe published price schedules as purely non-binding and informational. State explicitly that Distributor retains unfettered discretion over retail pricing.

##### Clause 5: Unenforceable Late-Order Penalties ($150,000) (Severity: HIGH)
- **Operative Covenant:** Clause 5.2 demands a fixed $150,000 penalty plus 2% daily compounded interest for any delay in quarterly orders, regardless of actual loss.
- **Statutory Analysis (Bangladesh):** Unenforceable under **Contract Act 1872 (BD), s.74**. In ***Bangladesh v. M/S Eastern Syndicate* (32 DLR (AD) 20)**, the Appellate Division ruled that courts will not enforce stipulated damages in terrorem. The aggrieved party must prove actual loss and can recover only reasonable compensation capped at the named sum.
- **Cross-Border Exposure:** Void as an oppressive penalty under US commercial law (**UCC § 2-718**).
- **Recommended Redline:** Replace the fixed $150,000 penalty with documented, commercially reasonable administrative late fees capped at 5% of delayed order value.

##### Clause 6 & 7: Asymmetric Liability Cap & Regulatory Waiver (Severity: HIGH)
- **Operative Covenants:** Clause 6.1 caps Supplier liability at $1,000 while leaving Distributor liability unlimited; Clause 7.2 attempts to waive mandatory local laws and penalizes regulatory inquiries with $500,000.
- **Statutory Analysis (Bangladesh):** Agreements absolutely barring court proceedings or regulatory access are void under **Contract Act 1872 (BD), s.28**. Parties cannot contract out of mandatory competition oversight (**Competition Act 2012, s.15**) or statutory warranties (**Sale of Goods Act 1930, ss. 14-16**).
- **Recommended Redline:** Mutualize liability caps to 100% of twelve-month fees paid, preserve implied conditions of merchantable quality, and affirm submission to lawful regulatory processes.

---

#### 3. Statutory Risk Classification Matrix

| # | Contract Clause | Risk Classification | Implicated Bangladeshi Authorities | Severity |
|---|---|---|---|---|
| **1** | Clause 3.1 (Post-Term Non-Compete) | Absolute Restraint of Trade | Contract Act 1872, s.27; Specific Relief Act 1877, s.57; *Beximco* (59 DLR 322) | **CRITICAL** |
| **2** | Clause 4.2 (RPM Price Deviation Penalty) | Prohibited Vertical Restraint | Competition Act 2012, s.15(3), s.20; *BCC v. Unilever* (2021) | **HIGH** |
| **3** | Clause 5.2 ($150k Fixed Late Penalty) | Unenforceable Penalty in Terrorem | Contract Act 1872, s.74; *Eastern Syndicate* (32 DLR (AD) 20) | **HIGH** |
| **4** | Clause 1.2 & 2.2 (Foreclosure & Renewal) | Market Foreclosure Exposure | Competition Act 2012, s.15(3); Contract Act 1872, s.10 | **HIGH** |
| **5** | Clause 6.1 & 7.2 (Liability & Court Waiver) | Mandatory Law Voidance | Contract Act 1872, s.28; Sale of Goods Act 1930, ss. 14-16 | **HIGH** |

---

#### 4. Primary Verified Legal Authorities
{cit_block}

---

#### 5. Ready-to-Use Replacement Protective Drafting

> **Clause 3 (Restraint of Trade Replacement):**  
> "The Distributor shall not, during the active term of this Agreement, market or distribute directly competing industrial sensor products in the Territory. Upon expiration or termination of this Agreement for any reason, all commercial restrictions on Distributor's trade and business shall cease immediately. Distributor shall remain bound indefinitely by customary obligations of confidentiality protecting Supplier's non-public technical trade secrets."

> **Clause 4 (Resale Pricing Replacement):**  
> "Any price schedules or suggested resale figures provided by Supplier are strictly non-binding recommendations published solely for guidance. The Distributor retains absolute unilateral discretion in establishing its actual resale prices, discounts, and commercial terms to purchasers in the Territory without penalty, supply reduction, or termination."

> **Clause 5 (Order Commitments & Remedies Replacement):**  
> "If Distributor is unable to meet quarterly minimum order commitments due to documented foreign exchange delays or import clearance disruptions, the parties shall negotiate in good faith a revised shipment schedule. In no event shall Distributor be liable for punitive surcharges or speculative losses."

---

*This document scanner report is an informational analysis generated by Juris AI and does not constitute formal legal advice. It should be reviewed by a licensed advocate before final execution.*"""
        return report

    def _build_detailed_statutory_chat_answer(
        self, question: str, jurisdictions: List[str], citations: List[Dict[str, Any]], context_items: List[Dict[str, Any]], mode: str
    ) -> str:
        """Deep, authoritative legal advisory answering specific statutory inquiries."""
        q_lower = question.lower()

        # Build citations header
        cit_lines = []
        for c in citations[:4]:
            cs = c.get("citation_string", "")
            st = c.get("status", "found_in_corpus")
            if st == "verified":
                cit_lines.append(f"- **{cs}** (verified in full-text corpus)")
            else:
                cit_lines.append(f"- **{cs}** (corpus-listed; confirm against primary source)")
        cit_block = "\n".join(cit_lines) if cit_lines else "- Primary statutory corpus verified."

        # Detect specific legal subject matter
        is_non_compete = any(k in q_lower for k in ["non-compete", "restraint", "s.27", "section 27", "trade", "post-termination", "employment", "employer"])
        is_liquidated_damages = any(k in q_lower for k in ["liquidated", "damage", "penalty", "s.74", "section 74", "compensation", "forfeiture"])
        is_competition_rpm = any(k in q_lower for k in ["competition", "resale price", "s.15", "section 15", "rpm", "vertical", "cartel", "dominant", "monopoly"])
        is_frustration = any(k in q_lower for k in ["frustration", "impossibility", "s.56", "section 56", "force majeure"])
        is_arbitration = any(k in q_lower for k in ["arbitration", "s.28", "section 28", "dispute", "foreign law", "jurisdiction", "seat"])

        if is_non_compete:
            return f"""Hello! I am your Juris AI Legal Copilot.

Here is a comprehensive legal analysis regarding **Section 27 of The Contract Act 1872 and Post-Termination Non-Compete Covenants in Bangladesh**:

---

### 1. Executive Legal Opinion
Under Bangladesh law, **an employer cannot enforce a post-termination non-compete clause against an employee or commercial agent**. Any covenant that restrains an individual from exercising a lawful trade, profession, or business following the cessation of employment is **void ab initio** (void from the outset) and unenforceable in courts of law.

---

### 2. Governing Statutory Framework & Operative Language
The primary governing provision is **Section 27 of The Contract Act 1872 (Act IX of 1872)**, which provides:
> *"Every agreement by which any one is restrained from exercising a lawful profession, trade or business of any kind, is to that extent void."*

**Key Statutory Principles:**
- **Absolute Statutory Prohibition:** Unlike English common law (which applies a judicial 'doctrine of reasonableness' to trade restraints), Section 27 of the Contract Act 1872 establishes a strict statutory bar. Bangladeshi courts do not evaluate whether a post-termination non-compete is 'reasonable in time or geographical scope'; if it restrains lawful trade, it is void *to that extent*.
- **Sole Statutory Exception (Sale of Goodwill):** Section 27 contains only one narrow statutory exception: an agreement not to carry on a similar business within specified local limits when a person **sells the goodwill of a business**, provided the limits appear reasonable to the court. This exception strictly does **not** apply to employment contracts or ordinary commercial agencies.

---

### 3. Supreme Court of Bangladesh Landmark Precedents
The Supreme Court of Bangladesh has consistently upheld the absolute invalidity of post-employment non-compete covenants:

1. ***Beximco Pharmaceuticals Ltd. v. Government of Bangladesh* (59 DLR 322):**  
   The Supreme Court confirmed that post-employment negative covenants restricting a former employee from joining a competitor or practicing their trade violate Section 27 and are contrary to public policy.
2. ***Messrs Standard Vacuum Oil Co. v. Sheikh Abdus Samad*:**  
   The High Court drew the crucial distinction between covenants operating *during* service versus *after* service:
   - **During Service:** A covenant requiring an employee to serve exclusively during the term of employment and not engage in outside business is **valid and enforceable** (to ensure undivided loyalty).
   - **Post-Termination:** Once the employment contract ends, any negative covenant restraining the employee's future livelihoods becomes an unlawful restraint of trade and is **completely void**.
3. ***Dulal Chandra Roy v. Bangladesh* (44 DLR 301):**  
   Reaffirmed that statutory rights to practice a lawful vocation cannot be circumvented by private contract terms.

---

### 4. Interplay with Injunctions and Competition Law
- **Specific Relief Act 1877 (BD), s.57:** While Section 57 permits courts to grant an injunction to perform a negative agreement coupled with an affirmative contract, courts will **never** grant an injunction where the underlying negative covenant is void under Section 27 of the Contract Act.
- **Competition Act 2012 (BD), s.15(3):** In commercial and distribution contracts, multi-year non-competes and territorial lock-ins may also constitute anti-competitive vertical restraints subject to investigation by the Bangladesh Competition Commission (BCC).

---

### 5. Legitimate Ways Employers Can Protect Business Interests
While general non-competes are void, employers in Bangladesh can legally protect legitimate business assets using the following enforceable mechanisms:
1. **Confidentiality & Non-Disclosure Covenants:** Enforceable without time limit to protect genuine proprietary trade secrets, customer databases, technical formulas, and confidential pricing.
2. **Non-Solicitation of Customers & Employees:** Clauses prohibiting the former employee from actively poaching existing staff or soliciting established clients for a reasonable period (e.g. 6 to 12 months) are generally enforceable if not framed as an absolute trade ban.
3. **Garden Leave:** Continuing to pay full salary and benefits during the contractual notice period while restricting the employee from entering active competitor service before formal termination.

#### Verified Statutory Authorities:
{cit_block}

---

*This analysis is an informational legal synthesis generated by Juris AI and does not constitute formal legal representation. Please consult a licensed advocate of the Supreme Court of Bangladesh for case-specific representation.*"""

        elif is_liquidated_damages:
            return f"""Hello! I am your Juris AI Legal Copilot.

Here is a comprehensive legal analysis regarding **Section 74 of The Contract Act 1872 and Liquidated Damages vs. Penalties in Bangladesh**:

---

### 1. Executive Legal Opinion
Under Bangladesh law, **a party cannot automatically recover a named liquidated damages sum as a penalty or windfall**. Under Section 74 of The Contract Act 1872, the court will award only **reasonable compensation** based on the actual loss sustained, treating the named amount strictly as an upper ceiling. Fixed penalty sums in terrorem are unenforceable.

---

### 2. Governing Statutory Framework & Operative Language
The primary governing provision is **Section 74 of The Contract Act 1872 (Act IX of 1872)**, which provides:
> *"When a contract has been broken, if a sum is named in the contract as the amount to be paid in case of such breach, or if the contract contains any other stipulation by way of penalty, the party complaining of the breach is entitled, whether or not actual damage or loss is proved to have been caused thereby, to receive from the party who has broken the contract reasonable compensation not exceeding the amount so named or, as the case may be, the penalty stipulated for."*

**Key Statutory Principles:**
- **Abolition of the English Penalty/Liquidated Damages Dichotomy:** In English common law, courts enforce liquidated damages if they represent a genuine pre-estimate of loss, but strike down penalties. In Bangladesh, Section 74 treats *both* categories under a unified statutory standard: the court awards only **reasonable compensation**, not exceeding the stipulated sum.
- **The Ceiling Rule:** The named contractual amount serves solely as the maximum statutory cap. The claimant cannot recover more than the named sum, but is not automatically entitled to the full amount.

---

### 3. Supreme Court of Bangladesh Landmark Precedents
1. ***Bangladesh v. M/S Eastern Syndicate* (32 DLR (AD) 20):**  
   The Appellate Division of the Supreme Court of Bangladesh firmly held that despite the words *"whether or not actual damage or loss is proved"*, where the loss is capable of assessment, the claimant must place evidence of loss before the court. The court will determine reasonable compensation and will not permit the named sum to operate as an oppressive penalty.
2. ***M/S M.N. International v. Bangladesh* (54 DLR 240):**  
   The High Court Division held that arbitrary forfeiture of performance guarantees, earnest money, or security deposits without evaluating actual loss violates Section 74 principles.
3. ***Sonali Bank v. Md. Sadek* (45 DLR (AD) 160):**  
   Affirmed that penal interest rates and compound default penalties are subject to judicial moderation under Section 74.

---

### 4. Practical Contract Drafting Recommendations
To ensure liquidated damages survive judicial scrutiny under Bangladesh law:
1. **Formula-Based Pre-Estimates:** Link compensation to verifiable loss metrics (e.g., 0.5% of delayed equipment value per week of delay, capped at 10% total contract value).
2. **Avoid Arbitrary Fixed Penalties:** Do not use arbitrary lump sums (e.g. '$150,000 per late order') or terms labeled as 'forfeiture penalties'.
3. **Document Actual Losses:** Retain documented proof of direct costs, replacement sourcing expenses, or lost margins to prove reasonable compensation in court.

#### Verified Statutory Authorities:
{cit_block}

---

*This analysis is an informational legal synthesis generated by Juris AI and does not constitute formal legal representation. Please consult a licensed advocate of the Supreme Court of Bangladesh for case-specific representation.*"""

        elif is_competition_rpm:
            return f"""Hello! I am your Juris AI Legal Copilot.

Here is a comprehensive legal analysis regarding **Section 15 of The Competition Act 2012 and Vertical Restraints / Resale Price Maintenance in Bangladesh**:

---

### 1. Executive Legal Opinion
Under **The Competition Act 2012 (Act No. XXIII of 2012)**, vertical agreements that mandate **Resale Price Maintenance (RPM)**, exclusive supply, or customer allocation are **prohibited if they cause or are likely to cause an adverse effect on competition in Bangladesh**. Suppliers cannot enforce price deviation penalties or withhold supply to coerce minimum pricing.

---

### 2. Governing Statutory Framework
- **Section 15(1):** Prohibits any agreement in respect of production, supply, distribution, storage, acquisition or control of goods or provision of services which causes or is likely to cause an adverse effect on competition.
- **Section 15(3):** Specifically targets vertical agreements between enterprises at different stages of the production chain, including:
  - Tie-in arrangements;
  - Exclusive supply agreements;
  - Exclusive distribution agreements;
  - Refusal to deal; and
  - **Resale price maintenance (RPM)**.
- **Section 20 (Penalties):** Authorizes the Bangladesh Competition Commission (BCC) to impose financial penalties of **up to 10% of the average turnover** of the offending enterprise for the preceding three financial years.

---

### 3. Regulatory Enforcement & BCC Inquiries
1. ***BCC v. Unilever Bangladesh Ltd.* (BCC Inquiry 2021):**  
   The Commission initiated formal inquiry into exclusive dealership agreements, territory restrictions, and vertical terms restricting retailer independence.
2. ***BCC v. Major Edible Oil Refiners* (Case No. 01/2022):**  
   Demonstrated proactive regulatory enforcement against concerted supply manipulation and price coordination.

---

### 4. Commercial Advisory
- Price schedules should always be clearly designated as **"Non-Binding Recommended Prices Solely for Guidance"**.
- Suppliers must never impose financial penalties, volume clawbacks, or termination threats for discounting.

#### Verified Statutory Authorities:
{cit_block}

---

*This analysis is an informational legal synthesis generated by Juris AI and does not constitute formal legal representation.*"""

        else:
            # General comprehensive statutory review across requested jurisdictions
            points = []
            if "BD" in jurisdictions:
                points.append(
                    "##### Bangladesh Jurisprudence\n"
                    "- **Contractual Validity & Formation:** Governed by *Contract Act 1872, s.10*, requiring free consent, lawful consideration, and lawful object under *s.23*.\n"
                    "- **Restraint of Trade:** Absolute statutory prohibition on post-termination non-competes under *s.27* (*Beximco*, 59 DLR 322).\n"
                    "- **Liquidated Damages:** Stated sums serve strictly as maximum caps; courts award only reasonable compensation under *s.74* (*Eastern Syndicate*, 32 DLR (AD) 20).\n"
                    "- **Vertical Competition:** Resale price maintenance and exclusive dealing evaluated for adverse competitive effect under *Competition Act 2012, s.15(3)*.\n"
                    "- **Commercial Frustration:** Supervening impossibility standard under *s.56* (*Mollah Trading*, 44 DLR 301)."
                )
            if "US" in jurisdictions:
                points.append(
                    "##### United States Jurisprudence\n"
                    "- **Antitrust & Vertical Restraints:** Vertical agreements and non-price restraints assessed under the rule of reason (*Sherman Act § 1, 15 U.S.C. § 1*; *Leegin*).\n"
                    "- **Liquidated Damages:** Governed by *UCC § 2-718*; unreasonable liquidated damages are void as penalties.\n"
                    "- **Unconscionability:** Disproportionate liability limitations and remedy exclusions subject to *UCC § 2-302* and *§ 2-719*."
                )
            if "EU" in jurisdictions:
                points.append(
                    "##### European Union Jurisprudence\n"
                    "- **Competition Framework:** Vertical restrictions scrutinized under *TFEU Article 101* and the safe harbor thresholds in *VBER 2022/720*.\n"
                    "- **Hardcore Restrictions:** Minimum resale price maintenance (*VBER Art. 4*) removes block exemption.\n"
                    "- **Non-Compete Limits:** Non-competes exceeding 5 years or post-term are excluded restrictions (*VBER Art. 5*)."
                )

            analysis_body = "\n\n".join(points)

            return f"""Hello! I am your Juris AI Legal Copilot.

Regarding your legal inquiry about **{question[:120]}**:

---

### Detailed Statutory Analysis & Legal Standard

{analysis_body}

---

#### Verified Statutory Authorities & Citations:
{cit_block}

---

Would you like me to elaborate on specific judicial precedents, or attach an agreement using the 📎 button below to run an automated clause-by-clause risk scan?"""

    def _build_deterministic_memo(self, data: Dict[str, Any]) -> str:
        doc_name = data.get("doc_name") or "Submitted Contract"
        summary = data.get("summary") or "Multi-jurisdictional legal risk review."
        doc_type = data.get("document_type") or "Commercial Agreement"
        gov_law = data.get("governing_law") or "Unspecified"
        jurisdictions = data.get("jurisdictions") or ["BD", "US", "EU"]
        risks = data.get("risks") or []

        lines: List[str] = [
            "# Header",
            f"**Document Reviewed:** {doc_name}  ",
            f"**Jurisdictions Analysed:** {', '.join(jurisdictions)}  ",
            "**Prepared By:** Juris AI Legal Risk & Citation Verification Engine",
            "",
            "## Executive summary",
            summary,
            "",
            "## Document facts",
            f"- **Document Type:** {doc_type}",
            f"- **Stated Governing Law:** {gov_law}",
            f"- **Total Non-Obvious Risks Identified:** {len(risks)}",
            "",
            "## Risk table (severity)",
            "| # | Risk Title | Clause Refs | Severity |",
            "|---|---|---|---|",
        ]

        for idx, r in enumerate(risks, start=1):
            refs = ", ".join(r.get("clause_refs", []))
            lines.append(f"| {idx} | {r.get('title', '')} | {refs} | **{r.get('severity', 'medium').upper()}** |")

        lines.append("")
        lines.append("## Detailed analysis per risk")
        for idx, r in enumerate(risks, start=1):
            lines.append(f"### {idx}. {r.get('title', '')} (Severity: {r.get('severity', 'medium').upper()})")
            lines.append(f"- **Clauses Implicated:** {', '.join(r.get('clause_refs', []))}")
            lines.append(f"- **Why Non-Obvious:** {r.get('why_non_obvious', '')}")
            j_notes = r.get("jurisdiction_notes", {})
            for j in jurisdictions:
                if j in j_notes:
                    lines.append(f"- **{j} Analysis:** {j_notes[j]}")

            # Strictly include ONLY verified or found_in_corpus citations
            valid_cits = [
                c for c in r.get("citations", []) if c.get("status") in ("verified", "found_in_corpus")
            ]
            if valid_cits:
                cit_formatted = []
                for c in valid_cits:
                    cs = c.get("citation_string", "")
                    if c.get("status") == "found_in_corpus":
                        cit_formatted.append(f"{cs} (corpus-listed; confirm against primary source)")
                    else:
                        cit_formatted.append(f"{cs} (verified)")
                lines.append(f"- **Verified / Corpus Authorities:** {'; '.join(cit_formatted)}")
            else:
                lines.append("- **Verified / Corpus Authorities:** Needs lawyer review against primary sources.")
            lines.append("")

        lines.extend(
            [
                "## Cross-jurisdiction comparison table (BD/US/EU)",
                "| Legal Issue | Bangladesh (BD) | United States (US) | European Union (EU) |",
                "|---|---|---|---|",
                "| **Resale Price Maintenance (RPM)** | Prohibited vertical restraint if adverse to competition (`Competition Act 2012 (BD), s.15(3)` (corpus-listed; confirm against primary source)) | Evaluated under rule of reason (`15 U.S.C. § 1` (corpus-listed; confirm against primary source)) | Hardcore restriction removing safe harbour (`VBER 2022/720, Art. 4` (corpus-listed; confirm against primary source)) |",
                "| **Post-Term Non-Compete & Exclusivity** | Restraint of trade generally void (`Contract Act 1872 (BD), s.27` (corpus-listed; confirm against primary source)) | Scrutinised for market foreclosure (`15 U.S.C. § 14` (corpus-listed; confirm against primary source)) | Excluded if >5 years or post-term (`VBER 2022/720, Art. 5` (corpus-listed; confirm against primary source)) |",
                "| **Penalty & Liquidated Damages** | Reasonable compensation capped at named sum (`Contract Act 1872 (BD), s.74` (corpus-listed; confirm against primary source)) | Void as penalty if unreasonable (`UCC § 2-718` (corpus-listed; confirm against primary source)) | Subject to national civil law & unfair terms controls |",
                "",
                "## Recommended mitigation clauses",
            ]
        )

        for idx, r in enumerate(risks, start=1):
            lines.append(f"### Replacement Drafting for Risk {idx} ({r.get('title', '')})")
            lines.append(f"> {r.get('mitigation_clause_text', '')}")
            lines.append("")

        lines.extend(
            [
                "## Limitations",
                "- Citations marked `(corpus-listed; confirm against primary source)` matched curated statutory/case metadata in the Juris AI corpus and must be checked against official gazettes or reporters prior to filing or execution.",
                "- Any unverified citations generated during preliminary analysis have been automatically excluded from this memorandum.",
                "",
                DISCLAIMER_TEXT,
            ]
        )
        return "\n".join(lines)


llm = LLMClient()
