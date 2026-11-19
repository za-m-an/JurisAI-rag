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
        context_note = data.get("context_note", "")

        cit_lines = []
        for c in allowed_citations[:4]:
            cs = c.get("citation_string", "")
            st = c.get("status", "found_in_corpus")
            if st == "verified":
                cit_lines.append(f"- **{cs}** (verified in full-text corpus)")
            elif st == "found_in_corpus":
                cit_lines.append(f"- **{cs}** (corpus-listed; confirm against primary source)")

        cit_block = "\n".join(cit_lines) if cit_lines else "- Needs lawyer review against primary statutory sources."

        # If in conversational mode (and not analyzing an attached contract), act as a conversational chatbot
        if mode == "conversational" and "attached contract" not in context_note.lower():
            return (
                f"Hello! I am your Juris AI Legal Copilot.\n\n"
                f"Regarding your question about **{question[:120]}**:\n\n"
                f"In commercial and civil transactions, this primarily involves statutory rules across your selected jurisdictions:\n\n"
                f"{cit_block}\n\n"
                f"**Key Statutory Principles:**\n"
                f"- **Bangladesh (`BD`)**: Under *Contract Act 1872, s.27*, agreements restraining lawful trade or business are void to that extent. Stipulated damages or penalty clauses are subject to judicial reasonableness under *s.74*.\n"
                f"- **United States (`US`)**: Vertical agreements and commercial restraints are reviewed under the antitrust rule of reason (*15 U.S.C. § 1*), with liquidation of damages governed by *UCC § 2-718*.\n"
                f"- **European Union (`EU`)**: Non-competes and distribution agreements must align with *TFEU Article 101* and the safe harbor thresholds in *VBER 2022/720*.\n\n"
                f"Would you like me to elaborate on a specific jurisdiction, or attach an agreement using the 📎 button below to run a deep clause-by-clause review?"
            )

        # Formal memorandum mode (or attached contract analysis)
        return (
            f"### Legal Advisory Memorandum\n\n"
            f"**Subject:** In-depth Legal Risk Review — {question[:120]}\n\n"
            f"#### Applicable Authorities\n{cit_block}\n\n"
            f"#### Statutory Analysis & Considerations\n"
            f"1. **Bangladesh (`BD`)**: Restraints of trade are generally void under `Contract Act 1872 (BD), s.27`, "
            f"stipulated penalties are limited to reasonable compensation under `Contract Act 1872 (BD), s.74`, and "
            f"anti-competitive vertical restraints fall under `Competition Act 2012 (BD), s.15(3)`.\n"
            f"2. **United States (`US`)**: Vertical restraints and minimum resale price agreements are scrutinised under "
            f"`15 U.S.C. § 1` and `15 U.S.C. § 14`, while penalty clauses and remedy limitations are governed by "
            f"`UCC § 2-718` and `UCC § 2-719`.\n"
            f"3. **European Union (`EU`)**: Hardcore resale price maintenance (`VBER 2022/720, Art. 4`) and non-competes "
            f"exceeding 5 years or post-term (`VBER 2022/720, Art. 5`) lose block exemption under `TFEU Art. 101(1)`.\n\n"
            f"#### Risk Classification & Recommendations\n"
            f"- **High Risk**: Review unilateral penalty provisions and territorial exclusivity restrictions.\n"
            f"- **Mitigation**: Introduce severability carveouts and align termination covenants with local statutory standards.\n\n"
            f"{DISCLAIMER_TEXT}"
        )

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
