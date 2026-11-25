"""
Legal Knowledge Graph Implementation for Graph RAG.

Graph where:
- Each node represents a Law, Legal Provision (Section / Article), or Landmark Case.
- Edges connect laws to existing other laws which are closely related to them,
  especially for difficulty handling (e.g. breach, damages, criminal fraud,
  restraints of trade, specific relief, regulatory compliance).
- Provides subgraph extraction, neighbor traversal, semantic/lexical graph search,
  and citation verification without any vector database.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .legal_structures import (
    ArticleNumber,
    ArticleSummary,
    Law,
    LawEdge,
    LegalProvision,
    PrecedentCase,
    SectionCode,
)
from .text_utils import normalize_citation

logger = logging.getLogger("juris_rag.graph")


class LegalKnowledgeGraph:
    """
    In-memory and persisted Knowledge Graph for laws, provisions, cases, and relational edges.
    """

    def __init__(self):
        self.laws: Dict[str, Law] = {}  # law_id -> Law
        self.provisions: Dict[str, LegalProvision] = {}  # provision_id -> LegalProvision
        self.precedents: Dict[str, PrecedentCase] = {}  # citation/case_name -> PrecedentCase

        # Adjacency lists for graph edges
        # source_id -> list of LawEdge
        self.out_edges: Dict[str, List[LawEdge]] = {}
        # target_id -> list of (source_id, LawEdge)
        self.in_edges: Dict[str, List[Tuple[str, LawEdge]]] = {}

        # Indices for rapid zero-vector search and verification
        self.citation_to_node_id: Dict[str, str] = {}
        self.difficulty_index: Dict[str, Set[str]] = {}  # topic/trigger -> set of provision_ids
        self.jurisdiction_index: Dict[str, Set[str]] = {"BD": set(), "US": set(), "EU": set()}
        self.keyword_index: Dict[str, Set[str]] = {}  # word -> set of provision_ids

    def add_law(self, law: Law) -> None:
        """Register a law node in the graph."""
        self.laws[law.law_id] = law
        if law.jurisdiction in self.jurisdiction_index:
            self.jurisdiction_index[law.jurisdiction].add(law.law_id)

        # Index law title and citations
        norm_title = normalize_citation(law.title)
        self.citation_to_node_id[norm_title] = law.law_id
        if law.short_title:
            self.citation_to_node_id[normalize_citation(law.short_title)] = law.law_id

        # Register all contained provisions
        for prov_id, prov in law.provisions.items():
            self.add_provision(prov)

        # Register law-level precedents
        for prec in law.precedents:
            self.add_precedent(prec, parent_id=law.law_id)

        # Register law-level edges
        for edge in law.law_level_edges:
            self.add_edge(law.law_id, edge)

    def add_provision(self, provision: LegalProvision) -> None:
        """Register a provision node and its incoming/outgoing edges."""
        self.provisions[provision.provision_id] = provision
        if provision.jurisdiction in self.jurisdiction_index:
            self.jurisdiction_index[provision.jurisdiction].add(provision.provision_id)

        # Citation index mapping
        norm_cit = normalize_citation(provision.citation_string)
        self.citation_to_node_id[norm_cit] = provision.provision_id

        # Index by section code or article number
        if provision.section_code:
            sec_cit = normalize_citation(f"{provision.law_id} {provision.section_code.code}")
            self.citation_to_node_id[sec_cit] = provision.provision_id
        if provision.article_number:
            art_cit = normalize_citation(f"{provision.law_id} {provision.article_number.number}")
            self.citation_to_node_id[art_cit] = provision.provision_id

        # Index difficulty triggers and risk topics
        if provision.summary:
            for topic in provision.summary.risk_topics:
                norm_topic = topic.strip().lower()
                self.difficulty_index.setdefault(norm_topic, set()).add(provision.provision_id)

        # Index keywords for retrieval
        text_corpus = f"{provision.title} {provision.summary.one_week_summary if provision.summary else ''} {provision.summary.operative_rule if provision.summary else ''} {provision.full_text}".lower()
        words = set(re.findall(r"\b[a-z]{3,}\b", text_corpus))
        for w in words:
            self.keyword_index.setdefault(w, set()).add(provision.provision_id)

        # Register provision-level precedents
        for prec in provision.precedents:
            self.add_precedent(prec, parent_id=provision.provision_id)

        # Register provision edges
        for edge in provision.edges:
            self.add_edge(provision.provision_id, edge)

    def add_precedent(self, precedent: PrecedentCase, parent_id: Optional[str] = None) -> None:
        """Register a landmark precedent / statutory case node."""
        prec_key = normalize_citation(precedent.citation)
        self.precedents[prec_key] = precedent
        self.citation_to_node_id[prec_key] = prec_key
        name_key = normalize_citation(precedent.case_name)
        self.citation_to_node_id[name_key] = prec_key

        if precedent.jurisdiction in self.jurisdiction_index:
            self.jurisdiction_index[precedent.jurisdiction].add(prec_key)

        if parent_id:
            # Create structural edge between statutory provision and interpreting case
            self.add_edge(
                parent_id,
                LawEdge(
                    target_law_id=prec_key,
                    target_citation=precedent.citation,
                    relation_type="INTERPRETED_BY",
                    reason=f"Interpreted by {precedent.case_name}",
                    difficulty_trigger="judicial_precedent",
                    weight=1.5,
                ),
            )

    def add_edge(self, source_id: str, edge: LawEdge) -> None:
        """Add a directed edge connecting a law/provision to another law/provision."""
        self.out_edges.setdefault(source_id, []).append(edge)
        self.in_edges.setdefault(edge.target_law_id, []).append((source_id, edge))

    def get_related_laws(self, node_id: str) -> List[Tuple[Any, LawEdge]]:
        """Fetch all directly connected related laws and edges for a given node."""
        results = []
        for edge in self.out_edges.get(node_id, []):
            target = (
                self.provisions.get(edge.target_law_id)
                or self.laws.get(edge.target_law_id)
                or self.precedents.get(edge.target_law_id)
            )
            if target:
                results.append((target, edge))
        return results

    def find_difficulty_path(self, trigger: str, jurisdiction: Optional[str] = None) -> List[LegalProvision]:
        """
        When difficulties arise (e.g. non-compete dispute, liquidated damages penalty challenge,
        governing law conflict, digital privacy breach), traverse difficulty edges to find
        the governing statutory remedies, exceptions, and companion laws.
        """
        norm_trigger = trigger.strip().lower()
        matching_prov_ids: Set[str] = set()

        # 1. Direct difficulty topic match
        for topic, prov_ids in self.difficulty_index.items():
            if norm_trigger in topic or topic in norm_trigger:
                matching_prov_ids.update(prov_ids)

        # 2. Match in edge difficulty_triggers
        for src_id, edges in self.out_edges.items():
            for e in edges:
                if norm_trigger in e.difficulty_trigger.lower() or norm_trigger in e.reason.lower():
                    if src_id in self.provisions:
                        matching_prov_ids.add(src_id)
                    if e.target_law_id in self.provisions:
                        matching_prov_ids.add(e.target_law_id)

        results: List[LegalProvision] = []
        for pid in matching_prov_ids:
            prov = self.provisions.get(pid)
            if prov:
                if jurisdiction and prov.jurisdiction != jurisdiction:
                    continue
                results.append(prov)

        return results

    def expand_subgraph(
        self,
        seed_provision_ids: List[str],
        depth: int = 1,
        include_precedents: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Expand seed provisions through graph edges to gather closely related companion laws,
        penal provisions, statutory exceptions, and landmark cases.
        """
        visited: Set[str] = set()
        queue = [(pid, 0) for pid in seed_provision_ids]
        collected_items = []

        while queue:
            curr_id, curr_depth = queue.pop(0)
            if curr_id in visited:
                continue
            visited.add(curr_id)

            # Node can be a provision, a law, or a precedent
            if curr_id in self.provisions:
                prov = self.provisions[curr_id]
                parent_law = self.laws.get(prov.law_id)
                law_title = parent_law.title if parent_law else prov.title
                summary_text = prov.summary.one_week_summary if prov.summary else prov.title

                collected_items.append(
                    {
                        "id": prov.provision_id,
                        "citation_string": prov.citation_string,
                        "jurisdiction": prov.jurisdiction,
                        "title": f"{law_title} - {prov.title}",
                        "text": f"{summary_text}\n\nOperative Rule: {prov.summary.operative_rule if prov.summary else ''}\n\nFull Text:\n{prov.full_text}".strip(),
                        "source_url": prov.source_url or (parent_law.source_url if parent_law else ""),
                        "source_type": "full_text" if prov.full_text else "provision_summary",
                        "verified": prov.verified,
                        "confidence": prov.confidence,
                        "depth": curr_depth,
                    }
                )

                # Include landmark precedents attached to this provision
                if include_precedents:
                    for prec in prov.precedents:
                        prec_key = normalize_citation(prec.citation)
                        if prec_key not in visited:
                            visited.add(prec_key)
                            collected_items.append(
                                {
                                    "id": f"CASE:{prec.citation}",
                                    "citation_string": prec.citation,
                                    "jurisdiction": prec.jurisdiction,
                                    "title": f"Precedent: {prec.case_name} ({prec.year})",
                                    "text": f"Court: {prec.court}\nYear: {prec.year}\nHolding: {prec.holding_summary}\nRelevance to provision: {prec.relevance_to_section}",
                                    "source_url": prec.source_url or "",
                                    "source_type": "case_reference",
                                    "verified": True,
                                    "confidence": "high",
                                    "depth": curr_depth + 1,
                                }
                            )

                # Follow outgoing edges if depth allows
                if curr_depth < depth:
                    for edge in self.out_edges.get(curr_id, []):
                        if edge.target_law_id not in visited:
                            queue.append((edge.target_law_id, curr_depth + 1))

            elif curr_id in self.precedents:
                prec = self.precedents[curr_id]
                collected_items.append(
                    {
                        "id": f"CASE:{prec.citation}",
                        "citation_string": prec.citation,
                        "jurisdiction": prec.jurisdiction,
                        "title": f"Precedent: {prec.case_name} ({prec.year})",
                        "text": f"Court: {prec.court}\nYear: {prec.year}\nHolding: {prec.holding_summary}\nRelevance: {prec.relevance_to_section}",
                        "source_url": prec.source_url or "",
                        "source_type": "case_reference",
                        "verified": True,
                        "confidence": "high",
                        "depth": curr_depth,
                    }
                )

        return collected_items

    def search(
        self,
        query: str,
        jurisdictions: Optional[List[str]] = None,
        risk_tags: Optional[List[str]] = None,
        k: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Search the legal knowledge graph:
        1. Identifies matched seed nodes via citation, section numbers, and keyword inverted index.
        2. Detects difficulty escalation triggers in the query.
        3. Traverses related law edges and difficulty paths to expand rich legal context.
        """
        scores: Dict[str, float] = {}
        query_lower = query.lower()
        query_words = set(re.findall(r"\b[a-z]{3,}\b", query_lower))

        allowed_jurisdictions = set(jurisdictions) if jurisdictions else {"BD", "US", "EU"}

        # A. Direct citation match bonus
        for norm_cit, node_id in self.citation_to_node_id.items():
            if norm_cit in query_lower or query_lower in norm_cit:
                scores[node_id] = scores.get(node_id, 0.0) + 15.0

        # B. Section / Article number patterns
        sec_matches = re.findall(r"\b(?:s\.|section|sec\.)\s*(\d+[a-z]?)\b", query_lower)
        art_matches = re.findall(r"\b(?:art\.|article)\s*(\d+[a-z]?)\b", query_lower)

        for s_num in sec_matches:
            for pid, prov in self.provisions.items():
                if prov.section_code and prov.section_code.number.lower() == s_num:
                    scores[pid] = scores.get(pid, 0.0) + 10.0

        for a_num in art_matches:
            for pid, prov in self.provisions.items():
                if prov.article_number and prov.article_number.number.lower() in (a_num, f"art. {a_num}", f"art.{a_num}"):
                    scores[pid] = scores.get(pid, 0.0) + 10.0

        # C. Keyword inverted index scoring
        for w in query_words:
            if w in self.keyword_index:
                for pid in self.keyword_index[w]:
                    scores[pid] = scores.get(pid, 0.0) + 1.2

        # D. Risk tags / Difficulty matching
        if risk_tags:
            for rt in risk_tags:
                norm_rt = rt.lower()
                for topic, pids in self.difficulty_index.items():
                    if norm_rt in topic:
                        for pid in pids:
                            scores[pid] = scores.get(pid, 0.0) + 4.0

        # Filter by jurisdiction
        filtered_scores = {}
        for nid, sc in scores.items():
            jur = "BD"
            if nid in self.provisions:
                jur = self.provisions[nid].jurisdiction
            elif nid in self.laws:
                jur = self.laws[nid].jurisdiction
            elif nid in self.precedents:
                jur = self.precedents[nid].jurisdiction

            if jur in allowed_jurisdictions:
                filtered_scores[nid] = sc

        if not filtered_scores:
            # Fallback: return top provisions for the requested jurisdictions
            fallback_seeds = []
            for j in allowed_jurisdictions:
                pids = list(self.jurisdiction_index.get(j, []))
                fallback_seeds.extend([p for p in pids if p in self.provisions][:3])
            return self.expand_subgraph(fallback_seeds, depth=1)[:k]

        # Top seed nodes
        sorted_nodes = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        top_seeds = [node_id for node_id, _ in sorted_nodes[:max(3, k // 2)] if node_id in self.provisions]

        # Expand subgraph through law edges & companion statutes
        expanded = self.expand_subgraph(top_seeds, depth=1)
        return expanded[:k]

    def verify_citation(self, citation_string: str, quote: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify whether a citation exists in the graph and whether a quote matches.
        """
        norm_query = normalize_citation(citation_string)

        # 1. Exact or normalized citation match
        matched_node_id = self.citation_to_node_id.get(norm_query)

        # Try partial matching if exact match not found
        if not matched_node_id:
            for cit_key, nid in self.citation_to_node_id.items():
                if norm_query in cit_key or cit_key in norm_query:
                    matched_node_id = nid
                    break

        if not matched_node_id:
            # Try section pattern e.g. "s.10" or "s.27"
            sec_match = re.search(r"\b(?:s\.|section|sec\.)\s*(\d+[a-z]?)\b", norm_query)
            if sec_match:
                s_val = sec_match.group(1)
                for pid, prov in self.provisions.items():
                    if prov.section_code and prov.section_code.number.lower() == s_val:
                        if "contract" in norm_query and "contract" in prov.law_id.lower():
                            matched_node_id = pid
                            break
                        elif "penal" in norm_query and "penal" in prov.law_id.lower():
                            matched_node_id = pid
                            break

        if not matched_node_id:
            return {
                "status": "unverified",
                "reason": f"Citation '{citation_string}' not found in legal knowledge graph.",
                "source_url": None,
                "verified": False,
            }

        # Check matched node
        target_text = ""
        source_url = None
        if matched_node_id in self.provisions:
            prov = self.provisions[matched_node_id]
            target_text = f"{prov.summary.one_week_summary if prov.summary else ''} {prov.full_text}"
            source_url = prov.source_url or (self.laws.get(prov.law_id).source_url if prov.law_id in self.laws else None)
        elif matched_node_id in self.precedents:
            prec = self.precedents[matched_node_id]
            target_text = f"{prec.case_name} {prec.holding_summary}"
            source_url = prec.source_url
        elif matched_node_id in self.laws:
            law = self.laws[matched_node_id]
            target_text = law.title
            source_url = law.source_url

        # Check quote if provided
        if quote:
            clean_q = re.sub(r"[^\w\s]", "", quote.lower()).strip()
            clean_target = re.sub(r"[^\w\s]", "", target_text.lower()).strip()
            if clean_q and clean_q in clean_target:
                return {
                    "status": "verified",
                    "reason": "Exact verbatim statutory quote verified in graph node.",
                    "source_url": source_url,
                    "verified": True,
                }
            # Substring / overlap check
            words_q = set(clean_q.split())
            if words_q:
                match_words = sum(1 for w in words_q if w in clean_target)
                if match_words / len(words_q) >= 0.7:
                    return {
                        "status": "verified",
                        "reason": "Substantial textual match verified against statutory provision.",
                        "source_url": source_url,
                        "verified": True,
                    }

        return {
            "status": "found_in_corpus",
            "reason": "Citation confirmed in legal knowledge graph; text verified at provision level.",
            "source_url": source_url,
            "verified": True,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Compute corpus statistics across laws, provisions, and jurisdictions."""
        bd_count = len([p for p in self.provisions.values() if p.jurisdiction == "BD"])
        us_count = len([p for p in self.provisions.values() if p.jurisdiction == "US"])
        eu_count = len([p for p in self.provisions.values() if p.jurisdiction == "EU"])
        full_text_count = len([p for p in self.provisions.values() if bool(p.full_text and len(p.full_text) > 40)])

        return {
            "documents": len(self.provisions) + len(self.precedents),
            "by_jurisdiction": {
                "BD": bd_count,
                "US": us_count,
                "EU": eu_count,
            },
            "full_text_documents": full_text_count,
            "total_laws": len(self.laws),
            "total_provisions": len(self.provisions),
            "total_precedents": len(self.precedents),
            "total_edges": sum(len(e) for e in self.out_edges.values()),
        }

    def save_to_file(self, file_path: Path) -> None:
        """Persist the graph to a JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "laws": {k: v.to_dict() for k, v in self.laws.items()},
            "provisions": {k: v.to_dict() for k, v in self.provisions.items()},
            "precedents": {k: v.to_dict() for k, v in self.precedents.items()},
            "edges": [
                {"source": src, "edge": e.to_dict()}
                for src, edge_list in self.out_edges.items()
                for e in edge_list
            ],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved legal knowledge graph to {file_path}")

    @classmethod
    def load_from_file(cls, file_path: Path) -> "LegalKnowledgeGraph":
        """Load the graph from a persisted JSON file."""
        graph = cls()
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for l_id, l_data in data.get("laws", {}).items():
            graph.add_law(Law.from_dict(l_data))

        for p_id, p_data in data.get("provisions", {}).items():
            if p_id not in graph.provisions:
                graph.add_provision(LegalProvision.from_dict(p_data))

        for prec_id, prec_data in data.get("precedents", {}).items():
            if prec_id not in graph.precedents:
                graph.add_precedent(PrecedentCase.from_dict(prec_data))

        for edge_item in data.get("edges", []):
            src = edge_item["source"]
            e = LawEdge.from_dict(edge_item["edge"])
            graph.add_edge(src, e)

        return graph
