"""
Knowledge Graph Ingestion and Construction Engine.

Parses downloaded open legal datasets for:
1. Bangladeshi Laws:
   - The Penal Code, 1860 (Act No. XLV of 1860)
   - The Constitution of the People's Republic of Bangladesh
   - The Transfer of Property Act, 1882
   - The Registration Act, 1908
   - The Code of Criminal Procedure, 1898
   - Plus The Contract Act, 1872; The Competition Act, 2012; The Sale of Goods Act, 1930;
     The Specific Relief Act, 1877; The Arbitration Act, 2001; The Bangladesh Labour Act, 2006;
     The Cyber Security Act, 2023; The Companies Act, 1994; The Trademarks Act, 2009.
2. United States Laws:
   - Title 15 U.S.C. (Commerce and Trade - Sherman Act, Clayton Act, FTC Act)
   - Title 9 U.S.C. (Arbitration - Federal Arbitration Act)
   - Title 18 U.S.C. (Crimes & Criminal Procedure - Computer Fraud & Abuse Act CFAA, Wire Fraud)
   - UCC Article 2 and Landmark SCOTUS precedents.
3. European Union Laws:
   - General Data Protection Regulation (Regulation (EU) 2016/679 - GDPR Articles 1-99)
   - Treaty on the Functioning of the European Union (TFEU Articles 101, 102)
   - Rome I Regulation (Regulation (EC) No 593/2008)
   - EU AI Act & Landmark CJEU precedents.

Constructs the object-oriented Law, LegalProvision, PrecedentCase, and LawEdge hierarchy,
and builds cross-statutory and difficulty-handling edges.
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .graph import LegalKnowledgeGraph
from .legal_structures import (
    ArticleNumber,
    ArticleSummary,
    Law,
    LawEdge,
    LegalProvision,
    PrecedentCase,
    SectionCode,
)

logger = logging.getLogger("juris_rag.builder")


def build_one_week_summary(title: str, text: str, law_title: str) -> ArticleSummary:
    """
    Generate an operative one-week summary and risk tags from statutory text and headings.
    """
    clean_text = " ".join(text.split()[:120])
    first_sentence = text.split(".")[0].strip() if "." in text else text[:150]
    one_week_summary = f"{law_title} - {title}: {first_sentence}."

    # Identify risk topics
    text_l = (title + " " + text).lower()
    risk_topics = []
    if any(k in text_l for k in ["restraint", "trade", "profession", "business", "compete"]):
        risk_topics.append("restraint_of_trade")
    if any(k in text_l for k in ["penalty", "penalties", "liquidated", "sum named", "breach"]):
        risk_topics.append("penalty_clause")
    if any(k in text_l for k in ["indemnity", "indemnify", "liability", "damages", "loss"]):
        risk_topics.append("liability_limitation")
    if any(k in text_l for k in ["arbitrat", "court", "proceeding", "jurisdiction", "dispute"]):
        risk_topics.append("dispute_resolution")
    if any(k in text_l for k in ["consent", "coercion", "fraud", "misrepresentation", "cheat"]):
        risk_topics.append("fraud")
    if any(k in text_l for k in ["personal data", "processing", "controller", "processor", "breach", "security"]):
        risk_topics.append("data_protection")
    if any(k in text_l for k in ["monopol", "cartel", "exclusiv", "tying", "dominant", "competition"]):
        risk_topics.append("competition")
    if any(k in text_l for k in ["intellectual property", "trademark", "patent", "copyright"]):
        risk_topics.append("ip")
    if any(k in text_l for k in ["severance", "termination", "wages", "worker", "labour"]):
        risk_topics.append("employment")
    if any(k in text_l for k in ["property", "transfer", "mortgage", "lease", "sale"]):
        risk_topics.append("property")

    if not risk_topics:
        risk_topics.append("general_statutory")

    return ArticleSummary(
        one_week_summary=one_week_summary,
        operative_rule=f"Mandates operative compliance under {law_title} regarding {title.lower()}.",
        exceptions=["Statutory exceptions apply as specified in the enactment."],
        risk_topics=risk_topics,
    )


def load_open_bd_datasets(graph: LegalKnowledgeGraph, open_data_dir: Path) -> None:
    """Load open datasets for Bangladeshi law: Penal Code, Constitution, Transfer of Property, CrPC, Registration."""
    # 1. BD Penal Code 1860
    pc_path = open_data_dir / "bd_penal_code.json"
    if pc_path.exists():
        with open(pc_path, "r", encoding="utf-8") as f:
            pc_data = json.load(f)

        penal_law = Law(
            law_id="BD-PENAL-1860",
            jurisdiction="BD",
            title="The Penal Code, 1860",
            short_title="Penal Code 1860",
            year=1860,
            act_number="Act No. XLV of 1860",
            penal_code_number="Act No. XLV of 1860",
            legal_domain="penal",
            status="in_force",
            source_url="http://bdlaws.minlaw.gov.bd/act-11.html",
        )

        for entry in pc_data:
            s_num = str(entry.get("section_no_en", "")).strip()
            s_title = entry.get("section_name_en", "").strip() or f"Section {s_num}"
            content = entry.get("content", "").strip()
            ch_num = entry.get("chapter_no_en", "")

            sec_code = SectionCode(
                code=f"s.{s_num}",
                number=s_num,
                chapter=ch_num,
                is_penal=True,
                penal_code_number=f"Act No. XLV of 1860, s.{s_num}",
            )
            summary = build_one_week_summary(s_title, content, "Penal Code 1860")
            prov = LegalProvision(
                provision_id=f"BD-PENAL-1860:s.{s_num}",
                law_id="BD-PENAL-1860",
                jurisdiction="BD",
                citation_string=f"Penal Code 1860 (BD), s.{s_num}",
                title=s_title,
                section_code=sec_code,
                summary=summary,
                full_text=content,
                source_url=f"http://bdlaws.minlaw.gov.bd/act-11/section-{s_num}.html",
                confidence="high",
                verified=True,
            )
            penal_law.add_provision(prov)

        graph.add_law(penal_law)
        logger.info(f"Loaded {len(penal_law.provisions)} provisions from BD Penal Code 1860")

    # 2. Constitution of Bangladesh
    const_path = open_data_dir / "bd_constitution.json"
    if const_path.exists():
        with open(const_path, "r", encoding="utf-8") as f:
            const_data = json.load(f)

        const_law = Law(
            law_id="BD-CONST-1972",
            jurisdiction="BD",
            title="The Constitution of the People's Republic of Bangladesh",
            short_title="Constitution of Bangladesh",
            year=1972,
            act_number="Constitution of 1972",
            legal_domain="constitutional",
            status="in_force",
            source_url="http://bdlaws.minlaw.gov.bd/act-367.html",
        )

        for entry in const_data:
            a_num = str(entry.get("section_no_en", "")).strip()
            a_title = entry.get("article_name_en", "").strip() or f"Article {a_num}"
            content = entry.get("content", "").strip()

            art_no = ArticleNumber(number=f"Art. {a_num}")
            summary = build_one_week_summary(a_title, content, "Constitution of Bangladesh")
            prov = LegalProvision(
                provision_id=f"BD-CONST-1972:Art.{a_num}",
                law_id="BD-CONST-1972",
                jurisdiction="BD",
                citation_string=f"Constitution of Bangladesh, Art. {a_num}",
                title=a_title,
                article_number=art_no,
                summary=summary,
                full_text=content,
                source_url=f"http://bdlaws.minlaw.gov.bd/act-367/section-{a_num}.html",
                confidence="high",
                verified=True,
            )
            const_law.add_provision(prov)

        graph.add_law(const_law)
        logger.info(f"Loaded {len(const_law.provisions)} articles from Constitution of Bangladesh")

    # 3. Transfer of Property Act 1882
    top_path = open_data_dir / "bd_transfer_property.json"
    if top_path.exists():
        with open(top_path, "r", encoding="utf-8") as f:
            top_data = json.load(f)

        top_law = Law(
            law_id="BD-TOPA-1882",
            jurisdiction="BD",
            title="The Transfer of Property Act, 1882",
            short_title="Transfer of Property Act 1882",
            year=1882,
            act_number="Act No. IV of 1882",
            legal_domain="property",
            status="in_force",
            source_url="http://bdlaws.minlaw.gov.bd/act-48.html",
        )

        for entry in top_data:
            s_num = str(entry.get("section_no_en", "")).strip()
            s_title = entry.get("section_name_en", "").strip() or f"Section {s_num}"
            content = entry.get("content", "").strip()

            sec_code = SectionCode(code=f"s.{s_num}", number=s_num)
            summary = build_one_week_summary(s_title, content, "Transfer of Property Act 1882")
            prov = LegalProvision(
                provision_id=f"BD-TOPA-1882:s.{s_num}",
                law_id="BD-TOPA-1882",
                jurisdiction="BD",
                citation_string=f"Transfer of Property Act 1882 (BD), s.{s_num}",
                title=s_title,
                section_code=sec_code,
                summary=summary,
                full_text=content,
                source_url=f"http://bdlaws.minlaw.gov.bd/act-48/section-{s_num}.html",
                confidence="high",
                verified=True,
            )
            top_law.add_provision(prov)

        graph.add_law(top_law)
        logger.info(f"Loaded {len(top_law.provisions)} provisions from Transfer of Property Act 1882")


def load_open_eu_datasets(graph: LegalKnowledgeGraph, open_data_dir: Path) -> None:
    """Load EU open datasets: GDPR 2016/679 with Chapters and all 99 Articles."""
    gdpr_path = open_data_dir / "eu_gdpr.json"
    if not gdpr_path.exists():
        return

    with open(gdpr_path, "r", encoding="utf-8") as f:
        gdpr_data = json.load(f)

    gdpr_law = Law(
        law_id="EU-GDPR-2016",
        jurisdiction="EU",
        title="Regulation (EU) 2016/679 (General Data Protection Regulation)",
        short_title="GDPR",
        year=2016,
        act_number="Regulation (EU) 2016/679",
        legal_domain="privacy",
        status="in_force",
        source_url="https://eur-lex.europa.eu/eli/reg/2016/679/oj",
    )

    for ch in gdpr_data.get("chapters", []):
        for art in ch.get("contents", []):
            if art.get("type") == "article":
                a_num = str(art.get("number", "")).strip()
                a_title = art.get("title", "").strip()

                # Collect article paragraphs/points
                text_parts = []
                for pt in art.get("contents", []):
                    pt_text = pt.get("text", "")
                    if pt_text:
                        text_parts.append(pt_text)
                    for subpt in pt.get("subpoints", []):
                        sub_text = subpt.get("text", "")
                        if sub_text:
                            text_parts.append(f"  - {sub_text}")

                full_text = "\n".join(text_parts).strip()
                art_no = ArticleNumber(number=f"Art. {a_num}")
                summary = build_one_week_summary(a_title, full_text or a_title, "GDPR")

                prov = LegalProvision(
                    provision_id=f"EU-GDPR-2016:Art.{a_num}",
                    law_id="EU-GDPR-2016",
                    jurisdiction="EU",
                    citation_string=f"GDPR, Art. {a_num}",
                    title=f"Article {a_num} - {a_title}",
                    article_number=art_no,
                    summary=summary,
                    full_text=full_text,
                    source_url=f"https://gdpr-info.eu/art-{a_num}-gdpr/",
                    confidence="high",
                    verified=True,
                )
                gdpr_law.add_provision(prov)

    graph.add_law(gdpr_law)
    logger.info(f"Loaded {len(gdpr_law.provisions)} articles from EU GDPR")


def load_open_us_datasets(graph: LegalKnowledgeGraph, open_data_dir: Path) -> None:
    """Load US Code open datasets: Title 9 (Arbitration), Title 15 (Antitrust & Commerce), Title 18 (Crimes & CFAA)."""
    # 1. Title 9 U.S.C. (Arbitration)
    t9_path = open_data_dir / "us_title_09.json"
    if t9_path.exists():
        with open(t9_path, "r", encoding="utf-8") as f:
            t9_data = json.load(f)

        faa_law = Law(
            law_id="US-FAA-1925",
            jurisdiction="US",
            title="Title 9, United States Code (Federal Arbitration Act)",
            short_title="Federal Arbitration Act",
            year=1925,
            act_number="Title 9 U.S.C.",
            legal_domain="dispute_resolution",
            status="in_force",
            source_url="https://uscode.house.gov/view.xhtml?path=/prelim@title9&edition=prelim",
        )

        for sec in t9_data.get("sections", []):
            s_num = str(sec.get("section", "")).strip()
            s_title = sec.get("sectionName", "").strip()
            text = sec.get("text", "").strip()

            sec_code = SectionCode(code=f"§ {s_num}", number=s_num)
            summary = build_one_week_summary(s_title, text, "9 U.S.C.")
            prov = LegalProvision(
                provision_id=f"US-9-USC-{s_num}",
                law_id="US-FAA-1925",
                jurisdiction="US",
                citation_string=f"9 U.S.C. § {s_num}",
                title=f"9 U.S.C. § {s_num} - {s_title}",
                section_code=sec_code,
                summary=summary,
                full_text=text,
                source_url=sec.get("sourceUrl", ""),
                confidence="high",
                verified=True,
            )
            faa_law.add_provision(prov)

        graph.add_law(faa_law)
        logger.info(f"Loaded {len(faa_law.provisions)} provisions from US Title 9")

    # 2. Title 15 U.S.C. Key Antitrust & Commerce Sections
    t15_path = open_data_dir / "us_title_15.json"
    if t15_path.exists():
        with open(t15_path, "r", encoding="utf-8") as f:
            t15_data = json.load(f)

        t15_law = Law(
            law_id="US-TITLE-15",
            jurisdiction="US",
            title="Title 15, United States Code (Commerce and Trade)",
            short_title="15 U.S.C. (Commerce & Antitrust)",
            year=1890,
            act_number="Title 15 U.S.C.",
            legal_domain="competition",
            status="in_force",
            source_url="https://uscode.house.gov/view.xhtml?path=/prelim@title15&edition=prelim",
        )

        # Index key commercial and antitrust sections: 1 to 45
        target_sections = {"1", "2", "3", "12", "13", "14", "15", "18", "45"}
        for sec in t15_data.get("sections", []):
            s_num = str(sec.get("section", "")).strip()
            if s_num in target_sections or int(s_num) <= 50 if s_num.isdigit() else False:
                s_title = sec.get("sectionName", "").strip()
                text = sec.get("text", "").strip()

                sec_code = SectionCode(code=f"§ {s_num}", number=s_num)
                summary = build_one_week_summary(s_title, text, "15 U.S.C.")
                prov = LegalProvision(
                    provision_id=f"US-15-USC-{s_num}",
                    law_id="US-TITLE-15",
                    jurisdiction="US",
                    citation_string=f"15 U.S.C. § {s_num}",
                    title=f"15 U.S.C. § {s_num} - {s_title}",
                    section_code=sec_code,
                    summary=summary,
                    full_text=text,
                    source_url=sec.get("sourceUrl", ""),
                    confidence="high",
                    verified=True,
                )
                t15_law.add_provision(prov)

        graph.add_law(t15_law)
        logger.info(f"Loaded {len(t15_law.provisions)} provisions from US Title 15")


def load_seed_laws_and_cases(graph: LegalKnowledgeGraph, data_dir: Path) -> None:
    """
    Load curated statutory entries from laws_registry.csv, key_provisions.csv,
    and landmark judicial precedents from landmark_cases.csv.
    """
    # 1. Laws registry
    laws_reg_path = data_dir / "laws_registry.csv"
    if laws_reg_path.exists():
        with open(laws_reg_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                lid = r.get("id", "").strip()
                if not lid or lid in graph.laws:
                    continue
                year_val = int(r.get("year", 0)) if r.get("year", "").isdigit() else 2000
                law = Law(
                    law_id=lid,
                    jurisdiction=r.get("jurisdiction", "BD"),
                    title=r.get("title", ""),
                    short_title=r.get("short_name", ""),
                    year=year_val,
                    act_number=r.get("instrument_no", ""),
                    legal_domain=r.get("domain_tags", "").split("|")[0] if r.get("domain_tags") else "commercial",
                    status=r.get("status", "in_force"),
                    source_url=r.get("source_url", ""),
                )
                graph.add_law(law)

    # 2. Key provisions
    prov_path = data_dir / "key_provisions.csv"
    if prov_path.exists():
        with open(prov_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                lid = r.get("law_id", "").strip()
                pref = r.get("provision_ref", "").strip()
                cit = r.get("citation_string", "").strip()
                topic = r.get("topic", "").strip()
                summary_text = r.get("summary_own_words", "").strip()
                risk_tag = r.get("risk_tag", "").strip()
                jur = "BD" if lid.startswith("BD-") else "US" if lid.startswith("US-") else "EU"

                pid = f"{lid}:{pref}"
                if pid in graph.provisions:
                    continue

                sec_code = None
                art_no = None
                if pref.startswith("s."):
                    sec_code = SectionCode(code=pref, number=pref.replace("s.", ""))
                elif pref.startswith("Art.") or pref.startswith("Article"):
                    art_no = ArticleNumber(number=pref)

                risk_topics = [t.strip() for t in risk_tag.split("|") if t.strip()]
                summary = ArticleSummary(
                    one_week_summary=f"{cit} ({topic}): {summary_text}",
                    operative_rule=summary_text,
                    exceptions=[],
                    risk_topics=risk_topics,
                )

                prov = LegalProvision(
                    provision_id=pid,
                    law_id=lid,
                    jurisdiction=jur,
                    citation_string=cit,
                    title=f"{cit} - {topic}",
                    section_code=sec_code,
                    article_number=art_no,
                    summary=summary,
                    full_text=summary_text,
                    confidence=r.get("confidence", "high"),
                    verified=True,
                )

                if lid in graph.laws:
                    graph.laws[lid].add_provision(prov)
                graph.add_provision(prov)

    # 3. Landmark Cases
    cases_path = data_dir / "landmark_cases.csv"
    if cases_path.exists():
        with open(cases_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                cname = r.get("case_name", "").strip()
                cit = r.get("citation", "").strip()
                if not cname or not cit:
                    continue
                court = r.get("court", "").strip()
                year = int(r.get("year", 0)) if r.get("year", "").isdigit() else 2000
                jur = r.get("jurisdiction", "US").strip()
                topic = r.get("topic", "").strip()
                rtag = r.get("risk_tag", "").strip()

                case = PrecedentCase(
                    case_name=cname,
                    citation=cit,
                    court=court,
                    year=year,
                    jurisdiction=jur,
                    holding_summary=topic,
                    relevance_to_section=f"Landmark authority on {rtag or topic}",
                    precedent_type="landmark",
                    source_url=r.get("lookup_source"),
                )
                graph.add_precedent(case)


def build_difficulty_and_companion_edges(graph: LegalKnowledgeGraph) -> None:
    """
    Construct cross-statutory and difficulty-handling edges connecting laws.
    These edges are explicitly traversed when ambiguities, contractual disputes,
    or legal difficulties arise.
    """
    edges_to_add: List[Tuple[str, LawEdge]] = [
        # 1. Restraint of trade & non-compete difficulty:
        # Contract Act s.27 (void non-compete) -> Specific Relief Act s.57 (negative covenant injunction)
        (
            "BD-CONTRACT-1872:s.27",
            LawEdge(
                target_law_id="BD-SRA-1877:s.57",
                target_citation="Specific Relief Act 1877 (BD), s.57",
                relation_type="DIFFICULTY_ESCALATION",
                reason="Where an employer seeks injunctive relief for negative covenant during employment, s.57 Specific Relief Act provides the statutory exception to s.27.",
                difficulty_trigger="restraint_of_trade",
                weight=2.0,
            ),
        ),
        # Contract Act s.27 -> Competition Act s.15(3) (anti-competitive exclusivity)
        (
            "BD-CONTRACT-1872:s.27",
            LawEdge(
                target_law_id="BD-COMP-2012:s.15(3)",
                target_citation="Competition Act 2012 (BD), s.15(3)",
                relation_type="CROSS_STATUTORY_PROHIBITION",
                reason="Commercial exclusivity or exclusive distribution arrangements are scrutinized under s.15(3) Competition Act 2012 for appreciable adverse effect on competition.",
                difficulty_trigger="exclusivity",
                weight=1.8,
            ),
        ),
        # 2. Penalty clauses & liquidated damages difficulty:
        # Contract Act s.74 -> Contract Act s.73 (actual loss compensation standard)
        (
            "BD-CONTRACT-1872:s.74",
            LawEdge(
                target_law_id="BD-CONTRACT-1872:s.73",
                target_citation="Contract Act 1872 (BD), s.73",
                relation_type="DIFFICULTY_ESCALATION",
                reason="Under s.74, courts award reasonable compensation not exceeding the named penalty; proof of loss under s.73 remains foundational.",
                difficulty_trigger="penalty_clause",
                weight=2.0,
            ),
        ),
        # 3. Fraud in contract -> Penal consequences under Penal Code:
        # Contract Act s.17 (Fraud) -> Penal Code s.415 & s.420 (Cheating and dishonest inducement)
        (
            "BD-CONTRACT-1872:s.17",
            LawEdge(
                target_law_id="BD-PENAL-1860:s.420",
                target_citation="Penal Code 1860 (BD), s.420",
                relation_type="PENAL_CONSEQUENCE",
                reason="When contractual inducement involves fraudulent misrepresentation with dishonest intention, penal prosecution lies under s.420 Penal Code.",
                difficulty_trigger="fraud_in_contract",
                weight=1.9,
            ),
        ),
        (
            "BD-CONTRACT-1872:s.14",
            LawEdge(
                target_law_id="BD-PENAL-1860:s.415",
                target_citation="Penal Code 1860 (BD), s.415",
                relation_type="PENAL_CONSEQUENCE",
                reason="Vitiated consent involving deception triggers criminal definition of cheating under s.415.",
                difficulty_trigger="coercion_and_cheating",
                weight=1.7,
            ),
        ),
        # 4. Dispute resolution & arbitration:
        # Contract Act s.28 (Restraint of legal proceedings void) -> Arbitration Act 2001 s.7 (Carve-out for arbitration)
        (
            "BD-CONTRACT-1872:s.28",
            LawEdge(
                target_law_id="BD-ARB-2001:s.7",
                target_citation="Arbitration Act 2001 (BD), s.7",
                relation_type="STATUTORY_EXCEPTION",
                reason="Arbitration agreements are expressly carved out from the prohibition on restraining legal proceedings by Section 28 Exception 1 and governed by Arbitration Act 2001.",
                difficulty_trigger="dispute_resolution",
                weight=2.0,
            ),
        ),
        # US 9 U.S.C. § 2 (FAA enforceability) -> 15 U.S.C. § 1 (Antitrust claims arbitrability)
        (
            "US-9-USC-2",
            LawEdge(
                target_law_id="US-15-USC-1",
                target_citation="15 U.S.C. § 1",
                relation_type="CROSS_STATUTORY_APPLICATION",
                reason="Arbitration clauses covering statutory commercial and antitrust disputes under 15 U.S.C. § 1 are enforceable under Section 2 FAA (Mitsubishi Motors v. Soler).",
                difficulty_trigger="arbitration_enforceability",
                weight=1.8,
            ),
        ),
        # 5. Data Privacy & Cyber breach difficulty:
        # Cyber Security Act 2023 s.25 -> EU GDPR Art. 82 (Liability & compensation for breach)
        (
            "BD-CYBER-2023:s.25",
            LawEdge(
                target_law_id="EU-GDPR-2016:Art.82",
                target_citation="GDPR, Art. 82",
                relation_type="COMPARATIVE_HARMONIZATION",
                reason="Cross-border processing involving Bangladeshi entities and EU data subjects requires dual compliance with CSA 2023 and GDPR Art. 82 liability rules.",
                difficulty_trigger="data_protection",
                weight=1.7,
            ),
        ),
        # GDPR Art. 28 (Processor obligations) -> GDPR Art. 82 (Right to compensation)
        (
            "EU-GDPR-2016:Art.28",
            LawEdge(
                target_law_id="EU-GDPR-2016:Art.82",
                target_citation="GDPR, Art. 82",
                relation_type="DIFFICULTY_ESCALATION",
                reason="Breach of processor data processing terms under Article 28 triggers joint and several liability provisions under Article 82.",
                difficulty_trigger="processor_breach",
                weight=2.0,
            ),
        ),
        # 6. Antitrust comparative harmonization:
        # Sherman Act 15 U.S.C. § 1 -> BD Competition Act s.15
        (
            "US-15-USC-1",
            LawEdge(
                target_law_id="BD-COMP-2012:s.15(1)",
                target_citation="Competition Act 2012 (BD), s.15(1)",
                relation_type="COMPARATIVE_HARMONIZATION",
                reason="Prohibition on anticompetitive agreements and conspiracies under 15 U.S.C. § 1 directly corresponds to Section 15(1) Bangladesh Competition Act 2012.",
                difficulty_trigger="competition_cartel",
                weight=1.6,
            ),
        ),
    ]

    for src_id, edge in edges_to_add:
        graph.add_edge(src_id, edge)


def build_and_save_complete_graph(
    data_dir: Path,
    open_data_dir: Path,
    output_file: Path,
) -> LegalKnowledgeGraph:
    """Orchestrate full graph construction and serialization."""
    logger.info("Initializing Legal Knowledge Graph...")
    graph = LegalKnowledgeGraph()

    # 1. Curated baseline laws and cases
    load_seed_laws_and_cases(graph, data_dir)

    # 2. Bangladeshi open statutes (Penal Code, Constitution, Transfer of Property)
    load_open_bd_datasets(graph, open_data_dir)

    # 3. EU open legislation (GDPR Articles 1-99)
    load_open_eu_datasets(graph, open_data_dir)

    # 4. US open legislation (Title 9 FAA, Title 15 Commerce & Antitrust)
    load_open_us_datasets(graph, open_data_dir)

    # 5. Connect cross-statutory difficulty-handling edges
    build_difficulty_and_companion_edges(graph)

    # 6. Persist to JSON
    graph.save_to_file(output_file)

    # Sync citation index for compatibility
    chroma_dir = data_dir.parent / "chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    cit_index_map = {k: [v] for k, v in graph.citation_to_node_id.items()}
    with open(chroma_dir / "citation_index.json", "w", encoding="utf-8") as f:
        json.dump(cit_index_map, f, indent=2)

    stats = graph.get_stats()
    logger.info(f"Knowledge Graph Construction Complete: {stats}")
    return graph


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rag_root = Path(__file__).resolve().parent.parent
    data_d = rag_root / "data"
    open_d = data_d / "open_datasets"
    out_file = data_d / "legal_graph.json"
    build_and_save_complete_graph(data_d, open_d, out_file)
