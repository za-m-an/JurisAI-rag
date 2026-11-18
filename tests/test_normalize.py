"""Unit tests for citation normalisation and clause splitting (Phase 1)."""

from app.text_utils import normalize_citation, split_into_clauses


def test_normalize_section_symbol_and_usc():
    assert normalize_citation("15 U.S.C. § 1") == "15 usc s 1"
    assert normalize_citation("15 USC s. 1") == "15 usc s 1"
    assert normalize_citation("15 U.S.C. Section 1") == "15 usc s 1"
    assert normalize_citation("15 U.S.C. sec. 1") == "15 usc s 1"


def test_normalize_sherman_act_distinct_from_bare_usc():
    # "Sherman Act, 15 U.S.C. § 1" must NOT loosely fuzzy-match "15 U.S.C. § 1" via normalization alone
    norm_full = normalize_citation("Sherman Act, 15 U.S.C. § 1")
    norm_bare = normalize_citation("15 U.S.C. § 1")
    assert norm_full == "sherman act 15 usc s 1"
    assert norm_bare == "15 usc s 1"
    assert norm_full != norm_bare


def test_normalize_bd_competition_act_variants():
    target = "competition act 2012 (bd) s 15(1)"
    assert normalize_citation("Competition Act 2012 (BD), s.15(1)") == target
    assert normalize_citation("Competition Act 2012 (BD), Section 15(1)") == target
    assert normalize_citation("Competition Act 2012 (BD), sec. 15(1)") == target
    assert normalize_citation("Competition Act 2012 (BD), § 15(1)") == target


def test_normalize_eu_article_variants():
    target = "tfeu art 101(1)"
    assert normalize_citation("TFEU Art. 101(1)") == target
    assert normalize_citation("TFEU Art.101(1)") == target
    assert normalize_citation("TFEU Article 101(1)") == target
    assert normalize_citation("tfeu   article   101(1)") == target


def test_normalize_vber_and_regulations():
    target = "vber 2022720 art 4"
    assert normalize_citation("VBER 2022/720, Art. 4") == target
    assert normalize_citation("VBER 2022/720, Article 4") == target


def test_normalize_ucc_provisions():
    assert normalize_citation("UCC § 2-302") == "ucc s 2302"
    assert normalize_citation("UCC Section 2-302") == "ucc s 2302"
    assert normalize_citation("UCC s. 2-302") == "ucc s 2302"


def test_normalize_plural_sections():
    assert normalize_citation("Competition Act 2012 (BD), ss.17-18") == "competition act 2012 (bd) ss 1718"
    assert normalize_citation("Competition Act 2012 (BD), Sections 17-18") == "competition act 2012 (bd) ss 1718"
    assert normalize_citation("Sherman Antitrust Act, 15 U.S.C. §§ 1-7") == "sherman antitrust act 15 usc ss 17"


def test_normalize_case_citations():
    assert (
        normalize_citation("Leegin Creative Leather Products v. PSKS, 551 U.S. 877")
        == "leegin creative leather products v psks 551 us 877"
    )
    assert (
        normalize_citation("United States v. Microsoft Corp., 253 F.3d 34 (D.C. Cir.)")
        == "united states v microsoft corp 253 f3d 34 (dc cir)"
    )


def test_normalize_empty_and_whitespace():
    assert normalize_citation("") == ""
    assert normalize_citation("   \n\t  ") == ""


def test_split_into_clauses_structured_and_fallback():
    structured = "1. Appointment\nThe supplier appoints distributor.\n\n2.1 Exclusivity\nExclusive territory."
    clauses = split_into_clauses(structured)
    assert len(clauses) == 2
    assert clauses[0][0] == "1."
    assert clauses[1][0] == "2.1"

    unstructured = "First paragraph with no heading.\n\nSecond paragraph with terms."
    p_clauses = split_into_clauses(unstructured)
    assert len(p_clauses) == 2
    assert p_clauses[0][0] == "P1"
    assert p_clauses[1][0] == "P2"
