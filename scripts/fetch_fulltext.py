"""Fetch full text for rows in laws_registry.csv and write structure-aware chunks as JSONL.
UNTESTED from the build sandbox (network was restricted) - run locally and fix selectors as needed.
Usage: python fetch_fulltext.py            (uses laws_registry.csv in same folder)
Output: corpus_chunks.jsonl  (one chunk per Article/Section where detectable)
Bangladesh: bdlaws.minlaw.gov.bd -> open each Act page, copy/scrape the English text into
            bd_texts/<law_id>.txt (e.g. BD-COMP-2012.txt), then re-run; this script chunks those files.
"""
import csv, json, re, time, pathlib, requests
from bs4 import BeautifulSoup
HERE = pathlib.Path(__file__).parent
H = {"User-Agent": "LegalCopilot-MVP/0.1 (research prototype)"}
def clean(t): return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", t)).strip()
def fetch_html(url):
    r = requests.get(url, headers=H, timeout=60); r.raise_for_status()
    return clean(BeautifulSoup(r.text, "html.parser").get_text("\n"))
def eurlex_url(celex): return f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"
SPLIT = re.compile(r"(?m)^\s*(Article\s+\d+[A-Za-z]?|Section\s+\d+[A-Za-z]?|\d{1,3}\.\s+[A-Z])")
def chunk(text, max_chars=3500):
    parts = SPLIT.split(text)
    chunks, head = [], ""
    if len(parts) < 3: parts = ["", text]
    for i in range(1, len(parts), 2):
        label, body = parts[i].strip(), parts[i+1] if i+1 < len(parts) else ""
        body = clean(body)
        for j in range(0, max(len(body), 1), max_chars):
            chunks.append((label, body[j:j+max_chars]))
    return chunks
out = open(HERE / "corpus_chunks.jsonl", "w", encoding="utf-8")
for row in csv.DictReader(open(HERE / "laws_registry.csv", encoding="utf-8-sig")):
    lid, acq = row["id"], row["text_acquisition"]
    try:
        if acq == "fetch_eurlex" and row["celex"]: text = fetch_html(eurlex_url(row["celex"]))
        elif acq in ("fetch_uscode", "fetch_html") and row["source_url"]: text = fetch_html(row["source_url"])
        elif row["jurisdiction"] == "BD":
            p = HERE / "bd_texts" / f"{lid}.txt"
            if not p.exists(): print("MISSING manual text:", p.name); continue
            text = p.read_text(encoding="utf-8")
        else: print("SKIP", lid, acq); continue
    except Exception as e:
        print("FAIL", lid, e); continue
    for n, (label, body) in enumerate(chunk(text)):
        out.write(json.dumps(dict(law_id=lid, jurisdiction=row["jurisdiction"], title=row["title"],
            chunk_id=f"{lid}#{n:04d}", provision_label=label, text=body, source_url=row["source_url"],
            retrieved=time.strftime("%Y-%m-%d"), status=row["status"]), ensure_ascii=False) + "\n")
    print("OK", lid); time.sleep(1.0)
