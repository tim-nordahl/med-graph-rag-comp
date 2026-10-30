"""Network fetcher for PubMed abstracts"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
import requests
from pathlib import Path

from src.io_utils import write_jsonl

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _text(el) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for art in root.iter("PubmedArticle"):
        pmid = _text(art.find(".//MedlineCitation/PMID"))
        title = _text(art.find(".//Article/ArticleTitle"))
        parts = []
        for ab in art.findall(".//Article/Abstract/AbstractText"):
            label = ab.get("Label")
            body = _text(ab)
            if body:
                parts.append(f"{label}: {body}" if label else body)
        out.append({"doc_id": pmid, "title": title, "text": " ".join(parts)})
    return out


def fetch_pubmed(dataset_dir, email="", api_key="", batch=100):
    d = Path(dataset_dir)
    pmids = [p.strip() for p in (d / "pmids.txt").read_text().splitlines() if p.strip()]
    delay = 0.11 if api_key else 0.4
    rows = []
    for i in range(0, len(pmids), batch):
        ids = pmids[i:i + batch]
        params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
        if email:
            params["email"] = email
        if api_key:
            params["api_key"] = api_key
        r = requests.post(EUTILS, data=params, timeout=60)
        r.raise_for_status()
        rows.extend(parse_pubmed_xml(r.text))
        time.sleep(delay)
    write_jsonl(d / "docs.jsonl", rows)
    empty = sum(1 for r in rows if not r["text"].strip())
    return {"requested": len(pmids), "fetched": len(rows), "without_abstract": empty}