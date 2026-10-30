from __future__ import annotations
from pathlib import Path
from src.io_utils import read_json, write_jsonl
import random


def _flatten(x):
    if isinstance(x, (list, tuple)):
        out = []
        for e in x:
            out.extend(_flatten(e))
        return out
    return [str(x)]


def answer_groups(qtype: str, exact) -> list:
    if qtype == "yesno":
        return [[str(exact).strip().lower()]]
    if qtype == "factoid":
        return [[s for s in _flatten(exact) if s.strip()]]
    if qtype == "list":
        groups = []
        for e in exact:
            g = [s for s in _flatten(e) if s.strip()]
            if g:
                groups.append(g)
        return groups
    return []


def pmid_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def prepare_bioasq(input_path, out_dir, n=200, seed=13, types=("yesno", "factoid", "list")):
    data = read_json(input_path)
    rows = []
    for item in data["questions"]:
        qtype = item.get("type")
        if qtype not in types or "exact_answer" not in item or not item.get("documents"):
            continue
        groups = answer_groups(qtype, item["exact_answer"])
        if not groups:
            continue
        rows.append({
            "qid": str(item["id"]),
            "dataset": "bioasq",
            "qtype": qtype,
            "question": item["body"].strip(),
            "answer_groups": groups,
            "gold_doc_ids": sorted({pmid_from_url(u) for u in item["documents"]}),
            "extra": {},
        })
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]
    out = Path(out_dir)
    write_jsonl(out / "questions.jsonl", rows)
    pmids = sorted({p for r in rows for p in r["gold_doc_ids"]})
    (out / "pmids.txt").write_text("\n".join(pmids) + "\n")
    return {"questions": len(rows), "pmids": len(pmids)}
