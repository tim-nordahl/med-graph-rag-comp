"""Chunk text and build corpus from gold documents of selected and unselected questions"""


from __future__ import annotations
from pathlib import Path

from src.io_utils import read_jsonl, write_jsonl, write_json


def chunk_text(text: str, chunk_words: int = 180, overlap: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = max(1, chunk_words - overlap)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_words]
        chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


def build_corpus(dataset_dir, chunk_words=180, overlap=30, require_all_gold=False):
    d = Path(dataset_dir)
    questions = read_jsonl(d / "questions.jsonl")
    docs = {r["doc_id"]: r for r in read_jsonl(d / "docs.jsonl") if r.get("text", "").strip()}

    kept, dropped = [], 0
    for q in questions:
        gold = q.get("gold_doc_ids", [])
        avail = [g for g in gold if g in docs]
        if gold and (not avail or (require_all_gold and len(avail) < len(gold))):
            dropped += 1
            continue
        q = dict(q)
        q["gold_doc_ids"] = avail
        kept.append(q)

    needed = []
    seen = set()
    for q in kept:
        for g in q["gold_doc_ids"]:
            if g not in seen:
                seen.add(g)
                needed.append(g)
    extras = [doc_id for doc_id in docs if doc_id not in seen] if (d / "use_extra_distractors").exists() else []

    passages = []
    for doc_id in needed + extras:
        doc = docs[doc_id]
        for i, ch in enumerate(chunk_text(doc["text"], chunk_words, overlap)):
            passages.append({"pid": f"{doc_id}::{i}", "doc_id": doc_id, "title": doc.get("title", ""), "text": ch})

    write_jsonl(d / "corpus.jsonl", passages)
    write_jsonl(d / "questions_eval.jsonl", kept)
    stats = {"questions_in": len(questions), "questions_kept": len(kept), "questions_dropped_missing_gold": dropped,
             "documents": len(needed) + len(extras), "passages": len(passages),
             "require_all_gold": require_all_gold, "chunk_words": chunk_words, "overlap": overlap}
    write_json(d / "corpus_stats.json", stats)
    return stats
