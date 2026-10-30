"""Prep GraphRAG-Bench (Medical) textbook style document into chunks to be treated as their own document.
Questions can then be paired with documents that contain evidence text. Tiers introduced to question-document
pairs to reflect confidence - can be tuned as hyperparameter. If question's evidence not clearly related
to any document then question omitted and replaced by default.

tiers are ranked in order of confidence: ngram, ngram_partial, token, none

sample call:
    python -m src.data.graphragbench --questions Q.json --corpus C.json --out data/graphrag_medical
        --n 200 --types fact complex
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

from src.data.corpus import chunk_text
from src.io_utils import read_jsonl, write_json, write_jsonl

TEXT_KEYS = ("context", "text", "content", "body")
TITLE_KEYS = ("corpus_name", "title", "name")
STOP = frozenset(
    "a an the of in on at to for and or but is are was were be been being by with as from that this these those it "
    "its into than then so such not no can may might will would should could also which who whom whose what when "
    "where how have has had do does did their there they them he she his her we our you your i".split())


def _flatten_strs(x) -> list:
    if x is None:
        return []
    if isinstance(x, str):
        return [x] if x.strip() else []
    if isinstance(x, (list, tuple)):
        out = []
        for e in x:
            out.extend(_flatten_strs(e))
        return out
    if isinstance(x, dict):
        out = []
        for v in x.values():
            out.extend(_flatten_strs(v))
        return out
    return [str(x)]


def _load_any(path):
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return read_jsonl(p)
    return json.loads(p.read_text(encoding="utf-8"))


def load_corpus_docs(path, text_field=None, title_field=None) -> list:
    p = Path(path)
    if p.suffix.lower() == ".txt":
        return [{"title": p.stem, "text": p.read_text(encoding="utf-8")}]
    data = _load_any(p)
    rows = data if isinstance(data, list) else [data]
    docs = []
    for i, r in enumerate(rows):
        if isinstance(r, str):
            docs.append({"title": f"doc{i}", "text": r})
            continue
        tkey = text_field or next((k for k in TEXT_KEYS if k in r), None)
        if tkey is None:
            raise ValueError(f"Cannot find a text field in corpus row {i}; keys: {list(r)}. Use --text-field.")
        nkey = title_field or next((k for k in TITLE_KEYS if k in r), None)
        docs.append({"title": str(r[nkey]) if nkey else f"doc{i}", "text": str(r[tkey])})
    return docs


def _norm(s) -> str:
    s = unicodedata.normalize("NFKC", str(s)).replace("\u00ad", "")
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    return s.lower()


def _tokens(s) -> list:
    return re.findall(r"\w+", _norm(s))


def _shingles(tokens: list, n: int = 4) -> set:
    if not tokens:
        return set()
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _slug(s: str) -> str:
    return re.sub(r"\W+", "_", s).strip("_")[:40] or "doc"


def build_chunks(docs, chunk_words=180, overlap=30) -> list:
    passages = []
    for d in docs:
        slug = _slug(d["title"])
        for i, ch in enumerate(chunk_text(d["text"], chunk_words, overlap)):
            pid = f"{slug}::{i}"
            passages.append({"pid": pid, "doc_id": pid, "title": d["title"], "text": ch})
    return passages


class ChunkIndex:
    def __init__(self, passages):
        self.pids = [p["pid"] for p in passages]
        toks = [_tokens(p["text"]) for p in passages]
        self.sh = [_shingles(t) for t in toks]
        self.sets = [set(t) for t in toks]
        self.n = len(passages)
        df = Counter()
        for s in self.sets:
            df.update(s)
        self._df = df

    def idf(self, t: str) -> float:
        return math.log((self.n + 1) / max(self._df.get(t, 0), 1))   # words absent from the corpus weigh like df=1


def _scores(ev: str, idx: ChunkIndex):
    """Per-chunk 4-gram containment and (when the evidence has >= 4 content words) IDF-weighted word recall."""
    toks = _tokens(ev)
    sh = _shingles(toks)
    cont = [len(sh & cs) / len(sh) for cs in idx.sh] if sh else [0.0] * idx.n
    content = sorted({t for t in toks if t not in STOP and len(t) > 1})
    rec = None
    if len(content) >= 4:
        w = {t: idx.idf(t) for t in content}
        tot = sum(w.values())
        rec = [sum(w[t] for t in content if t in s) / tot for s in idx.sets]
    return cont, rec


def match_one(ev, idx, min_containment=0.5, fallback_min=0.2, token_min=0.7):
    """Returns (chunk_indices, tier, best_containment, best_token_recall_or_None)."""
    cont, rec = _scores(ev, idx)
    best4 = max(cont) if cont else 0.0
    bestr = max(rec) if rec else None
    hits = [j for j, c in enumerate(cont) if c >= min_containment]
    if hits:
        return hits, "ngram", best4, bestr
    if cont and best4 >= fallback_min:
        return [max(range(len(cont)), key=cont.__getitem__)], "ngram_partial", best4, bestr
    if token_min is not None and rec and bestr >= token_min:
        cut = max(token_min, bestr - 0.05)
        return [j for j, r in enumerate(rec) if r >= cut], "token", best4, bestr
    return [], "none", best4, bestr


def _select_questions(raw, types):
    wanted = [t.lower() for t in types] if types else None
    seen, rows = set(), []
    for i, r in enumerate(raw):
        qid = str(r.get("id", i))
        qt = str(r.get("question_type", ""))
        if qid in seen or (wanted and not any(w in qt.lower() for w in wanted)):
            continue
        seen.add(qid)
        if r.get("answer") is None or not str(r.get("question", "")).strip():
            continue
        rows.append((qid, qt, r))
    return rows


def prepare_graphragbench(questions_path, corpus_path, out_dir, n=200, seed=13, types=None, chunk_words=180,
                          overlap=30, min_containment=0.5, fallback_min=0.2, token_min=0.7, keep_unmatched=False,
                          text_field=None, title_field=None, dataset_name="graphrag_medical") -> dict:
    raw = _load_any(questions_path)
    if isinstance(raw, dict):
        raw = raw.get("questions") or list(raw.values())
    passages = build_chunks(load_corpus_docs(corpus_path, text_field, title_field), chunk_words, overlap)
    idx = ChunkIndex(passages)

    type_counts = Counter(str(r.get("question_type", "")) for r in raw)
    pool = _select_questions(raw, types)
    random.Random(seed).shuffle(pool)

    out_q, best4s, bestrs, gold_sizes, tier_counts = [], [], [], [], Counter()
    examined = skipped_no_evidence = skipped_unmatched = 0
    for qid, qt, r in pool:
        if len(out_q) >= n:
            break
        examined += 1
        ev = _flatten_strs(r.get("evidence"))
        if not ev and not keep_unmatched:
            skipped_no_evidence += 1
            continue
        gold, tiers, q4, qr = set(), [], [], []
        for e in ev:
            hits, tier, b4, br = match_one(e, idx, min_containment, fallback_min, token_min)
            gold.update(idx.pids[j] for j in hits)
            tiers.append(tier)
            q4.append(b4)
            if br is not None:
                qr.append(br)
        if not gold and not keep_unmatched:
            skipped_unmatched += 1
            continue
        tier_counts.update(tiers)
        best4s.extend(q4)
        bestrs.extend(qr)
        if gold:
            gold_sizes.append(len(gold))
        ans = r.get("answer")
        answer = [str(a) for a in ans] if isinstance(ans, list) else [str(ans)]
        out_q.append({"qid": qid, "dataset": dataset_name, "qtype": "open", "question": str(r["question"]).strip(),
                      "answer_groups": [answer], "gold_doc_ids": sorted(gold),
                      "extra": {"question_type": qt, "source": str(r.get("source", "")),
                                "match": {"tiers": tiers}}})

    out = Path(out_dir)
    write_jsonl(out / "corpus.jsonl", passages)
    write_jsonl(out / "questions_eval.jsonl", out_q)
    ans_words = [len(" ".join(g[0] for g in q["answer_groups"]).split()) for q in out_q]
    by_type = Counter(q["extra"]["question_type"] for q in out_q)
    stats = {"questions_in_file": len(raw), "question_type_counts": dict(type_counts), "types_filter": types,
             "eligible_pool": len(pool), "candidates_examined": examined,
             "skipped_no_evidence": skipped_no_evidence, "skipped_evidence_unmatched": skipped_unmatched,
             "questions_requested": n, "questions_selected": len(out_q), "shortfall": max(0, n - len(out_q)),
             "selected_by_type": dict(by_type),
             "with_gold_chunks": len(gold_sizes), "evidence_snippets_by_tier": dict(tier_counts),
             "questions_with_any_strict_tier": sum(1 for q in out_q if any(t in ("ngram", "ngram_partial") for t in q["extra"]["match"]["tiers"])),
             "median_gold_chunks": statistics.median(gold_sizes) if gold_sizes else None,
             "median_best_containment": round(statistics.median(best4s), 3) if best4s else None,
             "median_best_token_recall": round(statistics.median(bestrs), 3) if bestrs else None,
             "median_answer_words": statistics.median(ans_words) if ans_words else None,
             "passages": len(passages), "chunk_words": chunk_words, "overlap": overlap,
             "min_containment": min_containment, "token_min": token_min, "keep_unmatched": keep_unmatched}
    write_json(out / "corpus_stats.json", stats)
    if stats["shortfall"]:
        print(f"WARNING: only {len(out_q)}/{n} questions have gold chunks; the eligible pool is exhausted. "
              f"Add --types, lower --token-min, or fix the corpus file.")
    return stats


def _best_window(chunk: str, ev: str, extra: int = 12) -> str:
    """The stretch of `chunk` that shares the most content words with the evidence (for eyeballing mismatches)."""
    content = {t for t in _tokens(ev) if t not in STOP and len(t) > 1}
    words = chunk.split()
    width = min(len(words), max(20, len(ev.split()) + extra))
    hit = [1 if (_tokens(w) and _tokens(w)[0] in content) else 0 for w in words]
    best, at, cur = -1, 0, sum(hit[:width])
    for i in range(0, len(words) - width + 1):
        if i:
            cur += hit[i + width - 1] - hit[i - 1]
        if cur > best:
            best, at = cur, i
    return " ".join(words[at:at + width])


def diagnose(questions_path, corpus_path, n=8, types=None, seed=13, chunk_words=180, overlap=30,
             text_field=None, title_field=None, max_questions=300, show=True) -> dict:
    """Explain WHY evidence fails to match: paraphrase (high word recall, zero 4-grams) vs absent (low recall)."""
    raw = _load_any(questions_path)
    if isinstance(raw, dict):
        raw = raw.get("questions") or list(raw.values())
    passages = build_chunks(load_corpus_docs(corpus_path, text_field, title_field), chunk_words, overlap)
    idx = ChunkIndex(passages)
    rows = _select_questions(raw, types)
    rng = random.Random(seed)
    rng.shuffle(rows)
    items = []
    for qid, qt, r in rows[:max_questions]:
        for e in _flatten_strs(r.get("evidence")):
            cont, rec = _scores(e, idx)
            b4 = max(cont) if cont else 0.0
            jr = max(range(len(rec)), key=rec.__getitem__) if rec else None
            items.append({"qid": qid, "type": qt, "evidence": e, "best_4gram": b4,
                          "best_word_recall": rec[jr] if rec else None, "best_chunk": idx.pids[jr] if rec else None})
    total = len(items)
    zero4 = sum(1 for i in items if i["best_4gram"] == 0)
    rec_vals = [i["best_word_recall"] for i in items if i["best_word_recall"] is not None]
    summary = {"evidence_snippets": total, "share_zero_4gram_overlap": round(zero4 / total, 3) if total else None,
               "word_recall>=0.9": sum(v >= 0.9 for v in rec_vals), "word_recall 0.7-0.9": sum(0.7 <= v < 0.9 for v in rec_vals),
               "word_recall 0.5-0.7": sum(0.5 <= v < 0.7 for v in rec_vals), "word_recall<0.5": sum(v < 0.5 for v in rec_vals),
               "too_short_for_word_match": total - len(rec_vals)}
    if show:
        print(json.dumps(summary, indent=1))
        pid2text = {p["pid"]: p["text"] for p in passages}
        bad = [i for i in items if i["best_4gram"] < 0.5]
        rng.shuffle(bad)
        for i in bad[:n]:
            print("\n--- qid", i["qid"], "|", i["type"], "| best 4-gram", round(i["best_4gram"], 2), "| best word recall",
                  None if i["best_word_recall"] is None else round(i["best_word_recall"], 2))
            print("EVIDENCE:", i["evidence"][:300])
            if i["best_chunk"]:
                print("CLOSEST TEXT in", i["best_chunk"], ":", _best_window(pid2text[i["best_chunk"]], i["evidence"])[:500])
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prepare GraphRAG-Bench (Medical) data into chunks")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--types", nargs="*", default=None,
                    help="keep question_type values containing any of these substrings (case-insensitive)")
    ap.add_argument("--chunk-words", type=int, default=180)
    ap.add_argument("--overlap", type=int, default=30)
    ap.add_argument("--min-containment", type=float, default=0.5)
    ap.add_argument("--token-min", type=float, default=0.7, help="word-recall threshold for paraphrased evidence")
    ap.add_argument("--no-token-match", action="store_true", help="disable the word-recall tier (strict labels only)")
    ap.add_argument("--keep-unmatched", action="store_true",
                    help="old behaviour: keep the first n questions even if they have no gold chunk (default: skip and "
                         "replace them with the next eligible question)")
    ap.add_argument("--text-field", default=None)
    ap.add_argument("--title-field", default=None)
    ap.add_argument("--diagnose", type=int, default=0, metavar="N",
                    help="print why evidence fails to match, with N examples, then exit (writes nothing)")
    a = ap.parse_args(argv)
    if a.diagnose:
        diagnose(a.questions, a.corpus, a.diagnose, a.types, a.seed, a.chunk_words, a.overlap, a.text_field, a.title_field)
        return
    print(json.dumps(prepare_graphragbench(
        a.questions, a.corpus, a.out, a.n, a.seed, a.types, a.chunk_words, a.overlap, a.min_containment,
        token_min=None if a.no_token_match else a.token_min, keep_unmatched=a.keep_unmatched,
        text_field=a.text_field, title_field=a.title_field), indent=1))


if __name__ == "__main__":
    main()