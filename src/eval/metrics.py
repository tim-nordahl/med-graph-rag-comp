"""Define metrics for summary
ans_em:
ans_f1:
ans_contains:
"""

from __future__ import annotations
import numpy as np
import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")

def retrieval_metrics(q, hits, pid2doc: dict) -> dict:
    """Document-level retrieval eval at the cut-off already applied to `hits`.

    doc_recall: fraction of gold docs retrieved. all_gold: every gold doc retrieved (the multi-hop number).
    hit_any: at least one gold doc retrieved. All None when the dataset has no evidence labels.
    """
    gold = set(q.gold_doc_ids)
    if not gold:
        return {"doc_recall": None, "all_gold": None, "hit_any": None}
    got = {pid2doc[h.pid] for h in hits if h.pid in pid2doc}
    found = gold & got
    return {"doc_recall": len(found) / len(gold), "all_gold": float(found == gold), "hit_any": float(bool(found))}

def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def contains(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    return bool(g) and f" {g} " in f" {p} "


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def parse_yesno(text: str):
    m = re.search(r"\b(yes|no)\b", text.lower())
    return m.group(1) if m else None


def _split_items(pred: str) -> list:
    return [x.strip() for x in re.split(r"[,;\n]| and ", pred) if x.strip()]


def answer_scores(pred: str, q) -> dict:
    groups = [[g for g in grp if g] for grp in q.answer_groups]
    groups = [g for g in groups if g]
    if not groups:
        return {"ans_em": 0.0, "ans_f1": 0.0, "ans_contains": 0.0}
    if q.qtype == "yesno":
        ok = float(parse_yesno(pred) == normalize(groups[0][0]))
        return {"ans_em": ok, "ans_f1": ok, "ans_contains": ok}
    if q.qtype == "list":
        matched = sum(any(contains(pred, s) for s in grp) for grp in groups)
        rec = matched / len(groups)
        items = _split_items(pred)
        prec = (sum(any(normalize(it) == normalize(s) or contains(it, s) for grp in groups for s in grp)
                    for it in items) / len(items)) if items else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"ans_em": float(rec == 1.0 and prec == 1.0), "ans_f1": f1, "ans_contains": rec}
    syn = [s for grp in groups for s in grp]
    em = float(any(normalize(pred) == normalize(s) for s in syn))
    cont = float(any(contains(pred, s) for s in syn))
    f1 = max(token_f1(pred, s) for s in syn)
    return {"ans_em": em, "ans_f1": f1, "ans_contains": cont}

def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2)), "n": int(v.size)}


def paired_bootstrap(a, b, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return {"diff": None, "lo": None, "hi": None, "n": 0}
    d = np.array([x - y for x, y in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return {"diff": float(d.mean()), "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2)), "n": int(d.size)}


def cohen_kappa(a: list, b: list) -> float:
    assert len(a) == len(b) and a, "need two equal-length non-empty label lists"
    n = len(a)
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
