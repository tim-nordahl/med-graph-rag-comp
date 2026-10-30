"""Citation metrics
Utilizes LLM Judge to determine if claim is supported by its cited passages
claim_support_recall = supported sentences / all sentences
claim_citation_precision = relevant citations / all citations
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.generate import CITE_RE

JUDGE_PROMPT = """You are verifying citations in a biomedical answer.

Statement: {statement}

Passages:
{passages}

Do the passages, taken together, fully support the statement? Statements that are only partly supported, or that go beyond the passages, count as UNSUPPORTED.
Reply with exactly one word: SUPPORTED or UNSUPPORTED."""


@dataclass
class Claim:
    text: str
    cites: list = field(default_factory=list)   # 1-based indices into the generation context


def split_claims(body: str, min_words: int = 4) -> list:
    body = re.sub(r"(?im)^\s*final answer:.*$", "", body)
    pieces = []
    for line in body.splitlines():
        line = line.strip()
        if line:
            pieces.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", line))
    merged = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        no_cites = CITE_RE.sub("", p).strip(" .,;")
        if not no_cites and merged:
            merged[-1] += " " + p
        else:
            merged.append(p)
    claims = []
    for p in merged:
        text = CITE_RE.sub("", p).strip()
        if len(text.split()) < min_words:
            continue
        cites = []
        for m in CITE_RE.finditer(p):
            i = int(m.group(1))
            if i not in cites:
                cites.append(i)
        claims.append(Claim(text=text, cites=cites))
    return claims

class SupportJudge:
    def __init__(self, llm):
        self.llm = llm

    def supports(self, statement: str, passage_texts: list) -> bool:
        if not passage_texts:
            return False
        block = "\n".join(f"[{i}] {t}" for i, t in enumerate(passage_texts, 1))
        out = self.llm.chat([{"role": "user", "content": JUDGE_PROMPT.format(statement=statement, passages=block)}],
                            max_tokens=8, temperature=0.0)
        up = out.upper()
        if re.search(r"\bUNSUPPORTED\b", up):
            return False
        return bool(re.search(r"\bSUPPORTED\b", up))


def judge_claims(claims, context_pids, pid2text, judge: SupportJudge, compute_precision=True, qid="", system=""):
    """Returns (metrics_dict, judgment_rows)."""
    rows, supported_n, rel_n, cite_n = [], 0, 0, 0
    for c in claims:
        valid = [i for i in c.cites if 1 <= i <= len(context_pids)]
        texts = [pid2text[context_pids[i - 1]] for i in valid]
        joint = judge.supports(c.text, texts) if valid else False
        per_cite = []
        if valid:
            cite_n += len(valid)
            if joint:
                supported_n += 1
                if len(valid) == 1:
                    per_cite = [True]
                elif compute_precision:
                    for j in range(len(valid)):
                        alone = judge.supports(c.text, [texts[j]])
                        without = judge.supports(c.text, texts[:j] + texts[j + 1:])
                        per_cite.append(not ((not alone) and without))
                else:
                    per_cite = [None] * len(valid)
            else:
                per_cite = [False] * len(valid)
            rel_n += sum(1 for x in per_cite if x)
        rows.append({"qid": qid,
                     "system": system,
                     "claim": c.text,
                     "cites": [context_pids[i - 1] for i in valid],
                     "cite_texts": texts,
                     "supported": joint,
                     "per_cite_relevant": per_cite,
                     "llm_verdict": "SUPPORTED" if joint else "UNSUPPORTED"})
    n = len(claims)
    m = {"n_claims": n,
         "claim_support_recall": (supported_n / n) if n else None,
         "claim_citation_precision": (rel_n / cite_n) if (cite_n and compute_precision) else None}
    return m, rows
