"""Pure-Python BM25. Needs no models or GPU: used for smoke tests and as a cheap lexical baseline."""
from __future__ import annotations

import math
import re
from collections import defaultdict

from src.retrievers.base import Retriever
from src.schema import Hit, RetrievalResult

_TOK = re.compile(r"\w+")


def tokenize(s: str) -> list[str]:
    return _TOK.findall(s.lower())


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def index(self, passages):
        self.pids = [p.pid for p in passages]
        self.postings = defaultdict(list)
        self.dl = []
        for i, p in enumerate(passages):
            toks = tokenize(f"{p.title} {p.text}")
            self.dl.append(len(toks))
            tf = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t, c in tf.items():
                self.postings[t].append((i, c))
        self.n = len(passages)
        self.avgdl = sum(self.dl) / max(1, self.n)

    def _scores(self, query: str):
        scores = defaultdict(float)
        for t in set(tokenize(query)):
            post = self.postings.get(t)
            if not post:
                continue
            idf = math.log(1 + (self.n - len(post) + 0.5) / (len(post) + 0.5))
            for i, c in post:
                denom = c + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += idf * c * (self.k1 + 1) / denom
        return scores

    def retrieve(self, questions, k):
        out = []
        for q in questions:
            sc = self._scores(q.question)
            top = sorted(sc.items(), key=lambda kv: -kv[1])[:k]
            out.append(RetrievalResult(qid=q.qid, hits=[Hit(self.pids[i], float(s)) for i, s in top]))
        return out
