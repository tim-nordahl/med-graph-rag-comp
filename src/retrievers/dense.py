from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.retrievers.base import Retriever
from src.schema import Hit, RetrievalResult


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[-1])
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


class DenseRetriever(Retriever):
    """Plain dense baseline: MedCPT query encoder on the raw question, MedCPT article encoder on passages."""
    name = "dense"

    def __init__(self, encoder, cache_dir=None):
        self.encoder = encoder
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def index(self, passages):
        self.passages = passages
        key = hashlib.sha1("|".join(p.pid for p in passages).encode()).hexdigest()[:12]
        cache = self.cache_dir / f"emb_{key}.npy" if self.cache_dir else None
        if cache is not None and cache.exists():
            self.emb = np.load(cache)
            return
        self.emb = self.encoder.encode_articles([(p.title, p.text) for p in passages])
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.emb)

    def _search(self, query_texts, k):
        q = self.encoder.encode_queries(query_texts)
        scores = q @ self.emb.T
        return [(topk_indices(s, k), s) for s in scores]

    def retrieve(self, questions, k):
        res = []
        for q, (idx, s) in zip(questions, self._search([q.question for q in questions], k)):
            res.append(RetrievalResult(qid=q.qid, hits=[Hit(self.passages[i].pid, float(s[i])) for i in idx]))
        return res
