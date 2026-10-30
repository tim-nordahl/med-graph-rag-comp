from __future__ import annotations

from src.retrievers.dense import DenseRetriever
from src.schema import Hit, RetrievalResult

RATIONALE_PROMPT = (
    "Write a brief rationale (2-3 sentences) stating the key biomedical facts and concepts needed to answer the "
    "question below. Do not hedge and do not mention that you are unsure.\n\nQuestion: {question}\n\nRationale:"
)


class Rag2StyleRetriever(DenseRetriever):
    name = "rag2"

    def __init__(self, encoder, llm, cache_dir=None, query_mode="rationale", rerank=False,
                 candidates=50, cross_encoder=None, workers=8):
        super().__init__(encoder, cache_dir)
        assert query_mode in ("rationale", "question+rationale")
        self.llm, self.query_mode, self.rerank = llm, query_mode, rerank
        self.candidates, self.cross_encoder, self.workers = candidates, cross_encoder, workers

    def retrieve(self, questions, k):
        prompts = [[{"role": "user", "content": RATIONALE_PROMPT.format(question=q.question)}] for q in questions]
        rationales = [r.strip() for r in self.llm.map(prompts, workers=self.workers, max_tokens=160, desc="rationales")]
        if self.query_mode == "rationale":
            queries = [r or q.question for q, r in zip(questions, rationales)]
        else:
            queries = [f"{q.question} {r}".strip() for q, r in zip(questions, rationales)]
        n = max(k, self.candidates) if self.rerank else k
        out = []
        for q, rat, qt, (idx, s) in zip(questions, rationales, queries, self._search(queries, n)):
            hits = [Hit(self.passages[i].pid, float(s[i])) for i in idx]
            if self.rerank and self.cross_encoder is not None:
                arts = [(self.passages[i].title, self.passages[i].text) for i in idx]
                ce = self.cross_encoder.score(qt, arts)
                order = sorted(range(len(hits)), key=lambda j: -ce[j])
                hits = [Hit(hits[j].pid, float(ce[j])) for j in order]
            out.append(RetrievalResult(qid=q.qid, hits=hits[:k], trace={"rationale": rat}))
        return out
