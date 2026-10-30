from __future__ import annotations

import ast
import json
import os
import re
import sys
import warnings

from src.retrievers.base import Retriever
from src.schema import Hit, RetrievalResult


def _clean_str(x):
    if x is None:
        return None
    s = x if isinstance(x, str) else str(x)
    return s if s.strip() else None


ZERO_COUNTS = {"dropped_entities": 0, "dropped_triples": 0, "null_lists": 0,
               "chunks_total": 0, "chunks_no_entities": 0, "chunks_no_triples": 0}


def sanitize_openie(all_openie_info) -> dict:
    """Clean HippoRAG's OpenIE output IN PLACE (the same list is reused after saving).

    Failure modes handled: (1) null / empty / non-string entries inside entity or triple lists (LLMs emit
    {"named_entities": ["x", null]}); (2) a whole list being None because parsing that chunk's LLM output failed.
    Both crash HippoRAG's stats code (`len(None)`). Non-string scalars are converted to strings. Returns counts so
    the extraction failure rate can be reported: chunks_no_entities / chunks_no_triples are chunks that end up with
    nothing extracted (e.g. after a parse error), which silently thins the graph.
    """
    c = dict(ZERO_COUNTS)
    for chunk in all_openie_info:
        c["chunks_total"] += 1
        ents = chunk.get("extracted_entities")
        if ents is None:
            chunk["extracted_entities"] = ents = []
            c["null_lists"] += 1
        elif isinstance(ents, list):
            clean = [x for x in map(_clean_str, ents) if x is not None]
            c["dropped_entities"] += len(ents) - len(clean)
            ents[:] = clean
        trips = chunk.get("extracted_triples")
        if trips is None:
            chunk["extracted_triples"] = trips = []
            c["null_lists"] += 1
        elif isinstance(trips, list):
            kept = []
            for t in trips:
                if isinstance(t, (list, tuple)):
                    parts = [_clean_str(x) for x in t]
                    if parts and all(x is not None for x in parts):
                        kept.append(parts)
                        continue
                c["dropped_triples"] += 1
            trips[:] = kept
        if isinstance(ents, list) and not ents:
            c["chunks_no_entities"] += 1
        if isinstance(trips, list) and not trips:
            c["chunks_no_triples"] += 1
    return c


def safe_literal_eval(text):
    """Replacement for eval() on LLM output. HippoRAG parses model replies with eval(), which fails on JSON
    `null`/`true`/`false` (NameError) and executes arbitrary code. Try JSON first, then Python literals."""
    if not isinstance(text, str):
        raise TypeError("safe_literal_eval expects a string")
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    fixed = re.sub(r"\bnull\b", "None", text)
    fixed = re.sub(r"\btrue\b", "True", fixed)
    fixed = re.sub(r"\bfalse\b", "False", fixed)
    return ast.literal_eval(fixed)


def install_safe_eval() -> bool:
    """Shadow the builtin eval() inside hipporag's OpenIE module (its calls resolve `eval` as a module global)."""
    try:
        import importlib
        mod = importlib.import_module("hipporag.information_extraction.openie_openai")
    except Exception:
        return False
    mod.eval = safe_literal_eval
    return True


def install_save_wrapper(HippoRAG) -> bool:
    """Patch HippoRAG.save_openie_results at CLASS level so every instance sanitizes OpenIE output first."""
    orig = getattr(HippoRAG, "save_openie_results", None)
    if orig is None:
        return False
    if getattr(orig, "_medgraph_wrapped", False):
        return True

    def _save(hippo_self, all_openie_info, *a, **k):
        c = sanitize_openie(all_openie_info)
        target = getattr(hippo_self, "_medgraph_counts", None)
        if target is not None:
            for key, v in c.items():
                target[key] += v
        if any(v for key, v in c.items() if key != "chunks_total"):
            warnings.warn(f"HippoRAG OpenIE output summary before saving: {c}. Chunks with no entities/triples "
                          "had unparseable LLM output; report these counts in your methods.")
        return orig(hippo_self, all_openie_info, *a, **k)

    _save._medgraph_wrapped = True
    HippoRAG.save_openie_results = _save
    return True

class HippoRAG2Retriever(Retriever):
    name = "hipporag2"

    def __init__(self, save_dir, llm_cfg: dict, embedding_model_name="facebook/contriever",
                 embedding_base_url=None, extra_kwargs=None):
        self.save_dir = str(save_dir)
        self.llm_cfg = llm_cfg
        self.embedding_model_name = embedding_model_name
        self.embedding_base_url = embedding_base_url
        self.extra_kwargs = extra_kwargs or {}

    @staticmethod
    def _doc(p) -> str:
        return f"{p.title}\n{p.text}" if p.title else p.text

    def index(self, passages):
        try:
            from hipporag import HippoRAG
        except ImportError as e:
            raise ImportError("hipporag is not installed in this environment. See README, 'HippoRAG 2 setup'.") from e
        key_env = self.llm_cfg.get("api_key_env", "LLM_API_KEY")
        if key_env in os.environ:
            os.environ.setdefault("OPENAI_API_KEY", os.environ[key_env])  # HippoRAG's OpenAI client reads this
        kwargs = dict(save_dir=self.save_dir, llm_model_name=self.llm_cfg["model"],
                      llm_base_url=self.llm_cfg["base_url"], embedding_model_name=self.embedding_model_name)
        if self.embedding_base_url:
            kwargs["embedding_base_url"] = self.embedding_base_url
        kwargs.update(self.extra_kwargs)
        self.patches = {"safe_eval": install_safe_eval(), "save_sanitizer": install_save_wrapper(HippoRAG)}
        self.hippo = HippoRAG(**kwargs)
        self.openie_sanitized = dict(ZERO_COUNTS)
        self.hippo._medgraph_counts = self.openie_sanitized
        print(f"[medgraph] HippoRAG patches active: {self.patches}", file=sys.stderr)
        docs = [self._doc(p) for p in passages]
        self.text2pid = {self._doc(p): p.pid for p in passages}
        self.text2pid_stripped = {t.strip(): pid for t, pid in self.text2pid.items()}
        try:
            self.hippo.index(docs=docs)
        except ZeroDivisionError as e:
            raise RuntimeError(
                "HippoRAG's OpenIE extracted no entities (it divided by zero while saving results). Every LLM call "
                "during indexing most likely failed: check 'total_prompt_tokens' in the progress bars (0 means no call "
                "succeeded). Run with OPENAI_LOG=info to see HTTP status codes, fix the cause, delete "
                f"{self.save_dir}, and rerun.") from e

    def retrieve(self, questions, k):
        try:
            res = self.hippo.retrieve(queries=[q.question for q in questions], num_to_retrieve=k)
        except ValueError as e:
            fe = getattr(self.hippo, "fact_embeddings", None)
            if fe is not None and len(fe) == 0:
                raise RuntimeError(
                    "HippoRAG's graph has 0 extracted facts, so OpenIE (entity/triple extraction) produced nothing. "
                    "This is almost always LLM-call failures or unparseable model output during indexing, not a "
                    f"retrieval bug. Fix the LLM errors, delete {self.save_dir} (it caches the empty results), "
                    "and rerun indexing.") from e
            raise
        sols = res[0] if isinstance(res, tuple) else res
        out = []
        for q, sol in zip(questions, sols):
            hits, unmapped = [], 0
            for text, score in zip(sol.docs, sol.doc_scores):
                pid = self.text2pid.get(text) or self.text2pid_stripped.get(text.strip())
                if pid is None:
                    unmapped += 1
                    continue
                hits.append(Hit(pid, float(score)))
            out.append(RetrievalResult(qid=q.qid, hits=hits[:k], trace={"unmapped_hits": unmapped, "openie_sanitized": getattr(self, "openie_sanitized", None),
                                              "hipporag_patches": getattr(self, "patches", None)}))
        return out