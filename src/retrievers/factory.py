from __future__ import annotations

from pathlib import Path


def build_retriever(name: str, cfg: dict, generator_llm, generator_cfg: dict, cache_dir, run_dir):
    rc = cfg.get("retrievers", {}).get(name, {})
    if name == "bm25":
        from src.retrievers.bm25 import BM25Retriever
        return BM25Retriever()
    if name in ("dense", "rag2"):
        from src.retrievers.medcpt import MedCPTEncoder, MedCPTCrossEncoder
        enc = MedCPTEncoder(query_model=rc.get("query_encoder", "ncbi/MedCPT-Query-Encoder"),
                            article_model=rc.get("article_encoder", "ncbi/MedCPT-Article-Encoder"),
                            device=rc.get("device"), batch_size=rc.get("batch_size", 32))
        if name == "dense":
            from src.retrievers.dense import DenseRetriever
            return DenseRetriever(enc, cache_dir=cache_dir)
        from src.retrievers.rag2 import Rag2StyleRetriever
        ce = MedCPTCrossEncoder(device=rc.get("device")) if rc.get("rerank") else None
        return Rag2StyleRetriever(enc, generator_llm, cache_dir=cache_dir,
                                  query_mode=rc.get("query_mode", "rationale"), rerank=rc.get("rerank", False),
                                  candidates=rc.get("candidates", 50), cross_encoder=ce)
    if name == "hipporag2":
        from src.retrievers.hipporag2 import HippoRAG2Retriever
        return HippoRAG2Retriever(save_dir=Path(run_dir) / "hipporag_store", llm_cfg=generator_cfg,
                                  embedding_model_name=rc.get("embedding_model_name", "facebook/contriever"),
                                  embedding_base_url=rc.get("embedding_base_url"),
                                  extra_kwargs=rc.get("extra_kwargs"))
    raise ValueError(f"Unknown system: {name}")
