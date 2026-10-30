import numpy as np

from src.retrievers.hipporag2 import HippoRAG2Retriever
from src.schema import Question


class _EmptyGraphHippo:
    fact_embeddings = np.zeros(0)

    def retrieve(self, queries, num_to_retrieve):
        raise ValueError("shapes (0,) and (768,) not aligned")


def test_empty_graph_gives_actionable_error():
    r = HippoRAG2Retriever(save_dir="/tmp/x", llm_cfg={})
    r.hippo, r.text2pid, r.text2pid_stripped = _EmptyGraphHippo(), {}, {}
    q = Question("1", "t", "open", "q?", [["a"]], [])
    try:
        r.retrieve([q], 5)
    except RuntimeError as e:
        assert "0 extracted facts" in str(e)
    else:
        raise AssertionError("expected RuntimeError")