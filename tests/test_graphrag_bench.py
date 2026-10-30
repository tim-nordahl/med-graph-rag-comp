import json

from src.data.corpus import chunk_text
from src.data.graphrag_bench import load_corpus_docs, prepare_graphrag_bench, _flatten_strs
from src.io_utils import read_jsonl


def _setup(tmp_path):
    words = [f"w{i}" for i in range(600)]
    text = " ".join(words)
    (tmp_path / "corpus.json").write_text(json.dumps([{"corpus_name": "NCCN", "context": text}]))
    ev_inside = " ".join(words[250:290])            # lies fully inside the chunk starting at word 240
    qs = [
        {"id": "a", "source": "s", "question": "Q fact?", "answer": "ans a", "question_type": "Fact Retrieval",
         "evidence": [ev_inside], "evidence_relations": []},
        {"id": "b", "source": "s", "question": "Q unmatched?", "answer": "ans b", "question_type": "Fact Retrieval",
         "evidence": ["completely different words that never occur anywhere in the corpus text"]},
        {"id": "c", "source": "s", "question": "Q complex?", "answer": "ans c", "question_type": "Complex Reasoning",
         "evidence": ev_inside},
        {"id": "d", "source": "s", "question": "Q summary?", "answer": "ans d", "question_type": "Contextual Summarize",
         "evidence": []},
        {"id": "a", "source": "s", "question": "duplicate id", "answer": "x", "question_type": "Fact Retrieval", "evidence": []},
    ]
    (tmp_path / "q.json").write_text(json.dumps(qs))
    return words


def test_chunk_level_gold_and_stats(tmp_path):
    _setup(tmp_path)
    stats = prepare_graphrag_bench(tmp_path / "q.json", tmp_path / "corpus.json", tmp_path / "out",
                                  n=10, chunk_words=100, overlap=20)
    qs = {q["qid"]: q for q in read_jsonl(tmp_path / "out" / "questions_eval.jsonl")}
    passages = read_jsonl(tmp_path / "out" / "corpus.jsonl")
    assert len(passages) == len(chunk_text(" ".join(f"w{i}" for i in range(600)), 100, 20))
    assert all(p["doc_id"] == p["pid"] for p in passages)          # chunk == document
    assert qs["a"]["gold_doc_ids"] == ["NCCN::3"] and qs["c"]["gold_doc_ids"] == ["NCCN::3"]
    assert qs["b"]["gold_doc_ids"] == [] and qs["d"]["gold_doc_ids"] == []
    assert stats["evidence_unmatched"] == 1 and stats["no_evidence"] == 1
    assert stats["questions_selected"] == 4                        # duplicate id removed
    assert qs["a"]["extra"]["question_type"] == "Fact Retrieval"


def test_type_filter_and_drop_unmatched(tmp_path):
    _setup(tmp_path)
    prepare_graphrag_bench(tmp_path / "q.json", tmp_path / "corpus.json", tmp_path / "out", n=10,
                          types=["fact", "complex"], chunk_words=100, overlap=20, drop_unmatched=True)
    ids = sorted(q["qid"] for q in read_jsonl(tmp_path / "out" / "questions_eval.jsonl"))
    assert ids == ["a", "c"]                                       # summary filtered, unmatched dropped


def test_corpus_loading_variants(tmp_path):
    (tmp_path / "c.txt").write_text("plain text corpus")
    assert load_corpus_docs(tmp_path / "c.txt")[0]["text"] == "plain text corpus"
    (tmp_path / "c.json").write_text(json.dumps({"title": "T", "text": "body"}))
    assert load_corpus_docs(tmp_path / "c.json") == [{"title": "T", "text": "body"}]
    assert _flatten_strs([["a", "b"], None, "c"]) == ["a", "b", "c"]