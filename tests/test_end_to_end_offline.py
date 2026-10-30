"""Whole pipeline offline: BM25 retriever + scripted fake LLMs (no network, no models)."""
import json

from src.data.corpus import build_corpus, chunk_text
from src.io_utils import read_json, read_jsonl, write_jsonl
from src.llm import FunctionLLM
from src.pipeline import run_evaluate, run_generate, run_retrieve
from src.report import make_report
from src.retrievers.bm25 import BM25Retriever
from src.schema import load_dataset


def _write_dataset(root):
    docs = [
        {"doc_id": "1", "title": "Aspirin", "text": "Aspirin irreversibly inhibits cyclooxygenase enzymes in platelets."},
        {"doc_id": "2", "title": "Insulin", "text": "Insulin lowers blood glucose by promoting cellular uptake of glucose."},
        {"doc_id": "3", "title": "Statins", "text": "Statins inhibit HMG-CoA reductase and lower LDL cholesterol."},
    ]
    qs = [
        {"qid": "q1", "dataset": "t", "qtype": "yesno", "question": "Does aspirin inhibit cyclooxygenase?",
         "answer_groups": [["yes"]], "gold_doc_ids": ["1"], "extra": {}},
        {"qid": "q2", "dataset": "t", "qtype": "factoid", "question": "Which enzyme do statins inhibit?",
         "answer_groups": [["HMG-CoA reductase"]], "gold_doc_ids": ["3"], "extra": {}},
        {"qid": "q3", "dataset": "t", "qtype": "factoid", "question": "Which hormone lowers blood glucose?",
         "answer_groups": [["insulin"]], "gold_doc_ids": ["2", "999"], "extra": {}},  # 999 has no abstract
    ]
    write_jsonl(root / "docs.jsonl", docs)
    write_jsonl(root / "questions.jsonl", qs)


def test_chunking():
    assert chunk_text("a b c", 10, 2) == ["a b c"]
    ch = chunk_text(" ".join(map(str, range(100))), 40, 10)
    assert len(ch) == 3 and ch[0].split()[0] == "0" and ch[-1].split()[-1] == "99"


def test_build_corpus_filters_missing_gold(tmp_path):
    _write_dataset(tmp_path)
    s = build_corpus(tmp_path, require_all_gold=True)
    assert s["questions_kept"] == 2 and s["questions_dropped_missing_gold"] == 1
    s = build_corpus(tmp_path, require_all_gold=False)
    assert s["questions_kept"] == 3
    q3 = [q for q in read_jsonl(tmp_path / "questions_eval.jsonl") if q["qid"] == "q3"][0]
    assert q3["gold_doc_ids"] == ["2"]


def gen_fn(messages):
    u = messages[-1]["content"]
    if "Passages:" not in u:
        return "From memory only.\nFINAL ANSWER: yes"
    top = u.split("Passages:")[1].split("Question:")[0].strip().splitlines()[0]  # "[1] (Title) text"
    answer = "yes" if "Aspirin" in top else ("HMG-CoA reductase" if "Statins" in top else "insulin")
    return f"{top.split(') ', 1)[1]} [1]\nFINAL ANSWER: {answer}"


def judge_fn(messages):
    c = messages[0]["content"]
    stmt = c.split("Statement:")[1].split("Passages:")[0].strip()
    block = c.split("Passages:")[1]
    return "SUPPORTED" if stmt.split()[0] in block else "UNSUPPORTED"


def test_full_pipeline(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    _write_dataset(ds)
    build_corpus(ds)
    questions, passages = load_dataset(ds)
    run_dir = tmp_path / "runs" / "ds" / "bm25"
    run_retrieve(BM25Retriever(), questions, passages, 2, run_dir)
    gen, judge = FunctionLLM(gen_fn, "gen"), FunctionLLM(judge_fn, "judge")
    run_generate(questions, passages, gen, 2, run_dir)
    rows = run_evaluate(questions, passages, judge, 2, run_dir, system="bm25")
    assert len(rows) == 3
    assert all(r["ans_score"] == 1.0 for r in rows)
    assert all(r["hit_any"] == 1.0 for r in rows)
    assert all(r["claim_support_recall"] == 1.0 for r in rows)
    assert all(r["cite_hit_any"] == 1.0 for r in rows)
    summ = read_json(run_dir / "summary.json")
    assert summ["n_questions"] == 3 and summ["eval"]["ans_score"]["mean"] == 1.0
    assert (run_dir / "judgments.jsonl").exists()
    assert len(read_jsonl(run_dir / "usage.jsonl")) == 2

    # no-retrieval control: no citations -> citation eval absent, answer metric still computed
    nr = tmp_path / "runs" / "ds" / "no_retrieval"
    nr.mkdir(parents=True)
    run_generate(questions, passages, gen, 2, nr, with_passages=False)
    rows = run_evaluate(questions, passages, None, 2, nr, has_citations=False, system="no_retrieval")
    assert "claim_support_recall" not in rows[0] and "ans_score" in rows[0]

    report = make_report(tmp_path / "runs", baseline="no_retrieval", prices={"gen": (1.0, 1.0)})
    assert "bm25" in report and "no_retrieval" in report and "Paired differences" in report
