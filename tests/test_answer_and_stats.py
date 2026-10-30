from src.eval.metrics import bootstrap_ci, cohen_kappa, paired_bootstrap, retrieval_metrics, answer_scores, normalize, parse_yesno, token_f1
from src.schema import Hit, Question


def Q(qtype, groups, gold=()):
    return Question("1", "t", qtype, "q?", groups, list(gold))


def test_normalize_and_f1():
    assert normalize("The  BRCA-1, gene!") == "brca 1 gene"
    assert token_f1("brca1 gene", "brca1 gene") == 1.0
    assert token_f1("a", "b") == 0.0


def test_yesno():
    assert parse_yesno("Yes, it does") == "yes"
    assert answer_scores("no", Q("yesno", [["no"]]))["ans_f1"] == 1.0
    assert answer_scores("yes", Q("yesno", [["no"]]))["ans_f1"] == 0.0
    assert answer_scores("unclear", Q("yesno", [["yes"]]))["ans_f1"] == 0.0


def test_factoid_synonyms_and_contains():
    q = Q("factoid", [["Fam20C", "family with sequence similarity 20C"]])
    assert answer_scores("fam20c", q)["ans_em"] == 1.0
    s = answer_scores("The kinase is Fam20C in humans", q)
    assert s["ans_contains"] == 1.0 and s["ans_score"] == 1.0 and s["ans_em"] == 0.0
    assert answer_scores("something else", q)["ans_score"] == 0.0


def test_list_scoring():
    q = Q("list", [["aspirin"], ["ibuprofen", "advil"], ["naproxen"]])
    s = answer_scores("aspirin, advil", q)
    assert abs(s["ans_contains"] - 2 / 3) < 1e-9
    assert s["ans_f1"] > 0.7


def test_retrieval_metrics():
    pid2doc = {"a::0": "A", "b::0": "B", "c::0": "C"}
    q = Q("open", [["x"]], gold=["A", "B"])
    m = retrieval_metrics(q, [Hit("a::0", 1), Hit("c::0", 0.5)], pid2doc)
    assert m == {"doc_recall": 0.5, "all_gold": 0.0, "hit_any": 1.0}
    m = retrieval_metrics(q, [Hit("a::0", 1), Hit("b::0", 0.5)], pid2doc)
    assert m["all_gold"] == 1.0
    assert retrieval_metrics(Q("open", [["x"]]), [], pid2doc)["doc_recall"] is None


def test_bootstrap_and_kappa():
    ci = bootstrap_ci([1, 0, 1, 1, None, 0, 1, 1])
    assert ci["n"] == 7 and ci["lo"] <= ci["mean"] <= ci["hi"]
    assert bootstrap_ci([None])["mean"] is None
    d = paired_bootstrap([1, 1, 1, 0], [0, 0, 1, 0])
    assert d["diff"] == 0.5 and d["n"] == 4
    assert cohen_kappa(["S", "S", "U", "U"], ["S", "S", "U", "U"]) == 1.0
    assert abs(cohen_kappa(["S", "S", "U", "U"], ["S", "U", "S", "U"])) < 1e-9
