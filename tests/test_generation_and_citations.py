from src.eval.citation import SupportJudge, doc_level_citation, judge_claims, split_claims
from src.generate import build_messages, parse_generation
from src.llm import FunctionLLM
from src.schema import Passage, Question


def test_parse_generation():
    raw = "Aspirin inhibits COX [1]. It reduces clotting [2][1]. See also [9].\nFINAL ANSWER: Yes"
    p = parse_generation(raw, 3)
    assert p.answer == "Yes"
    assert p.cited_indices == [1, 2] and p.invalid == [9]
    assert "FINAL ANSWER" not in p.body


def test_parse_generation_without_final_line():
    p = parse_generation("It is BRCA1 [1].", 2)
    assert p.answer == "It is BRCA1." and p.cited_indices == [1]


def test_build_messages_with_and_without_passages():
    q = Question("1", "t", "yesno", "Is X true?", [["yes"]], [])
    ps = [Passage("a::0", "a", "Title A", "text a")]
    with_ctx = build_messages(q, ps)[1]["content"]
    assert "[1] (Title A) text a" in with_ctx and "'yes' or 'no'" in with_ctx
    no_ctx = build_messages(q, [])[1]["content"]
    assert "Passages:" not in no_ctx


def test_split_claims_merges_dangling_citations_and_skips_final():
    body = "Aspirin blocks COX enzymes in platelets [1]. It lowers clot risk. [2]\nYes.\nFINAL ANSWER: yes"
    claims = split_claims(body)
    assert [c.cites for c in claims] == [[1], [2]]
    assert claims[1].text == "It lowers clot risk."


def test_doc_level_citation():
    q = Question("1", "t", "open", "q", [["x"]], ["A", "B"])
    pid2doc = {"a::0": "A", "c::0": "C"}
    r = doc_level_citation(q, ["a::0", "c::0"], pid2doc)
    assert r["cite_doc_precision"] == 0.5 and r["cite_doc_recall"] == 0.5 and r["cite_hit_any"] == 1.0
    r = doc_level_citation(q, [], pid2doc)
    assert r["cite_doc_precision"] is None and r["cite_doc_recall"] == 0.0 and r["n_cited"] == 0


def _judge(rule):
    """rule(statement, passages_block) -> bool"""
    def fn(messages):
        c = messages[0]["content"]
        stmt = c.split("Statement:")[1].split("Passages:")[0].strip()
        block = c.split("Passages:")[1].split("Do the passages")[0]
        return "SUPPORTED" if rule(stmt, block) else "UNSUPPORTED"
    return SupportJudge(FunctionLLM(fn))


def test_unsupported_phrase_not_misread():
    j = _judge(lambda s, b: False)
    assert j.supports("s", ["p"]) is False


def test_judge_claims_alce_precision():
    # claim cites [1] (supports) and [2] (irrelevant): joint supported; [2] alone unsupported and removing it keeps support.
    ctx = ["p1", "p2"]
    texts = {"p1": "ALPHA fact", "p2": "unrelated"}
    j = _judge(lambda s, b: "ALPHA" in b)
    claims = split_claims("Alpha thing is true here [1][2]. Beta thing is claimed [1].")
    m, rows = judge_claims(claims, ctx, texts, j, compute_precision=True)
    # claim 1 supported by p1; citation [1] relevant, [2] irrelevant. claim 2 ("Beta") cites [1] with ALPHA -> judge says supported (rule is crude)
    assert rows[0]["per_cite_relevant"] == [True, False]
    assert m["n_claims"] == 2
    assert m["claim_support_recall"] == 1.0
    assert abs(m["claim_citation_precision"] - 2 / 3) < 1e-9


def test_uncited_and_unsupported_claims_count_against_recall():
    ctx = ["p1"]
    texts = {"p1": "nothing relevant"}
    j = _judge(lambda s, b: False)
    claims = split_claims("This statement has no citation at all. This one cites a passage [1].")
    m, rows = judge_claims(claims, ctx, texts, j)
    assert m["claim_support_recall"] == 0.0 and m["claim_citation_precision"] == 0.0
