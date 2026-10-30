import statistics
from src.io_utils import read_jsonl
qs = {q["qid"]: q for q in read_jsonl("data/graphrag_medical/questions_eval.jsonl")}
gens = read_jsonl("runs/graphrag_medical/rag2/generations.jsonl")
per = {r["qid"]: r for r in read_jsonl("runs/graphrag_medical/rag2/per_question_rag2_bioasq.jsonl")}
print("median gold words:", statistics.median(len(qs[g["qid"]]["answer_groups"][0][0].split()) for g in gens),
      "| median pred words:", statistics.median(len(g["answer"].split()) for g in gens),
      "| mean ans_f1:", round(sum(r["ans_f1"] for r in per.values()) / len(per), 3))
for g in gens[:5]:
    q = qs[g["qid"]]
    print("Q   :", q["question"]); print("GOLD:", q["answer_groups"][0][0][:250])
    print("PRED:", g["answer"][:250]); print("F1  :", round(per[g["qid"]]["ans_f1"], 2), "\n")