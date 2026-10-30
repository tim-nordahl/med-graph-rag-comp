# MedCite

Evaluating answer correctness **and citation faithfulness** for different retrieval patterns on biomedical QA.

| System | What it is here                                                                                                                                                                                                                                                         |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `hipporag2` | Adapter around the official HippoRAG package (graph + Personalized PageRank).                                                                                                                                                                                           |
| `rag2` | **RAG²-style**: LLM-written rationale used as the query to a MedCPT retriever (+ optional MedCPT cross-encoder). Does *not* include the paper's perplexity-filtered retriever training; swap in an official checkpoint if one exists. Call it "RAG²-style" in write-ups. |
| `bm25` | Pure-Python lexical baseline.                                                                                                                                                                                                                                           |

Datasets: **GraphRAG-Bench (Medical)** (multi-hop, textbook), **BioASQ** (mostly single-hop, PubMed, gold documents).

**Experimental and Control variables:** Retrieval systems are the experimental condition. Passages 
are held consistent across retrieval systems. Citations are made by a generator, not a native mechanism per
retrieval system.

## Metrics 

- **Answer** (`ans_em`, `ans_f1`): unofficial local proxies.
- **Retrieval** (`doc_recall`): gold documents found in top-k.
- **Claim-level citation (LLM judge)**: each answer *sentence* is checked against the passages it cites, jointly.
  `claim_support_recall` = supported sentences / sentences (uncited sentences count as unsupported).
  `claim_citation_precision` = ALCE-style relevant-citation rate. Units are sentences.

## Setup (Linux)

```bash
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt && pip install -e .
export LLM_API_KEY=...            # any OpenAI-compatible provider; edit base_url/model in configs/default.yaml
python -m pytest -q               # offline tests, no network or GPU needed
```

### HippoRAG 2 setup (separate environment recommended)
```bash
python3.10 -m venv .venv-hippo && source .venv-hippo/bin/activate # Hipporag requires torch version 2.5.1 which requires python 3.10
python3.10 -m pip install torch==2.5.1
python3.10 -m pip install hipporag            # check the HippoRAG README for current instructions
python3.10 -m pip install -e . --no-deps && pip install numpy pyyaml openai requests tqdm
```
## Data

BioASQ needs registration; download the training JSON from the BioASQ site yourself.
```bash
python -m src prepare-bioasq --input BioASQ-trainingXb.json --out data/bioasq --n 200
python -m src fetch-pubmed   --dataset-dir data/bioasq --email you@example.com
python -m src build-corpus   --dataset-dir data/bioasq
```
```bash
 python -m src.data.graphrag_bench --questions [questions_path]     
    --corpus [corpus_path] --out [out_path] --n 200 --types fact complex```
````
## Running
```bash
python -m src run --dataset bioasq --system dense --limit 20     # pilot
for s in no_retrieval bm25 rag2; do python -m src run --dataset bioasq --system $s; done
python -m src run --dataset graphrag_medical --system dense --limit 20     # pilot
for s in no_retrieval bm25 rag2; do python -m src run --dataset graphrag_medical --system $s; done
python -m src report --baseline bm25                            # tables + paired differences + est. API cost
```

```bash
# in .venv-hippo
python3.10 -m src run --dataset bioasq --system hipporag2 --stages retrieve
python3.10 -m src run --dataset graphrag_medical --system hipporag2 --stages retrieve
# back in .venv
source .venv/bin/activate
python -m src run --dataset bioasq --system hipporag2 --stages generate evaluate
python -m src run --dataset graphrag_medical --system hipporag2 --stages generate evaluate
```
**Verify the adapter** (`medcite/retrievers/hipporag2.py`) against your installed version: constructor arguments, `.index(docs=...)`,
`.retrieve(queries=..., num_to_retrieve=...)` and the `.docs` / `.doc_scores` fields. Inspect `unmapped_hits` in `retrieval.jsonl` traces;
if it is non-zero the text-to-id mapping is failing. **Spot-check extracted triples** with your chosen LLM before the full run.
HippoRAG's own OpenIE calls bypass this repo's token counter, so check provider dashboard spend details.

Everything LLM-related is cached in `.cache/llm/*.sqlite`, so reruns after a crash cost nothing for completed calls.
Outputs per run: `runs/<dataset>/<system>/{retrieval,generations,per_question,judgments}.jsonl`, `summary.json`, `usage.jsonl`.
