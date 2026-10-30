# Convenience targets. Set LLM_API_KEY (and edit configs/default.yaml) first.
SYSTEMS_MAIN = no_retrieval bm25 dense rag2
DS ?= bioasq

test:
	python -m pytest -q

pilot:   ## 20-question pilot of one system on one dataset
	python -m medcite run --dataset $(DS) --system dense --limit 20

all-main:  ## everything except HippoRAG 2 (which runs in its own env, see README)
	for s in $(SYSTEMS_MAIN); do python -m medcite run --dataset $(DS) --system $$s; done

report:
	python -m medcite report --baseline dense
