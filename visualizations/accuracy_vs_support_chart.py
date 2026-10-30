import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# Display name -> file name
SYSTEMS = {
    "HippoRAG2": "per_question_hippo_bioasq.jsonl",
    "RAG2": "per_question_rag2_bioasq.jsonl",
    "BM25": "per_question_bm25_bioasq.jsonl",
}

CORRECT_THRESHOLD = 0.5  # ans_f1 >= this counts as correct
SUPPORTED_THRESHOLD = 1.0  # claim_support_recall >= this counts as fully supported

GROUPS = [
    ("Correct, fully supported", "#2a78d6"),
    ("Wrong, fully supported", "#eb6834"),
    ("Correct, not fully supported", "#1baf7a"),
    ("Wrong, not fully supported", "#eda100"),
]


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def quadrant_counts(rows, metric):
    counts = [0, 0, 0, 0]
    for r in rows:
        correct = r[metric] >= CORRECT_THRESHOLD
        supported = (r["claim_support_recall"] or 0) >= SUPPORTED_THRESHOLD
        idx = (0 if correct else 1) + (0 if supported else 2)
        counts[idx] += 1
    return counts


def pearson(rows, metric):
    x = np.array([r["claim_support_recall"] or 0 for r in rows], dtype=float)
    y = np.array([r[metric] for r in rows], dtype=float)
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="folder containing the JSONL files")
    ap.add_argument("--out", default="accuracy_vs_support.png")
    ap.add_argument("--metric", default="ans_score",
                    help="answer metric to threshold, e.g. ans_score or ans_f1")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="significance level for the correlation star")
    args = ap.parse_args()

    names, counts, totals, rs, ps = [], [], [], [], []
    for name, fname in SYSTEMS.items():
        rows = load(Path(args.dir) / str.lower(name) / fname)
        names.append(name)
        counts.append(quadrant_counts(rows, args.metric))
        totals.append(len(rows))
        r, p = pearson(rows, args.metric)
        rs.append(r)
        ps.append(p)

    counts = np.array(counts)

    fig, ax = plt.subplots(figsize=(10, 4))
    y = np.arange(len(names))[::-1]
    left = np.zeros(len(names))

    for i, (label, color) in enumerate(GROUPS):
        vals = counts[:, i]
        ax.barh(y, vals, left=left, height=0.5, color=color,
                edgecolor="white", linewidth=2, label=label)
        for yi, v, l in zip(y, vals, left):
            if v >= 12:  # skip labels on very thin segments
                ax.text(l + v / 2, yi, str(v), ha="center", va="center",
                        fontsize=9, color="white")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\nr = {r:.2f}{'*' if p < args.alpha else ''}" for n, r, p in zip(names, rs, ps)])
    ax.set_xlim(0, max(totals))
    ax.set_xlabel("Questions")
    ax.set_title(f"Answer correctness ({args.metric}) vs. Citation Support (LLM Judge)", loc="left",
                 fontsize=12, pad=10)

    ax.legend(ncol=4, fontsize=8, frameon=False, handlelength=1,
              loc="upper left", bbox_to_anchor=(0, -0.22))

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=1)
    ax.set_axisbelow(True)

    fig.text(0.01, 0.01,
             f"Correct = {args.metric} ≥ {CORRECT_THRESHOLD}. "
             f"Fully supported = claim_support_recall ≥ {SUPPORTED_THRESHOLD}. "
             "r = Pearson correlation of the two scores.",
             fontsize=8, color="#898781")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
