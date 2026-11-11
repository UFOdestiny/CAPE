"""Grader-independent answer clusters for short open answers (GPU stage, equivalence ablation only).

Two answers to the same question are equivalent when an NLI model (DeBERTa-large-MNLI, as in semantic entropy)
predicts entailment in both directions between "<question> <answer a>" and "<question> <answer b>"; identical
answer strings are equivalent without a model call. Responses are clustered greedily against the first member of
each cluster, as ``uqbench.verify.cluster_answers`` does; a response without an answer is its own cluster. Clusters
of every QA question (test sets and training pools) go to ``OUTPUT_DIR/<model>/clusters_nli/<dataset>.jsonl``;
the grader (alias match) and its labels are unchanged.

Usage::

    hf download microsoft/deberta-large-mnli --local-dir $MODELS_DIR/deberta-large-mnli
    python -m uqbench.pipeline.nli --models qwen3-8b r1-distill-qwen-7b gpt-oss-20b
"""
from __future__ import annotations

import argparse

from uqbench.config import MODELS_DIR, generated, load_yaml, run_dir
from uqbench.io import read_jsonl, write_jsonl

NLI_MODEL = "deberta-large-mnli"   # microsoft/deberta-large-mnli
BATCH = 256


def greedy_clusters(preds: list[str | None], equal) -> list[int]:
    reps, out = [], []
    for i, a in enumerate(preds):
        if a is None:
            reps.append(None)
            out.append(len(reps) - 1)
            continue
        hit = next((c for c, j in enumerate(reps) if j is not None and (preds[j] == a or equal(preds[j], a))), None)
        if hit is None:
            reps.append(i)
            hit = len(reps) - 1
        out.append(hit)
    return out


def entailment(pairs: list[tuple[str, str]], tok, model) -> list[bool]:
    import torch

    label = {v.lower(): k for k, v in model.config.id2label.items()}["entailment"]
    out = []
    for s in range(0, len(pairs), BATCH):
        enc = tok([p for p, _ in pairs[s : s + BATCH]], [h for _, h in pairs[s : s + BATCH]], padding=True,
                  truncation=True, max_length=256, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out += (model(**enc).logits.argmax(-1) == label).tolist()
    return out


def cluster_dataset(rows: list[dict], tok, model) -> dict[str, list[int]]:
    """Every ordered pair of distinct answers within a question is scored once, then clustered greedily."""
    pairs = {}
    for r in rows:
        answers = sorted({s["pred"] for s in r["samples"] if s["pred"] is not None})
        for a in answers:
            for b in answers:
                if a != b:
                    pairs[(r["question"], a, b)] = None
    keys = list(pairs)
    for key, v in zip(keys, entailment([(f"{q} {a}", f"{q} {b}") for q, a, b in keys], tok, model), strict=True):
        pairs[key] = v
    return {r["uid"]: greedy_clusters([s["pred"] for s in r["samples"]],
                                      lambda a, b, q=r["question"]: pairs[(q, a, b)] and pairs[(q, b, a)])
            for r in rows}


def load(model: str, dataset: str) -> dict[str, list[int]]:
    path = run_dir(model) / "clusters_nli" / f"{dataset}.jsonl"
    return {r["uid"]: r["clusters"] for r in read_jsonl(path)} if path.exists() else {}


def main() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    bench = load_yaml("benchmark")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=bench["backbones"])
    ap.add_argument("--datasets", nargs="+", default=generated(["triviaqa", *bench["training_pools"]]))
    args = ap.parse_args()
    path = str(MODELS_DIR / NLI_MODEL)
    tok = AutoTokenizer.from_pretrained(path)
    nli = AutoModelForSequenceClassification.from_pretrained(path).cuda().eval()
    for m in args.models:
        out = run_dir(m) / "clusters_nli"
        out.mkdir(parents=True, exist_ok=True)
        for name in args.datasets:
            rows = [r for r in read_jsonl(run_dir(m) / "gen" / f"{name}.jsonl") if r["task"] == "qa"]
            if not rows:
                continue
            clusters = cluster_dataset(rows, tok, nli)
            write_jsonl(out / f"{name}.jsonl", ({"uid": u, "clusters": c} for u, c in clusters.items()))
            print(f"[nli] {m}/{name}: {len(rows)} questions", flush=True)


if __name__ == "__main__":
    main()
