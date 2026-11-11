"""Cost-sensitive answering with the plug-in Bayes rule, from the per-response verdicts (``results/<model>/responses``).

Each question is answered with a method's most probable answer cluster, or abstained on. A correct answer gains 1,
a wrong one costs (1 - alpha) / alpha and abstaining costs 0, so answering iff the answer's probability is at least
1 - alpha is the Bayes decision for a calibrated forecast: the rule has no parameter to tune and reads each method's
(validation-calibrated or projected) probabilities as they are. A method's probability of an answer is the mean of
its per-response probabilities over the cluster (for a coherent forecast, the cluster's mass).

* Utility (per question), coverage, risk among the answered, and on how many questions without a correct sample
  (all of them, and the unanimous ones) the method still answers.
* Answer mode ``own`` (each method's most probable cluster) and ``majority`` (every method scores the majority answer,
  so only the confidence differs).
* Hindsight threshold: the best threshold per pair chosen on the test labels, an upper bound that separates ranking
  from calibration; ``regret`` is its utility minus the plug-in utility.
* Validation threshold (mode ``validation``): each method's threshold chosen on the training pool's validation
  questions for each alpha, for the methods whose validation scores are stored (CAPE and the published estimators,
  raw and projected).
* Paired aggregate bootstrap of CAPE minus each rival in the mean utility over pairs (All, ID, OOD); the backbones
  answer the same questions, so each draw resamples a test set's questions once for all three backbones.

Writes ``results/abstention.csv`` (per pair) and ``results/abstention_paired.csv``.

Usage::

    python -m uqbench.studies.abstention
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from uqbench.config import RESULTS_DIR, load_yaml, scores_dir

BENCH = load_yaml("benchmark")
BACKBONES = BENCH["backbones"]
SETS = BENCH["test_sets"]
OOD = ("gpqa", "gpqa_ext", "supergpqa")
ALPHAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
MODES = ("own", "majority")
PUBLISHED = [m for m in BENCH["methods"]["baselines"] + BENCH["methods"]["references"]
             if m not in BENCH["methods"]["constructed"]]
# label -> probability column of the per-response verdicts
METHODS = {"CAPE": "p:CAPE",
           **{v: f"p:{v}" for v in ("CAPE w/o null", "CAPE w/o votes", "CAPE w/o probe")},
           "Comb.+C": "p:Stacker + probe, gated +Coh", "Modular": "p:Coherent stacker + learned null",
           "Combiner": "p:Stacker", "Combiner+C": "p:Stacker+Coh", "DeepConf×SC": "p:DeepConf×SC",
           **{m: f"p:{m}" for m in PUBLISHED}, **{f"{m}+C": f"p:{m}+Coh" for m in PUBLISHED}}
RIVALS = [m for m in METHODS if m != "CAPE"]
VALIDATED = ["CAPE", *PUBLISHED, *(f"{m}+C" for m in PUBLISHED)]
K = 8


def load_pair(model: str, name: str) -> dict:
    """Arrays of shape (questions, K) ordered by uid, so the backbones share question indices."""
    d = pd.read_parquet(RESULTS_DIR / model / "responses" / f"{name}.parquet").sort_values(["uid", "k"])
    assert (d.groupby("uid").size() == K).all()
    out = {"y": d["correct"].to_numpy(float), "cluster": d["cluster"].to_numpy(),
           "answered": ~d["no_answer"].to_numpy(bool), **{m: d[c].to_numpy(float) for m, c in METHODS.items()}}
    return {k: v.reshape(-1, K) for k, v in out.items()}


def load_validation(model: str) -> dict:
    """The validation questions of the training pool (K = 4) in the layout of :func:`load_pair`, with the stored
    calibrated scores of ``VALIDATED`` (projected ones projected here)."""
    from cape.coherence import project_coherent
    from uqbench.records import load_pool

    z = np.load(scores_dir(model) / "val.npz")
    val = [q for q in load_pool(model, with_replay=False, with_followup=False) if q.split == "val"]
    assert (np.repeat([q.uid for q in val], [q.k for q in val]) == z["uid"]).all()
    k = val[0].k
    has = np.array([[s["pred"] is not None for s in q.samples] for q in val])
    out = {"y": np.stack([q.labels for q in val]), "cluster": np.array([q.clusters for q in val]), "answered": has}
    for m in VALIDATED:
        p = z[f"{m.removesuffix('+C')}__prob"].reshape(-1, k)
        if m.endswith("+C"):
            p = np.stack([project_coherent(pi, q.clusters, h) for pi, q, h in zip(p, val, has, strict=True)])
        out[m] = p
    return out


def best_threshold(conf: np.ndarray, correct: np.ndarray, alpha: float) -> float:
    """The threshold of :func:`hindsight` (answer iff confidence >= it); inf when abstaining everywhere is best."""
    order = np.argsort(-conf, kind="stable")
    gain = np.cumsum(np.where(correct[order] == 1, 1.0, -(1 - alpha) / alpha))
    ends = np.r_[np.flatnonzero(np.diff(conf[order]) != 0), len(conf) - 1]
    j = ends[np.argmax(gain[ends])]
    return float(conf[order][j]) if gain[j] > 0 else float("inf")


def validation_thresholds(model: str) -> dict[tuple[str, float], float]:
    a = load_validation(model)
    table = cluster_table(a)
    out = {}
    for m in VALIDATED:
        conf, correct = choices(a, table, m, "own")
        out.update({(m, alpha): best_threshold(conf, correct, alpha) for alpha in ALPHAS})
    return out


def cluster_table(a: dict) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per question: member masks of the answered clusters, their sizes and correctness."""
    out = []
    for cl, ok, y in zip(a["cluster"], a["answered"], a["y"], strict=True):
        ids = np.unique(cl[ok])
        masks = np.array([(cl == c) & ok for c in ids]).reshape(len(ids), len(cl))
        out.append((masks, masks.sum(1), np.array([y[m].max() for m in masks])))
    return out


def choices(a: dict, table, m: str, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Per question: the probability of the answer given and whether it is correct (0, 0 without an answer)."""
    conf, correct = np.zeros(len(table)), np.zeros(len(table))
    for i, (masks, size, ok) in enumerate(table):
        if not len(size):
            continue
        p = masks @ a[m][i] / size
        j = int(np.argmax(size)) if mode == "majority" else int(np.lexsort((-size, -p))[0])
        conf[i], correct[i] = p[j], ok[j]
    return conf, correct


def hindsight(conf: np.ndarray, correct: np.ndarray, alpha: float) -> float:
    """Best mean utility over thresholds on the confidence (answer the top-ranked questions)."""
    order = np.argsort(-conf, kind="stable")
    gain = np.cumsum(np.where(correct[order] == 1, 1.0, -(1 - alpha) / alpha))
    ends = np.r_[np.flatnonzero(np.diff(conf[order]) != 0), len(conf) - 1]   # cut only between distinct values
    return float(max(0.0, gain[ends].max()) / len(conf))


def score_set(args) -> tuple[str, list[dict], dict]:
    """Per-pair rows and per-question utilities {(backbone, method, alpha, mode): array} of one test set."""
    name, thresholds = args
    rows, per_q = [], {}
    for bb in BACKBONES:
        a = load_pair(bb, name)
        table = cluster_table(a)
        none = np.array([not ok.any() for _, _, ok in table])
        unanimous = np.array([len(size) == 1 and size[0] == K for _, size, _ in table])
        for m in METHODS:
            for mode in (*MODES, "validation"):
                if mode == "validation" and m not in VALIDATED:
                    continue
                conf, correct = choices(a, table, m, "own" if mode == "validation" else mode)
                for alpha in ALPHAS:
                    cut = thresholds[bb][(m, alpha)] if mode == "validation" else 1 - alpha
                    u = np.where(conf >= cut, np.where(correct == 1, 1.0, -(1 - alpha) / alpha), 0.0)
                    ans = conf >= cut
                    per_q[(bb, m, alpha, mode)] = u
                    rows.append({"backbone": bb, "dataset": name, "method": m, "mode": mode, "alpha": alpha,
                                 "questions": len(u), "utility": u.mean(), "coverage": ans.mean(),
                                 "risk": float((ans & (correct == 0)).sum() / max(ans.sum(), 1)),
                                 "answered_none_correct": float((ans & none).mean()),
                                 "answered_unanimous_wrong": float((ans & none & unanimous).mean()),
                                 "hindsight": hindsight(conf, correct, alpha),
                                 "oracle": float(correct.mean())})
    return name, rows, per_q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=BENCH["bootstrap"])
    ap.add_argument("--workers", type=int, default=len(SETS))
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers) as ex:
        thresholds = dict(zip(BACKBONES, ex.map(validation_thresholds, BACKBONES), strict=True))
        scored = list(ex.map(score_set, [(name, thresholds) for name in SETS]))
    df = pd.DataFrame([r for _, rows, _ in scored for r in rows])
    df["regret"] = df["hindsight"] - df["utility"]
    df.to_csv(RESULTS_DIR / "abstention.csv", index=False, float_format="%.5f")

    per_q = {name: u for name, _, u in scored}
    rng = np.random.default_rng(BENCH["seed"])
    nq = {name: len(per_q[name][(BACKBONES[0], "CAPE", ALPHAS[0], "own")]) for name in SETS}
    # row 0: the observed sample
    draws = {name: np.vstack([np.arange(n), rng.integers(0, n, (args.boot, n))]) for name, n in nq.items()}
    groups = {"All": SETS, "ID": [d for d in SETS if d not in OOD], "OOD": list(OOD)}
    rows = []
    for mode in (*MODES, "validation"):
        rivals = RIVALS if mode != "validation" else [m for m in VALIDATED if m != "CAPE"]
        for alpha in ALPHAS:
            boot = {(bb, d, m): per_q[d][(bb, m, alpha, mode)][draws[d]].mean(1)
                    for d in SETS for bb in BACKBONES for m in ["CAPE", *rivals]}   # (B + 1,) mean utility per pair
            for g, sets in groups.items():
                pairs = [(bb, d) for d in sets for bb in BACKBONES]

                def mean_over_pairs(m, pairs=pairs, boot=boot):
                    return np.mean([boot[(bb, d, m)] for bb, d in pairs], axis=0)

                cape = mean_over_pairs("CAPE")
                for r in rivals:
                    diff = cape - mean_over_pairs(r)
                    lo, hi = np.percentile(diff[1:], [2.5, 97.5])
                    better = sum(per_q[d][(bb, "CAPE", alpha, mode)].mean() > per_q[d][(bb, r, alpha, mode)].mean()
                                 + 1e-12 for bb, d in pairs)
                    rows.append({"mode": mode, "alpha": alpha, "group": g, "rival": r, "cape": cape[0],
                                 "rival_utility": cape[0] - diff[0], "gain": diff[0], "lo": lo, "hi": hi,
                                 "p_le_0": float(np.mean(diff[1:] <= 0)), "pairs_better": better,
                                 "pairs": len(pairs)})
        print(f"[abstention] {mode} done", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "abstention_paired.csv", index=False, float_format="%.5f")


if __name__ == "__main__":
    main()
