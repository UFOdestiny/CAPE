"""Robustness of the main comparison, from the per-response verdicts (``results/<model>/responses``).

* Aggregation: mean AUROC / Brier / ECE of the main methods over the 30 pairs, weighted by questions,
  balanced over domains (math, multiple choice, short-answer QA), and with each source left out.
* Paired aggregate test: CAPE minus a rival (or one of its ablation rows) in the mean over pairs (ID, OOD, All).
  The backbones answer the same questions, so each bootstrap draw resamples the questions of a test set once for
  all three backbones.
* Projection on questions whose labels violate (A1)-(A2): share of questions and the per-question Brier change.
* Question-level selective answering: each coherent method answers with its most probable cluster; answer
  accuracy, AURC and accuracy at 50% / 80% coverage, with the same paired aggregate bootstrap.

Writes ``results/robustness_{aggregate,paired,violations,selective,selective_paired}.csv``.

Usage::

    python -m uqbench.studies.robustness
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from uqbench.config import RESULTS_DIR, load_yaml

BENCH = load_yaml("benchmark")
BACKBONES = BENCH["backbones"]
SETS = BENCH["test_sets"]
OOD = ("gpqa", "gpqa_ext", "supergpqa")
DOMAIN = {**{d: "math" for d in BENCH["math_sets"]}, "gpqa": "mc", "gpqa_ext": "mc", "mmlupro": "mc",
          "supergpqa": "mc", "triviaqa": "qa"}
SOURCE = {"math500": "MATH", "math_rest": "MATH", "olympiadbench": "OlympiadBench", "minerva": "Minerva",
          "aime": "AIME", "gpqa": "GPQA", "gpqa_ext": "GPQA", "mmlupro": "MMLU-Pro", "supergpqa": "SuperGPQA",
          "triviaqa": "TriviaQA"}
# label -> (column of the probability, column of the AUROC score)
METHODS = {"CAPE": ("p:CAPE", "raw:CAPE"),
           "Coherent combiner": ("p:Stacker + probe, gated +Coh", "p:Stacker + probe, gated +Coh"),
           "Modular": ("p:Coherent stacker + learned null", "p:Coherent stacker + learned null"),
           "Combiner": ("p:Stacker", "raw:Stacker"),
           "SEP+C": ("p:SEP+Coh", "p:SEP+Coh"),
           "ReasonProbe+C": ("p:ReasonProbe+Coh", "p:ReasonProbe+Coh"),
           "DeepConf+C": ("p:DeepConf+Coh", "p:DeepConf+Coh"),
           "DeepConf×SC": ("p:DeepConf×SC", "raw:DeepConf×SC"),
           "Sem. Energy": ("p:SemEnergy", "raw:SemEnergy"),
           "SC": ("p:SC", "raw:SC"),
           # CAPE's ablation rows (uqbench.pipeline.evaluate), tested with the same paired bootstrap
           **{v: (f"p:{v}", f"raw:{v}") for v in ("CAPE w/o null", "CAPE w/o votes", "CAPE w/o probe")}}
RIVALS = [m for m in METHODS if m != "CAPE"]
# coherent forecasts whose most probable cluster answers the question
SELECTIVE = {"CAPE": "p:CAPE", "Coherent combiner": "p:Stacker + probe, gated +Coh",
             "Modular": "p:Coherent stacker + learned null", "Combiner+C": "p:Stacker+Coh",
             "SEP+C": "p:SEP+Coh", "DeepConf+C": "p:DeepConf+Coh", "SC": "p:SC+Coh"}
INCOHERENT = ["DeepConf", "CDG", "SelfCertainty", "LRMVerbal", "ReasonProbe", "SNNE", "SEP", "ORM", "SemEnergy",
              "CoCoA", "SE"]
K = 8


def fast_auroc(score: np.ndarray, y: np.ndarray) -> float:
    pos = y.sum()
    if pos == 0 or pos == len(y):
        return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos)))


def fast_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    gap = np.bincount(idx, weights=y - p, minlength=n_bins)
    return float(np.abs(gap).sum() / len(p))


def load_set(name: str) -> dict:
    """Per backbone: arrays of shape (questions, K) ordered by uid, so the backbones share question indices."""
    out = {}
    for bb in BACKBONES:
        d = pd.read_parquet(RESULTS_DIR / bb / "responses" / f"{name}.parquet").sort_values(["uid", "k"])
        assert (d.groupby("uid").size() == K).all()
        cols = {"y": d["correct"].to_numpy(float), "cluster": d["cluster"].to_numpy(),
                "answered": ~d["no_answer"].to_numpy(bool)}
        for m, (pcol, scol) in METHODS.items():
            cols[f"p|{m}"], cols[f"s|{m}"] = d[pcol].to_numpy(float), d[scol].to_numpy(float)
        for m, pcol in SELECTIVE.items():
            cols[f"sel|{m}"] = d[pcol].to_numpy(float)
        for m in INCOHERENT:
            cols[f"cal|{m}"], cols[f"coh|{m}"] = d[f"p:{m}"].to_numpy(float), d[f"p:{m}+Coh"].to_numpy(float)
        out[bb] = {k: v.reshape(-1, K) for k, v in cols.items()}
    return out


def response_metrics(a: dict, idx: np.ndarray) -> dict[tuple[str, str], float]:
    y = a["y"][idx].ravel()
    res = {}
    for m in METHODS:
        p = a[f"p|{m}"][idx].ravel()
        res[(m, "auroc")] = fast_auroc(a[f"s|{m}"][idx].ravel(), y)
        res[(m, "brier")] = float(np.mean((p - y) ** 2))
        res[(m, "ece")] = fast_ece(p, y)
    return res


def answer_choice(a: dict, m: str) -> tuple[np.ndarray, np.ndarray]:
    """Per question: mass of the most probable answered cluster and whether it is correct."""
    p, cl, ok, y = a[f"sel|{m}"], a["cluster"], a["answered"], a["y"]
    conf, correct = np.zeros(len(p)), np.zeros(len(p))
    for i in range(len(p)):
        best, best_key = None, None
        for c in np.unique(cl[i][ok[i]]):
            mem = (cl[i] == c) & ok[i]
            key = (p[i][mem].mean(), mem.sum())
            if best_key is None or key > best_key:
                best, best_key = mem, key
        if best is not None:
            conf[i], correct[i] = best_key[0], y[i][best][0]
    return conf, correct


def selective_metrics(conf: np.ndarray, ok: np.ndarray) -> dict[str, float]:
    order = np.lexsort((np.random.default_rng(0).random(len(conf)), -conf))
    hits = np.cumsum(ok[order])
    n = len(ok)
    acc_at = lambda c: float(hits[max(int(round(c * n)), 1) - 1] / max(int(round(c * n)), 1))  # noqa: E731
    return {"accuracy": float(ok.mean()), "aurc": float(np.mean(1 - hits / np.arange(1, n + 1))),
            "acc_at_50": acc_at(0.5), "acc_at_80": acc_at(0.8)}


def boot_set(args) -> tuple[str, np.ndarray, np.ndarray, dict]:
    """Bootstrap draws of one test set: response metrics (B, backbones, methods, 3) and selective metrics."""
    name, n_boot, seed = args
    data = load_set(name)
    nq = next(iter(data.values()))["y"].shape[0]
    choice = {bb: {m: answer_choice(a, m) for m in SELECTIVE} for bb, a in data.items()}
    rng = np.random.default_rng(seed + SETS.index(name))
    resp = np.full((n_boot + 1, len(BACKBONES), len(METHODS), 3), np.nan)
    sel = np.full((n_boot + 1, len(BACKBONES), len(SELECTIVE), 4), np.nan)
    for b in range(n_boot + 1):
        idx = np.arange(nq) if b == 0 else rng.integers(0, nq, nq)   # draw 0: the observed sample
        for i, bb in enumerate(BACKBONES):
            r = response_metrics(data[bb], idx)
            resp[b, i] = [[r[(m, k)] for k in ("auroc", "brier", "ece")] for m in METHODS]
            for j, m in enumerate(SELECTIVE):
                s = selective_metrics(choice[bb][m][0][idx], choice[bb][m][1][idx])
                sel[b, i, j] = [s["accuracy"], s["aurc"], s["acc_at_50"], s["acc_at_80"]]
    info = {bb: (nq, float(a["y"].mean())) for bb, a in data.items()}   # questions, accuracy
    return name, resp, sel, info


def violations(name: str) -> list[dict]:
    rows = []
    for bb, a in load_set(name).items():
        y, cl, ok = a["y"], a["cluster"], a["answered"]
        bad = np.zeros(len(y), bool)
        for i in range(len(y)):
            correct = {c for c in np.unique(cl[i][ok[i]]) if y[i][(cl[i] == c) & ok[i]].max() == 1}
            mixed = any(np.ptp(y[i][(cl[i] == c) & ok[i]]) > 0 for c in np.unique(cl[i][ok[i]]))
            bad[i] = mixed or len(correct) > 1 or y[i][~ok[i]].max(initial=0) == 1
        for m in INCOHERENT:
            d = ((a[f"coh|{m}"] - y) ** 2).sum(1) - ((a[f"cal|{m}"] - y) ** 2).sum(1)   # per-question change
            rows.append({"backbone": bb, "dataset": name, "method": m, "questions": len(y),
                         "violating": int(bad.sum()), "raised": int((d[bad] > 1e-9).sum()),
                         "lowered": int((d[bad] < -1e-9).sum()), "change_violating": float(d[bad].sum()),
                         "change_all": float(d.sum())})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=BENCH["bootstrap"])
    ap.add_argument("--workers", type=int, default=len(SETS))
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers) as ex:
        draws = {n: (r, s, i) for n, r, s, i in ex.map(boot_set, [(n, args.boot, BENCH["seed"]) for n in SETS])}
        viol = [row for rows in ex.map(violations, SETS) for row in rows]
    pd.DataFrame(viol).to_csv(RESULTS_DIR / "robustness_violations.csv", index=False, float_format="%.5f")

    pairs = [(bb, d) for d in SETS for bb in BACKBONES]
    resp = np.stack([draws[d][0][:, BACKBONES.index(bb)] for bb, d in pairs], axis=1)   # (B, pairs, methods, 3)
    sel = np.stack([draws[d][1][:, BACKBONES.index(bb)] for bb, d in pairs], axis=1)
    nq = np.array([draws[d][2][bb][0] for bb, d in pairs], float)
    metrics, methods = ("auroc", "brier", "ece"), list(METHODS)

    # aggregation schemes on the observed sample
    obs = resp[0]
    schemes = {"pairs (unweighted)": np.ones(len(pairs)), "questions": nq}
    dom = np.array([DOMAIN[d] for _, d in pairs])
    schemes["domains (balanced)"] = np.array([1 / (dom == x).sum() for x in dom])
    src = np.array([SOURCE[d] for _, d in pairs])
    for s in dict.fromkeys(src):
        schemes[f"without {s}"] = (src != s).astype(float)
    for bb in BACKBONES:
        schemes[f"without {bb}"] = np.array([b != bb for b, _ in pairs], float)
    lowest = int(np.argmin([draws[d][2][bb][1] for bb, d in pairs]))
    schemes["without the lowest-accuracy pair"] = (np.arange(len(pairs)) != lowest).astype(float)
    rows = []
    for name, w in schemes.items():
        for j, m in enumerate(methods):
            rows.append({"scheme": name, "method": m,
                         **{k: float(np.average(obs[:, j, i], weights=w)) for i, k in enumerate(metrics)}})
    agg = pd.DataFrame(rows)
    agg.to_csv(RESULTS_DIR / "robustness_aggregate.csv", index=False, float_format="%.4f")

    # paired aggregate bootstrap of CAPE minus each rival (positive favors CAPE)
    groups = {"ID": np.array([d not in OOD for _, d in pairs]), "OOD": np.array([d in OOD for _, d in pairs]),
              "All": np.ones(len(pairs), bool)}
    rows = []
    for g, mask in groups.items():
        for r in RIVALS:
            j = methods.index(r)
            for i, k in enumerate(metrics):
                sign = 1.0 if k == "auroc" else -1.0
                diff = sign * (np.nanmean(resp[:, mask, 0, i], 1) - np.nanmean(resp[:, mask, j, i], 1))
                lo, hi = np.percentile(diff[1:], [2.5, 97.5])
                rows.append({"group": g, "rival": r, "metric": k, "gain": diff[0], "lo": lo, "hi": hi,
                             "p_le_0": float(np.mean(diff[1:] <= 0))})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "robustness_paired.csv", index=False, float_format="%.5f")

    # question-level selective answering
    sel_names, sel_metrics = list(SELECTIVE), ("accuracy", "aurc", "acc_at_50", "acc_at_80")
    rows = [{"backbone": bb, "dataset": d, "method": m, **dict(zip(sel_metrics, sel[0, p, j], strict=True))}
            for p, (bb, d) in enumerate(pairs) for j, m in enumerate(sel_names)]
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "robustness_selective.csv", index=False, float_format="%.5f")
    rows = []
    for j, r in enumerate(sel_names[1:], start=1):
        for i, k in enumerate(sel_metrics):
            sign = -1.0 if k == "aurc" else 1.0
            diff = sign * (sel[:, :, 0, i].mean(1) - sel[:, :, j, i].mean(1))
            lo, hi = np.percentile(diff[1:], [2.5, 97.5])
            rows.append({"rival": r, "metric": k, "cape": sel[0, :, 0, i].mean(), "rival_value": sel[0, :, j, i].mean(),
                         "gain": diff[0], "lo": lo, "hi": hi,
                         "pairs_better": int((sign * (sel[0, :, 0, i] - sel[0, :, j, i]) > 0).sum())})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "robustness_selective_paired.csv", index=False, float_format="%.5f")
    print(agg[agg.scheme.isin(["pairs (unweighted)", "questions", "domains (balanced)"])].to_string())


if __name__ == "__main__":
    main()
