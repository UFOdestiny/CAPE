"""Dependence study: shared-prefix sampling changes how the K responses depend on each other, not what a response is.

Each regime samples the K responses of a question as continuations of one response of the base run after the first
``fraction`` of the median trace length (``uqbench.pipeline.generate.shared_prefixes``; fraction 0 is the main run).
The cut is a stopping time, so every response is still an exact draw from the model: any single-response
estimator has the same per-response calibration in every regime, while the probability that no sampled answer is
correct changes with the dependence (prop. "single-response scores do not identify none correct"). Three stages:

* ``structure`` (generations only): per-response accuracy (invariant), correctness ICC, unanimity, the error rate of
  unanimous answers, the none-correct rate, self-consistency and the vote-share floor, and the effective number of
  independent responses (the iid K at which the main run reaches the same vote-share floor / none-correct rate).
* ``methods`` (needs replay; SemEnergy also the energy stage): estimators fitted once on the iid training pool and
  applied unchanged to every regime: response metrics, the per-response calibration of the probe, and the
  none-correct metrics of coherent estimators.
* ``refit``: CAPE's answer model refitted per regime by 5-fold cross-fitting over questions (probe kept), with the
  vote exponent α and the within-cluster evidence correlation r learned, against the frozen model and against the
  projected probe whose none-correct residual is recalibrated on the same folds.
* ``bootstrap``: question-level bootstrap (stratified by test set, the same questions in every regime) of the change
  of each metric from the iid run to every shared-prefix regime, and of the paired gaps between methods.

Usage::

    python -m uqbench.studies.dependence --base qwen3-8b --stages structure methods refit
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from baselines.base import labels_of
from uqbench import metrics
from uqbench.calibration import select_calibrator
from uqbench.config import RESULTS_DIR, load_yaml, model_config, run_dir
from uqbench.diagnostics import event_scores
from uqbench.records import load_generated, load_pool
from uqbench.studies.decision import agreement_floor, vote_shares

FRACTIONS = (0.25, 0.5, 0.75)
SETS = ("math500", "olympiadbench", "minerva", "aime", "gpqa", "triviaqa")
FOLDS = 5    # cross-fitting folds of the refit stage
DRAWS = 20   # random K'-subsets of the main run per question for the iid reference curves
NO_FOLLOWUP = {"ptrue_logp1": None, "ptrue_logp0": None, "verbal_conf": None}


def available(base: str) -> dict[str, float]:
    regimes = {base: 0.0, **{f"{base}-pf{round(100 * f)}": f for f in FRACTIONS}}
    return {m: f for m, f in regimes.items() if all((run_dir(m) / "gen" / f"{d}.jsonl").exists() for d in SETS)}


def load(model: str, dataset: str, uids: set[str] | None = None, with_replay: bool = False) -> list:
    """Questions of one regime. Math labels of the shared-prefix runs come from the same rules and LLM grader as the
    main run (``uqbench.labeling.regrade``, verdicts reused per answer string)."""
    shared = "prefix" in model_config(model)
    qs = load_generated(model, dataset, with_replay=with_replay, with_followup=not shared and with_replay)
    if shared:   # no follow-up stage on these runs; CAPE's probe model reads no follow-up column
        qs = [replace(q, followup=[NO_FOLLOWUP] * q.k) for q in qs]
    return [q for q in qs if uids is None or q.uid in uids]


def common_uids(regimes) -> dict[str, set[str]]:
    out = {}
    for d in SETS:
        sets = [{q.uid for q in load(m, d)} for m in regimes]
        out[d] = set.intersection(*sets)
    return out


def icc(z: np.ndarray) -> float:
    """One-way ANOVA intraclass correlation of correctness, questions x K responses."""
    k = z.shape[1]
    msb = k * z.mean(1).var(ddof=1)
    msw = ((z - z.mean(1, keepdims=True)) ** 2).sum() / (z.shape[0] * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw)) if msb + msw > 0 else float("nan")


def unanimous(q) -> bool:
    return all(s["pred"] is not None for s in q.samples) and len(set(q.clusters)) == 1


def structure_row(qs) -> dict:
    z = np.stack([q.labels for q in qs])
    y = labels_of(qs)
    v = vote_shares(qs)
    una = [q for q in qs if unanimous(q)]
    return {"questions": len(qs), "accuracy": float(z.mean()), "icc": icc(z),
            "none_rate": float((z.max(1) == 0).mean()),
            "unanimous_share": len(una) / len(qs),
            "unanimous_error": float(np.mean([q.labels[0] == 0 for q in una])) if una else np.nan,
            "distinct_answers": float(np.mean([len({c for c, s in zip(q.clusters, q.samples, strict=True)
                                                    if s["pred"] is not None}) for q in qs])),
            "sc_auroc": metrics.auroc(v, y), "sc_brier": metrics.brier(v, y), "vote_floor": agreement_floor(v, y)}


def iid_curve(qs, rng) -> pd.DataFrame:
    """Vote-share floor and none-correct rate of the main run at K' = 1..K (random subsets; the stored clustering
    restricted to the subset)."""
    rows = []
    for k in range(1, qs[0].k + 1):
        floors, nones = [], []
        for _ in range(DRAWS if k < qs[0].k else 1):
            sub = [q.subset(sorted(rng.choice(q.k, k, replace=False).tolist())) for q in qs]
            floors.append(agreement_floor(vote_shares(sub), labels_of(sub)))
            nones.append(np.mean([s.labels.max() == 0 for s in sub]))
        rows.append({"K": k, "vote_floor": float(np.mean(floors)), "none_rate": float(np.mean(nones))})
    return pd.DataFrame(rows)


def effective_k(curve: pd.DataFrame, column: str, value: float) -> float:
    """K' at which the (monotone decreasing) iid curve reaches ``value``, linearly interpolated in K'."""
    ks, vs = curve["K"].to_numpy(float), curve[column].to_numpy()
    if value >= vs[0]:
        return 1.0
    if value <= vs[-1]:
        return float(ks[-1])
    i = int(np.argmax(vs <= value))
    return float(ks[i - 1] + (vs[i - 1] - value) / (vs[i - 1] - vs[i]))


def stage_structure(base, regimes, uids) -> None:
    rng = np.random.default_rng(0)
    rows, curves = [], []
    for d in SETS:
        curve = iid_curve(load(base, d, uids[d]), rng)
        curves.append(curve.assign(dataset=d))
        for m, f in regimes.items():
            r = {"dataset": d, "regime": m, "fraction": f, **structure_row(load(m, d, uids[d]))}
            r["k_eff_floor"] = effective_k(curve, "vote_floor", r["vote_floor"])
            r["k_eff_none"] = effective_k(curve, "none_rate", r["none_rate"])
            rows.append(r)
        print(f"[dependence] structure {d} done", flush=True)
    out = RESULTS_DIR / base
    pd.DataFrame(rows).to_csv(out / "dependence_structure.csv", index=False, float_format="%.4f")
    pd.concat(curves).to_csv(out / "dependence_iid_curve.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def fit_estimators(base: str):
    """Estimators fitted on the iid pool, their validation calibration maps, and the projected single-response ones."""
    from baselines import SC, SEP, DeepConf, DeepConfSC, ReasonProbe, SelfCertainty, SemEnergy
    from cape import CAPE, CoherentProjection
    from cape.model import HiddenProbe

    pool = load_pool(base)
    train, val = [q for q in pool if q.split == "train"], [q for q in pool if q.split == "val"]
    yv, gv = labels_of(val), np.repeat([q.uid for q in val], [q.k for q in val])
    singles = [DeepConf(), SelfCertainty(), ReasonProbe(), SEP()]
    others = [SemEnergy(), DeepConfSC(), SC(), CAPE(), CAPE(use_null=False, name="CAPE w/o null", main=False)]
    methods, cal = {}, {}
    for m in singles + others:
        m.fit(train, val)
        cal[m.name] = select_calibrator(m.score_all(val), yv, gv, m.probabilistic)[0]
        methods[m.name] = m
        print(f"[dependence] fitted {m.name}", flush=True)
    for m in singles:
        p = CoherentProjection(m, cal[m.name])
        methods[p.name], cal[p.name] = p, (lambda x: x)
    probe = HiddenProbe().fit(train, val)
    pcal = select_calibrator(np.concatenate([probe.logits(q) for q in val]), yv, gv, False)[0]
    return methods, cal, probe, pcal


def none_metrics(prefix: str, e: np.ndarray, none: np.ndarray, una: np.ndarray) -> dict:
    return {f"{prefix}none_auroc": metrics.auroc(e, none), f"{prefix}none_brier": float(np.mean((e - none) ** 2)),
            f"{prefix}none_mean": float(e.mean()),
            f"{prefix}none_mean_unanimous": float(e[una].mean()) if una.any() else np.nan,
            f"{prefix}none_brier_unanimous": float(np.mean((e[una] - none[una]) ** 2)) if una.any() else np.nan,
            f"{prefix}none_brier_split": float(np.mean((e[~una] - none[~una]) ** 2)) if (~una).any() else np.nan}


def stage_methods(base, regimes, uids) -> None:
    from cape.coherence import project_coherent

    bench = load_yaml("benchmark")
    methods, cal, probe, pcal = fit_estimators(base)
    rows, scores = [], []
    for d in SETS:
        for m, f in regimes.items():
            qs = load(m, d, uids[d], with_replay=True)
            if m != base:
                from uqbench.artifacts import load_energy
                ene = load_energy(m, d)
                qs = [replace(q, energy=ene.get(q.uid, [])) for q in qs]
            y = labels_of(qs)
            none = np.array([float(q.labels.max() == 0) for q in qs])
            una = np.array([unanimous(q) for q in qs])
            s = np.concatenate([probe.logits(q) for q in qs])
            p = pcal(s)
            start, proj = 0, []
            for q in qs:
                proj.append(project_coherent(p[start:start + q.k], q.clusters,
                                             np.array([x["pred"] is not None for x in q.samples])))
                start += q.k
            proj = np.concatenate(proj)
            scored = {"Probe": (s, p), "Probe+Coh": (proj, proj)}
            for name, est in methods.items():
                if name == "SemEnergy" and any(len(q.energy) != q.k for q in qs):
                    continue
                conf = est.score_all(qs)
                scored[name] = (conf, cal[name](conf))
            for name, (conf, prob) in scored.items():
                row = {"dataset": d, "regime": m, "fraction": f, "method": name,
                       "auroc": metrics.auroc(conf, y), "brier": metrics.brier(prob, y),
                       "ece": metrics.ece(prob, y, bench["ece_bins"])}
                n = len(qs)
                kept = {"dataset": d, "fraction": f, "method": name, "uids": [q.uid for q in qs], "none": none,
                        "conf": np.reshape(conf, (n, -1)), "prob": np.reshape(prob, (n, -1)), "y": y.reshape(n, -1)}
                if name in ("Probe+Coh", "CAPE") or name.endswith("+Coh"):
                    ev = event_scores(prob, qs)
                    row.update(none_metrics("", ev["residual"], none, una))
                    row.update(none_metrics("top_", ev["1-top"], none, una))
                    kept.update(none_cape=ev["residual"], none_top=ev["1-top"])
                rows.append(row)
                scores.append(kept)
            print(f"[dependence] methods {d} {m} done", flush=True)
            del qs
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / base / "dependence_methods.csv", index=False, float_format="%.4f")
    pd.to_pickle(scores, scores_path(base, "methods"))
    print(df.pivot_table(index=["method"], columns="fraction", values=["auroc", "brier"]).round(4).to_string())


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p) - np.log(1 - p)


def projected_events(qs, probe, pcal) -> dict[str, np.ndarray]:
    """Residual and one-minus-top-answer none-correct scores of the calibrated, projected probe."""
    from cape.coherence import project_coherent

    proj = [project_coherent(pcal(probe.logits(q)), q.clusters, np.array([s["pred"] is not None for s in q.samples]))
            for q in qs]
    return event_scores(np.concatenate(proj), qs)


def stage_refit(base, regimes, uids) -> None:
    """Question-level cross-fitting within each regime (all sets pooled; folds fixed by question id, hence the same
    questions in every regime)."""
    import torch

    from cape.model import CAPE, AnswerModel
    from uqbench.calibration import Platt

    pool = load_pool(base)
    train, val = [q for q in pool if q.split == "train"], [q for q in pool if q.split == "val"]
    frozen = CAPE()
    frozen.fit(train, val)
    probe = frozen.probe
    yv, gv = labels_of(val), np.repeat([q.uid for q in val], [q.k for q in val])
    pcal = select_calibrator(np.concatenate([probe.logits(q) for q in val]), yv, gv, False)[0]
    del pool, train, val
    rows, params, refit_scores = [], [], []
    for m, f in regimes.items():
        qs = [q for d in SETS for q in load(m, d, uids[d], with_replay=True)]
        fold = np.array([sum(map(ord, q.uid)) % FOLDS for q in qs])
        variants: dict[str, list] = {"CAPE (frozen)": [], "CAPE (refit)": [], "CAPE (refit, learned α)": [],
                                     "CAPE (refit, learned α, r)": [],
                                     "Probe+Coh residual (recalibrated)": [], "Probe+Coh 1-top (recalibrated)": []}
        order = []
        for k in range(FOLDS):
            fit_q = [q for q, g in zip(qs, fold, strict=True) if g != k]
            test_q = [q for q, g in zip(qs, fold, strict=True) if g == k]
            rng = np.random.default_rng(k)
            is_val = rng.random(len(fit_q)) < 0.2
            tr = [q for q, v in zip(fit_q, is_val, strict=True) if not v]
            va = [q for q, v in zip(fit_q, is_val, strict=True) if v]
            refit = AnswerModel(probe=probe).fit(tr, va)
            alpha_only = AnswerModel(probe=probe, alpha=None).fit(tr, va)
            learned = AnswerModel(probe=probe, alpha=None, learn_r=True).fit(tr, va)
            d = learned.describe()
            params.append({"regime": m, "fraction": f, "fold": k, "alpha_only": alpha_only.describe()["alpha"],
                           "alpha": d["alpha"], "r": d["r"],
                           "theta_refit": refit.describe()["theta"], "theta_frozen": frozen.full.describe()["theta"]})
            # the projected probe's derived none-correct scores, Platt-recalibrated on the same in-regime folds
            fit_ev, test_ev = projected_events(fit_q, probe, pcal), projected_events(test_q, probe, pcal)
            none_fit = np.array([float(q.labels.max() == 0) for q in fit_q])
            cal_ev = {kind: Platt().fit(logit(fit_ev[kind]), none_fit)(logit(test_ev[kind])) for kind in fit_ev}
            for i, q in enumerate(test_q):
                order.append(q)
                for name, model in (("CAPE (frozen)", frozen.full), ("CAPE (refit)", refit),
                                    ("CAPE (refit, learned α)", alpha_only), ("CAPE (refit, learned α, r)", learned)):
                    logp, M = model.log_outcomes(q)
                    pr = torch.nan_to_num(logp.exp(), nan=0.0)
                    resp = (M * pr[: M.shape[0], None]).sum(0)[: q.k].numpy()
                    variants[name].append((resp, float(1 - pr[: M.shape[0]].sum())))
                variants["Probe+Coh residual (recalibrated)"].append((None, float(cal_ev["residual"][i])))
                variants["Probe+Coh 1-top (recalibrated)"].append((None, float(cal_ev["1-top"][i])))
        y = labels_of(order)
        none = np.array([float(q.labels.max() == 0) for q in order])
        una = np.array([unanimous(q) for q in order])
        for name, vals in variants.items():
            row = {"regime": m, "fraction": f, "method": name}
            if vals[0][0] is not None:
                resp = np.concatenate([v[0] for v in vals])
                row.update(auroc=metrics.auroc(resp, y), brier=metrics.brier(resp, y), ece=metrics.ece(resp, y))
            row.update(none_metrics("", np.array([v[1] for v in vals]), none, una))
            rows.append(row)
        refit_scores.append({"fraction": f, "uids": [q.uid for q in order], "none": none, "una": una,
                             **{name: np.array([v[1] for v in vals]) for name, vals in variants.items()}})
        print(f"[dependence] refit {m} done", flush=True)
        del qs
    out = RESULTS_DIR / base
    pd.DataFrame(rows).to_csv(out / "dependence_refit.csv", index=False, float_format="%.4f")
    pd.DataFrame(params).to_csv(out / "dependence_refit_params.csv", index=False)
    pd.to_pickle(refit_scores, scores_path(base, "refit"))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(params).to_string(index=False))


def scores_path(base: str, stage: str):
    """Per-question scores of a stage, kept for the bootstrap stage."""
    out = run_dir(base) / "dependence"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{stage}_scores.pkl"


BOOT_METHODS = ("CAPE", "CAPE w/o null", "SC", "SemEnergy", "DeepConf×SC", "Probe", "Probe+Coh",
                "DeepConf", "ReasonProbe", "SEP")


def stage_bootstrap(base, regimes, uids) -> None:
    """Paired question-level bootstrap. Response metrics are means over the test sets (as in the methods table);
    the none-correct metrics pool the questions of all sets (as in the refit table)."""
    n_boot, rng = load_yaml("benchmark")["bootstrap"], np.random.default_rng(0)
    order = {d: sorted(uids[d]) for d in SETS}
    table, saved = {}, pd.read_pickle(scores_path(base, "methods"))   # (method, fraction, dataset) -> arrays
    for r in saved:
        if r["method"] in BOOT_METHODS:
            pos = {u: i for i, u in enumerate(r["uids"])}
            idx = [pos[u] for u in order[r["dataset"]]]
            table[r["method"], r["fraction"], r["dataset"]] = {k: np.asarray(v)[idx] for k, v in r.items()
                                                                if k in ("conf", "prob", "y", "none", "none_top",
                                                                         "none_cape")}
    fracs = sorted({r["fraction"] for r in saved})

    def response_metrics(draw):
        out = {}
        for (name, f, d), a in table.items():
            i = draw[d]
            conf, prob, y = a["conf"][i].ravel(), a["prob"][i].ravel(), a["y"][i].ravel()
            out.setdefault((name, f), []).append((metrics.auroc(conf, y), metrics.brier(prob, y)))
        return {k: np.mean(v, axis=0) for k, v in out.items()}

    def none_gap(draw):   # none-correct Brier of CAPE minus that of the projected probe's 1 - top
        out = {}
        for f in fracs:
            cape = np.concatenate([table["CAPE", f, d]["none_cape"][draw[d]] for d in SETS])
            top = np.concatenate([table["Probe+Coh", f, d]["none_top"][draw[d]] for d in SETS])
            none = np.concatenate([table["CAPE", f, d]["none"][draw[d]] for d in SETS])
            out[f] = np.mean((cape - none) ** 2) - np.mean((top - none) ** 2)
        return out

    full = {d: np.arange(len(order[d])) for d in SETS}
    draws = [{d: rng.integers(0, len(order[d]), len(order[d])) for d in SETS} for _ in range(n_boot)]
    rows = []

    def add(kind, method, f, metric, est, boot):
        boot = np.asarray(boot)
        rows.append({"kind": kind, "method": method, "fraction": f, "metric": metric, "estimate": est,
                     "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5),
                     "p_le_0": float((boot <= 0).mean())})

    point, boots = response_metrics(full), [response_metrics(dr) for dr in draws]
    for name in BOOT_METHODS:
        for f in fracs[1:]:
            if (name, f) not in point:
                continue
            for j, metric in enumerate(("auroc", "brier")):
                add("change vs iid", name, f, metric, point[name, f][j] - point[name, 0.0][j],
                    [b[name, f][j] - b[name, 0.0][j] for b in boots])

    def null_effect(m, f, j):   # extra change of CAPE without the outside option over CAPE
        return (m["CAPE w/o null", f][j] - m["CAPE w/o null", 0.0][j]) - (m["CAPE", f][j] - m["CAPE", 0.0][j])

    for f in fracs[1:]:
        for j, metric in enumerate(("auroc", "brier")):
            add("change: w/o null minus CAPE", "CAPE w/o null - CAPE", f, metric, null_effect(point, f, j),
                [null_effect(b, f, j) for b in boots])
    gap, gaps = none_gap(full), [none_gap(dr) for dr in draws]
    for f in fracs:
        add("none-correct Brier gap", "CAPE - Probe+Coh 1-top", f, "none_brier", gap[f], [g[f] for g in gaps])

    # refit stage: frozen minus refitted CAPE, none-correct Brier over all and over unanimous questions
    pooled = [u for d in SETS for u in order[d]]
    by_set = {d: np.array([pooled.index(u) for u in order[d]]) for d in SETS}
    for r in pd.read_pickle(scores_path(base, "refit")):
        pos = {u: i for i, u in enumerate(r["uids"])}
        idx = np.array([pos[u] for u in pooled])
        none, una = r["none"][idx], r["una"][idx]
        err = (r["CAPE (frozen)"][idx] - none) ** 2 - (r["CAPE (refit, learned α)"][idx] - none) ** 2
        bs = [np.concatenate([by_set[d][dr[d]] for d in SETS]) for dr in draws]
        add("frozen minus refit", "CAPE", r["fraction"], "none_brier", err.mean(), [err[b].mean() for b in bs])
        add("frozen minus refit", "CAPE", r["fraction"], "none_brier_unanimous", err[una].mean(),
            [err[b[una[b]]].mean() for b in bs])
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / base / "dependence_bootstrap.csv", index=False, float_format="%.4f")
    print(df.round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="qwen3-8b", help="main run whose shared-prefix runs are compared")
    ap.add_argument("--stages", nargs="+", default=["structure"],
                    choices=["structure", "methods", "refit", "bootstrap"])
    args = ap.parse_args()
    regimes = available(args.base)
    print(f"[dependence] regimes {regimes}", flush=True)
    uids = common_uids(regimes)
    for stage in args.stages:
        {"structure": stage_structure, "methods": stage_methods, "refit": stage_refit,
         "bootstrap": stage_bootstrap}[stage](args.base, regimes, uids)


if __name__ == "__main__":
    main()
