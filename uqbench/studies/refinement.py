"""Information beyond calibration, from saved scores (``$OUTPUT_DIR/<model>/scores/*.npz``).

* Refinement: every method's score is recalibrated out of sample on the test set itself (isotonic
  regression fitted on one half of the questions and applied to the other, two folds, averaged over
  random splits). The resulting Brier score removes the pool-to-test calibration shift and measures
  only the information a score carries. Self-consistency recalibrated this way is the attainable
  (cross-fitted) agreement floor; a method below it carries information beyond agreement.
* Null signal: CAPE's mass on "none correct", pi_null = 1 - sum of its answer posteriors, as a
  question-level detector of "no sampled answer is correct", against the vote-based alternative
  1 - (largest vote share) and the analogous quantity from every other estimator's coherent projection.

* Supervised "none correct" control: a logistic regression on question-level features (vote shares,
  number of clusters, answered fraction, answer entropy and summaries of the Stacker's per-response
  probabilities), trained on the pool's validation questions to predict that no sampled answer is correct.
* "None correct" classifier on CAPE's evidence: logistic regression or gradient-boosted trees (chosen by
  cross-validated AUROC on the pool) on question-level summaries of CAPE's 22 trace statistics and the
  out-of-fold hidden-state probe logit, plus vote shares that do not depend on K, trained on the whole pool
  (train and validation questions) to predict that no sampled answer is correct: the same evidence and
  labels as CAPE, with "none correct" learned separately from the answers.
* Coherent stacker with a learned outside option: the coherent stacker's answer masses renormalized
  to 1 - P(none correct) of the classifier on CAPE's evidence, per question; AUROC and Brier score against CAPE.
* Selective answering: per question, each coherent method answers with its most probable cluster and
  abstains below a confidence threshold; question-level accuracy at full coverage, at 50% coverage and
  the area under the risk-coverage curve.
* Projection steps: the AUROC and Brier score of each estimator's calibrated probabilities after zeroing
  answerless responses, after also averaging within clusters, and after the full projection (capping);
  for comparison, the product of the calibrated probability with the vote share of the response's answer
  (an ad hoc fusion with agreement) and the projection of a constant score (agreement alone).

Writes ``results/<model>/refinement.csv``, ``results/<model>/null_signal.csv``,
``results/<model>/selective_qa.csv``, ``results/<model>/projection_steps.csv`` and the per-question
``results/<model>/questions.csv`` (unanimity, correctness, CAPE's answer confidence and null mass) used by
the paper's figures.

Usage::

    python -m uqbench.studies.refinement --model qwen3-8b
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from cape.coherence import project_coherent
from uqbench.config import RESULTS_DIR, load_yaml, scores_dir
from uqbench.metrics import aurc, auroc, ece
from uqbench.records import load_pool, load_questions
from uqbench.studies.decision import MAIN

CONTROLS = ["Stacker+Coh"]
SELECTIVE = {"CAPE": "CAPE__prob", "SC": "SC+Coh", "CISC": "CISC+Coh", "DeepConf+C": "DeepConf+Coh",
             "DeepConf×SC+C": "DeepConf×SC+Coh", "Stacker+C": "Stacker+Coh",
             "Coherent stacker": "Stacker + probe, gated +Coh"}
SUPERVISED_INPUTS = ["DeepConf", "CDG", "SelfCertainty", "LRMVerbal", "CISC", "ReasonProbe", "SNNE", "DeepConf-Vote",
                     "ConsistCal", "SEP", "ORM", "SemEnergy", "DeepConf×SC", "CoCoA", "Stacker", "SC", "SE"]
STEP_METHODS = ["DeepConf", "CDG", "SelfCertainty", "LRMVerbal", "ReasonProbe", "SNNE", "SEP", "ORM", "SemEnergy",
                "CoCoA", "SE", "DeepConf×SC", "Stacker"]


def crossfit_isotonic(score: np.ndarray, y: np.ndarray, qid: np.ndarray, repeats: int = 10, seed: int = 0) -> float:
    """Brier score of the score after out-of-sample isotonic recalibration (question-level two-fold splits)."""
    rng = np.random.default_rng(seed)
    uids = np.unique(qid)
    out = []
    for _ in range(repeats):
        fold = dict(zip(uids, rng.integers(0, 2, len(uids)), strict=True))
        f = np.array([fold[u] for u in qid])
        pred = np.empty(len(y))
        for k in (0, 1):
            tr, te = f != k, f == k
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(score[tr], y[tr])
            pred[te] = iso.predict(score[te])
        out.append(float(np.mean((pred - y) ** 2)))
    return float(np.mean(out))


def null_masses(questions, prob: np.ndarray) -> np.ndarray:
    """Per question, 1 - sum over answer clusters of the (coherent) per-response probability."""
    out, start = [], 0
    for q in questions:
        p = prob[start:start + q.k]
        start += q.k
        seen = {}
        for c, s, v in zip(q.clusters, q.samples, p, strict=True):
            if s["pred"] is not None:
                seen.setdefault(c, []).append(v)
        out.append(1.0 - sum(float(np.mean(v)) for v in seen.values()))
    return np.array(out)


def top_share(questions) -> np.ndarray:
    out = []
    for q in questions:
        counts = {}
        for c, s in zip(q.clusters, q.samples, strict=True):
            if s["pred"] is not None:
                counts[c] = counts.get(c, 0) + 1
        out.append(max(counts.values()) / q.k if counts else 0.0)
    return np.array(out)


def question_features(questions, stacker: np.ndarray) -> np.ndarray:
    """Question-level features for the supervised "none correct" control."""
    rows, start = [], 0
    for q in questions:
        p = stacker[start:start + q.k]
        start += q.k
        groups = {}
        for i, (c, smp) in enumerate(zip(q.clusters, q.samples, strict=True)):
            if smp["pred"] is not None:
                groups.setdefault(c, []).append(i)
        shares = np.array(sorted((len(v) / q.k for v in groups.values()), reverse=True)) if groups else np.zeros(1)
        means = [float(np.mean(p[v])) for v in groups.values()] or [0.0]
        answered = [i for v in groups.values() for i in v]
        ent = float(-(shares[shares > 0] * np.log(shares[shares > 0])).sum())
        rows.append([shares[0], shares[1] if len(shares) > 1 else 0.0, len(groups) / q.k, len(answered) / q.k, ent,
                     max(means), float(np.mean(p[answered])) if answered else 0.0,
                     float(np.std(p[answered])) if answered else 0.0, sum(means)])
    return np.array(rows)


def rich_question_features(questions, z) -> np.ndarray:
    """``question_features`` plus, for every estimator, the best and worst answer-cluster mean of its
    probabilities and their mean and spread over the responses."""
    cols = [question_features(questions, z["Stacker__prob"])]
    for m in SUPERVISED_INPUTS:
        p, rows, start = z[f"{m}__prob"], [], 0
        for q in questions:
            v = p[start:start + q.k]
            start += q.k
            groups = {}
            for i, (c, smp) in enumerate(zip(q.clusters, q.samples, strict=True)):
                if smp["pred"] is not None:
                    groups.setdefault(c, []).append(i)
            means = [float(np.mean(v[g])) for g in groups.values()] or [0.0]
            rows.append([max(means), min(means), float(np.mean(v)), float(np.std(v))])
        cols.append(np.array(rows))
    return np.hstack(cols)


def evidence_features(questions, probe) -> np.ndarray:
    """Question-level features for the separately learned "none correct" classifier: K-invariant vote
    features, and the mean, minimum and maximum over answered responses (and the mean over the largest
    cluster) of CAPE's trace statistics and of the hidden-state probe logit."""
    from cape.model import scalar_features

    rows = []
    for q in questions:
        x = np.c_[scalar_features(q), probe.logits(q)]
        groups = {}
        for i, (c, smp) in enumerate(zip(q.clusters, q.samples, strict=True)):
            if smp["pred"] is not None:
                groups.setdefault(c, []).append(i)
        shares = np.array(sorted((len(v) / q.k for v in groups.values()), reverse=True)) if groups else np.zeros(1)
        ent = float(-(shares[shares > 0] * np.log(shares[shares > 0])).sum()) / np.log(q.k)
        votes = [shares[0], shares[1] if len(shares) > 1 else 0.0, len(groups) / q.k,
                 sum(len(v) for v in groups.values()) / q.k, ent]
        if groups:
            ans = [i for v in groups.values() for i in v]
            top = max(groups.values(), key=len)
            summ = np.r_[x[ans].mean(0), x[ans].min(0), x[ans].max(0), x[top].mean(0)]
        else:
            summ = np.r_[x.mean(0), x.min(0), x.max(0), x.mean(0)]
        rows.append(np.r_[votes, summ])
    return np.nan_to_num(np.array(rows), nan=0.0, posinf=0.0, neginf=0.0)


def fit_evidence_classifier(pool, seed: int):
    """The separately learned "none correct" classifier on CAPE's evidence (module docstring)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from cape.model import HiddenProbe

    train, val = [q for q in pool if q.split == "train"], [q for q in pool if q.split == "val"]
    probe = HiddenProbe().fit(train, val)   # out-of-fold on train, out-of-sample on validation questions
    x, y = evidence_features(pool, probe), none_correct(pool)
    candidates = {"LR": make_pipeline(StandardScaler(), LogisticRegressionCV(
                      Cs=np.logspace(-3, 2, 12), cv=5, scoring="roc_auc", max_iter=5000)),
                  "GBT": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                        l2_regularization=1.0, random_state=seed)}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    name = max(candidates, key=lambda k: cross_val_score(candidates[k], x, y, cv=cv, scoring="roc_auc").mean())
    print(f"[refinement] evidence classifier: {name}", flush=True)
    return candidates[name].fit(x, y), probe


def none_correct(questions) -> np.ndarray:
    return np.array([float(not any(lab == 1 for lab in q.labels)) for q in questions])


def answer_choice(questions, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per question: confidence of the most probable answer cluster and whether that answer is correct."""
    conf, correct, start = [], [], 0
    for q in questions:
        p = prob[start:start + q.k]
        start += q.k
        groups = {}
        for i, (c, smp) in enumerate(zip(q.clusters, q.samples, strict=True)):
            if smp["pred"] is not None:
                groups.setdefault(c, []).append(i)
        if not groups:
            conf.append(0.0)
            correct.append(0)
            continue
        best = max(groups.values(), key=lambda idx: (float(np.mean(p[idx])), len(idx)))
        conf.append(float(np.mean(p[best])))
        correct.append(int(q.labels[best[0]] == 1))
    return np.array(conf), np.array(correct)


def projection_steps(questions, prob: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """AUROC / Brier after (1) zeroing answerless responses, (2) also averaging within clusters."""
    zeroed, averaged, start = prob.copy(), prob.copy(), 0
    for q in questions:
        sl = slice(start, start + q.k)
        start += q.k
        has = np.array([smp["pred"] is not None for smp in q.samples])
        z = zeroed[sl]
        z[~has] = 0.0
        a = z.copy()
        for c in {c for c, h in zip(q.clusters, has, strict=True) if h}:
            idx = [i for i, (cc, h) in enumerate(zip(q.clusters, has, strict=True)) if cc == c and h]
            a[idx] = z[idx].mean()
        zeroed[sl], averaged[sl] = z, a
    return {"auroc_zero": auroc(zeroed, y), "brier_zero": float(np.mean((zeroed - y) ** 2)),
            "auroc_avg": auroc(averaged, y), "brier_avg": float(np.mean((averaged - y) ** 2))}


def main() -> None:
    bench = load_yaml("benchmark")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    score_dir = scores_dir(args.model)
    ref_rows, null_rows, q_rows, sel_rows, step_rows, stack_rows = [], [], [], [], [], []
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    zv = np.load(score_dir / "val.npz")
    val_q = [q for q in load_pool(args.model, with_replay=False, with_followup=False)
             if q.split == "val"]
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegressionCV

    y_null = none_correct(val_q)
    null_clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(
        question_features(val_q, zv["Stacker__prob"]), y_null)
    x_rich = rich_question_features(val_q, zv)
    rich_clfs = {"supervised, all estimators (LR)": make_pipeline(
                     StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 2, 12), cv=5, scoring="roc_auc",
                                                            max_iter=5000)).fit(x_rich, y_null),
                 "supervised, all estimators (GBT)": HistGradientBoostingClassifier(
                     max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                     random_state=bench["seed"]).fit(x_rich, y_null)}
    ev_clf, ev_probe = fit_evidence_classifier(load_pool(args.model), bench["seed"])
    for name in bench["test_sets"]:
        z = np.load(score_dir / f"{name}.npz")
        y, uid = z["labels"].astype(float), z["uid"]
        row = {"dataset": name}
        for m in MAIN + CONTROLS:
            key = f"{m}__prob" if f"{m}__prob" in z.files else m
            if key in z.files:
                row[m] = crossfit_isotonic(z[key], y, uid, seed=bench["seed"])
        ref_rows.append(row)

        questions = load_questions(args.model, name, with_replay=False, with_followup=False)
        none = np.array([float(not any(lab == 1 for lab in q.labels)) for q in questions])
        scores = dict(z)
        if (score_dir / f"controls_{name}.npz").exists():
            scores.update(np.load(score_dir / f"controls_{name}.npz"))
        for name_m, key in SELECTIVE.items():
            if key in scores:
                conf, ok = answer_choice(questions, scores[key])
                order = np.argsort(-conf, kind="stable")
                half = order[: len(order) // 2]
                sel_rows.append({"dataset": name, "method": name_m, "accuracy": ok.mean(),
                                 "accuracy_at_50": ok[half].mean(), "aurc": aurc(conf, ok),
                                 "auroc": roc_auc_score(ok, conf) if 0 < ok.sum() < len(ok) else np.nan,
                                 "brier": float(np.mean((conf - ok) ** 2))})
        share, const, start = np.zeros(len(y)), np.zeros(len(y)), 0
        for q in questions:
            has = np.array([smp["pred"] is not None for smp in q.samples])
            counts = {}
            for c, h in zip(q.clusters, has, strict=True):
                if h:
                    counts[c] = counts.get(c, 0) + 1
            share[start:start + q.k] = [counts[c] / q.k if h else 0.0 for c, h in zip(q.clusters, has, strict=True)]
            const[start:start + q.k] = project_coherent(np.full(q.k, 0.5), q.clusters, has)
            start += q.k
        step_rows.append({"dataset": name, "method": "constant", "auroc_coh": auroc(const, y),
                          "brier_coh": float(np.mean((const - y) ** 2))})
        for m in STEP_METHODS:
            if f"{m}__prob" in z.files and f"{m}+Coh" in z.files:
                p = z[f"{m}__prob"]
                step_rows.append({"dataset": name, "method": f"{m} x share", "auroc_coh": auroc(p * share, y),
                                  "brier_coh": float(np.mean((p * share - y) ** 2))})
                step_rows.append({"dataset": name, "method": m, "auroc_prob": auroc(p, y),
                                  "brier_prob": float(np.mean((p - y) ** 2)), **projection_steps(questions, p, y),
                                  "auroc_coh": auroc(z[f"{m}+Coh"], y),
                                  "brier_coh": float(np.mean((z[f"{m}+Coh"] - y) ** 2)),
                                  "ece_coh": ece(z[f"{m}+Coh"], y)})
        ctrl = score_dir / f"controls_{name}.npz"
        full = load_questions(args.model, name)       # with replay and follow-up: CAPE's evidence
        ev_null = ev_clf.predict_proba(evidence_features(full, ev_probe))[:, 1]
        if ctrl.exists():
            cs = np.load(ctrl)["Stacker + probe, gated +Coh"]
            s_null = ev_null   # the strongest separately learned "none correct" probability
            combo, start = np.zeros(len(y)), 0
            for q, sn in zip(questions, s_null, strict=True):
                p = cs[start:start + q.k]
                mass = 1.0 - null_masses([q], p)[0]
                combo[start:start + q.k] = p / mass * (1 - sn) if mass > 1e-9 else 0.0
                start += q.k
            for m, v in (("Coherent stacker", cs), ("Coherent stacker + learned null", combo),
                         ("CAPE", z["CAPE__prob"])):
                stack_rows.append({"dataset": name, "method": m, "auroc": auroc(v, y),
                                   "brier": float(np.mean((v - y) ** 2)), "ece": ece(v, y)})
            # per-response scores of the modular implementation and its per-question "none correct" probability
            np.savez_compressed(score_dir / f"modular_{name}.npz", **{"Coherent stacker + learned null": combo},
                                null_classifier=ev_null)
        cape_null = null_masses(questions, z["CAPE__prob"])
        brier_of = {"brier_sc": z["SC__prob"], "brier_deepconf": z["DeepConf+Coh"], "brier_cape": z["CAPE__prob"]}
        if ctrl.exists():
            brier_of["brier_stacker"] = np.load(ctrl)["Stacker + probe, gated +Coh"]
            brier_of["brier_modular"] = combo   # the coherent stacker with the separate "none correct" classifier
        dc_null = null_masses(questions, z["DeepConf+Coh"])
        stack_p = np.load(ctrl)["Stacker + probe, gated +Coh"] if ctrl.exists() else z["Stacker+Coh"]
        stack_null = null_masses(questions, stack_p)   # the coherent stacker, as in every table
        sup_null = {label: clf.predict_proba(rich_question_features(questions, z))[:, 1]
                    for label, clf in rich_clfs.items()}
        start = 0
        for i, (q, pn, dn, sn) in enumerate(zip(questions, cape_null, dc_null, stack_null, strict=True)):
            answered = [c for c, smp in zip(q.clusters, q.samples, strict=True) if smp["pred"] is not None]
            q_rows.append({"dataset": name, "uid": q.uid,
                           "unanimous": len(answered) == q.k and len(set(answered)) == 1,
                           "correct": bool(q.labels.max() == 1),
                           "cape_top": float(z["CAPE__prob"][start:start + q.k].max()),
                           "sc_top": float(z["SC__prob"][start:start + q.k].max()),
                           "deepconf_top": float(z["DeepConf+Coh"][start:start + q.k].max()),
                           "cape_null": float(pn), "deepconf_null": float(dn), "stacker_null": float(sn),
                           "stacker_top": float(stack_p[start:start + q.k].max()),
                           "evidence_null": float(ev_null[i]),
                           "sup_lr_null": float(sup_null["supervised, all estimators (LR)"][i]),
                           "sup_gbt_null": float(sup_null["supervised, all estimators (GBT)"][i]),
                           **{k: float(np.mean((v[start:start + q.k] - q.labels) ** 2)) for k, v in brier_of.items()}})
            start += q.k
        if 0 < none.sum() < len(none):
            nrow = {"dataset": name, "questions": len(questions), "none_correct": int(none.sum()),
                    "1 - top share": roc_auc_score(none, 1 - top_share(questions)),
                    "supervised": roc_auc_score(none, null_clf.predict_proba(
                        question_features(questions, z["Stacker__prob"]))[:, 1])}
            for label in rich_clfs:
                nrow[label] = roc_auc_score(none, sup_null[label])
            nrow["evidence classifier"] = roc_auc_score(none, ev_null)
            offsets = np.cumsum([0] + [q.k for q in questions[:-1]])
            for label, key in (("CAPE", "CAPE__prob"), ("DeepConf+Coh", "DeepConf+Coh")):
                if key in z.files:   # the other read-out: one minus the probability of the top answer
                    nrow[f"1 - top {label}"] = roc_auc_score(none, 1 - np.array(
                        [z[key][o:o + q.k].max() for o, q in zip(offsets, questions, strict=True)]))
            for label, p_none in (("CAPE", cape_null), ("evidence classifier", ev_null),
                                  ("coherent stacker", stack_null)):   # "none correct" as a forecast
                nrow[f"brier {label}"] = float(np.mean((np.clip(p_none, 0, 1) - none) ** 2))
            nrow["1 - top coherent stacker"] = roc_auc_score(none, 1 - np.array(
                [stack_p[o:o + q.k].max() for o, q in zip(np.cumsum([0] + [q.k for q in questions[:-1]]),
                                                           questions, strict=True)]))
            for m in ["CAPE", *[f"{b}+Coh" for b in MAIN if b != "CAPE"]]:
                key = "CAPE__prob" if m == "CAPE" else m      # projections are coherent as stored
                if key in z.files:
                    nrow[m] = roc_auc_score(none, null_masses(questions, z[key]))
            if ctrl.exists():   # coherent controls of the ladder (uqbench.studies.controls), both read-outs
                cz = np.load(ctrl)
                for key in ("CAPE agreement only", "Stacker +Coh", "Stacker + probe +Coh",
                            "Stacker + probe, gated +Coh"):
                    if key in cz.files:
                        nrow[key] = roc_auc_score(none, null_masses(questions, cz[key]))
                        nrow[f"1 - top {key}"] = roc_auc_score(none, 1 - np.array(
                            [cz[key][o:o + q.k].max() for o, q in zip(offsets, questions, strict=True)]))
            for key in ("CISC+Coh",):
                if key in z.files:
                    nrow[f"1 - top {key}"] = roc_auc_score(none, 1 - np.array(
                        [z[key][o:o + q.k].max() for o, q in zip(offsets, questions, strict=True)]))
            null_rows.append(nrow)
    out = RESULTS_DIR / args.model
    pd.DataFrame(ref_rows).to_csv(out / "refinement.csv", index=False, float_format="%.4f")
    pd.DataFrame(null_rows).to_csv(out / "null_signal.csv", index=False, float_format="%.4f")
    pd.DataFrame(q_rows).to_csv(out / "questions.csv", index=False, float_format="%.6f")
    pd.DataFrame(sel_rows).to_csv(out / "selective_qa.csv", index=False, float_format="%.4f")
    pd.DataFrame(step_rows).to_csv(out / "projection_steps.csv", index=False, float_format="%.4f")
    pd.DataFrame(stack_rows).to_csv(out / "stacker_null.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
