"""Verifier audit of consistent errors (``results/<model>/verifier_audit.csv``).

Consistent errors (all K answers equivalent and wrong) motivate the outside option, so we check how
many of them a looser checker would accept, and whether the separation of unanimous questions survives
without them. Looser checks: for math, units and LaTeX wrappers stripped and a relative numeric
tolerance of 1% or 5% (Minerva-style reference answers are often rounded); for TriviaQA, containment of
an alias in the answer (or vice versa) or token F1 >= 0.5 with an alias. Multiple choice is exact by
construction and not audited.

Usage::

    python -m uqbench.studies.audit --model qwen3-8b
"""
from __future__ import annotations

import argparse
import ast
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from uqbench.config import RESULTS_DIR, load_yaml, scores_dir
from uqbench.records import load_generations
from uqbench.reporting.results import low_parse

NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?")
WRAP = re.compile(r"\\(?:mathrm|text|textbf|mathbf|operatorname)\{([^}]*)\}")


def _clean_math(s: str) -> str:
    s = WRAP.sub(r"\1", s or "")
    s = re.sub(r"\\[,;!]|\\ |~|\$|\\left|\\right", "", s)
    s = s.replace("\\%", "").replace("%", "").replace("^\\circ", "").replace("\\circ", "")
    return s.replace(",", "").strip()


def _numbers(s: str) -> list[float]:
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    out = []
    for m in NUM.findall(s):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def math_close(pred: str, ref: str, tol: float) -> bool:
    p, r = _clean_math(pred), _clean_math(ref)
    if p and p == r:
        return True
    pn, rn = _numbers(p), _numbers(r)
    if len(pn) != 1 or len(rn) != 1:
        return False
    a, b = pn[0], rn[0]
    return abs(a - b) <= tol * max(abs(b), 1e-9)


def _tokens(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()


def qa_close(pred: str, aliases: list[str]) -> bool:
    p = " ".join(_tokens(pred))
    for a in aliases:
        a = " ".join(_tokens(a))
        if not a or not p:
            continue
        if a in p or p in a:
            return True
        pt, at = p.split(), a.split()
        common = sum(min(pt.count(t), at.count(t)) for t in set(pt))
        if common and 2 * common / (len(pt) + len(at)) >= 0.5:
            return True
    return False


SINGLE = ("DeepConf", "SelfCertainty", "ReasonProbe", "LRMVerbal", "CDG", "SEP", "ORM")


def audit(model: str) -> pd.DataFrame:

    bench = load_yaml("benchmark")
    rows, pooled, flagged = [], [], set()
    for name in bench["test_sets"]:
        recs = load_generations(model, name)
        if not recs:
            continue
        if recs[0]["task"] not in ("math", "qa"):   # multiple choice: exact by construction
            z = np.load(scores_dir(model) / f"{name}.npz")
            cols = {m: z[f"{m}__prob"] for m in ("CAPE", "SC", *SINGLE) if f"{m}__prob" in z.files}
            q = pd.DataFrame({"uid": z["uid"], "y": z["labels"], "share": z["SC"], **cols}).groupby("uid").mean()
            if name not in low_parse(model):
                pooled.append(q[q.share >= 1 - 1e-9].assign(dataset=name))
            continue
        z = np.load(scores_dir(model) / f"{name}.npz")
        cols = {m: z[f"{m}__prob"] for m in ("CAPE", "SC", *SINGLE) if f"{m}__prob" in z.files}
        q = pd.DataFrame({"uid": z["uid"], "y": z["labels"], "share": z["SC"], **cols}).groupby("uid").mean()
        q = q[q.share >= 1 - 1e-9]
        q["cape"] = q["CAPE"]
        if name not in low_parse(model):
            pooled.append(q.assign(dataset=name))
        flags = {"tol1": set(), "tol5": set()}
        for r in recs:
            if r["uid"] not in q.index or q.loc[r["uid"], "y"] > 0.5:
                continue
            pred = next((s["pred"] for s in r["samples"] if s["pred"] is not None), "")
            if r["task"] == "qa":
                aliases = r["answer"] if isinstance(r["answer"], list) else ast.literal_eval(r["answer"])
                if qa_close(pred, aliases):
                    flags["tol1"].add(r["uid"])
                    flags["tol5"].add(r["uid"])
            else:
                for key, tol in (("tol1", 0.01), ("tol5", 0.05)):
                    if math_close(pred, str(r["answer"]), tol):
                        flags[key].add(r["uid"])
        flagged |= {(name, u) for u in flags["tol5"]}
        wrong = int((q.y <= 0.5).sum())
        keep = q[~q.index.isin(flags["tol5"])]
        rows.append({"dataset": name, "unanimous": len(q), "wrong": wrong,
                     "flag_tol1": len(flags["tol1"]), "flag_tol5": len(flags["tol5"]),
                     "auroc_cape": roc_auc_score((q.y > 0.5).astype(int), q.cape) if q.y.nunique() > 1 else np.nan,
                     "auroc_cape_clean": roc_auc_score((keep.y > 0.5).astype(int), keep.cape)
                     if (keep.y > 0.5).nunique() > 1 else np.nan})
    every = pd.concat(pooled)
    every["y"] = (every.y > 0.5).astype(int)
    clean = every[[(d, u) not in flagged for d, u in zip(every.dataset, every.index, strict=True)]]
    single = [m for m in SINGLE if m in every]
    for tag, q in (("all", every), ("all_clean", clean)):
        rows.append({"dataset": tag, "unanimous": len(q), "wrong": int((q.y == 0).sum()),
                     "auroc_cape": roc_auc_score(q.y, q.CAPE),
                     "auroc_best_single": max(roc_auc_score(q.y, q[m]) for m in single),
                     "auroc_sc": roc_auc_score(q.y, q.SC)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    df = audit(args.model)
    df.to_csv(RESULTS_DIR / args.model / "verifier_audit.csv", index=False, float_format="%.4f")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
