"""Read access to the result tables in ``results/`` for the summary, the paper's tables and figures.

Backbones, datasets and method lists come from ``uqbench/configs/benchmark.yaml``; display names from
``uqbench/configs/models.yaml`` (backbones) and :data:`DATASETS` (test sets).
"""
from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd

from uqbench.config import RESULTS_DIR, load_yaml, model_config, scores_dir

BENCH = load_yaml("benchmark")
BACKBONES = {m: model_config(m)["name"] for m in BENCH["backbones"]}    # reported backbones -> display name
ABLATION_MODEL = BENCH["ablation_backbone"]
LOW_PARSE = BENCH["low_parse"]
MATH_SETS = tuple(BENCH["math_sets"])
BASELINES = list(BENCH["methods"]["baselines"])
REFERENCES = list(BENCH["methods"]["references"])
CONSTRUCTED = list(BENCH["methods"]["constructed"])
SINGLE_TRACE = tuple(BENCH["methods"]["single_response"])
COMPETITORS = BASELINES + REFERENCES          # every estimator CAPE is compared with
# test sets in table order, with the short names printed in table headers
DATASETS = {"math_all": "Math", "math500": "MATH-500", "math_rest": "MATH (non-500)", "olympiadbench": "Olymp.",
            "aime": "AIME", "minerva": "Minerva", "gpqa": "GPQA", "gpqa_ext": "GPQA-Ext.", "mmlupro": "MMLU-Pro",
            "supergpqa": "SuperGPQA", "triviaqa": "TriviaQA"}
CLEAN_SETS = ["gpqa", "gpqa_ext", "supergpqa"]            # sources absent from the training pool
POOL_SETS = ["math500", "math_rest", "olympiadbench", "minerva", "aime", "mmlupro", "triviaqa"]   # sources in the pool


@cache
def low_parse(model: str) -> frozenset[str]:
    """Test sets on which more than LOW_PARSE of the backbone's responses have no extractable answer."""
    path = RESULTS_DIR / model / "answer_structure.csv"
    if not path.exists():
        return frozenset()
    st = pd.read_csv(path).set_index("dataset")
    excluded = set(st.index[st["no_answer_rate"] > LOW_PARSE])
    parts = [d for d in MATH_SETS if d in st.index]
    if parts:  # pooled math: response-weighted no-answer rate of its components
        w = st.loc[parts, "questions"]
        if (st.loc[parts, "no_answer_rate"] * w).sum() / w.sum() > LOW_PARSE:
            excluded.add("math_all")
    return frozenset(excluded)


def metric_frames() -> dict[str, pd.DataFrame]:
    """``results/<model>/metrics.csv`` of every reported backbone that has one."""
    return {m: pd.read_csv(RESULTS_DIR / m / "metrics.csv") for m in BACKBONES
            if (RESULTS_DIR / m / "metrics.csv").exists()}


def frames_of(name: str) -> pd.DataFrame:
    """``results/<model>/<name>.csv`` of every reported backbone, high-parse test sets only, with a ``model``
    column."""
    rows = []
    for m in BACKBONES:
        path = RESULTS_DIR / m / f"{name}.csv"
        if path.exists():
            d = pd.read_csv(path)
            rows.append(d[~d.dataset.isin(low_parse(m))].assign(model=m))
    out = pd.concat(rows) if rows else pd.DataFrame()
    return with_strongest_supervised(out) if name == "null_signal" else out


def with_strongest_supervised(nul: pd.DataFrame) -> pd.DataFrame:
    """Report, as "supervised", the supervised "none correct" classifier with the highest mean AUROC over all
    high-parse pairs (the variant most favourable to the baseline)."""
    cols = [c for c in nul.columns if c.startswith("supervised")]
    if not cols:
        return nul
    return nul.assign(supervised=nul[nul[cols].mean().idxmax()])


def with_null_signals(q: pd.DataFrame) -> pd.DataFrame:
    """Per-question "none correct" signals: the vote and coherent-stacker complements of the top answer, and the
    stronger (by mean AUROC over pairs) of the two supervised classifiers on all estimators."""
    from sklearn.metrics import roc_auc_score

    q = q.assign(none=(~q.correct.astype(bool)).astype(float), vote=1 - q.sc_top)
    if "stacker_top" in q:
        q = q.assign(stack_top=1 - q.stacker_top)
    if "deepconf_top" in q:
        q = q.assign(dc_top=1 - q.deepconf_top)
    sup = [c for c in ("sup_lr_null", "sup_gbt_null") if c in q]
    if sup:
        def mean_auroc(col):
            groups = [g.dropna(subset=[col]) for _, g in q.groupby(["model", "dataset"])]
            return np.mean([roc_auc_score(g.none, g[col]) for g in groups if 0 < g.none.sum() < len(g)])
        q = q.assign(sup_null=q[max(sup, key=mean_auroc)])
    return q


def escalation_recall(q: pd.DataFrame, col: str, budget: float, by=("model", "dataset"),
                      dropna: bool = True) -> list[float]:
    """Escalate the ``budget`` share of each pair's questions with the highest ``col``: per pair, the share of the
    questions without a correct sample (``none``) that are caught."""
    vals = []
    for _, g in (q.dropna(subset=[col]) if dropna else q).groupby(list(by)):
        if g.none.sum() > 0:
            top = g.sort_values(col, ascending=False, kind="stable").head(max(1, round(budget * len(g))))
            vals.append(top.none.sum() / g.none.sum())
    return vals


def unanimous_confidence(model: str) -> pd.Series:
    """CAPE's mean confidence on unanimous-correct, unanimous-wrong and split questions (high-parse test sets,
    weighted by questions)."""
    nm = pd.read_csv(RESULTS_DIR / model / "null_mass.csv")
    nm = nm[(nm.method == "CAPE") & ~nm.dataset.isin(low_parse(model)) & (nm.questions > 0)].dropna()
    return nm.groupby("group").apply(lambda x: np.average(x.mean_confidence, weights=x.questions),
                                     include_groups=False)


def token_stats(model: str) -> dict:
    """Mean prompt and response length (tokens) over all test-set responses of a backbone."""
    from uqbench.records import load_generations

    prompt, resp = [], []
    for name in BENCH["test_sets"]:
        for r in load_generations(model, name, partial=True):
            prompt.append(len(r["prompt_ids"]))
            resp += [len(s["token_ids"]) for s in r["samples"]]
    return {"prompt": float(np.mean(prompt)), "response": float(np.mean(resp))}


def scores(model: str, dataset: str):
    """Saved per-response scores (``$OUTPUT_DIR/<model>/scores/<dataset>.npz``) or None."""
    path = scores_dir(model) / f"{dataset}.npz"
    return np.load(path) if path.exists() else None
