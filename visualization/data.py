"""Data shared by the paper's tables and figures (``visualization/``) beyond the per-backbone result tables of
``uqbench.reporting.results``: the paired significance tests of the main comparison, the attribution tables of the two
principles with their recalibration control, and the teaser's worked example."""
from __future__ import annotations

from functools import cache

import pandas as pd

from uqbench.config import RESULTS_DIR
from uqbench.reporting.results import BACKBONES, BENCH, COMPETITORS, CONSTRUCTED, DATASETS, low_parse, metric_frames
from uqbench.stats import bh_threshold, paired_bootstrap
from uqbench.studies.refinement import STEP_METHODS

HIGHER = {"auroc": True, "brier": False, "ece": False}   # the three main metrics, higher is better?
PUBLISHED = [m for m in STEP_METHODS if m not in CONSTRUCTED]   # incoherent published estimators


@cache
def significance() -> tuple[list[tuple], dict]:
    """Paired bootstrap of CAPE against the strongest competitor (by AUROC and by Brier score) per pair, and the
    Benjamini-Hochberg thresholds over the independent high-parse pairs (pooled math excluded)."""
    entries, p = [], {"p": [], "p_brier": []}
    for model, df in metric_frames().items():
        for d in [d for d in DATASETS if d in set(df["dataset"])]:
            sub = df[df.dataset == d].set_index("method")
            pool = [x for x in COMPETITORS if x in sub.index]
            rival = sub.loc[pool, "auroc"].idxmax()
            r = paired_bootstrap(model, d, "CAPE", rival, BENCH["bootstrap"], BENCH["seed"])
            rival_b = sub.loc[pool, "brier"].idxmin()
            rb = paired_bootstrap(model, d, "CAPE", rival_b, BENCH["bootstrap"], BENCH["seed"], metric="brier")
            entries.append((model, d, rival, r, rival_b, rb))
            if d not in low_parse(model) | {"math_all"}:   # independent (backbone, test set) pairs
                p["p"].append((2 * min(r["p_le0"], 1 - r["p_le0"]), r["delta"]))
                p["p_brier"].append((2 * min(rb["p_le0"], 1 - rb["p_le0"]), rb["delta"]))
    return entries, {"thr": {k: bh_threshold(v) for k, v in p.items()}}


def significance_frame() -> pd.DataFrame:
    """Full per-setting tests behind the main-result markers, including intervals and comparison identities."""
    entries, counts = significance()
    rows = []
    for model, dataset, rival_a, test_a, rival_b, test_b in entries:
        in_family = dataset not in low_parse(model) | {"math_all"}
        for metric, rival, test, key in (("auroc", rival_a, test_a, "p"),
                                         ("brier", rival_b, test_b, "p_brier")):
            p = 2 * min(test["p_le0"], 1 - test["p_le0"])
            threshold = counts["thr"][key]
            rows.append({"model": model, "dataset": dataset, "metric": metric, "rival": rival,
                         **test, "p_two_sided": p, "bh_threshold": threshold, "in_family": in_family,
                         "significant": in_family and p <= threshold,
                         "bootstrap_draws": BENCH["bootstrap"], "seed": BENCH["seed"]})
    return pd.DataFrame(rows)


def attribution_frames() -> dict[str, pd.DataFrame]:
    """The three attribution tables (``uqbench.studies.attribution``) of every backbone, restricted to its high-parse
    pairs."""
    out = {}
    for name in ("safety", "null", "null_paired"):
        rows = []
        for m in BACKBONES:
            path = RESULTS_DIR / m / f"attribution_{name}.csv"
            if path.exists():
                d = pd.read_csv(path)
                rows.append(d[~d.dataset.isin(low_parse(m) | {"math_all"})].assign(model=m))
        out[name] = pd.concat(rows) if rows else pd.DataFrame()
    return out


UNANIMOUS_DOT = 0.9   # CAPE's confidence threshold marked by the dot on the consistent-error curve
