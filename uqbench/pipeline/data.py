"""Benchmark datasets: raw HF downloads -> unified problem records.

Every record is a dict with keys ``uid, dataset, split, task, question, answer, meta`` where
``task`` is ``"math"`` (free-form answer checked by math_verify), ``"mc"`` (option letter) or
``"qa"`` (short open answer matched against normalised aliases; ``answer`` is then a list).

Usage::

    python -m uqbench.pipeline.data   # writes DATASETS_DIR/processed/<name>.jsonl, including the further test
                                      # questions (``*_heldout``, benchmark.yaml: test_union) and the training pools
"""
from __future__ import annotations

import argparse
import json
import random
import re

import pandas as pd

from uqbench.config import DATASETS_DIR
from uqbench.verify import extract_boxed

RAW = DATASETS_DIR / "raw"
PROCESSED = DATASETS_DIR / "processed"

NONMATH_SIZE = 1000
POOL = "math_pool"
POOL_SIZE = 3000
MIXED_POOL = "mixed_pool"
MIXED_POOL_SIZE = 500      # per source (MMLU-Pro non-math, TriviaQA)
VAL_FRACTION = 0.2
SEED = 0

MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
QA_SUFFIX = ("\n\nPlease reason step by step, and put your final answer (a short entity or phrase) "
             "within \\boxed{}.")
MC_SUFFIX = (
    "\n\nPlease reason step by step, and put the letter of your final answer choice "
    "within \\boxed{} (e.g., \\boxed{A})."
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _record(uid, dataset, split, task, question, answer, **meta):
    answer = [str(a) for a in answer] if isinstance(answer, list) else str(answer).strip()
    return {"uid": uid, "dataset": dataset, "split": split, "task": task,
            "question": question, "answer": answer, "meta": meta}


def user_prompt(record: dict) -> str:
    """User-turn text sent to the backbone (chat template is applied by the generator)."""
    suffix = {"mc": MC_SUFFIX, "qa": QA_SUFFIX}.get(record["task"], MATH_SUFFIX)
    return record["question"] + suffix


def load_math500() -> list[dict]:
    rows = [json.loads(line) for line in open(RAW / "HuggingFaceH4_MATH-500/test.jsonl")]
    return [_record(f"math500-{i}", "math500", "test", "math", r["problem"], r["answer"],
                    level=r["level"], subject=r["subject"], source_id=r["unique_id"])
            for i, r in enumerate(rows)]


def load_olympiadbench() -> list[dict]:
    """Text-only English OlympiadBench (OE_TO_maths_en_COMP), single-answer problems only."""
    df = pd.read_parquet(RAW / "math-ai_olympiadbench/test.parquet")
    out = []
    for _, r in df.iterrows():
        if r["image_1"] is not None or bool(r["is_multiple_answer"]):
            continue
        question = r["question"] if pd.isna(r["context"]) else f"{r['context']}\n\n{r['question']}"
        if isinstance(r["unit"], str) and r["unit"].strip():
            question += f"\n\n(Give the answer in {r['unit'].strip()}.)"
        out.append(_record(f"olympiadbench-{r['id']}", "olympiadbench", "test", "math", question,
                           r["final_answer"][0], answer_type=r["answer_type"]))
    return out


def load_minerva() -> list[dict]:
    rows = [json.loads(line) for line in open(RAW / "math-ai_minervamath/test.jsonl")]
    return [_record(f"minerva-{i}", "minerva", "test", "math", r["question"], r["answer"])
            for i, r in enumerate(rows)]


def load_aime() -> list[dict]:
    """AIME 2024 + 2025 + 2026 (90 problems), pooled into one stress-test set."""
    out = []
    a24 = pd.read_parquet(RAW / "math-ai_aime24/test-00000-of-00001.parquet")
    for _, r in a24.iterrows():
        out.append(_record(f"aime-2024-{r['id']}", "aime", "test", "math", r["problem"],
                           extract_boxed(r["solution"]), year=2024))
    for line in open(RAW / "math-ai_aime25/test.jsonl"):
        r = json.loads(line)
        out.append(_record(f"aime-2025-{r['id']}", "aime", "test", "math", r["problem"], r["answer"], year=2025))
    a26 = pd.read_parquet(RAW / "MathArena_aime_2026/data/train-00000-of-00001.parquet")
    for _, r in a26.iterrows():
        out.append(_record(f"aime-2026-{r['problem_idx']}", "aime", "test", "math", r["problem"],
                           r["answer"], year=2026))
    return out


def load_gpqa() -> list[dict]:
    """GPQA-Diamond with options shuffled by a fixed per-question RNG (seeded by row index)."""
    df = pd.read_csv(RAW / "Idavidrein_gpqa/gpqa_diamond.csv")
    out = []
    for i, r in df.iterrows():
        options = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = list(range(4))
        random.Random(SEED * 100_003 + i).shuffle(order)
        letters = "ABCD"
        body = "\n".join(f"{letters[k]}) {str(options[j]).strip()}" for k, j in enumerate(order))
        gold = letters[order.index(0)]
        out.append(_record(f"gpqa-{i}", "gpqa", "test", "mc", f"{r['Question'].strip()}\n\n{body}", gold,
                           domain=r.get("High-level domain")))
    return out


def _stratified(df: pd.DataFrame, by: str, n: int) -> pd.DataFrame:
    """Proportional stratified sample of ``n`` rows (fixed seed)."""
    frac = n / len(df)
    parts = [g.sample(n=max(1, round(len(g) * frac)), random_state=SEED) for _, g in df.groupby(by, sort=True)]
    return pd.concat(parts).sort_index()


def _mc_question(question: str, options: list[str]) -> str:
    letters = "ABCDEFGHIJ"
    body = "\n".join(f"{letters[k]}) {str(o).strip()}" for k, o in enumerate(options))
    return f"{question.strip()}\n\n{body}"


def load_mmlupro() -> list[dict]:
    """MMLU-Pro test, non-math categories, 1000 questions stratified by category (10 options)."""
    df = pd.read_parquet(RAW / "TIGER-Lab_MMLU-Pro/data/test-00000-of-00001.parquet")
    df = _stratified(df[df["category"] != "math"], "category", NONMATH_SIZE)
    return [_record(f"mmlupro-{r['question_id']}", "mmlupro", "test", "mc",
                    _mc_question(r["question"], list(r["options"])), r["answer"], category=r["category"])
            for _, r in df.iterrows()]


def load_supergpqa() -> list[dict]:
    """SuperGPQA, 1000 questions stratified by discipline (up to 10 options)."""
    df = pd.read_json(RAW / "m-a-p_SuperGPQA/SuperGPQA-all.jsonl", lines=True)
    df = _stratified(df, "discipline", NONMATH_SIZE)
    return [_record(f"supergpqa-{r['uuid']}", "supergpqa", "test", "mc",
                    _mc_question(r["question"], list(r["options"])), r["answer_letter"],
                    discipline=r["discipline"], difficulty=r["difficulty"])
            for _, r in df.iterrows()]


def load_triviaqa() -> list[dict]:
    """TriviaQA rc.nocontext validation, 1000 random questions; gold = normalised aliases."""
    path = sorted((RAW / "mandarjoshi_trivia_qa/rc.nocontext").glob("validation-*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in path], ignore_index=True)
    df = df.sample(n=NONMATH_SIZE, random_state=SEED).sort_index()
    return [_record(f"triviaqa-{r['question_id']}", "triviaqa", "test", "qa", r["question"],
                    sorted(set(r["answer"]["normalized_aliases"]) | {r["answer"]["normalized_value"]}))
            for _, r in df.iterrows()]


def load_math_pool(exclude: list[dict]) -> list[dict]:
    """Stratified MATH-train sample (level x subject), disjoint from all test questions."""
    seen = {_norm(r["question"]) for r in exclude if r["task"] == "math"}
    frames = []
    for path in sorted((RAW / "EleutherAI_hendrycks_math").glob("*/train-*.parquet")):
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df["answer"] = df["solution"].map(extract_boxed)
    df = df[df["answer"].notna() & ~df["problem"].map(_norm).isin(seen)].reset_index(drop=True)
    frac = POOL_SIZE / len(df)
    strata = [g.sample(n=max(1, round(len(g) * frac)), random_state=SEED)
              for _, g in df.groupby(["level", "type"], sort=True)]
    sampled = pd.concat(strata).sample(frac=1.0, random_state=SEED).reset_index()
    n_val = int(round(len(sampled) * VAL_FRACTION))
    out = []
    for k, r in sampled.iterrows():
        split = "val" if k < n_val else "train"
        out.append(_record(f"math_pool-{r['index']}", POOL, split, "math", r["problem"], r["answer"],
                           level=r["level"], subject=r["type"]))
    return out


def load_mixed_pool(exclude: list[dict]) -> list[dict]:
    """Ablation pool: MMLU-Pro non-math and TriviaQA questions disjoint from every test question."""
    seen = {r["uid"] for r in exclude} | {_norm(r["question"]) for r in exclude}
    df = pd.read_parquet(RAW / "TIGER-Lab_MMLU-Pro/data/test-00000-of-00001.parquet")
    df = df[(df["category"] != "math") & ~df["question_id"].map(lambda i: f"mmlupro-{i}").isin(seen)]
    mc = [_record(f"mixed_pool-mmlupro-{r['question_id']}", MIXED_POOL, "", "mc",
                  _mc_question(r["question"], list(r["options"])), r["answer"], source="mmlupro")
          for _, r in _stratified(df, "category", MIXED_POOL_SIZE).iterrows()]
    path = sorted((RAW / "mandarjoshi_trivia_qa/rc.nocontext").glob("validation-*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in path], ignore_index=True)
    df = df[~df["question"].map(_norm).isin(seen)].drop_duplicates("question_id")
    qa = [_record(f"mixed_pool-triviaqa-{r['question_id']}", MIXED_POOL, "", "qa", r["question"],
                  sorted(set(r["answer"]["normalized_aliases"]) | {r["answer"]["normalized_value"]}),
                  source="triviaqa")
          for _, r in df.sample(n=MIXED_POOL_SIZE, random_state=SEED).iterrows()]
    rows = pd.Series(mc + qa).sample(frac=1.0, random_state=SEED).tolist()
    n_val = int(round(len(rows) * VAL_FRACTION))
    return [{**r, "split": "val" if k < n_val else "train"} for k, r in enumerate(rows)]


HELDOUT_SIZE = {"math_heldout": 500, "mmlupro_heldout": 1000, "supergpqa_heldout": 1000, "triviaqa_heldout": 1000}
HELDOUT_SEED = SEED + 1


def load_heldout(exclude: list[dict]) -> dict[str, list[dict]]:
    """Further test questions, disjoint from the first test sample and both pools: MATH test without MATH-500,
    GPQA-Extended without Diamond, and new samples of MMLU-Pro, SuperGPQA and TriviaQA drawn as their first test
    samples are. ``benchmark.yaml: test_union`` merges them into the test sets."""
    seen = {r["uid"] for r in exclude} | {_norm(r["question"]) for r in exclude}
    first = lambda t: _norm(t.split("\n\n")[0])  # noqa: E731  (MC stems; math problems that share a stem)
    seen |= {first(r["question"]) for r in exclude}
    out = {}
    frames = [pd.read_parquet(p) for p in sorted((RAW / "EleutherAI_hendrycks_math").glob("*/test-*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    math500 = {_norm(r["question"]) for r in load_math500()}
    df["answer"] = df["solution"].map(extract_boxed)
    df = df[df["answer"].notna() & ~df["problem"].map(_norm).isin(seen | math500)
            & ~df["problem"].map(first).isin(seen)].reset_index(drop=True)
    frac = HELDOUT_SIZE["math_heldout"] / len(df)
    df = pd.concat([g.sample(n=max(1, round(len(g) * frac)), random_state=HELDOUT_SEED)
                    for _, g in df.groupby(["level", "type"], sort=True)]).sort_index()
    out["math_heldout"] = [_record(f"math_heldout-{i}", "math_heldout", "test", "math", r["problem"], r["answer"],
                                   level=r["level"], subject=r["type"]) for i, r in df.iterrows()]
    diamond = {_norm(q) for q in pd.read_csv(RAW / "Idavidrein_gpqa/gpqa_diamond.csv")["Question"]}
    ext = pd.read_csv(RAW / "Idavidrein_gpqa/gpqa_extended.csv")
    rows = []
    for i, r in ext.iterrows():
        if _norm(r["Question"]) in diamond:
            continue
        options = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        order = list(range(4))
        random.Random(HELDOUT_SEED * 100_003 + i).shuffle(order)
        body = "\n".join(f"{'ABCD'[k]}) {str(options[j]).strip()}" for k, j in enumerate(order))
        rows.append(_record(f"gpqa_heldout-{i}", "gpqa_heldout", "test", "mc", f"{r['Question'].strip()}\n\n{body}",
                            "ABCD"[order.index(0)], domain=r.get("High-level domain")))
    out["gpqa_heldout"] = rows
    df = pd.read_parquet(RAW / "TIGER-Lab_MMLU-Pro/data/test-00000-of-00001.parquet")
    df = df[(df["category"] != "math") & ~df["question_id"].map(lambda i: f"mmlupro-{i}").isin(seen)
            & ~df["question_id"].map(lambda i: f"mixed_pool-mmlupro-{i}").isin(seen)
            & ~df["question"].map(_norm).isin(seen)]
    frac = HELDOUT_SIZE["mmlupro_heldout"] / len(df)
    df = pd.concat([g.sample(n=max(1, round(len(g) * frac)), random_state=HELDOUT_SEED)
                    for _, g in df.groupby("category", sort=True)]).sort_index()
    out["mmlupro_heldout"] = [_record(f"mmlupro_heldout-{r['question_id']}", "mmlupro_heldout", "test", "mc",
                                      _mc_question(r["question"], list(r["options"])), r["answer"],
                                      category=r["category"]) for _, r in df.iterrows()]
    df = pd.read_json(RAW / "m-a-p_SuperGPQA/SuperGPQA-all.jsonl", lines=True)
    df = df[~df["uuid"].map(lambda u: f"supergpqa-{u}").isin(seen) & ~df["question"].map(_norm).isin(seen)]
    frac = HELDOUT_SIZE["supergpqa_heldout"] / len(df)
    df = pd.concat([g.sample(n=max(1, round(len(g) * frac)), random_state=HELDOUT_SEED)
                    for _, g in df.groupby("discipline", sort=True)]).sort_index()
    out["supergpqa_heldout"] = [_record(f"supergpqa_heldout-{r['uuid']}", "supergpqa_heldout", "test", "mc",
                                        _mc_question(r["question"], list(r["options"])), r["answer_letter"],
                                        discipline=r["discipline"], difficulty=r["difficulty"])
                                for _, r in df.iterrows()]
    path = sorted((RAW / "mandarjoshi_trivia_qa/rc.nocontext").glob("validation-*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in path], ignore_index=True)
    df = df[~df["question"].map(_norm).isin(seen)].drop_duplicates("question_id").drop_duplicates("question")
    df = df.sample(n=HELDOUT_SIZE["triviaqa_heldout"], random_state=HELDOUT_SEED).sort_index()
    out["triviaqa_heldout"] = [_record(f"triviaqa_heldout-{r['question_id']}", "triviaqa_heldout",
                                       "test", "qa", r["question"],
                                       sorted(set(r["answer"]["normalized_aliases"])
                                              | {r["answer"]["normalized_value"]}))
                               for _, r in df.iterrows()]
    return out


LOADERS = {"math500": load_math500, "olympiadbench": load_olympiadbench, "minerva": load_minerva,
           "aime": load_aime, "gpqa": load_gpqa, "mmlupro": load_mmlupro, "supergpqa": load_supergpqa,
           "triviaqa": load_triviaqa}


def load(name: str) -> list[dict]:
    """Read a processed dataset (run ``python -m uqbench.pipeline.data`` first)."""
    path = PROCESSED / f"{name}.jsonl"
    return [json.loads(line) for line in open(path)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tests = {name: fn() for name, fn in LOADERS.items()}
    every_test = [r for rows in tests.values() for r in rows]
    pools = {POOL: load_math_pool(every_test), MIXED_POOL: load_mixed_pool(every_test)}
    used = every_test + [r for rows in pools.values() for r in rows]
    datasets = {**tests, **pools, **load_heldout(used)}
    for name, rows in datasets.items():
        with open(PROCESSED / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        splits = pd.Series([r["split"] for r in rows]).value_counts().to_dict()
        print(f"{name:14s} {len(rows):5d} {splits}")


if __name__ == "__main__":
    main()
