"""CPU stage: re-grade stored math answers without re-running any GPU stage.

1. ``rules``: the current ``uqbench.verify`` rules (a leading definition such as "z(t) =" dropped, approximate
   quantities equal within 1%) relabel wrong answers they accept, and merge answer clusters whose representatives
   they equate (label-free; ids renumbered in order of first appearance; judge outputs mapped onto the merged ids).
2. ``submit`` / ``status`` / ``apply``: an LLM grader (Claude Haiku 5.5 with adaptive thinking, Message Batches API)
   sees the problem, the reference and each distinct answer the rules still reject (pairs of plain numbers are left
   to the rules) and decides whether the answer is correct. Only labels change: clusters never use the grader,
   which sees the reference.
3. ``pairs-submit`` / ``status`` / ``pairs-apply``: the same model, without the reference, judges whether two
   answers to the same question are equivalent, for every pair of cluster representatives (answers of runs whose
   responses are pooled, such as qwen3-8b and qwen3-8b-x8, together); equivalent pairs go to
   ``OUTPUT_DIR/math_equiv/equivalent.jsonl``, which ``uqbench.verify.cluster_answers`` reads, and the stored
   clusters merge accordingly. Grading and clustering then use the same notion of equivalence (assumption (A1)).

Every changed label is listed in ``OUTPUT_DIR/<model>/regrade.jsonl`` (``source`` = rules | llm); the cluster cache
and judge outputs before the first run are kept in ``OUTPUT_DIR/<model>/pre_regrade/``. Raw grader results live in
``OUTPUT_DIR/math_regrade/``.

Usage::

    uqbench regrade rules --models qwen3-8b r1-distill-qwen-7b gpt-oss-20b qwen3-8b-x8 claude-haiku-5-5
    python -m uqbench.labeling.regrade submit --models ...
    python -m uqbench.labeling.regrade status [--wait]
    python -m uqbench.labeling.regrade apply  --models ...
    uqbench regrade pairs-submit --groups qwen3-8b,qwen3-8b-x8 r1-distill-qwen-7b gpt-oss-20b claude-haiku-5-5
    uqbench regrade pairs-apply  --models qwen3-8b r1-distill-qwen-7b gpt-oss-20b qwen3-8b-x8 claude-haiku-5-5
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

from uqbench.config import OUTPUT_DIR, run_dir
from uqbench.io import read_jsonl, write_jsonl
from uqbench.labeling.common import remap_judge
from uqbench.labeling.extract import MODEL
from uqbench.verify import _decimal, llm_equivalent, math_equal

SKIP = ("dev",)   # generated datasets no result uses
ROOT = OUTPUT_DIR / "math_regrade"
PROMPT = ("You grade the final answer to a math or science problem against the reference answer.\n\n"
          "Problem:\n{question}\n\nReference answer: {answer}\n\nCandidate answer: {pred}\n\n"
          "The candidate is correct only if it answers the quantity the problem asks for and is equivalent to the "
          "reference: an algebraically equivalent expression, equivalent notation, an equivalent equation, reordered "
          "terms, renamed arbitrary constants, or extra units or words around the same value. When the problem "
          "states a form, range or condition (for example an angle in [0, 2pi), simplest form, a specific case), the "
          "candidate must meet it; a general formula that still needs substitution does not answer a specific case. "
          "When the reference is an exact value (integer, fraction, radical, symbolic expression), the candidate "
          "must be exactly equal to it, not merely close in value. Only when the reference is a decimal approximation "
          "of a measured or computed quantity may the candidate differ from it by at most 1%. Check equivalence "
          "carefully, then give a one-sentence reason and the verdict.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reason", "correct"],
          "properties": {"reason": {"type": "string"}, "correct": {"type": "boolean"}}}
PAIR_ROOT = OUTPUT_DIR / "math_equiv"
PAIR_PROMPT = ("Two final answers to the same math or science problem are given. Decide whether they state the same "
               "answer.\n\nProblem:\n{question}\n\nAnswer A: {a}\n\nAnswer B: {b}\n\n"
               "They are the same if they are equivalent answers to what the problem asks: an algebraically "
               "equivalent expression, equivalent notation, an equivalent equation, reordered terms, renamed "
               "arbitrary constants, or extra units or words around the same value. Exact values must be exactly "
               "equal, not merely close; two decimal approximations of the same measured or computed quantity are the "
               "same within 1%. Do not judge whether either answer is correct. Check equivalence carefully, then give "
               "a one-sentence reason and the verdict.")
PAIR_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reason", "same"],
               "properties": {"reason": {"type": "string"}, "same": {"type": "boolean"}}}


def math_files(model: str):
    for path in sorted((run_dir(model) / "gen").glob("*.jsonl")):
        if path.stem in SKIP:
            continue
        rows = list(read_jsonl(path))
        if rows and rows[0]["task"] == "math":
            yield path, rows


def backup(root) -> None:
    dest = root / "pre_regrade"
    if not dest.exists():
        for sub in ("clusters", "judge", "judge_think"):
            if (root / sub).exists():
                shutil.copytree(root / sub, dest / sub)


def merge(clusters: list[int], preds: list[str | None], same=None) -> list[int]:
    """Greedily merge clusters into the first earlier cluster whose representative is equivalent (by the rules, or by
    ``same``)."""
    same = same or (lambda a, b: math_equal(a, b) or math_equal(b, a))
    reps, root = {}, {}
    for c, p in zip(clusters, preds, strict=True):
        if c in root:
            continue
        hit = next((r for r, rp in reps.items() if p is not None and rp is not None and same(rp, p)), None)
        root[c] = hit if hit is not None else c
        if hit is None:
            reps[c] = p
    order = {}
    for c in clusters:
        order.setdefault(root[c], len(order))
    return [order[root[c]] for c in clusters]


def rewrite(root, name: str, rows: list[dict], new: dict[str, list[int]] | None) -> None:
    """The generations, and when clusters changed the cluster cache and the judge outputs."""
    write_jsonl(root / "gen" / f"{name}.jsonl", rows)
    cache = root / "clusters" / f"{name}.jsonl"
    if not new or not cache.exists():
        return
    old = {r["uid"]: r["clusters"] for r in read_jsonl(cache)}
    write_jsonl(cache, ({"uid": u, "clusters": new.get(u, c)} for u, c in old.items()))
    for judge in ("judge", "judge_think"):
        for shard in sorted((root / judge / "_shards").glob(f"{name}.*.jsonl")):
            write_jsonl(shard, [remap_judge(j, old[j["uid"]], new[j["uid"]]) if j["uid"] in new else j
                                for j in read_jsonl(shard)])


def log(root, changes: list[dict]) -> None:
    path = root / "regrade.jsonl"
    write_jsonl(path, (list(read_jsonl(path)) if path.exists() else []) + changes)


def rules(args) -> None:
    for model in args.models:
        root = run_dir(model)
        backup(root)
        changes = []
        for path, rows in math_files(model):
            cache = root / "clusters" / f"{path.stem}.jsonl"
            old = {r["uid"]: r["clusters"] for r in read_jsonl(cache)} if cache.exists() else {}
            new, flips = {}, 0
            for r in rows:
                preds = [s["pred"] for s in r["samples"]]
                for k, s in enumerate(r["samples"]):
                    if not s["correct"] and s["pred"] is not None and math_equal(str(r["answer"]), s["pred"]):
                        s["correct"] = 1
                        flips += 1
                        changes.append({"dataset": path.stem, "uid": r["uid"], "sample": k, "pred": s["pred"],
                                        "answer": r["answer"], "source": "rules"})
                if r["uid"] in old:
                    merged = merge(old[r["uid"]], preds)
                    if merged != old[r["uid"]]:
                        new[r["uid"]] = merged
            rewrite(root, path.stem, rows, new)
            print(f"[regrade] {model}/{path.stem}: {flips} responses now correct, {len(new)} questions reclustered",
                  flush=True)
        log(root, changes)


def pending(models: list[str]):
    """(key, question record) per distinct wrong answer of a question, key = (model, dataset, uid, answer); pairs of
    plain numbers are left to the rules."""
    for model in models:
        for path, rows in math_files(model):
            for r in rows:
                plain = _decimal(str(r["answer"])) is not None
                seen = set()
                for s in r["samples"]:
                    p = s["pred"]
                    if s["correct"] or p is None or p in seen or (plain and _decimal(p) is not None):
                        continue
                    seen.add(p)
                    yield (model, path.stem, r["uid"], p), r


def params(content: str, schema: dict) -> dict:
    return {"model": MODEL, "max_tokens": 8192, "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
            "messages": [{"role": "user", "content": content}]}


def keys(root, batch: str) -> dict[str, tuple]:
    """custom id -> key of one submitted batch: (model, dataset, uid, answer) for grading, (uid, a, b) for pairs."""
    return {r["custom_id"]: tuple(r["key"]) for r in read_jsonl(root / f"{batch}.keys.jsonl")}


def results(root=ROOT) -> dict[tuple, dict]:
    out = {}
    for path in sorted(root.glob("*.results.jsonl")):
        key = keys(root, path.name.removesuffix(".results.jsonl"))
        out.update({key[r["custom_id"]]: r for r in read_jsonl(path)})
    return out


def send(root, items: list[tuple[tuple, dict]]) -> None:
    """Submit (key, request params) as one batch, skipping keys submitted before."""
    import anthropic
    from anthropic.types.messages.batch_create_params import Request

    root.mkdir(parents=True, exist_ok=True)
    done = {k for path in root.glob("*.keys.jsonl") for k in keys(root, path.name.removesuffix(".keys.jsonl")).values()}
    items = [(key, p) for key, p in items if key not in done]
    print(f"[regrade] {len(items)} requests")
    if not items:
        return
    batch = anthropic.Anthropic().messages.batches.create(
        requests=[Request(custom_id=f"r{i}", params=p) for i, (_, p) in enumerate(items)])
    write_jsonl(root / f"{batch.id}.keys.jsonl", ({"custom_id": f"r{i}", "key": list(key)}
                                                  for i, (key, _) in enumerate(items)))
    print(f"[regrade] submitted {batch.id}")


def graded() -> set[tuple]:
    """(dataset, uid, answer) already submitted for any run: the grader sees only the problem, reference and answer."""
    return {key[1:] for path in ROOT.glob("*.keys.jsonl")
            for key in keys(ROOT, path.name.removesuffix(".keys.jsonl")).values()}


def submit(args) -> None:
    done = graded()
    send(ROOT, [(key, params(PROMPT.format(question=r["question"][:4000], answer=r["answer"], pred=key[3]), SCHEMA))
                for key, r in pending(args.models) if key[1:] not in done])


def pairs(groups: list[list[str]]):
    """((uid, a, b), question) for every pair of distinct cluster representatives of a question, over the stored
    clusters of each group of runs whose responses are pooled; pairs of plain numbers are left to the rules."""
    seen = set()
    for models in groups:
        reps, text = {}, {}
        for model in models:
            for path, rows in math_files(model):
                cache = run_dir(model) / "clusters" / f"{path.stem}.jsonl"
                stored = {r["uid"]: r["clusters"] for r in read_jsonl(cache)}
                for r in rows:
                    first = {}
                    for c, s in zip(stored[r["uid"]], r["samples"], strict=True):
                        if s["pred"] is not None:
                            first.setdefault(c, s["pred"])
                    reps.setdefault(r["uid"], []).extend(p for p in first.values() if p not in reps[r["uid"]])
                    text[r["uid"]] = r["question"]
        for uid, answers in reps.items():
            for i, a in enumerate(answers):
                for b in answers[i + 1:]:
                    key = (uid, *sorted((a, b)))
                    if key not in seen and (_decimal(a) is None or _decimal(b) is None):   # plain numbers: the rules
                        seen.add(key)
                        yield key, text[uid]


def pairs_submit(args) -> None:
    send(PAIR_ROOT, [(key, params(PAIR_PROMPT.format(question=q[:4000], a=key[1], b=key[2]), PAIR_SCHEMA))
                     for key, q in pairs([g.split(",") for g in args.groups])])


def status(args) -> None:
    import anthropic

    client = anthropic.Anthropic()
    while True:
        open_batches = 0
        for path in sorted([*ROOT.glob("*.keys.jsonl"), *PAIR_ROOT.glob("*.keys.jsonl")]):
            batch_id = path.name.removesuffix(".keys.jsonl")
            out = path.parent / f"{batch_id}.results.jsonl"
            if out.exists():
                continue
            batch = client.messages.batches.retrieve(batch_id)
            print(f"{batch_id}: {batch.processing_status} {batch.request_counts}", flush=True)
            if batch.processing_status != "ended":
                open_batches += 1
                continue
            rows, cost = [], 0.0
            for r in client.messages.batches.results(batch_id):
                verdict = {"custom_id": r.custom_id, "type": r.result.type}
                if r.result.type == "succeeded":
                    m = r.result.message
                    try:
                        verdict.update(json.loads("".join(b.text for b in m.content if b.type == "text")))
                    except json.JSONDecodeError:
                        pass
                    cost += (m.usage.input_tokens * 0.05 + m.usage.output_tokens * 0.25) / 1e6
                rows.append(verdict)
            write_jsonl(out, rows)
            print(f"  downloaded {len(rows)} results, ${cost:.2f}", flush=True)
        if not args.wait or open_batches == 0:
            break
        time.sleep(args.interval)


def apply(args) -> None:
    got = {}   # (dataset, uid, answer) -> verdict, whichever run it was submitted for
    for key, v in results().items():
        got[key[1:]] = got.get(key[1:], False) or v.get("correct") is True
    for model in args.models:
        root = run_dir(model)
        changes = []
        for path, rows in math_files(model):
            flips = 0
            for r in rows:
                for k, s in enumerate(r["samples"]):
                    if not s["correct"] and got.get((path.stem, r["uid"], s["pred"])):
                        s["correct"] = 1
                        flips += 1
                        changes.append({"dataset": path.stem, "uid": r["uid"], "sample": k, "pred": s["pred"],
                                        "answer": r["answer"], "source": "llm"})
            rewrite(root, path.stem, rows, None)
            print(f"[regrade] {model}/{path.stem}: {flips} responses now correct (LLM grader)", flush=True)
        log(root, changes)


def pairs_apply(args) -> None:
    same = [{"uid": k[0], "a": k[1], "b": k[2]} for k, v in results(PAIR_ROOT).items() if v.get("same") is True]
    write_jsonl(PAIR_ROOT / "equivalent.jsonl", same)
    llm_equivalent.cache_clear()
    table = llm_equivalent()
    print(f"[regrade] {len(same)} equivalent pairs", flush=True)
    for model in args.models:
        root = run_dir(model)
        for path, rows in math_files(model):
            cache = root / "clusters" / f"{path.stem}.jsonl"
            old = {r["uid"]: r["clusters"] for r in read_jsonl(cache)}
            new = {}
            for r in rows:
                pairs_of = table.get(r["uid"], set())
                merged = merge(old[r["uid"]], [s["pred"] for s in r["samples"]],
                               lambda a, b, s=pairs_of: frozenset((a, b)) in s)
                if merged != old[r["uid"]]:
                    new[r["uid"]] = merged
            rewrite(root, path.stem, rows, new)
            print(f"[regrade] {model}/{path.stem}: {len(new)} questions reclustered", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("rules", "submit", "apply", "pairs-apply"):
        sub.add_parser(name).add_argument("--models", nargs="+", required=True)
    sub.add_parser("pairs-submit").add_argument("--groups", nargs="+", required=True,
                                                help="comma-separated runs whose responses are pooled")
    p = sub.add_parser("status")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()
    {"rules": rules, "submit": submit, "status": status, "apply": apply, "pairs-submit": pairs_submit,
     "pairs-apply": pairs_apply}[args.stage](args)


if __name__ == "__main__":
    main()
