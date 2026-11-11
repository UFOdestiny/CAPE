"""Answer extraction for open-domain QA responses that state their answer in prose instead of ``\\boxed{}``.

Claude Haiku 5.5 (Message Batches API) is shown the question and the end of the final response and copies the
answer the response gives, as a short phrase in the response's own words; it never sees the reference answer.
The extracted phrase then takes the place of the missing boxed answer: grading (exact alias match after
normalisation) and answer clustering are unchanged. Responses with an unclosed reasoning block stay unanswered.

Stages::

    python -m uqbench.labeling.extract submit --models qwen3-8b r1-distill-qwen-7b gpt-oss-20b claude-haiku-5-5
    python -m uqbench.labeling.extract status [--wait]
    uqbench extract apply --models ...     # labels, clusters and judge outputs of the affected questions

Raw results live in ``OUTPUT_DIR/qa_extract/``; changed samples carry ``pred_source = "extracted"``.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

from uqbench.config import OUTPUT_DIR, run_dir
from uqbench.io import read_jsonl, write_jsonl
from uqbench.labeling.common import remap_judge
from uqbench.verify import cluster_answers, is_correct, normalize_qa

MODEL = "claude-haiku-5-5"
DATASETS = ("triviaqa", "triviaqa_heldout", "mixed_pool")
TAIL = 2000    # characters of the final response shown to the extractor (a few responses repeat for 100K+)
ROOT = OUTPUT_DIR / "qa_extract"
PROMPT = ("Question: {question}\n\nResponse: {response}\n\n"
          "Copy the final answer that the response gives to the question: the shortest entity or phrase that answers "
          "it, in the response's own words. Do not correct or judge it. If the response gives no answer, return an "
          "empty string.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"],
          "properties": {"answer": {"type": "string"}}}


def pending(models: list[str]):
    """(custom id, question record, response) for every closed QA response without an extracted answer."""
    for model in models:
        for name in DATASETS:
            path = run_dir(model) / "gen" / f"{name}.jsonl"
            if not path.exists():
                continue
            rows = list(read_jsonl(path))
            counts = Counter(r["uid"] for r in rows)
            for r in rows:   # repeated question ids are dropped by uqbench.records
                if r["task"] != "qa" or counts[r["uid"]] > 1:
                    continue
                for k, s in enumerate(r["samples"]):
                    if s["closed"] and s["pred"] is None and s["response"].strip():
                        yield f"{model}__{r['uid']}__{k}", r, s["response"]


def params(record: dict, response: str) -> dict:
    return {"model": MODEL, "max_tokens": 64, "thinking": {"type": "disabled"},
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
            "messages": [{"role": "user", "content": PROMPT.format(question=record["question"],
                                                                     response=response[-TAIL:])}]}


def results() -> dict[str, str | None]:
    out = {}
    for path in sorted(ROOT.glob("*.results.jsonl")):
        for r in read_jsonl(path):
            out[r["custom_id"]] = r.get("answer")
    return out


def submit(args) -> None:
    import anthropic
    from anthropic.types.messages.batch_create_params import Request

    done = results()
    running = {cid for path in ROOT.glob("*.ids") if not path.with_suffix(".results.jsonl").exists()
               for cid in path.read_text().split()}
    items = [(cid, params(r, resp)) for cid, r, resp in pending(args.models) if cid not in done and cid not in running]
    print(f"[extract] {len(items)} requests")
    if not items:
        return
    client = anthropic.Anthropic()
    ROOT.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(items), 50_000):
        chunk = items[start:start + 50_000]
        batch = client.messages.batches.create(requests=[Request(custom_id=c, params=p) for c, p in chunk])
        (ROOT / f"{batch.id}.ids").write_text("\n".join(c for c, _ in chunk))
        print(f"[extract] submitted {batch.id} ({len(chunk)} requests)")


def status(args) -> None:
    import anthropic

    client = anthropic.Anthropic()
    while True:
        open_batches = 0
        for ids in sorted(ROOT.glob("*.ids")):
            out = ids.with_suffix(".results.jsonl")
            if out.exists():
                continue
            batch = client.messages.batches.retrieve(ids.stem)
            print(f"{ids.stem}: {batch.processing_status} {batch.request_counts}", flush=True)
            if batch.processing_status != "ended":
                open_batches += 1
                continue
            rows, cost = [], 0.0
            for r in client.messages.batches.results(ids.stem):
                answer = None
                if r.result.type == "succeeded":
                    m = r.result.message
                    text = "".join(b.text for b in m.content if b.type == "text")
                    try:
                        answer = json.loads(text)["answer"].strip() or None
                    except (json.JSONDecodeError, KeyError):
                        answer = None
                    cost += (m.usage.input_tokens * 0.05 + m.usage.output_tokens * 0.25) / 1e6
                rows.append({"custom_id": r.custom_id, "type": r.result.type, "answer": answer})
            write_jsonl(out, rows)
            print(f"  downloaded {len(rows)} results, ${cost:.2f}", flush=True)
        if not args.wait or open_batches == 0:
            break
        time.sleep(args.interval)


def apply(args) -> None:
    got = results()
    for model in args.models:
        root = run_dir(model)
        for name in DATASETS:
            path = root / "gen" / f"{name}.jsonl"
            if not path.exists():
                continue
            rows = list(read_jsonl(path))
            cache = root / "clusters" / f"{name}.jsonl"
            old = {r["uid"]: r["clusters"] for r in read_jsonl(cache)} if cache.exists() else {}
            changed, recovered, correct = {}, 0, 0
            for r in rows:
                for k, s in enumerate(r["samples"]):
                    answer = got.get(f"{model}__{r['uid']}__{k}")
                    if s["pred"] is None and answer and normalize_qa(answer):
                        s["pred"], s["pred_source"] = normalize_qa(answer), "extracted"
                        s["correct"] = int(is_correct(s["pred"], r["answer"], r["task"]))
                        changed[r["uid"]] = r
                        recovered += 1
                        correct += s["correct"]
            new = {u: cluster_answers([s["pred"] for s in r["samples"]], r["task"]) for u, r in changed.items()}
            write_jsonl(path, rows)
            if cache.exists():
                write_jsonl(cache, ({"uid": u, "clusters": new.get(u, c)} for u, c in old.items()))
            for judge in ("judge", "judge_think"):
                for shard in sorted((root / judge / "_shards").glob(f"{name}.*.jsonl")):
                    write_jsonl(shard, [remap_judge(j, old[j["uid"]], new[j["uid"]]) if j["uid"] in new else j
                                        for j in read_jsonl(shard)])
            print(f"[extract] {model}/{name}: {recovered} answers extracted ({correct} correct), "
                  f"{len(changed)} questions reclustered", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in ("submit", "apply"):
        sub.add_parser(name).add_argument("--models", nargs="+", required=True)
    p = sub.add_parser("status")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()
    {"submit": submit, "status": status, "apply": apply}[args.stage](args)


if __name__ == "__main__":
    main()
