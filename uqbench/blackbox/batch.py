"""Black-box run of Claude Haiku 5.5 on the benchmark questions through the Message Batches API.

The API returns neither log-probabilities nor hidden states and never returns the raw reasoning, so only the
final visible text of each response is kept. Prompts, answer extraction and grading are those of the open-weight
runs (``uqbench.pipeline.data``, ``uqbench.verify``); outputs follow ``uqbench.pipeline.generate`` so
``uqbench.records`` reads them.

Requests are keyed ``<uid>__<k>`` (``__v`` appended for the verbal-confidence follow-up). Every stage skips
requests that already have a result or sit in a running batch, so a pilot's responses are part of the full run.
Before submitting, the spend so far plus the projected cost of all pending and new requests must stay below
``--budget``.

Usage::

    python -m uqbench.blackbox.batch smoke                          # one synchronous request per request shape
    python -m uqbench.blackbox.batch submit --per-dataset 8 --k 2   # pilot
    python -m uqbench.blackbox.batch submit                         # everything not yet answered
    python -m uqbench.blackbox.batch status [--wait]                # poll batches, download ended ones
    python -m uqbench.blackbox.batch verbal                         # verbal-confidence follow-ups of finished responses
    python -m uqbench.blackbox.batch merge                          # OUTPUT_DIR/<MODEL>/{gen,followup/_shards}/
    python -m uqbench.blackbox.batch cost                           # spend and projection per dataset
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

from uqbench.config import load_yaml, run_dir
from uqbench.io import read_jsonl, write_jsonl
from uqbench.pipeline import data
from uqbench.verbal import VERBAL_CLASSES, VERBAL_PROMPT, parse_verbal
from uqbench.verify import extract_answer, is_correct

MODEL = "claude-haiku-5-5"
EFFORT = "medium"            # the model's default, pinned so later default changes do not alter the run
# Batch prices in $ per token (half the standard $0.10 / $0.50 per million for prompts up to 100K tokens).
PRICE_IN, PRICE_OUT = 0.05e-6, 0.25e-6
# Projection for a request kind without observed results yet: (input, output) tokens per request.
PRIOR = {"gen": (400, 8000), "verbal": (3000, 30)}
MAX_BATCH = 50_000           # requests per submitted batch (API limit 100,000 / 256 MB)

ROOT = run_dir(MODEL)
BATCHES = ROOT / "batches"


# ------------------------------------------------------------------ requests
@cache
def bench() -> dict:
    return load_yaml("benchmark")


def gen_params(record: dict) -> dict:
    return {"model": MODEL, "max_tokens": bench()["max_tokens"], "thinking": {"type": "adaptive"},
            "output_config": {"effort": EFFORT},
            "messages": [{"role": "user", "content": data.user_prompt(record)}]}


# The reply of the verbal follow-up is constrained to the class names, as decoding is for the open-weight backbones.
VERBAL_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["confidence"],
                 "properties": {"confidence": {"type": "string", "enum": [c.title() for c in VERBAL_CLASSES]}}}


def verbal_params(record: dict, response: str) -> dict:
    """The LRMVerbal two-turn proxy (``uqbench.verbal``): the finished response, then the confidence question,
    answered with thinking off."""
    return {"model": MODEL, "max_tokens": 64, "thinking": {"type": "disabled"},
            "output_config": {"format": {"type": "json_schema", "schema": VERBAL_SCHEMA}},
            "messages": [{"role": "user", "content": data.user_prompt(record)},
                         {"role": "assistant", "content": response},
                         {"role": "user", "content": VERBAL_PROMPT}]}


def datasets() -> list[str]:
    """Generated datasets behind the test sets and the training pools, in generation order."""
    b = bench()
    wanted = {p for n in b["test_sets"] + b["training_pools"] for p in b.get("test_union", {}).get(n, [n])}
    return [n for n in b["generation_order"] if n in wanted]


@cache
def questions(name: str) -> list[dict]:
    """Records of a generated dataset; uids that occur more than once are dropped (``uqbench.records`` drops them
    too, so sampling them would be wasted)."""
    rows = data.load(name)
    counts = Counter(r["uid"] for r in rows)
    return [r for r in rows if counts[r["uid"]] == 1]


def k_of(name: str) -> int:
    return bench()["samples"]["pool" if name in bench()["training_pools"] else "test"]


# ------------------------------------------------------------------ local state
def index() -> list[dict]:
    path = BATCHES / "index.jsonl"
    return list(read_jsonl(path)) if path.exists() else []


def results() -> dict[str, dict]:
    """Downloaded results by request id; a later succeeded result replaces an earlier failure."""
    out = {}
    for b in index():
        path = BATCHES / f"{b['id']}.results.jsonl"
        if path.exists():
            for r in read_jsonl(path):
                if r["custom_id"] not in out or r["type"] == "succeeded":
                    out[r["custom_id"]] = r
    return out


def pending() -> dict[str, str]:
    """Request ids in batches not yet downloaded, with their kind."""
    out = {}
    for b in index():
        if not (BATCHES / f"{b['id']}.results.jsonl").exists():
            out.update({cid: b["kind"] for cid in read_jsonl_ids(BATCHES / f"{b['id']}.ids")})
    return out


def read_jsonl_ids(path: Path) -> list[str]:
    return path.read_text().split()


def kind_of(cid: str) -> str:
    return "verbal" if cid.endswith("__v") else "gen"


# ------------------------------------------------------------------ cost
def cost_of(usage: dict) -> float:
    return usage["input_tokens"] * PRICE_IN + usage["output_tokens"] * PRICE_OUT


def per_request(res: dict[str, dict], lookup: dict[str, str]) -> dict[tuple[str, str], tuple[float, int]]:
    """Mean observed cost of a succeeded request per (kind, dataset), and the number observed."""
    acc = defaultdict(list)
    for cid, r in res.items():
        if r["type"] == "succeeded":
            acc[(kind_of(cid), lookup[cid.split("__")[0]])].append(cost_of(r["usage"]))
    return {key: (sum(v) / len(v), len(v)) for key, v in acc.items()}


def projected(cids: list[str], res: dict[str, dict], lookup: dict[str, str]) -> float:
    """Projected cost of requests not yet answered: the observed mean of their (kind, dataset) when at least
    eight results exist, else the conservative prior."""
    means = per_request(res, lookup)
    total = 0.0
    for cid in cids:
        kind = kind_of(cid)
        mean, n = means.get((kind, lookup[cid.split("__")[0]]), (0.0, 0))
        prior = PRIOR[kind][0] * PRICE_IN + PRIOR[kind][1] * PRICE_OUT
        total += mean if n >= 8 else prior
    return total


def spent(res: dict[str, dict]) -> float:
    return sum(cost_of(r["usage"]) for r in res.values() if r.get("usage"))


@cache
def uid_lookup() -> dict[str, str]:
    return {r["uid"]: name for name in datasets() for r in questions(name)}


# ------------------------------------------------------------------ stages
def client():
    import anthropic

    return anthropic.Anthropic()


def smoke(_args) -> None:
    """One synchronous request per request shape (streamed: the generation budget is large)."""
    record = data.load("math500")[0]
    c = client()
    with c.messages.stream(**gen_params(record)) as stream:
        msg = stream.get_final_message()
    text = "".join(b.text for b in msg.content if b.type == "text")
    pred = extract_answer(text, record["task"])
    print(f"gen: stop={msg.stop_reason} usage={msg.usage.input_tokens}/{msg.usage.output_tokens} "
          f"pred={pred!r} gold={record['answer']!r} correct={is_correct(pred, record['answer'], record['task'])}")
    print("     text tail:", text[-200:].replace("\n", " "))
    v = c.messages.create(**verbal_params(record, text))
    vtext = "".join(b.text for b in v.content if b.type == "text")
    print(f"verbal: stop={v.stop_reason} usage={v.usage.input_tokens}/{v.usage.output_tokens} "
          f"reply={vtext!r} conf={parse_verbal(vtext)}")
    usd = (msg.usage.input_tokens + v.usage.input_tokens) * 2 * PRICE_IN \
        + (msg.usage.output_tokens + v.usage.output_tokens) * 2 * PRICE_OUT
    print(f"smoke cost (standard price): ${usd:.4f}")


def submit_requests(kind: str, items: list[tuple[str, dict]], budget: float, dry: bool) -> None:
    res, pend, lookup = results(), pending(), uid_lookup()
    items = [(cid, p) for cid, p in items
             if cid not in pend and not (cid in res and res[cid]["type"] == "succeeded")]
    if not items:
        print(f"[{kind}] nothing to submit")
        return
    so_far = spent(res)
    queued = projected(list(pend), res, lookup)
    new = projected([cid for cid, _ in items], res, lookup)
    print(f"[{kind}] {len(items)} requests; spent ${so_far:.2f}, pending ${queued:.2f} ({len(pend)} requests), "
          f"this submission ~${new:.2f}; budget ${budget:.2f}")
    if so_far + queued + new > budget:
        raise SystemExit(f"[{kind}] refused: projected total ${so_far + queued + new:.2f} exceeds the budget")
    if dry:
        return
    from anthropic.types.messages.batch_create_params import Request

    c = client()
    BATCHES.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(items), MAX_BATCH):
        chunk = items[start:start + MAX_BATCH]
        batch = c.messages.batches.create(requests=[Request(custom_id=cid, params=p) for cid, p in chunk])
        (BATCHES / f"{batch.id}.ids").write_text("\n".join(cid for cid, _ in chunk))
        with open(BATCHES / "index.jsonl", "a") as f:
            f.write(json.dumps({"id": batch.id, "kind": kind, "n": len(chunk), "created": time.time()}) + "\n")
        print(f"[{kind}] submitted {batch.id} ({len(chunk)} requests)")


def submit(args) -> None:
    items = []
    for name in args.datasets or datasets():
        rows = questions(name)
        rows = rows[: args.per_dataset] if args.per_dataset else rows
        k = min(args.k, k_of(name)) if args.k else k_of(name)
        items += [(f"{r['uid']}__{i}", gen_params(r)) for r in rows for i in range(k)]
    submit_requests("gen", items, args.budget, args.dry_run)


def verbal(args) -> None:
    res = results()
    records = {r["uid"]: r for name in datasets() for r in questions(name)}
    items = []
    for cid, r in res.items():
        if kind_of(cid) == "gen" and r["type"] == "succeeded" and r["text"].strip():
            items.append((cid + "__v", verbal_params(records[cid.split("__")[0]], r["text"])))
    submit_requests("verbal", items, args.budget, args.dry_run)


def status(args) -> None:
    c = client()
    while True:
        open_batches = 0
        for b in index():
            out = BATCHES / f"{b['id']}.results.jsonl"
            if out.exists():
                continue
            batch = c.messages.batches.retrieve(b["id"])
            n = batch.request_counts
            print(f"{b['id']} [{b['kind']}] {batch.processing_status}: processing={n.processing} "
                  f"succeeded={n.succeeded} errored={n.errored} canceled={n.canceled} expired={n.expired}")
            if batch.processing_status != "ended":
                open_batches += 1
                continue
            rows = []
            for r in c.messages.batches.results(b["id"]):
                row = {"custom_id": r.custom_id, "type": r.result.type}
                if r.result.type == "succeeded":
                    m = r.result.message
                    row.update(text="".join(x.text for x in m.content if x.type == "text"),
                               stop_reason=m.stop_reason, usage={"input_tokens": m.usage.input_tokens,
                                                                 "output_tokens": m.usage.output_tokens})
                elif r.result.type == "errored":
                    row["error"] = str(r.result.error)
                rows.append(row)
            write_jsonl(out, rows)
            kinds = Counter(r["type"] for r in rows)
            print(f"  downloaded {len(rows)} results: {dict(kinds)}")
        if not args.wait or open_batches == 0:
            break
        time.sleep(args.interval)


def blocked_as_refusal(r: dict | None) -> dict | None:
    """An output blocked by the content filter is a declined response, scored like a refusal (no answer)."""
    if r and r["type"] == "errored" and "content filtering" in r["error"]:
        return {**r, "type": "succeeded", "text": "", "stop_reason": "content_filter",
                "usage": {"input_tokens": 0, "output_tokens": 0}}
    return r


def merge(args) -> None:
    """Generation records in the format of ``uqbench.pipeline.generate`` (questions with all K responses only) and the
    verbal follow-ups in the follow-up format that ``uqbench.records`` reads (``followup/_shards``). With
    ``--partial`` (pipeline tests on the pilot), each question keeps the responses it has and the output goes to
    ``OUTPUT_DIR/<MODEL>-pilot``."""
    res = results()
    root = run_dir(MODEL + "-pilot") if args.partial else ROOT
    for name in datasets():
        k = k_of(name)
        rows, folls = [], []
        for record in questions(name):
            got = [blocked_as_refusal(res.get(f"{record['uid']}__{i}")) for i in range(k)]
            if args.partial:
                got = [g for g in got if g and g["type"] == "succeeded"]
                k = len(got)
            if not got or not all(g and g["type"] == "succeeded" for g in got):
                continue
            samples = []
            for g in got:
                closed = g["stop_reason"] == "end_turn"
                pred = extract_answer(g["text"], record["task"]) if closed else None
                samples.append({"text": g["text"], "response": g["text"], "finish_reason": g["stop_reason"],
                                "closed": closed, "pred": pred, "usage": g["usage"],
                                "correct": int(is_correct(pred, record["answer"], record["task"]))})
            rows.append({**record, "prompt_ids": [], "samples": samples})
            verb = [res.get(f"{record['uid']}__{i}__v") for i in range(k)]
            folls.append({"uid": record["uid"], "samples": [
                {"ptrue_logp1": None, "ptrue_logp0": None, "verbal_text": v["text"] if v else None,
                 "verbal_conf": parse_verbal(v["text"]) if v and v["type"] == "succeeded" else None}
                for v in verb]})
        if not rows:
            continue
        write_jsonl(root / "gen" / f"{name}.jsonl", rows)
        write_jsonl(root / "followup" / "_shards" / f"{name}.0.jsonl", folls)
        n = sum(len(r["samples"]) for r in rows)
        acc = sum(s["correct"] for r in rows for s in r["samples"]) / n
        cut = sum(not s["closed"] for r in rows for s in r["samples"]) / n
        print(f"[merge] {name}: {len(rows)}/{len(questions(name))} questions, acc={acc:.3f}, unclosed={cut:.3f}")


def cost(_args) -> None:
    res, pend, lookup = results(), pending(), uid_lookup()
    means = per_request(res, lookup)
    print(f"{'kind':7} {'dataset':18} {'n':>6} {'$/req':>9} {'out tok':>8}")
    for (kind, name), (mean, n) in sorted(means.items()):
        outs = [r["usage"]["output_tokens"] for cid, r in res.items()
                if r["type"] == "succeeded" and kind_of(cid) == kind and lookup[cid.split("__")[0]] == name]
        print(f"{kind:7} {name:18} {n:6d} {mean:9.5f} {sum(outs) / len(outs):8.0f}")
    failed = Counter(r["type"] for r in res.values() if r["type"] != "succeeded")
    stops = Counter(r["stop_reason"] for r in res.values() if r["type"] == "succeeded")
    print(f"failed: {dict(failed)}  stop reasons: {dict(stops)}")
    todo = [f"{r['uid']}__{i}" for name in datasets() for r in questions(name) for i in range(k_of(name))]
    todo = [cid for cid in todo if not (cid in res and res[cid]["type"] == "succeeded") and cid not in pend]
    print(f"spent ${spent(res):.2f}; pending ${projected(list(pend), res, lookup):.2f} ({len(pend)}); "
          f"remaining generation ${projected(todo, res, lookup):.2f} ({len(todo)}); "
          f"verbal for all ~${projected([c + '__v' for c in todo], res, lookup):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("smoke")
    p = sub.add_parser("submit")
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--per-dataset", type=int, default=0, help="first N questions of each dataset (0 = all)")
    p.add_argument("--k", type=int, default=0, help="responses per question (0 = benchmark K)")
    p = sub.add_parser("verbal")
    for p in (sub.choices["submit"], sub.choices["verbal"]):
        p.add_argument("--budget", type=float, default=200.0)
        p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("status")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--interval", type=int, default=120)
    p = sub.add_parser("merge")
    p.add_argument("--partial", action="store_true")
    sub.add_parser("cost")
    args = ap.parse_args()
    stages = {"smoke": smoke, "submit": submit, "verbal": verbal, "status": status, "merge": merge, "cost": cost}
    stages[args.stage](args)


if __name__ == "__main__":
    main()
