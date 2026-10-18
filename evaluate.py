"""Score one extraction method on a test split.

Modes:
  zero_shot          base Qwen2.5-0.5B-Instruct, system prompt only
  few_shot           base model + 3 worked examples from the training set in the prompt
  lora               base model + LoRA adapter (--adapter adapters/main/epoch2)
  ollama:<model>     a larger model served by Ollama, zero-shot (e.g. ollama:qwen2.5:1.5b)
                     add --few_shot to give it the same 3 examples

Writes results/<split>__<tag>.jsonl (one row per example, with the raw output) and
appends one summary row to results/summary.jsonl.

Usage:
  python evaluate.py --mode zero_shot --split test_id
  python evaluate.py --mode lora --adapter adapters/main/epoch2 --split test_ood
"""
import argparse
import json
import time

from common import ROOT, build_messages, generate, load_jsonl, load_model, parse_json, score

OLLAMA = "http://localhost:11434"


def pick_few_shot(train):
    """3 fixed, deliberately different examples: per-group count, cell line, missing n."""
    def first(cond):
        return next(r for r in train if cond(r))
    return [
        first(lambda r: "per group" in r["text"] and r["label"]["organism"] == "Mus musculus"),
        first(lambda r: "cells" in r["text"] and r["label"]["organism"] == "Homo sapiens"),
        first(lambda r: r["label"]["sample_size"] is None),
    ]


def run_ollama(model, message_lists):
    import requests
    outs = []
    for msgs in message_lists:
        r = requests.post(f"{OLLAMA}/api/chat", json={
            "model": model, "messages": msgs, "stream": False,
            "options": {"temperature": 0, "num_predict": 64}}, timeout=300)
        outs.append(r.json()["message"]["content"])
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--split", default="test_id", choices=["val", "test_id", "test_ood"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--few_shot", action="store_true", help="with ollama:<model>, add the 3 examples")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N rows")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--tag", default=None, help="name for the output files (defaults to the mode)")
    args = ap.parse_args()

    rows = load_jsonl(args.split)[:args.limit]
    shots = pick_few_shot(load_jsonl("train")) if (args.mode == "few_shot" or args.few_shot) else None
    msgs = [build_messages(r["text"], few_shot=shots) for r in rows]

    tag = args.tag or args.mode.replace(":", "-") + ("-fewshot" if args.few_shot else "")
    if args.mode == "lora" and not args.tag:
        tag = "lora-" + args.adapter.rstrip("/").split("adapters/")[-1].replace("/", "-")

    t0 = time.time()
    if args.mode.startswith("ollama:"):
        raw = run_ollama(args.mode.split(":", 1)[1], msgs)
    else:
        if args.mode == "lora" and not args.adapter:
            ap.error("--mode lora needs --adapter")
        tok, model = load_model(args.adapter if args.mode == "lora" else None)
        t0 = time.time()   # exclude model load time
        raw = generate(tok, model, msgs, batch_size=args.batch_size)
    secs = time.time() - t0

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    totals = {"json_valid": 0, "json_strict": 0, "organism": 0, "assay": 0, "sample_size": 0,
              "all_fields": 0}
    with open(out_dir / f"{args.split}__{tag}.jsonl", "w") as fh:
        for r, out in zip(rows, raw):
            pred, strict = parse_json(out)
            s = score(pred, r["label"])
            totals["json_valid"] += pred is not None
            totals["json_strict"] += strict
            for k in ("organism", "assay", "sample_size", "all_fields"):
                totals[k] += s[k]
            fh.write(json.dumps({"id": r["id"], "label": r["label"], "raw": out, "pred": pred,
                                 **s}) + "\n")

    n = len(rows)
    summary = {"split": args.split, "tag": tag, "n": n, "sec_per_example": round(secs / n, 2),
               **{k: round(v / n, 3) for k, v in totals.items()}}
    with open(out_dir / "summary.jsonl", "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
