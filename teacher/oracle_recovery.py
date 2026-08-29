"""Oracle recovery / ceiling diagnostic (gate-independent).

For each failed trajectory, sweep candidate fork points INCLUDING k=0 (whole rewrite),
let the teacher generate the suffix (rejection sampling: n tries, recovered if ANY is
correct), and record whether SOME fork point recovers it and which one (k*).

This measures the CEILING a perfect gate could reach — separate from how well the gate
is actually trained. Keeps the full band: band=0 (student hopeless) is expected to
recover via k=0 (teacher solves from scratch); band 0.2-0.6 via a late fork.

Reports oracle recovery overall and split by student pass-rate, plus the k*/R distribution.
Runs on the mcli box.
    python teacher/oracle_recovery.py --failures out/diag_nt.jsonl --teacher Qwen/Qwen3-8B \
        --fork-fracs 0,0.3,0.6,0.85 --n-samples 2 --out out/oracle.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_gen"))
from verify import is_correct  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures", required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--fork-fracs", default="0,0.3,0.6,0.85")
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--max-traj", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--out", default="out/oracle.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)
    rows = [json.loads(l) for l in open(args.failures)]
    if args.max_traj:
        rows = rows[: args.max_traj]
    fracs = [float(x) for x in args.fork_fracs.split(",")]

    # build all (row_idx, k, sample) generation requests
    prompts, meta = [], []
    for i, r in enumerate(rows):
        prompt_ids = tok(r["rendered_prompt"], add_special_tokens=False)["input_ids"]
        R = len(r["token_ids"])
        ks = sorted({max(0, min(int(f * R), R - 1)) for f in fracs})
        for k in ks:
            ctx = prompt_ids + r["token_ids"][:k]
            for _ in range(args.n_samples):
                prompts.append({"prompt_token_ids": ctx})
                meta.append((i, k, R))

    llm = LLM(model=args.teacher, tensor_parallel_size=1, gpu_memory_utilization=0.6)
    sp = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_new_tokens)
    outs = llm.generate(prompts, sampling_params=sp)

    # per trajectory: which k recovered (any sample)
    recov = defaultdict(set)   # row_idx -> set of k that recovered
    for (i, k, R), o in zip(meta, outs):
        r = rows[i]
        prefix_text = tok.decode(r["token_ids"][:k], skip_special_tokens=True)
        if is_correct(prefix_text + o.outputs[0].text, r["ground_truth"]):
            recov[i].add(k)

    # aggregate
    n = len(rows)
    n_recovered = sum(1 for i in range(n) if recov[i])
    # k* = LATEST recovering fork (keep max prefix); record k*/R
    kstar_frac, by_band = [], defaultdict(lambda: [0, 0])   # band -> [recovered, total]
    whole_rewrite_only = 0
    for i, r in enumerate(rows):
        R = len(r["token_ids"])
        pr = round(r.get("pass_rate", -1), 2)
        by_band[pr][1] += 1
        if recov[i]:
            by_band[pr][0] += 1
            kmax = max(recov[i])
            kstar_frac.append(kmax / R)
            if recov[i] == {0}:
                whole_rewrite_only += 1

    print(f"[oracle recovery] {n_recovered}/{n} = {n_recovered/n:.1%} of failed trajectories "
          f"recoverable by SOME candidate fork")
    print(f"[whole-rewrite-only] {whole_rewrite_only} recovered ONLY at k=0 "
          f"(student prefix unsalvageable -> teacher from scratch)")
    if kstar_frac:
        import statistics as s
        print(f"[k*/R among recovered] median={s.median(kstar_frac):.2f} "
              f"(0=whole-rewrite, high=keep long student prefix)")
    print("[recovery by student pass-rate band]")
    for pr in sorted(by_band):
        rec, tot = by_band[pr]
        print(f"  pass_rate={pr}: {rec}/{tot} = {rec/tot:.0%} recoverable")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"n": n, "recovered": n_recovered, "rate": n_recovered / n,
               "whole_rewrite_only": whole_rewrite_only,
               "kstar_frac": kstar_frac,
               "by_band": {str(k): v for k, v in by_band.items()}}, open(args.out, "w"))
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
