"""Dense per-position recoverability labels for SUPERVISED gate training.

For each failed trajectory, sweep a grid of fork points and record whether the teacher
recovers (any of n samples) at each. Output aligns with the load_pool order so it pairs
with the cached features. This is the supervised target: "recoverable-from-position-k".

Runs on the mcli box.
    python teacher/label_recoverability.py --failures out/pool500.jsonl --teacher Qwen/Qwen3-8B \
        --fracs 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 --n-samples 2 --out out/reco_labels.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_gen"))
from verify import is_correct  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures", required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--fracs", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--max-traj", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--out", default="out/reco_labels.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)
    raw = [json.loads(l) for l in open(args.failures)]
    if args.max_traj:
        raw = raw[: args.max_traj]
    # same filter as train_gate.load_pool (align with features)
    pool = [r for r in raw if len(r["token_ids"]) >= 2]
    fracs = [float(x) for x in args.fracs.split(",")]

    prompts, meta = [], []
    for i, r in enumerate(pool):
        pids = tok(r["rendered_prompt"], add_special_tokens=False)["input_ids"]
        R = len(r["token_ids"])
        for f in fracs:
            k = max(0, min(int(f * R), R - 1))
            for _ in range(args.n_samples):
                prompts.append({"prompt_token_ids": pids + r["token_ids"][:k]})
                meta.append((i, f, k))

    llm = LLM(model=args.teacher, tensor_parallel_size=args.tp, gpu_memory_utilization=0.6)
    sp = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_new_tokens)
    outs = llm.generate(prompts, sampling_params=sp)

    # per trajectory: frac -> recovered (any sample)
    labels = [{} for _ in pool]
    for (i, f, k), o in zip(meta, outs):
        r = pool[i]
        prefix = tok.decode(r["token_ids"][:k], skip_special_tokens=True)
        ok = is_correct(prefix + o.outputs[0].text, r["gt"] if "gt" in r else r["ground_truth"])
        key = str(f)
        labels[i][key] = labels[i].get(key, False) or ok

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"fracs": fracs, "labels": labels,
               "pass_rate": [r.get("pass_rate") for r in pool]}, open(args.out, "w"))

    # quick summary: recovery rate per frac + any-recoverable
    n = len(pool)
    print(f"[labeled {n} trajectories, {len(fracs)} fork points, n={args.n_samples}]")
    for f in fracs:
        rr = sum(1 for L in labels if L.get(str(f))) / n
        print(f"  frac={f}: recover {rr:.1%}")
    anyrec = sum(1 for L in labels if any(L.values())) / n
    print(f"  ANY fork (oracle): {anyrec:.1%}")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
