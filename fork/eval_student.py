"""Evaluate a student (base or a trained checkpoint) on the HELD-OUT test problems from
split.json — the exposure-bias experiment's judge. Generates n samples per test problem
and reports accuracy (mean pass-rate + solve-rate). Runs on the mcli box (vLLM).

    python fork/eval_student.py --model Qwen/Qwen3-1.7B --dataset out/dataset \
        --pool data/pool500.jsonl --n 8 --out out/eval_base.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_gen"))
from verify import is_correct  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local checkpoint path")
    ap.add_argument("--dataset", default="out/dataset")
    ap.add_argument("--pool", default="data/pool500.jsonl")
    ap.add_argument("--n", type=int, default=8, help="samples per test problem")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    split = json.load(open(os.path.join(args.dataset, "split.json")))
    test_pids = set(split["test_problem_ids"])

    # one (prompt, gt) per TEST problem, from the pool (dedup by problem_id)
    probs = {}
    for line in open(args.pool):
        r = json.loads(line)
        if r["problem_id"] in test_pids and r["problem_id"] not in probs:
            probs[r["problem_id"]] = (r["rendered_prompt"], r["ground_truth"])
    pids = sorted(probs)
    print(f"[eval] {len(pids)} held-out test problems, n={args.n}, model={args.model}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem_util)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=0.95, max_tokens=args.max_new_tokens)
    outs = llm.generate([probs[p][0] for p in pids], sampling_params=sp)   # rendered_prompt is already chat-templated

    per_prob, n_solved, pass_sum = [], 0, 0.0
    for p, o in zip(pids, outs):
        gt = probs[p][1]
        c = [is_correct(s.text, gt) for s in o.outputs]
        pr = sum(c) / len(c)
        per_prob.append({"problem_id": p, "pass_rate": pr})
        pass_sum += pr; n_solved += int(any(c))
    acc = pass_sum / len(pids)          # mean pass-rate
    solve = n_solved / len(pids)        # fraction solved at least once (pass@n)
    res = {"model": args.model, "n_problems": len(pids), "n": args.n,
           "mean_pass_rate": acc, "solve_rate_at_n": solve, "per_problem": per_prob}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[eval] mean_pass_rate={acc:.3f}  solve_rate@{args.n}={solve:.3f}  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
