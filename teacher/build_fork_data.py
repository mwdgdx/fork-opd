"""Path B, step 2: frozen external teacher (Qwen3-8B) constructs fork trajectories
from the student's failed rollouts, and measures the wall-clock / token cost that
our efficiency claim rests on.

For a failed trajectory of student tokens [0, T) and a fork point k:
    prefix  = student tokens [0, k)     (kept as-is; on-policy)
    suffix  = teacher generates from (prompt + prefix)   (off-policy)
The union prefix+suffix is the constructed training trajectory.

Fork policies (which k):
    whole      k = 0            -> teacher rewrites the whole thing (baseline)
    nofork     k = T            -> no teacher generation (pure on-policy)
    fixed_frac k = round(f * T) -> ablation
    fixed_k    k = K            -> ablation
(The learned gate will supply k later; this script covers the fixed policies.)

Qwen3-1.7B and Qwen3-8B share a tokenizer, so student token ids feed the teacher
directly. We measure teacher autoregressive-generation tokens and wall-clock — the
framework-independent efficiency metric (Relay-style), NOT FLOPs.

Runs on the mcli GPU box.

Example:
    python teacher/build_fork_data.py --failures out/failures.jsonl \
        --teacher Qwen/Qwen3-8B --fork-policy whole \
        --max-new-tokens 4096 --out out/fork_whole.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# allow importing the sibling verifier
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_gen"))
from verify import is_correct  # noqa: E402


def fork_index(policy: str, T: int, frac: float, k: int) -> int:
    if policy == "whole":
        return 0
    if policy == "nofork":
        return T
    if policy == "fixed_frac":
        return max(0, min(round(frac * T), T))
    if policy == "fixed_k":
        return max(0, min(k, T))
    raise ValueError(f"unknown fork policy: {policy}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures", required=True, help="jsonl from generate_rollouts.py")
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--fork-policy", default="whole",
                    choices=["whole", "nofork", "fixed_frac", "fixed_k"])
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    rows = [json.loads(l) for l in open(args.failures)]
    tok = AutoTokenizer.from_pretrained(args.teacher)

    # Build one teacher context (prompt + student prefix) per failed trajectory.
    contexts, metas = [], []
    for r in rows:
        student_ids = r["token_ids"]
        T = len(student_ids)
        k = fork_index(args.fork_policy, T, args.frac, args.k)
        prompt_ids = tok(r["rendered_prompt"], add_special_tokens=False)["input_ids"]
        context_ids = prompt_ids + student_ids[:k]  # shared vocab -> ids interchange
        contexts.append(context_ids)
        metas.append({"row": r, "T": T, "k": k, "prompt_len": len(prompt_ids)})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.fork_policy == "nofork":
        # pure on-policy: no teacher generation at all (k == T for every row)
        with open(args.out, "w") as f:
            for m in metas:
                r = m["row"]
                f.write(json.dumps({
                    "problem_id": r["problem_id"], "ground_truth": r["ground_truth"],
                    "rendered_prompt": r["rendered_prompt"],
                    "prefix_token_ids": r["token_ids"], "suffix_token_ids": [],
                    "fork_k": m["k"], "recovered": False,
                    "teacher_gen_tokens": 0,
                }) + "\n")
        print(f"[nofork] wrote {len(metas)} rows, teacher generated 0 tokens")
        return

    llm = LLM(model=args.teacher, tensor_parallel_size=args.tensor_parallel_size)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_new_tokens)

    t0 = time.perf_counter()
    results = llm.generate([{"prompt_token_ids": c} for c in contexts], sampling_params=sp)
    gen_wall_s = time.perf_counter() - t0

    total_gen_tokens = 0
    n_recovered = 0
    with open(args.out, "w") as f:
        for res, m in zip(results, metas):
            r = m["row"]
            out = res.outputs[0]
            suffix_ids = list(out.token_ids)
            total_gen_tokens += len(suffix_ids)
            # recovered = does prefix + teacher suffix reach the correct answer?
            prefix_text = tok.decode(r["token_ids"][: m["k"]], skip_special_tokens=True)
            full_text = prefix_text + out.text
            recovered = is_correct(full_text, r["ground_truth"])
            n_recovered += int(recovered)
            f.write(json.dumps({
                "problem_id": r["problem_id"], "ground_truth": r["ground_truth"],
                "rendered_prompt": r["rendered_prompt"],
                "prefix_token_ids": r["token_ids"][: m["k"]],
                "suffix_token_ids": suffix_ids,
                "fork_k": m["k"], "T": m["T"], "recovered": recovered,
                "teacher_gen_tokens": len(suffix_ids),
            }) + "\n")

    n = len(metas)
    print(f"[{args.fork_policy}] rows={n} recovered={n_recovered} ({n_recovered/n:.1%})")
    print(f"[EFFICIENCY] teacher gen tokens: total={total_gen_tokens} "
          f"mean={total_gen_tokens/n:.1f}/traj | gen wall-clock={gen_wall_s:.1f}s "
          f"({total_gen_tokens/gen_wall_s:.0f} tok/s)")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
