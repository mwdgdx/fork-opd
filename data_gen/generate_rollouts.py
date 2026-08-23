"""Milestone 1: generate student rollouts on DAPO-Math, verify, keep band-difficulty
problems, and save the FAILED trajectories that fork training will operate on.

Runs on the mcli GPU box (vLLM). Local arca box has no torch — deploy via git, run there.

Example:
    python data_gen/generate_rollouts.py \
        --model Qwen/Qwen3-1.7B \
        --dataset BytedTsinghua-SIA/DAPO-Math-17k \
        --num-problems 1000 --num-samples 8 \
        --temperature 1.0 --max-tokens 4096 \
        --band-low 0.2 --band-high 0.6 \
        --out out/failures.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from datasets import load_dataset

from verify import is_correct

INSTRUCTION = (
    "Solve the following math problem step by step. "
    "Put your final answer inside \\boxed{}."
)


def extract_question(row: dict, field: str) -> str:
    """Pull the problem text, tolerating verl-style chat `prompt` columns."""
    if field in row and row[field] is not None:
        val = row[field]
        if isinstance(val, list):  # chat messages -> last user turn
            for m in reversed(val):
                if isinstance(m, dict) and m.get("role") == "user":
                    return m["content"]
            return val[-1]["content"] if val else ""
        return str(val)
    for cand in ("prompt", "question", "problem"):
        if cand in row and row[cand] is not None:
            return extract_question(row, cand)
    raise KeyError(f"no question field found in row keys={list(row)}")


def extract_ground_truth(row: dict, field: str) -> str:
    if field in row and row[field] is not None:
        return str(row[field])
    rm = row.get("reward_model")
    if isinstance(rm, dict) and rm.get("ground_truth") is not None:
        return str(rm["ground_truth"])
    for cand in ("answer", "solution", "final_answer"):
        if cand in row and row[cand] is not None:
            return str(row[cand])
    raise KeyError(f"no ground-truth field found in row keys={list(row)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dataset", default="BytedTsinghua-SIA/DAPO-Math-17k")
    ap.add_argument("--split", default="train")
    ap.add_argument("--question-field", default="prompt")
    ap.add_argument("--answer-field", default="reward_model")
    ap.add_argument("--num-problems", type=int, default=1000)
    ap.add_argument("--num-samples", type=int, default=8, help="rollouts per problem (G)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--band-low", type=float, default=0.2)
    ap.add_argument("--band-high", type=float, default=0.6)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--out", default="out/failures.jsonl")
    args = ap.parse_args()

    # heavy imports here so `--help` works without torch installed
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    ds = load_dataset(args.dataset, split=args.split)
    if args.num_problems and args.num_problems < len(ds):
        ds = ds.select(range(args.num_problems))

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts, questions, gts = [], [], []
    for row in ds:
        q = extract_question(row, args.question_field)
        gt = extract_ground_truth(row, args.answer_field)
        chat = [
            {"role": "user", "content": f"{INSTRUCTION}\n\n{q}"},
        ]
        prompts.append(tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
        questions.append(q)
        gts.append(gt)

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    sp = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    results = llm.generate(prompts, sp)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    band_hist: Counter = Counter()
    n_band_problems = 0
    n_failures = 0
    with open(args.out, "w") as f:
        for pid, (res, q, gt, rendered) in enumerate(zip(results, questions, gts, prompts)):
            outs = res.outputs
            correct = [is_correct(o.text, gt) for o in outs]
            pass_rate = sum(correct) / len(correct)
            band_hist[round(pass_rate, 2)] += 1
            in_band = args.band_low <= pass_rate <= args.band_high
            if not in_band:
                continue
            n_band_problems += 1
            for o, ok in zip(outs, correct):
                if ok:
                    continue  # only failed trajectories are fork candidates
                n_failures += 1
                f.write(
                    json.dumps(
                        {
                            "problem_id": pid,
                            "question": q,
                            "ground_truth": gt,
                            "rendered_prompt": rendered,
                            "generation": o.text,
                            "token_ids": list(o.token_ids),
                            "num_gen_tokens": len(o.token_ids),
                            "pass_rate": pass_rate,
                        }
                    )
                    + "\n"
                )

    print(f"[done] band problems: {n_band_problems} | failed trajectories saved: {n_failures}")
    print(f"[pass-rate histogram] {dict(sorted(band_hist.items()))}")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
