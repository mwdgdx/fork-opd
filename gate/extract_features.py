"""Standalone feature extractor -> cache. Two variants (test the 'missing signal' idea):
  default     : teacher teacher-forces the student trajectory (teacher does NOT know the answer)
  --gold-aware: prepend 'The correct final answer is {gt}.' so the teacher's per-position
                assessment is ANSWER-AWARE (privileged, like SDPO/OPSD) -> features reflect
                'is the student on track TO THE KNOWN ANSWER'.

Cache aligns with train_gate.load_pool order (skip resp<2). Runs on the mcli box (HF, 1 GPU).
    python gate/extract_features.py --failures out/pool500.jsonl --teacher Qwen/Qwen3-8B \
        --gold-aware --out out/feats_gold.pt
"""

from __future__ import annotations

import argparse
import json
import os

import torch

TOPK = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures", required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--gold-aware", action="store_true")
    ap.add_argument("--max-traj", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from features import extract_features

    tok = AutoTokenizer.from_pretrained(args.teacher)
    raw = [json.loads(l) for l in open(args.failures)]
    if args.max_traj:
        raw = raw[: args.max_traj]
    pool = [r for r in raw if len(r["token_ids"]) >= 2]

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    hidden = teacher.config.hidden_size

    feats = []
    for i, r in enumerate(pool):
        prompt_ids = tok(r["rendered_prompt"], add_special_tokens=False)["input_ids"]
        if args.gold_aware:
            pre = tok(f"The correct final answer is {r['ground_truth']}.\n\n",
                      add_special_tokens=False)["input_ids"]
            prompt_ids = pre + prompt_ids
        f = extract_features(teacher, prompt_ids, r["token_ids"], topk=TOPK, device="cuda")
        feats.append(f.cpu())
        if (i + 1) % 100 == 0:
            print(f"  [features] {i+1}/{len(pool)}", flush=True)

    torch.save({"feats": feats, "hidden": hidden, "gold_aware": args.gold_aware}, args.out)
    print(f"[done] {len(feats)} trajectories, gold_aware={args.gold_aware} -> {args.out}")


if __name__ == "__main__":
    main()
