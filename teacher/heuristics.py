"""Heuristic fork-rule comparison: how well do FIXED rules recover, vs the oracle ceiling
and the learned gate? Answers "is there room for a learned gate to beat Relay-style rules?"

Rules (fork point k per failed trajectory):
  frac0.25/0.5/0.75 : fixed fraction of the trajectory
  first_disagree    : first position where teacher argmax != student token (agree==0)
                      -- the "start off-policy at first disagreement" baseline
  error_onset       : first position where p_teacher(student token) < thresh
                      -- teacher confidently disagrees (a localized-error heuristic)
For each rule we fork there, let the teacher generate the suffix, and check recovery.
Reuses the cached per-position features (no teacher re-forward) to compute the rule ks.

Runs on the mcli box.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_gen"))
from verify import is_correct  # noqa: E402

TOPK = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures", required=True)
    ap.add_argument("--features-cache", required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--max-traj", type=int, default=0)
    ap.add_argument("--p-teacher-thresh", type=float, default=0.1)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--out", default="out/heuristics.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)

    # Rebuild the pool with the SAME filter train_gate.load_pool used (skip resp<2), so
    # feats_list[i] aligns with pool[i].
    raw = [json.loads(l) for l in open(args.failures)]
    if args.max_traj:
        raw = raw[: args.max_traj]
    pool = []
    for r in raw:
        prompt_ids = tok(r["rendered_prompt"], add_special_tokens=False)["input_ids"]
        if len(r["token_ids"]) < 2:
            continue
        pool.append({"prompt_ids": prompt_ids, "resp_ids": r["token_ids"], "gt": r["ground_truth"]})

    blob = torch.load(args.features_cache)
    feats_list, hidden = blob["feats"], blob["hidden"]
    assert len(feats_list) == len(pool), f"feature/pool misalignment {len(feats_list)} vs {len(pool)}"
    p_col = hidden + TOPK + 0       # p_teacher(student token)
    agree_col = hidden + TOPK + 3   # teacher argmax == student token

    def rule_ks(feats, R):
        ks = {"frac0.25": int(0.25 * R), "frac0.5": int(0.5 * R), "frac0.75": int(0.75 * R)}
        dis = (feats[:, agree_col] == 0).nonzero()
        ks["first_disagree"] = int(dis[0].item()) if len(dis) else 0
        eo = (feats[:, p_col] < args.p_teacher_thresh).nonzero()
        ks["error_onset"] = int(eo[0].item()) if len(eo) else 0
        return {n: max(0, min(k, R - 1)) for n, k in ks.items()}

    # build all (rule, traj) generations
    prompts, meta = [], []
    for i, ex in enumerate(pool):
        R = len(ex["resp_ids"])
        for name, k in rule_ks(feats_list[i], R).items():
            prompts.append({"prompt_token_ids": ex["prompt_ids"] + ex["resp_ids"][:k]})
            meta.append((name, i, k, R))

    llm = LLM(model=args.teacher, tensor_parallel_size=args.tp, gpu_memory_utilization=0.6)
    sp = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_new_tokens)
    outs = llm.generate(prompts, sampling_params=sp)

    rec = defaultdict(lambda: [0, 0.0, 0])   # rule -> [recovered, sum_kR, total]
    for (name, i, k, R), o in zip(meta, outs):
        ex = pool[i]
        prefix_text = tok.decode(ex["resp_ids"][:k], skip_special_tokens=True)
        ok = is_correct(prefix_text + o.outputs[0].text, ex["gt"])
        rec[name][0] += int(ok)
        rec[name][1] += k / R
        rec[name][2] += 1

    print(f"[heuristic recovery] over {len(pool)} failed trajectories (1 teacher sample each)")
    for name in ["frac0.25", "frac0.5", "frac0.75", "first_disagree", "error_onset"]:
        r, ksum, tot = rec[name]
        print(f"  {name:16s}: recover {r}/{tot} = {r/tot:.1%}   mean_k/R={ksum/tot:.2f}")
    json.dump({n: {"recover": v[0], "mean_kR": v[1] / v[2], "total": v[2]} for n, v in rec.items()},
              open(args.out, "w"))
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
