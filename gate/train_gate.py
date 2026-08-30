"""Phase 1: warm up the fork gate on a pool of failed trajectories (contextual bandit).

Two phases (sequential, so the teacher lives in only one engine at a time):
  A. HF teacher forward over every failed trajectory -> per-position features (needs
     hidden states, which vLLM can't give). Cached to disk. HF teacher then freed.
  B. vLLM teacher for suffix generation. Bandit loop: gate samples a fork point k ->
     teacher generates the suffix from prompt+prefix[:k] -> verify recovery -> reward
        r = recovered * (k/R) - lambda_c * (suffix_len/max) - lambda_p * wrong_prefix_frac
     -> REINFORCE update on the gate. Saves the gate.

The gate reads only teacher features (student-weight-independent), so this warmup on the
base student's failures transfers to the evolving student in later co-training.

Runs on the mcli box.
    python gate/train_gate.py --failures out/failures.jsonl --teacher Qwen/Qwen3-8B \
        --max-traj 300 --epochs 4 --out out/gate.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_gen"))
from verify import is_correct  # noqa: E402
from features import extract_features, feature_dim  # noqa: E402
from model import ForkGate  # noqa: E402

TOPK = 20


def load_pool(path, tokenizer, max_traj):
    rows = [json.loads(l) for l in open(path)]
    if max_traj:
        rows = rows[:max_traj]
    pool = []
    for r in rows:
        prompt_ids = tokenizer(r["rendered_prompt"], add_special_tokens=False)["input_ids"]
        resp_ids = r["token_ids"]
        if len(resp_ids) < 2:
            continue
        pool.append({"prompt_ids": prompt_ids, "resp_ids": resp_ids,
                     "gt": r["ground_truth"]})
    return pool


def phase_a_features(pool, teacher_name, cache_path, device="cuda"):
    """HF teacher forward -> cache per-trajectory features. Returns hidden_size."""
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    hidden = teacher.config.hidden_size
    feats = []
    for i, ex in enumerate(pool):
        f = extract_features(teacher, ex["prompt_ids"], ex["resp_ids"], topk=TOPK, device=device)
        feats.append(f.cpu())
        if (i + 1) % 25 == 0:
            print(f"  [features] {i+1}/{len(pool)}", flush=True)
    torch.save({"feats": feats, "hidden": hidden}, cache_path)
    del teacher
    torch.cuda.empty_cache()
    print(f"[phase A] cached features for {len(feats)} trajectories -> {cache_path}")
    return hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures", required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--max-traj", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-c", type=float, default=0.05)    # tiny suffix-length (compute) tiebreak
    ap.add_argument("--lambda-fail", type=float, default=0.5)  # penalty for not recovering (enforces recovery)
    ap.add_argument("--seq-model", type=int, default=1)        # 1 = bidirectional attention; 0 = per-position MLP
    ap.add_argument("--gate-hidden", type=int, default=256)
    ap.add_argument("--entropy-beta", type=float, default=0.01)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gpu-mem-util", type=float, default=0.45)
    ap.add_argument("--tp", type=int, default=8, help="vLLM tensor-parallel size (use all GPUs)")
    ap.add_argument("--features-cache", default="out/gate_features.pt")
    ap.add_argument("--out", default="out/gate.pt")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher)

    pool = load_pool(args.failures, tok, args.max_traj)
    print(f"[pool] {len(pool)} failed trajectories")

    # ---- Phase A: features (HF teacher) ----
    if os.path.exists(args.features_cache):
        blob = torch.load(args.features_cache)
        feats_cpu, hidden = blob["feats"], blob["hidden"]
        print(f"[phase A] loaded cached features ({len(feats_cpu)})")
    else:
        hidden = phase_a_features(pool, args.teacher, args.features_cache)
        feats_cpu = torch.load(args.features_cache)["feats"]

    fdim = feature_dim(hidden, TOPK)
    p_teacher_col = hidden + TOPK  # index of p_teacher(student_token) scalar

    # ---- Phase B: vLLM teacher + bandit ----
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.teacher, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem_util)
    sp = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_new_tokens)

    gate = ForkGate(fdim, hidden=args.gate_hidden, seq_model=bool(args.seq_model)).cuda()
    opt = torch.optim.Adam(gate.parameters(), lr=args.lr)
    baseline = 0.0

    for ep in range(args.epochs):
        order = torch.randperm(len(pool)).tolist()
        # 1. sample a fork action per trajectory
        actions, gen_prompts, gen_meta = [], [], []
        for idx in order:
            ex = pool[idx]
            R = len(ex["resp_ids"])
            feats = feats_cpu[idx].cuda()
            a, _, _ = gate.sample(feats)         # a in [0, R] ; R == no-fork
            actions.append((idx, a))
            if a < R:                            # fork at k=a -> generate suffix
                ctx = ex["prompt_ids"] + ex["resp_ids"][:a]
                gen_prompts.append(ctx)
                gen_meta.append((idx, a))
        # 2. batch-generate suffixes
        outs = (llm.generate([{"prompt_token_ids": c} for c in gen_prompts], sampling_params=sp)
                if gen_prompts else [])
        rewards_by_idx = {}
        rec_cnt = 0
        for (idx, k), o in zip(gen_meta, outs):
            ex = pool[idx]; R = len(ex["resp_ids"])
            suffix = o.outputs[0]
            prefix_text = tok.decode(ex["resp_ids"][:k], skip_special_tokens=True)
            recovered = is_correct(prefix_text + suffix.text, ex["gt"])
            rec_cnt += int(recovered)
            # RECOVERY dominates: big fixed reward for recovering; keep-prefix + short-suffix
            # are only small tie-breakers AMONG recoveries. Not recovering -> penalty.
            if recovered:   # reward = fraction of prefix kept; maximize prefix AMONG recoveries
                r = (k / R) - args.lambda_c * (len(suffix.token_ids) / args.max_new_tokens)
            else:           # not recovering is strongly penalized -> gate can't fork-late-and-fail
                r = -args.lambda_fail
            rewards_by_idx[idx] = r
        # 3. REINFORCE update
        opt.zero_grad()
        rlist, klist = [], []
        epoch_loss = 0.0
        B = len(actions)
        CHUNK = 128     # backward per chunk so we never hold all B graphs (OOM at scale)
        for c0 in range(0, B, CHUNK):
            chunk_losses = []
            for idx, a in actions[c0:c0 + CHUNK]:
                ex = pool[idx]; R = len(ex["resp_ids"])
                r = rewards_by_idx.get(idx, 0.0)     # no-fork -> reward 0
                feats = feats_cpu[idx].cuda()
                logp, ent = gate.logprob_entropy(feats, a)
                chunk_losses.append(-logp * (r - baseline) - args.entropy_beta * ent)
                rlist.append(r); klist.append(a / R if a < R else 1.0)
            l = torch.stack(chunk_losses).sum() / B
            l.backward()                            # grad accumulates across chunks
            epoch_loss += float(l.item())
        opt.step()
        mean_r = sum(rlist) / len(rlist)
        baseline = 0.9 * baseline + 0.1 * mean_r
        print(f"[epoch {ep}] mean_reward={mean_r:.3f} recover={rec_cnt}/{len(gen_meta)} "
              f"mean_k/R={sum(klist)/len(klist):.2f} loss={epoch_loss:.3f}", flush=True)

    torch.save({"state_dict": gate.state_dict(), "feat_dim": fdim, "hidden": hidden, "topk": TOPK}, args.out)
    print(f"[done] gate saved -> {args.out}")


if __name__ == "__main__":
    main()
