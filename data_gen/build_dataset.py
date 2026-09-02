"""One-shot dataset builder for the fork-gate study. Run ONCE on the mcli box; then all
modelling / attribution is local (torch-free) so we never regenerate.

It produces, in a single session, everything the two downstream experiments need:

  A. Per-trajectory RAW features (store the rawest affordable thing; derive features
     offline). For every failed student trajectory, one teacher forward + one student
     forward, teacher-forced on the student's own tokens. Per response position we store
     BOTH models' top-k logprobs + a handful of full-distribution scalars that can't be
     reconstructed from top-k (entropy, exact KL both directions). Teacher hidden states
     (the "semantic why/where" signal) are stored for a stratified SUBSET (per-position
     hidden for all trajectories is ~80GB; a subset is enough to probe).

  B. Recovery LABELS + completion TEXT. For each policy in {whole_rewrite, relay(=first
     teacher-student disagreement), fixed-fork}, generate n teacher completions from the
     fork point, verify, and SAVE THE TEXT. This yields BOTH the soft recovery-rate label
     (for the gate) AND the correct constructed trajectories (for the premise experiment).
     (Last time only labels were kept, not text — do not repeat that.)

  C. train/test split BY problem_id (held-out generalisation for the premise experiment
     and honest grouped CV for the gate).

Two phases so the teacher lives in only one engine at a time (HF for hidden states, which
vLLM can't give; vLLM for fast generation).

    python data_gen/build_dataset.py --pool out/pool500.jsonl --out out/dataset \
        --teacher Qwen/Qwen3-8B --student Qwen/Qwen3-1.7B \
        --topk 64 --n-label 32 --hidden-subset 800 --test-frac 0.2 --tp 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from verify import is_correct  # noqa: E402

POLICIES = ["whole_rewrite", "relay", "fixed0.5"]


# --------------------------------------------------------------------------- IO
def load_pool(path, tokenizer, max_traj=0):
    rows = [json.loads(l) for l in open(path)]
    if max_traj:
        rows = rows[:max_traj]
    pool = []
    for i, r in enumerate(rows):
        prompt_ids = tokenizer(r["rendered_prompt"], add_special_tokens=False)["input_ids"]
        resp_ids = r["token_ids"]
        if len(resp_ids) < 2:
            continue
        pool.append({
            "traj_id": i,
            "problem_id": r["problem_id"],
            "prompt_ids": prompt_ids,
            "resp_ids": resp_ids,
            "gt": r["ground_truth"],
            "pass_rate": r["pass_rate"],
        })
    return pool


def stratified_subset(pool, k, seed=0):
    """Pick k trajectory ids, stratified by pass_rate so difficulty is covered."""
    if not k or k >= len(pool):
        return {ex["traj_id"] for ex in pool}
    by_pr = {}
    for ex in pool:
        by_pr.setdefault(round(ex["pass_rate"], 3), []).append(ex["traj_id"])
    rng = random.Random(seed)
    chosen = set()
    frac = k / len(pool)
    for pr, ids in by_pr.items():
        take = max(1, round(len(ids) * frac))
        chosen.update(rng.sample(ids, min(take, len(ids))))
    return chosen


# --------------------------------------------------------- Phase A: features (HF)
@torch.no_grad()
def per_position_features(teacher, student, prompt_ids, resp_ids, topk, want_hidden,
                          hidden_layers, device, kl_chunk=256):
    """Both models teacher-forced on prompt+resp. Returns a dict of per-position arrays
    over the R response positions, plus first_disagree and hidden states (if wanted)."""
    P, R = len(prompt_ids), len(resp_ids)
    ids = torch.tensor(prompt_ids + resp_ids, device=device).unsqueeze(0)
    dec = torch.tensor([max(P + r - 1, 0) for r in range(R)], device=device)  # [R]
    resp = torch.tensor(resp_ids, device=device)                              # [R]

    t_out = teacher(ids, output_hidden_states=want_hidden, use_cache=False)
    s_out = student(ids, use_cache=False)
    t_logits = t_out.logits[0][dec]        # [R, V]
    s_logits = s_out.logits[0][dec]        # [R, V]
    V = t_logits.shape[-1]

    feat = {}
    t_topk_ids = np.zeros((R, topk), np.int32); t_topk_lp = np.zeros((R, topk), np.float16)
    s_topk_ids = np.zeros((R, topk), np.int32); s_topk_lp = np.zeros((R, topk), np.float16)
    scal = {n: np.zeros(R, np.float32) for n in
            ["t_lp_tok", "s_lp_tok", "t_ent", "s_ent", "t_top1_lp", "agree", "rank",
             "kl_st", "kl_ts"]}

    for c0 in range(0, R, kl_chunk):                        # chunk over positions (memory)
        sl = slice(c0, min(c0 + kl_chunk, R))
        lp_t = torch.log_softmax(t_logits[sl].float(), dim=-1)   # [c, V]
        lp_s = torch.log_softmax(s_logits[sl].float(), dim=-1)
        r = resp[sl]
        t_lp_tok = lp_t.gather(1, r[:, None]).squeeze(1)
        s_lp_tok = lp_s.gather(1, r[:, None]).squeeze(1)
        p_t, p_s = lp_t.exp(), lp_s.exp()
        top1_lp, top1_idx = lp_t.max(dim=-1)
        tk_t = lp_t.topk(topk, dim=-1); tk_s = lp_s.topk(topk, dim=-1)
        t_topk_ids[sl] = tk_t.indices.cpu().numpy();  t_topk_lp[sl] = tk_t.values.half().cpu().numpy()
        s_topk_ids[sl] = tk_s.indices.cpu().numpy();  s_topk_lp[sl] = tk_s.values.half().cpu().numpy()
        scal["t_lp_tok"][sl] = t_lp_tok.cpu().numpy()
        scal["s_lp_tok"][sl] = s_lp_tok.cpu().numpy()
        scal["t_ent"][sl] = (-(p_t * lp_t).sum(-1)).cpu().numpy()
        scal["s_ent"][sl] = (-(p_s * lp_s).sum(-1)).cpu().numpy()
        scal["t_top1_lp"][sl] = top1_lp.cpu().numpy()
        scal["agree"][sl] = (top1_idx == r).float().cpu().numpy()
        scal["rank"][sl] = ((lp_t > t_lp_tok[:, None]).sum(-1).float() / V).cpu().numpy()
        scal["kl_st"][sl] = (p_s * (lp_s - lp_t)).sum(-1).cpu().numpy()   # KL(student||teacher)
        scal["kl_ts"][sl] = (p_t * (lp_t - lp_s)).sum(-1).cpu().numpy()   # KL(teacher||student)

    feat.update(scal)
    feat["t_topk_ids"] = t_topk_ids; feat["t_topk_lp"] = t_topk_lp
    feat["s_topk_ids"] = s_topk_ids; feat["s_topk_lp"] = s_topk_lp
    feat["resp_ids"] = np.asarray(resp_ids, np.int32)
    # first position where teacher's argmax != student's token = relay fork point
    dis = np.where(feat["agree"] < 0.5)[0]
    first_disagree = int(dis[0]) if len(dis) else R
    hidden = None
    if want_hidden:
        hs = t_out.hidden_states                       # tuple[L+1] of [1,N,H]
        layers = [hs[li][0][dec].half().cpu().numpy() for li in hidden_layers]
        hidden = np.stack(layers, axis=1)              # [R, n_layers, H]
    return feat, first_disagree, hidden


def phase_a(pool, teacher, student, args, dirs, device):
    hidden_ids = stratified_subset(pool, args.hidden_subset, seed=0)
    fork_points = {}
    for n, ex in enumerate(pool):
        want_hidden = ex["traj_id"] in hidden_ids
        feat, first_disagree, hidden = per_position_features(
            teacher, student, ex["prompt_ids"], ex["resp_ids"],
            args.topk, want_hidden, args.hidden_layers, device)
        np.savez_compressed(os.path.join(dirs["features"], f"{ex['traj_id']}.npz"), **feat)
        if hidden is not None:
            np.savez_compressed(os.path.join(dirs["hidden"], f"{ex['traj_id']}.npz"), hidden=hidden)
        fork_points[ex["traj_id"]] = {"R": len(ex["resp_ids"]), "first_disagree": first_disagree,
                                      "problem_id": ex["problem_id"], "pass_rate": ex["pass_rate"]}
        if (n + 1) % 50 == 0:
            print(f"  [phase A] {n+1}/{len(pool)}", flush=True)
    json.dump(fork_points, open(os.path.join(args.out, "fork_points.json"), "w"))
    print(f"[phase A] features for {len(pool)} traj; hidden for {len(hidden_ids)}", flush=True)
    return fork_points


# ------------------------------------------------ Phase B: labels + text (vLLM)
def fork_k(policy, R, first_disagree):
    if policy == "whole_rewrite":
        return 0
    if policy == "relay":
        return min(first_disagree, R - 1)
    if policy == "fixed0.5":
        return R // 2
    raise ValueError(policy)


def phase_b(pool, fork_points, args, dirs, tok):
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.teacher, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util)
    sp = SamplingParams(n=args.n_label, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_new_tokens)

    # build every (traj, policy) generation request
    reqs = []          # (traj_id, policy, k, prompt_token_ids, gt)
    by_traj = {ex["traj_id"]: ex for ex in pool}
    for ex in pool:
        fp = fork_points[str(ex["traj_id"])] if str(ex["traj_id"]) in fork_points \
            else fork_points[ex["traj_id"]]
        R = len(ex["resp_ids"])
        for pol in args.policies:
            k = fork_k(pol, R, fp["first_disagree"])
            ctx = ex["prompt_ids"] + ex["resp_ids"][:k]
            reqs.append((ex["traj_id"], pol, k, ctx, ex["gt"]))

    labels_f = open(os.path.join(args.out, "labels.jsonl"), "w")
    for c0 in range(0, len(reqs), args.gen_chunk):
        chunk = reqs[c0:c0 + args.gen_chunk]
        outs = llm.generate([{"prompt_token_ids": c[3]} for c in chunk], sampling_params=sp)
        for (traj_id, pol, k, ctx, gt), o in zip(chunk, outs):
            R = len(by_traj[traj_id]["resp_ids"])
            prefix = tok.decode(by_traj[traj_id]["resp_ids"][:k], skip_special_tokens=True)
            comps = []
            n_rec = 0
            for s in o.outputs:
                ok = is_correct(prefix + s.text, gt)
                n_rec += int(ok)
                comps.append({"text": s.text, "recovered": ok, "n_tok": len(s.token_ids)})
            # completion text (premise training data) — one file per (traj, policy)
            with open(os.path.join(dirs["completions"], f"{traj_id}__{pol}.jsonl"), "w") as cf:
                for c in comps:
                    cf.write(json.dumps(c) + "\n")
            labels_f.write(json.dumps({
                "traj_id": traj_id, "policy": pol, "fork_k": k, "R": R,
                "n": args.n_label, "recover_rate": n_rec / args.n_label,
            }) + "\n")
        print(f"  [phase B] {min(c0 + args.gen_chunk, len(reqs))}/{len(reqs)} reqs", flush=True)
    labels_f.close()
    print("[phase B] labels + completions written", flush=True)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", default="out/dataset")
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--student", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--max-traj", type=int, default=0, help="0 = all")
    # knob 1: label sampling
    ap.add_argument("--n-label", type=int, default=32, help="teacher samples per policy (soft label)")
    # knob 2: fork policies
    ap.add_argument("--policies", default=",".join(POLICIES),
                    help="comma list from {whole_rewrite,relay,fixed0.5}")
    # knob 3: hidden-state scope
    ap.add_argument("--hidden-subset", type=int, default=800,
                    help="store per-position teacher hidden for this many traj (0 = all)")
    ap.add_argument("--hidden-layers", default="mid,last",
                    help="which layers: comma list of ints or 'mid'/'last'")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--test-frac", type=float, default=0.2, help="held-out problems (premise)")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gpu-mem-util", type=float, default=0.6)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--gen-chunk", type=int, default=256)
    ap.add_argument("--skip-phase-a", action="store_true")
    ap.add_argument("--skip-phase-b", action="store_true")
    args = ap.parse_args()
    args.policies = [p for p in args.policies.split(",") if p]

    device = "cuda"
    dirs = {k: os.path.join(args.out, k) for k in ["features", "hidden", "completions"]}
    for d in [args.out, *dirs.values()]:
        os.makedirs(d, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher)
    pool = load_pool(args.pool, tok, args.max_traj)
    print(f"[pool] {len(pool)} trajectories, {len({e['problem_id'] for e in pool})} problems")

    # ---- C. train/test split by problem_id ----
    split_path = os.path.join(args.out, "split.json")
    if not os.path.exists(split_path):
        pids = sorted({ex["problem_id"] for ex in pool})
        rng = random.Random(0); rng.shuffle(pids)
        n_test = int(args.test_frac * len(pids))
        split = {"test_problem_ids": pids[:n_test], "train_problem_ids": pids[n_test:]}
        json.dump(split, open(split_path, "w"))
        print(f"[split] {len(split['train_problem_ids'])} train / {len(split['test_problem_ids'])} test problems")

    # ---- A. features (HF teacher + student) ----
    fork_points = None
    if not args.skip_phase_a:
        from transformers import AutoModelForCausalLM
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, torch_dtype=torch.bfloat16, device_map=device).eval()
        student = AutoModelForCausalLM.from_pretrained(
            args.student, torch_dtype=torch.bfloat16, device_map=device).eval()
        L = teacher.config.num_hidden_layers
        layer_map = {"mid": L // 2, "last": L}
        args.hidden_layers = [layer_map.get(x, None) or int(x) for x in args.hidden_layers.split(",")]
        print(f"[phase A] teacher L={L}, hidden layers stored = {args.hidden_layers}")
        fork_points = phase_a(pool, teacher, student, args, dirs, device)
        del teacher, student
        torch.cuda.empty_cache()

    # ---- B. recovery labels + completion text (vLLM teacher) ----
    if not args.skip_phase_b:
        if fork_points is None:
            fork_points = json.load(open(os.path.join(args.out, "fork_points.json")))
        phase_b(pool, fork_points, args, dirs, tok)

    json.dump(vars(args), open(os.path.join(args.out, "meta.json"), "w"), default=str)
    print(f"[done] dataset -> {args.out}")


if __name__ == "__main__":
    main()
