"""Study WHY the oracle recovers cases heuristics (early fork / whole-rewrite) miss.

Finds LATE-ONLY trajectories: teacher fork@0 (whole-rewrite) FAILS, but fork@late
(keep student prefix) RECOVERS. Saves the full text of each so we can read what is in
the student prefix that lets the teacher finish — but couldn't produce from scratch.

Runs on the mcli box.
    python teacher/study_late_only.py --failures out/pool500.jsonl --teacher Qwen/Qwen3-8B \
        --n-traj 400 --late-frac 0.6 --n-samples 2 --out out/late_only.jsonl
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
    ap.add_argument("--n-traj", type=int, default=400)
    ap.add_argument("--late-frac", type=float, default=0.6)
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--out", default="out/late_only.jsonl")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)
    rows = [json.loads(l) for l in open(args.failures)][: args.n_traj]

    # build generations: for each traj, k=0 (whole-rewrite) and k=late, n samples each
    prompts, meta = [], []
    for i, r in enumerate(rows):
        pids = tok(r["rendered_prompt"], add_special_tokens=False)["input_ids"]
        R = len(r["token_ids"])
        klate = max(1, min(int(args.late_frac * R), R - 1))
        for tag, k in [("k0", 0), ("klate", klate)]:
            for _ in range(args.n_samples):
                prompts.append({"prompt_token_ids": pids + r["token_ids"][:k]})
                meta.append((i, tag, k))

    llm = LLM(model=args.teacher, tensor_parallel_size=args.tp, gpu_memory_utilization=0.6)
    sp = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_new_tokens)
    outs = llm.generate(prompts, sampling_params=sp)

    # collect recovery + keep one recovered text per (traj, tag)
    rec = {}   # (i, tag) -> {"any": bool, "text": recovered-suffix-text-or-first}
    for (i, tag, k), o in zip(meta, outs):
        r = rows[i]
        prefix = tok.decode(r["token_ids"][:k], skip_special_tokens=True)
        ok = is_correct(prefix + o.outputs[0].text, r["ground_truth"])
        cur = rec.setdefault((i, tag), {"any": False, "text": "", "k": k})
        if ok and not cur["any"]:
            cur["any"], cur["text"] = True, o.outputs[0].text
        elif not cur["text"]:
            cur["text"] = o.outputs[0].text

    late_only = []
    for i, r in enumerate(rows):
        k0 = rec.get((i, "k0"), {"any": False})
        kl = rec.get((i, "klate"), {"any": False})
        if kl["any"] and not k0["any"]:   # late recovers, whole-rewrite fails
            R = len(r["token_ids"])
            klate = kl["k"]
            late_only.append({
                "problem_id": r["problem_id"], "ground_truth": r["ground_truth"],
                "pass_rate": r.get("pass_rate"),
                "question": r["question"],
                "student_prefix_kept": tok.decode(r["token_ids"][:klate], skip_special_tokens=True),
                "student_full": r["generation"],
                "teacher_from_scratch_FAILED": rec[(i, "k0")]["text"],
                "teacher_from_prefix_RECOVERED": kl["text"],
                "klate_frac": klate / R,
            })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for e in late_only:
            f.write(json.dumps(e) + "\n")

    n0 = sum(1 for i in range(len(rows)) if rec.get((i, "k0"), {}).get("any"))
    nl = sum(1 for i in range(len(rows)) if rec.get((i, "klate"), {}).get("any"))
    print(f"[over {len(rows)} traj] whole-rewrite(k=0) recovers: {n0} ({n0/len(rows):.1%}) | "
          f"late(k={args.late_frac}) recovers: {nl} ({nl/len(rows):.1%})")
    print(f"[LATE-ONLY] (late recovers, whole-rewrite fails): {len(late_only)} "
          f"({len(late_only)/len(rows):.1%}) -> saved to {args.out}")
    for e in late_only[:2]:
        print("\n" + "=" * 70)
        print("Q:", e["question"][:200])
        print("GT:", e["ground_truth"], "| kept prefix frac:", round(e["klate_frac"], 2))
        print("-- student prefix kept (tail) --\n", e["student_prefix_kept"][-400:])
        print("-- teacher FROM SCRATCH (failed, tail) --\n", e["teacher_from_scratch_FAILED"][-300:])
        print("-- teacher FROM PREFIX (recovered, tail) --\n", e["teacher_from_prefix_RECOVERED"][-300:])


if __name__ == "__main__":
    main()
