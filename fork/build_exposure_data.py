"""Build MATCHED training data for the exposure-bias experiment from the Round-8 dataset
(no re-generation — reuse the saved recovered-completion TEXT). Torch-free.

The idea's real, still-untested claim: does training the student on FORK trajectories
(student on-policy prefix + teacher suffix) beat training on WHOLE-REWRITE trajectories
(all-teacher)? We compare two arms on the SAME trajectories (matched), so the only thing
that differs is how the correct trajectory was constructed:

  whole-rewrite arm : prefix = ""            , suffix = teacher's frac0 recovered completion
  fork arm          : prefix = student[:k]   , suffix = teacher's completion at the LATEST
                      fork fraction that recovered (max kept on-policy prefix)

Only TRAIN problems (split.json) are used; TEST problems are held out for eval. A trajectory
is included only if BOTH arms have a recovered completion (fair, matched comparison).
Output rows are in fork/train.py's format (rendered_prompt, prefix_token_ids, suffix_token_ids).

    python fork/build_exposure_data.py --dataset out/dataset --pool data/pool500.jsonl \
        --tokenizer Qwen/Qwen3-1.7B --out-dir out/exposure
"""

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="out/dataset")
    ap.add_argument("--pool", default="data/pool500.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out-dir", default="out/exposure")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    split = json.load(open(os.path.join(args.dataset, "split.json")))
    train_pids = set(split["train_problem_ids"])

    pool = {}
    for i, line in enumerate(open(args.pool)):
        r = json.loads(line)
        pool[i] = r                                   # traj_id == line index (as generated)

    comp_dir = os.path.join(args.dataset, "completions")

    def recovered_text(tid, frac):
        """First recovered completion text for (traj, frac), or None."""
        p = os.path.join(comp_dir, f"{tid}__frac{frac}.jsonl")
        if not os.path.exists(p):
            return None
        for line in open(p):
            d = json.loads(line)
            if d.get("recovered"):
                return d["text"]
        return None

    os.makedirs(args.out_dir, exist_ok=True)
    fw = open(os.path.join(args.out_dir, "whole_train.jsonl"), "w")
    ff = open(os.path.join(args.out_dir, "fork_train.jsonl"), "w")
    n_matched = 0

    # discover which (traj, frac) recovered from the completion filenames
    by_traj = {}
    for p in glob.glob(os.path.join(comp_dir, "*__frac*.jsonl")):
        base = os.path.basename(p)[:-6]               # strip .jsonl
        tid_s, frac_s = base.split("__frac")
        by_traj.setdefault(int(tid_s), []).append(float(frac_s))

    for tid, fracs in by_traj.items():
        r = pool.get(tid)
        if r is None or r["problem_id"] not in train_pids:
            continue
        # whole-rewrite = frac 0.0 recovered
        whole_txt = recovered_text(tid, 0.0)
        # fork = latest frac>0 that recovered
        fork_txt, fork_frac = None, None
        for frac in sorted([f for f in fracs if f > 0.0], reverse=True):
            t = recovered_text(tid, frac)
            if t is not None:
                fork_txt, fork_frac = t, frac
                break
        if whole_txt is None or fork_txt is None:
            continue                                   # need BOTH arms (matched)
        n_matched += 1
        resp = r["token_ids"]; R = len(resp)
        k = max(0, min(int(fork_frac * R), R - 1))
        base = {"problem_id": r["problem_id"], "ground_truth": r["ground_truth"],
                "rendered_prompt": r["rendered_prompt"]}
        fw.write(json.dumps({**base, "prefix_token_ids": [],
                             "suffix_token_ids": tok(whole_txt, add_special_tokens=False)["input_ids"],
                             "fork_k": 0}) + "\n")
        ff.write(json.dumps({**base, "prefix_token_ids": resp[:k],
                             "suffix_token_ids": tok(fork_txt, add_special_tokens=False)["input_ids"],
                             "fork_k": k, "fork_frac": fork_frac}) + "\n")
    fw.close(); ff.close()
    print(f"[exposure] {n_matched} matched trajectories (both arms recovered), train problems only")
    print(f"[out] {args.out_dir}/whole_train.jsonl  {args.out_dir}/fork_train.jsonl")


if __name__ == "__main__":
    main()
