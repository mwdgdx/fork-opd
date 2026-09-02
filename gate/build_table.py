"""Turn the raw per-position dataset (from data_gen/build_dataset.py) into a flat
trajectory-level feature TABLE + target, for local attribution. Torch-free (numpy only).

The generator stores the RAW per-position arrays so we can derive any feature offline.
This is that derivation layer: each per-position scalar -> summary stats (mean/std/min/max/
last), plus divergence-specific summaries (disagreement rate, cumulative KL, position of
peak KL, mean top-k overlap between teacher and student). Edit `summarize` to explore more
features WITHOUT regenerating.

Target = soft recovery rate of a chosen policy (default whole_rewrite = the clean anchor).
Group key = problem_id (so downstream CV can split by problem, not trajectory).

    python gate/build_table.py --dataset out/dataset --policy whole_rewrite --out out/table.npz
"""

import argparse
import json
import os

import numpy as np

PER_POS = ["t_lp_tok", "s_lp_tok", "t_ent", "s_ent", "t_top1_lp", "agree", "rank",
           "kl_st", "kl_ts"]


def summarize(feat):
    """Per-position raw arrays -> a flat dict of trajectory-level features."""
    out = {}
    R = len(feat["agree"])
    for name in PER_POS:
        a = feat[name].astype(np.float64)
        out[f"{name}_mean"] = a.mean()
        out[f"{name}_std"] = a.std()
        out[f"{name}_min"] = a.min()
        out[f"{name}_max"] = a.max()
        out[f"{name}_last"] = a[-1]
    # divergence-specific
    out["disagree_rate"] = 1.0 - feat["agree"].mean()
    out["kl_st_sum"] = feat["kl_st"].sum()
    out["kl_ts_sum"] = feat["kl_ts"].sum()
    out["kl_st_peakpos"] = float(np.argmax(feat["kl_st"])) / max(R - 1, 1)
    out["resp_len"] = float(R)
    # mean top-k overlap (Jaccard) between teacher and student top-k per position
    tk_t, tk_s = feat["t_topk_ids"], feat["s_topk_ids"]
    jac = np.empty(R)
    for i in range(R):
        st, ss = set(tk_t[i].tolist()), set(tk_s[i].tolist())
        jac[i] = len(st & ss) / max(len(st | ss), 1)
    out["topk_overlap_mean"] = jac.mean()
    out["topk_overlap_min"] = jac.min()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="out/dataset")
    ap.add_argument("--policy", default="whole_rewrite")
    ap.add_argument("--out", default="out/table.npz")
    args = ap.parse_args()

    fork_points = json.load(open(os.path.join(args.dataset, "fork_points.json")))
    # target: recover_rate for the chosen policy
    rate = {}
    for line in open(os.path.join(args.dataset, "labels.jsonl")):
        d = json.loads(line)
        if d["policy"] == args.policy:
            rate[int(d["traj_id"])] = d["recover_rate"]

    rows, cols = [], None
    y, groups, pass_rate, traj_ids = [], [], [], []
    feat_dir = os.path.join(args.dataset, "features")
    for fn in sorted(os.listdir(feat_dir), key=lambda x: int(x.split(".")[0])):
        tid = int(fn.split(".")[0])
        if tid not in rate:
            continue
        feat = dict(np.load(os.path.join(feat_dir, fn)))
        s = summarize(feat)
        if cols is None:
            cols = sorted(s.keys())
        rows.append([s[c] for c in cols])
        y.append(rate[tid])
        fp = fork_points[str(tid)]
        groups.append(fp["problem_id"]); pass_rate.append(fp["pass_rate"]); traj_ids.append(tid)

    X = np.asarray(rows, np.float64)
    print(f"[table] {X.shape[0]} trajectories x {X.shape[1]} features; "
          f"{len(set(groups))} problems; policy={args.policy}")
    np.savez(args.out, X=X, y=np.asarray(y), groups=np.asarray(groups),
             pass_rate=np.asarray(pass_rate), traj_ids=np.asarray(traj_ids),
             cols=np.asarray(cols))
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
