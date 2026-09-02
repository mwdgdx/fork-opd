"""Turn the raw per-position dataset (from data_gen/build_dataset.py) into a flat TABLE
for local attribution. Torch-free (numpy only).

Design (see discussion): the gate predicts recovery as a function of the trajectory AND
the fork position k, so we fit Q(x, k/R) with position as a CONTINUOUS input — NOT a 6-way
classifier. So we emit ONE ROW PER (trajectory, frac): features = whole-trajectory summary
+ k/R + a summary of the kept prefix [0, k); target = that fork point's recovery rate.
Group key = problem_id (CV splits by problem, never by trajectory).

Edit `summarize()` / `prefix_features()` to explore features WITHOUT regenerating.

    python gate/build_table.py --dataset out/dataset --out out/table.npz
"""

import argparse
import json
import os

import numpy as np

PER_POS = ["t_lp_tok", "s_lp_tok", "t_ent", "s_ent", "t_top1_lp", "agree", "rank",
           "kl_st", "kl_ts"]


def summarize(feat):
    """Whole-trajectory summary — same for every fork point of this trajectory."""
    out = {}
    R = len(feat["agree"])
    for name in PER_POS:
        a = feat[name].astype(np.float64)
        out[f"{name}_mean"] = a.mean(); out[f"{name}_std"] = a.std()
        out[f"{name}_min"] = a.min();  out[f"{name}_max"] = a.max(); out[f"{name}_last"] = a[-1]
    out["disagree_rate"] = 1.0 - feat["agree"].mean()
    out["kl_st_sum"] = feat["kl_st"].sum(); out["kl_ts_sum"] = feat["kl_ts"].sum()
    out["kl_st_peakpos"] = float(np.argmax(feat["kl_st"])) / max(R - 1, 1)
    out["resp_len"] = float(R)
    tk_t, tk_s = feat["t_topk_ids"], feat["s_topk_ids"]
    jac = np.array([len(set(tk_t[i]) & set(tk_s[i])) / max(len(set(tk_t[i]) | set(tk_s[i])), 1)
                    for i in range(R)])
    out["topk_overlap_mean"] = jac.mean(); out["topk_overlap_min"] = jac.min()
    return out


def prefix_features(feat, k):
    """Features of the KEPT prefix [0, k) — these DO vary with the fork point k."""
    R = len(feat["agree"])
    k = max(1, min(k, R))
    pre = slice(0, k)
    return {
        "k_over_R": k / R,                                   # position (the key input)
        "pre_cum_kl_st": float(feat["kl_st"][pre].sum()),    # how far off-track by k
        "pre_disagree_rate": float(1.0 - feat["agree"][pre].mean()),
        "pre_kl_st_mean": float(feat["kl_st"][pre].mean()),
        "pre_kl_st_at_k": float(feat["kl_st"][k - 1]),       # divergence right at the fork
        "pre_agree_at_k": float(feat["agree"][k - 1]),
        "pre_t_ent_at_k": float(feat["t_ent"][k - 1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="out/dataset")
    ap.add_argument("--out", default="out/table.npz")
    args = ap.parse_args()

    fork_points = {}
    import glob
    for f in glob.glob(os.path.join(args.dataset, "fork_points*.json")):
        fork_points.update(json.load(open(f)))

    # labels: (traj_id, frac) -> (recover_rate, fork_k)
    labels = {}
    for line in open(os.path.join(args.dataset, "labels.jsonl")):
        d = json.loads(line)
        labels.setdefault(int(d["traj_id"]), []).append((d["frac"], d["recover_rate"], d["fork_k"]))

    feat_dir = os.path.join(args.dataset, "features")
    rows, cols = [], None
    y, groups, traj_ids, fracs, pass_rate = [], [], [], [], []
    for tid, entries in labels.items():
        fp = os.path.join(feat_dir, f"{tid}.npz")
        if not os.path.exists(fp):
            continue
        feat = dict(np.load(fp))
        whole = summarize(feat)
        meta = fork_points[str(tid)]
        for frac, rate, k in entries:
            row = dict(whole); row.update(prefix_features(feat, int(k)))
            if cols is None:
                cols = sorted(row.keys())
            rows.append([row[c] for c in cols])
            y.append(rate); groups.append(meta["problem_id"]); traj_ids.append(tid)
            fracs.append(frac); pass_rate.append(meta["pass_rate"])

    X = np.asarray(rows, np.float64)
    print(f"[table] {X.shape[0]} (traj,frac) rows x {X.shape[1]} features; "
          f"{len(set(traj_ids))} traj, {len(set(groups))} problems")
    np.savez(args.out, X=X, y=np.asarray(y), groups=np.asarray(groups),
             traj_ids=np.asarray(traj_ids), fracs=np.asarray(fracs),
             pass_rate=np.asarray(pass_rate), cols=np.asarray(cols))
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
