"""Local, torch-free attribution for the fork-gate. Runs on the feature table from
gate/build_table.py. Enforces the methodology we agreed on:

  Phase 1 (FIT FIRST): with ALL features + a strong model, how high can held-out AUC go?
    - grouped CV (split by problem_id, never by trajectory) so difficulty can't leak
    - anchors: 0.5 (constant) and pass_rate-only AUC (the free ~0.70 baseline)
    - shuffle-label control: must collapse to ~0.5, else we're overfitting/leaking
  Only if Phase 1 clearly beats the anchors do we bother with:

  Phase 2 (ATTRIBUTE): which features/groups carry the signal (never prune on solo AUC).
    - single-feature AUC (screening; promotes, never demotes)
    - leave-one-GROUP-out ablation (catches interaction-only features, evaluated in-context)
    - permutation importance on a held-out split

  --hidden: also probe the teacher hidden states (linear Ridge + small MLP), each with a
    shuffle control, to test whether the "semantic why/where" adds signal over the scalars.

    python gate/fit_and_attribute.py --table out/table.npz [--hidden out/dataset/hidden]
"""

import argparse
import os

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor

GROUPS = {  # column-prefix -> feature group
    "position": ("k_over_R", "pre_"),   # fork position + kept-prefix summary (varies with k)
    "student": ("s_lp_tok", "s_ent"),
    "teacher": ("t_lp_tok", "t_ent", "t_top1_lp", "agree", "rank", "disagree_rate"),
    "divergence": ("kl_st", "kl_ts", "topk_overlap"),
    "shape": ("resp_len", "kl_st_peakpos"),
}


def col_group(col):
    for g, prefixes in GROUPS.items():
        if any(col.startswith(p) for p in prefixes):
            return g
    return "other"


def cv_oof(X, y_soft, groups, n_splits, model_fn):
    """Out-of-fold predictions from grouped CV (train on soft rate, predict rate)."""
    oof = np.zeros(len(y_soft))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y_soft, groups):
        m = model_fn().fit(X[tr], y_soft[tr])
        oof[te] = m.predict(X[te])
    return oof


def gbdt():
    return GradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                     n_estimators=300, subsample=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="out/table.npz")
    ap.add_argument("--hidden", default="", help="dataset/hidden dir to also probe hidden states")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--anchor", type=float, default=0.70, help="pass_rate baseline AUC to beat")
    args = ap.parse_args()

    d = np.load(args.table, allow_pickle=True)
    X, y_soft, groups, pass_rate = d["X"], d["y"], d["groups"], d["pass_rate"]
    cols = list(d["cols"]); traj_ids = d["traj_ids"]
    y_bin = (y_soft > 0).astype(int)                     # "recoverable at all" (matches anchor)
    print(f"[data] {X.shape[0]} traj x {X.shape[1]} feats, {len(set(groups))} problems, "
          f"recoverable={y_bin.mean():.1%}\n")

    # ---------------- anchors ----------------
    auc_passrate = roc_auc_score(y_bin, pass_rate)
    print("=== anchors ===")
    print(f"  constant             AUC = 0.500")
    print(f"  pass_rate-only       AUC = {auc_passrate:.3f}   (free baseline / not deployable)")

    # ---------------- Phase 1: fit first ----------------
    oof = cv_oof(X, y_soft, groups, args.n_splits, gbdt)
    auc_full = roc_auc_score(y_bin, oof)
    # shuffle-label control
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y_soft))
    oof_sh = cv_oof(X, y_soft[perm], groups, args.n_splits, gbdt)
    auc_shuffle = roc_auc_score(y_bin[perm], oof_sh)
    print("\n=== Phase 1: FIT (all scalar features, GBDT, grouped CV) ===")
    print(f"  full model           AUC = {auc_full:.3f}")
    print(f"  shuffle-label control AUC = {auc_shuffle:.3f}   (must be ~0.5)")
    verdict = "BEATS" if auc_full > args.anchor + 0.02 else "does NOT beat"
    print(f"  -> {verdict} the pass_rate anchor ({args.anchor:.2f}). "
          f"{'Proceed to attribution.' if verdict=='BEATS' else 'Signal not (yet) above difficulty.'}")

    # ---------------- Phase 1b: single-feature screening ----------------
    print("\n=== single-feature AUC (screening; promotes only, never prunes) ===")
    solo = []
    for j, c in enumerate(cols):
        a = roc_auc_score(y_bin, X[:, j])
        solo.append((max(a, 1 - a), c, a))       # |0.5-centered| so inverse signals count
    for adj, c, a in sorted(solo, reverse=True)[:12]:
        print(f"  {c:22s} AUC={a:.3f}")

    # ---------------- Phase 2: group ablation ----------------
    print("\n=== leave-one-group-out ablation (drop from full = unique contribution) ===")
    for g in list(GROUPS) + ["other"]:
        keep = [j for j, c in enumerate(cols) if col_group(c) != g]
        if len(keep) == len(cols) or not keep:
            continue
        oof_g = cv_oof(X[:, keep], y_soft, groups, args.n_splits, gbdt)
        auc_g = roc_auc_score(y_bin, oof_g)
        print(f"  without {g:12s} AUC = {auc_g:.3f}   (drop {auc_full - auc_g:+.3f})")

    # ---------------- Phase 2b: permutation importance ----------------
    print("\n=== permutation importance (held-out split) ===")
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
                  .split(X, y_soft, groups))
    m = gbdt().fit(X[tr], y_soft[tr])
    base = roc_auc_score(y_bin[te], m.predict(X[te]))
    imp = []
    for j in range(X.shape[1]):
        Xp = X[te].copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        imp.append((base - roc_auc_score(y_bin[te], m.predict(Xp)), cols[j]))
    for drop, c in sorted(imp, reverse=True)[:12]:
        print(f"  {c:22s} AUC drop when shuffled = {drop:+.3f}")

    # ---------------- optional: hidden-state probe ----------------
    if args.hidden:
        print("\n=== hidden-state probe (does 'semantic why/where' add over scalars?) ===")
        # table has many rows per traj; aggregate to a per-traj target for the probe
        from collections import defaultdict
        tr_rows = defaultdict(list)
        for i, t in enumerate(traj_ids):
            tr_rows[int(t)].append(i)
        tr_rate = {t: float(np.mean([y_soft[i] for i in idxs])) for t, idxs in tr_rows.items()}
        tr_recov = {t: int(any(y_soft[i] > 0 for i in idxs)) for t, idxs in tr_rows.items()}
        tr_group = {t: groups[idxs[0]] for t, idxs in tr_rows.items()}
        H, ys, ybh, gh = [], [], [], []
        for fn in sorted(os.listdir(args.hidden)):
            tid = int(fn.split(".")[0])
            if tid not in tr_rows:
                continue
            hs = np.load(os.path.join(args.hidden, fn))["hidden"]   # [R, n_layers, H]
            H.append(hs.astype(np.float32).mean(0).ravel())         # mean-pool positions
            ys.append(tr_rate[tid]); ybh.append(tr_recov[tid]); gh.append(tr_group[tid])
        H = np.asarray(H); ys = np.asarray(ys); ybh = np.asarray(ybh); gh = np.asarray(gh)
        ns = min(args.n_splits, len(set(gh.tolist())))
        print(f"  hidden subset: {H.shape[0]} traj x {H.shape[1]} dims, {len(set(gh.tolist()))} problems")
        for name, mk in [("linear (Ridge)", lambda: Ridge(alpha=10.0)),
                         ("MLP", lambda: MLPRegressor((256,), alpha=1e-2, max_iter=300))]:
            auc_h = roc_auc_score(ybh, cv_oof(H, ys, gh, ns, mk))
            ph = rng.permutation(len(ys))
            auc_hs = roc_auc_score(ybh[ph], cv_oof(H, ys[ph], gh, ns, mk))
            print(f"  {name:16s} AUC={auc_h:.3f}   shuffle-control={auc_hs:.3f}")
        print("  -> compare to scalar-only full model above; higher = hidden adds signal.")


if __name__ == "__main__":
    main()
