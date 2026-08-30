"""Supervised gate: predict per-position 'recoverable-from-here' from teacher features,
using the dense recoverability labels (from label_recoverability.py). FAST — no vLLM in
the loop (cached features + precomputed labels), so we can iterate architectures/features.

Decisive test: can teacher features PREDICT where the recoverable fork is? And does the
gate's fork choice beat whole-rewrite / fixed@0.5 in recovery — WITHOUT new generation
(evaluated against the precomputed labels)?

    python gate/train_supervised.py --features out/gate_features_pool500.pt \
        --labels out/reco_labels.json --seq-model 1 --epochs 40 --out out/sup_gate.pt
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn


def sinusoidal_pe(R, dim, device):
    pos = torch.arange(R, device=device).float().unsqueeze(1)
    i = torch.arange(dim, device=device).float().unsqueeze(0)
    angle = pos / (10000.0 ** (2 * (i // 2) / dim))
    pe = torch.zeros(R, dim, device=device)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe


class SupGate(nn.Module):
    def __init__(self, feat_dim, hidden=256, seq_model=True, n_layers=2, n_heads=4):
        super().__init__()
        self.proj = nn.Linear(feat_dim + 1, hidden)   # +1 for k/R (normalized position)
        if seq_model:
            layer = nn.TransformerEncoderLayer(hidden, n_heads, hidden * 2, batch_first=True,
                                               activation="gelu", dropout=0.0)
            self.seq = nn.TransformerEncoder(layer, n_layers)
        else:
            self.seq = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)

    def forward(self, feats):                 # [R, D] -> per-position logit [R]
        R = feats.shape[0]
        posfrac = (torch.arange(R, device=feats.device).float() / max(R - 1, 1)).unsqueeze(1)
        x = self.proj(torch.cat([feats, posfrac], dim=-1))
        if isinstance(self.seq, nn.TransformerEncoder):
            x = x + sinusoidal_pe(R, x.shape[1], x.device)   # position-aware attention
            x = self.seq(x.unsqueeze(0)).squeeze(0)
        else:
            x = self.seq(x)
        return self.head(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--seq-model", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--heldout-frac", type=float, default=0.2)
    ap.add_argument("--out", default="out/sup_gate.pt")
    args = ap.parse_args()

    blob = torch.load(args.features)
    feats_list, hidden = blob["feats"], blob["hidden"]
    lab = json.load(open(args.labels))
    fracs, labels = lab["fracs"], lab["labels"]
    assert len(feats_list) == len(labels), f"misalign {len(feats_list)} vs {len(labels)}"
    D = feats_list[0].shape[1]

    n = len(feats_list)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g).tolist()
    n_ho = int(args.heldout_frac * n)
    ho, tr = set(perm[:n_ho]), perm[n_ho:]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SupGate(D, args.hidden, bool(args.seq_model)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    def labeled_positions(i):
        R = feats_list[i].shape[0]
        out = []
        for f in fracs:
            k = max(0, min(int(f * R), R - 1))
            out.append((k, float(labels[i][str(f)])))
        return out

    for ep in range(args.epochs):
        model.train()
        import random
        random.Random(ep).shuffle(tr)
        tot = 0.0
        for i in tr:
            feats = feats_list[i].to(dev)
            logits = model(feats)
            pos = labeled_positions(i)
            idx = torch.tensor([p[0] for p in pos], device=dev)
            tgt = torch.tensor([p[1] for p in pos], device=dev)
            loss = bce(logits[idx], tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[epoch {ep}] train_bce={tot/len(tr):.3f}", flush=True)

    # ---- held-out eval (against precomputed labels; no generation) ----
    model.eval()
    preds, tgts = [], []
    strat = {"whole_rewrite": [0, 0.0], "fixed0.5": [0, 0.0], "gate": [0, 0.0], "no_fork": 0}
    with torch.no_grad():
        for i in ho:
            R = feats_list[i].shape[0]
            logits = model(feats_list[i].to(dev))
            pos = labeled_positions(i)
            for (k, t) in pos:
                preds.append(torch.sigmoid(logits[k]).item()); tgts.append(t)
            lab_by_frac = {f: float(labels[i][str(f)]) for f in fracs}
            # whole-rewrite = frac 0.0 ; fixed0.5 = frac closest to 0.5
            strat["whole_rewrite"][0] += lab_by_frac[fracs[0]]; strat["whole_rewrite"][1] += fracs[0]
            f05 = min(fracs, key=lambda f: abs(f - 0.5))
            strat["fixed0.5"][0] += lab_by_frac[f05]; strat["fixed0.5"][1] += f05
            # gate: latest frac whose predicted prob > 0.5; recovered = its label
            probs = {f: torch.sigmoid(logits[max(0, min(int(f * R), R - 1))]).item() for f in fracs}
            chosen = [f for f in fracs if probs[f] > 0.5]
            if chosen:
                f = max(chosen)
                strat["gate"][0] += lab_by_frac[f]; strat["gate"][1] += f
            else:
                strat["no_fork"] += 1

    # AUC (manual)
    import statistics as s
    order = sorted(range(len(preds)), key=lambda j: preds[j])
    ranks = [0] * len(preds)
    for r, j in enumerate(order):
        ranks[j] = r + 1
    npos = sum(tgts); nneg = len(tgts) - npos
    auc = (sum(ranks[j] for j in range(len(tgts)) if tgts[j] > 0) - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else float("nan")
    acc = s.mean([1.0 if (preds[j] > 0.5) == (tgts[j] > 0) else 0.0 for j in range(len(preds))])
    m = len(ho)
    print(f"\n[held-out {m} traj] per-position: AUC={auc:.3f} acc={acc:.1%}")
    print("[held-out fork-strategy recovery (via labels, no generation)]")
    for name in ["whole_rewrite", "fixed0.5", "gate"]:
        rec, kf = strat[name]
        print(f"  {name:14s}: recover {rec/m:.1%}   mean_fork_frac={kf/m:.2f}")
    print(f"  gate no-fork chosen on {strat['no_fork']}/{m}")
    torch.save({"state_dict": model.state_dict(), "feat_dim": D, "hidden": args.hidden,
                "seq_model": bool(args.seq_model)}, args.out)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
