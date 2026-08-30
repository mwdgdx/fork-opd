"""The fork gate: reads per-position frozen-teacher features and picks a fork point.

Two variants:
  - seq_model=True (default): a small BIDIRECTIONAL transformer over the sequence of
    per-position features, so each position's score is informed by the WHOLE trajectory
    (before AND after) — locating the error onset from the global disagreement profile.
  - seq_model=False: a per-position MLP (each position scored in isolation; note the
    teacher hidden state is already causally contextualized, but there is no look-ahead
    and no cross-position comparison).

Action space: fork at response position j in [0, R) (j=0 = whole-rewrite), or R = no-fork.
Teacher is frozen; only this module trains.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ForkGate(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256, seq_model: bool = True,
                 n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(feat_dim, hidden)
        if seq_model:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
                batch_first=True, activation="gelu", dropout=0.0,
            )
            self.seq = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.seq = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Linear(hidden, 1)
        self.no_fork = nn.Parameter(torch.zeros(()))

    def action_logits(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: [R, D] -> action logits [R+1] (last entry = no-fork)."""
        x = self.proj(feats)                        # [R, H]
        if isinstance(self.seq, nn.TransformerEncoder):
            x = self.seq(x.unsqueeze(0)).squeeze(0)  # bidirectional over positions -> [R, H]
        else:
            x = self.seq(x)
        pos = self.head(x).squeeze(-1)              # [R]
        return torch.cat([pos, self.no_fork.view(1)])

    @torch.no_grad()
    def sample(self, feats: torch.Tensor):
        logits = self.action_logits(feats)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a)), float(dist.entropy())

    def logprob_entropy(self, feats: torch.Tensor, action: int):
        logits = self.action_logits(feats)
        dist = torch.distributions.Categorical(logits=logits)
        a = torch.tensor(action, device=logits.device)
        return dist.log_prob(a), dist.entropy()
