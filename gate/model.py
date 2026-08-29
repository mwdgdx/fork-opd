"""The fork gate: a small MLP on frozen-teacher per-position features.

Input: per-response-position feature vectors (teacher hidden state + logit-derived
signals; see features.py). Output: a distribution over fork actions
    action j in [0, R)  -> fork at response position j (keep student prefix [0,j),
                            teacher takes over from j)
    action R            -> no-fork (skip this trajectory; teacher adds nothing)
j=0 is whole-rewrite. The teacher is frozen; only this MLP trains.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ForkGate(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.no_fork = nn.Parameter(torch.zeros(()))  # learned no-fork logit

    def action_logits(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: [R, D] -> action logits [R+1] (last entry = no-fork)."""
        pos = self.net(feats).squeeze(-1)                 # [R]
        return torch.cat([pos, self.no_fork.view(1)])     # [R+1]

    @torch.no_grad()
    def sample(self, feats: torch.Tensor):
        logits = self.action_logits(feats)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a)), float(dist.entropy())

    def logprob_entropy(self, feats: torch.Tensor, action: int):
        """Differentiable log-prob + entropy for a taken action (for REINFORCE)."""
        logits = self.action_logits(feats)
        dist = torch.distributions.Categorical(logits=logits)
        a = torch.tensor(action, device=logits.device)
        return dist.log_prob(a), dist.entropy()
