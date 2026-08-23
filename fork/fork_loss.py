"""Milestone 2 (reference): the mixed fork loss for one constructed trajectory.

A constructed trajectory is  student_prefix[0:k] + teacher_suffix[k:T] :
    positions [0, k)  -> ON-policy  (student's own tokens): reverse KL toward teacher
    positions [k, T)  -> OFF-policy (teacher-generated tokens): imitation (CE / forward KL)

The single parameter `fork_k` recovers every arm for free:
    fork_k == 0   -> whole-rewrite  (all imitation, off-policy)
    fork_k >= T   -> no-fork        (all reverse KL, pure on-policy)
    0 < fork_k < T -> fork

Teacher is frozen: `teacher_logits` are treated as constants (detached upstream).
This is a reference for wiring into the SDPO/verl trainer, not a standalone loop.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """KL(pi_S || pi_T) per position (mode-seeking). Gradient flows through student only."""
    log_p_s = F.log_softmax(student_logits, dim=-1)
    log_p_t = F.log_softmax(teacher_logits, dim=-1)
    p_s = log_p_s.exp()
    return (p_s * (log_p_s - log_p_t)).sum(dim=-1)


def _forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """KL(pi_T || pi_S) per position (mass-covering)."""
    log_p_s = F.log_softmax(student_logits, dim=-1)
    log_p_t = F.log_softmax(teacher_logits, dim=-1)
    p_t = log_p_t.exp()
    return (p_t * (log_p_t - log_p_s)).sum(dim=-1)


def fork_loss(
    student_logits: torch.Tensor,   # [T, V]  student next-token logits over constructed seq
    token_ids: torch.Tensor,        # [T]     constructed-sequence token ids (targets)
    fork_k: int,
    teacher_logits: torch.Tensor | None = None,  # [T, V] needed for the prefix (reverse KL)
    prefix_divergence: str = "reverse_kl",       # "reverse_kl" | "forward_kl"
    loss_mask: torch.Tensor | None = None,        # [T] 1 for supervised positions
) -> torch.Tensor:
    """Mean mixed loss over supervised positions of one constructed trajectory."""
    T = student_logits.shape[0]
    fork_k = max(0, min(fork_k, T))
    per_pos = torch.zeros(T, device=student_logits.device, dtype=student_logits.dtype)

    # prefix [0, k): on-policy divergence between student and (frozen) teacher
    if fork_k > 0:
        if teacher_logits is None:
            raise ValueError("prefix loss needs teacher_logits over [0, fork_k)")
        div = _reverse_kl if prefix_divergence == "reverse_kl" else _forward_kl
        per_pos[:fork_k] = div(student_logits[:fork_k], teacher_logits[:fork_k].detach())

    # suffix [k, T): imitate the teacher's generated tokens (cross-entropy)
    if fork_k < T:
        ce = F.cross_entropy(
            student_logits[fork_k:], token_ids[fork_k:], reduction="none"
        )
        per_pos[fork_k:] = ce

    if loss_mask is not None:
        per_pos = per_pos * loss_mask
        denom = loss_mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(float(T), device=per_pos.device)
    return per_pos.sum() / denom


if __name__ == "__main__":  # tiny shape/degenerate-case sanity check (CPU)
    torch.manual_seed(0)
    T, V = 6, 100
    s = torch.randn(T, V, requires_grad=True)
    t = torch.randn(T, V)
    ids = torch.randint(0, V, (T,))
    for k in (0, 3, T):  # whole-rewrite / fork / no-fork
        l = fork_loss(s, ids, fork_k=k, teacher_logits=t)
        print(f"fork_k={k}: loss={l.item():.4f}")
