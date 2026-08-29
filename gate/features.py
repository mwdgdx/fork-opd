"""Per-position gate features from a frozen teacher, teacher-forced on the STUDENT's
trajectory. One forward pass, no generation. Vectorized over response positions.

For each response position r (a candidate fork point) we take the teacher's decision at
the token before it (position P+r-1) and build:
    [ teacher hidden state h   (H,) ]  semantic "is the reasoning still on track"
    [ teacher top-k logprobs   (k,) ]  shape of the teacher's distribution
    [ scalars                  (7,) ]
scalars = p_teacher(student_token), entropy, teacher top-1 prob, agree-flag
          (teacher argmax == student token), student-token rank (normalized),
          student logprob of its token, gap = logp_teacher - logp_student.

The student's *tokens* are always the teacher-forced input, so the gate is never
student-blind. student_logprobs (from generation) is optional; if absent, gap degrades
to logp_teacher and student_logprob is 0.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def extract_features(
    teacher, prompt_ids: list[int], response_ids: list[int],
    student_logprobs: list[float] | None = None, topk: int = 20, device: str = "cuda",
) -> torch.Tensor:
    P, R = len(prompt_ids), len(response_ids)
    if R == 0:
        return torch.empty(0)
    ids = torch.tensor(prompt_ids + response_ids, device=device).unsqueeze(0)
    out = teacher(ids, output_hidden_states=True, use_cache=False)
    logits = out.logits[0]                  # [N, V]
    hs = out.hidden_states[-1][0]           # [N, H]
    V = logits.shape[-1]

    dec = torch.tensor([max(P + r - 1, 0) for r in range(R)], device=device)  # [R]
    lp = torch.log_softmax(logits[dec].float(), dim=-1)   # [R, V]
    resp = torch.tensor(response_ids, device=device)      # [R]
    lp_tok = lp.gather(1, resp[:, None]).squeeze(1)       # [R]

    topk_lp = lp.topk(topk, dim=-1).values                # [R, topk]
    ent = -(lp.exp() * lp).sum(-1)                        # [R]
    top1_lp, top1_idx = lp.max(dim=-1)                    # [R]
    agree = (top1_idx == resp).float()                    # [R]
    rank = (lp > lp_tok[:, None]).sum(-1).float() / V     # [R]
    p_tok = lp_tok.exp()                                  # [R]
    if student_logprobs is not None:
        s_lp = torch.tensor(student_logprobs, device=device).float()
    else:
        s_lp = torch.zeros(R, device=device)
    gap = lp_tok - s_lp                                   # [R]

    h_dec = hs[dec].float()                               # [R, H]
    scal = torch.stack([p_tok, ent, top1_lp.exp(), agree, rank, s_lp, gap], dim=-1)  # [R,7]
    return torch.cat([h_dec, topk_lp, scal], dim=-1)      # [R, H+topk+7]


def feature_dim(hidden_size: int, topk: int = 20) -> int:
    return hidden_size + topk + 7
