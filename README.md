# fork-opd

Learned **fork-gate** for on-policy distillation: a small gate decides *where* a strong
teacher should take over a weak student's rollout. The prefix stays on-policy (student
tokens, reverse KL); the suffix is teacher-generated (imitation / forward KL). The gate
subsumes the two fixed strategies — `no-fork` = pure on-policy, `k=0` = whole-rewrite —
so it can only help if *where to fork* matters.

## Why it might beat whole-rewrite

Whole-rewrite (teacher rewrites the entire failed trajectory) produces the *best-looking*
trajectories but trains on 100% teacher tokens → train/inference distribution mismatch
(exposure bias, GKD 2024) → the student drifts at inference. Fork keeps the student's
on-distribution prefix and only hands off the failed suffix, so it inherits on-policy's
accuracy advantage while generating fewer teacher tokens (cheaper).

The advantage grows with the **teacher-student gap** and with how **localizable / late**
the errors are. A self-teacher (small gap) shows no exposure bias — which is exactly why
prior self-teacher OPD (SDPO) found whole-rewrite ≥ on-policy.

## Minimal experiment (what we are actually building first)

One training pipeline, `k` is a parameter, so the baselines fall out for free:

| arm | fork point | role |
|-----|-----------|------|
| whole-rewrite | `k = 0` | baseline to beat (= Improved SDPO with an **external** teacher) |
| pure on-policy | `no-fork` | free lower-bound sanity |
| **learned gate** | predicted `k` | our method |

Compare **final student accuracy + teacher-generation compute**. The gate ≥ whole-rewrite
on accuracy *and* cheaper ⇒ result.

Everything else (fixed-fraction fork, error-onset heuristic, self-teacher control,
teacher-size sweep, cross-pair transfer) is a later ablation, not part of basic validation.

## Setting

- student `Qwen3-1.7B`, teacher `Qwen3-8B` (large gap; swept later)
- train `DAPO-Math-17K`, test `MATH500 / AMC / AIME24-25 / OlympiadBench`
- weak feedback = final-answer correctness only
- difficulty band: student pass-rate ~20–60% (maximizes salvageable failures)

## Build milestones

1. **[in progress] data generation** — `data_gen/generate_rollouts.py`: student rollouts on
   DAPO-Math, verify answers, keep band-difficulty problems, save failed trajectories.
2. **fork training loop** — `fork/fork_loss.py`: mixed loss with `k` (k=0 / no-fork / fixed-k
   all covered). Built on the SDPO/verl trainer.
3. **learned gate** — probe on teacher hidden states; start supervised (cheap error-onset
   proxy), then contextual-bandit refinement.

Compute runs on an mcli GPU box (launched by the user); this repo is code + git only.
