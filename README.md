# fork-opd

Learned **fork-gate** for on-policy distillation: a small gate decides *where* a strong
teacher should take over a weak student's rollout. The prefix stays on-policy (student
tokens, reverse KL); the suffix is teacher-generated (imitation / forward KL). The gate
subsumes the two fixed strategies — `no-fork` = pure on-policy, `k=0` = whole-rewrite —
so it can only help if *where to fork* matters.

## Quickstart (on the mcli GPU box)

```bash
git clone git@github.com:mwdgdx/fork-opd.git && cd fork-opd
bash run.sh                       # fast SMOKE run (50 problems), whole-rewrite baseline arm
```

`run.sh` installs deps via the Databricks pip proxy, then runs the whole-rewrite arm
end-to-end: student rollouts → teacher rewrites (verify + wall-clock print) → train.
Dataset and models download from HF at runtime. Scale up with env vars, e.g.
`NUM_PROBLEMS=2000 MAX_TOKENS=8192 TP=2 bash run.sh`. `SKIP_SETUP=1` skips reinstall.

> Untested locally (this repo is authored on a CPU box). Expect to fix small issues on
> the first box run — most likely the DAPO field names (override `--question-field` /
> `--answer-field` in `generate_rollouts.py`) or GPU memory (`TP`, `MAX_TOKENS`).

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

## Build (Path B: lightweight offline/batched, frozen external teacher, vLLM generation)

We do NOT reuse the SDPO/verl online loop: its "teacher" is an EMA self-teacher + privileged
reprompt (small gap), whereas we need a strong external teacher (large gap). Path B keeps the
frozen 8B teacher offline (vLLM), constructs fork trajectories, and trains the student in a
light loop — the fork-vs-whole-rewrite comparison is unaffected, and it avoids the verl
teacher-worker lift. Scale to verl later if needed.

1. **[done] data generation** — `data_gen/generate_rollouts.py`: student rollouts on DAPO-Math,
   `math_verify` checking, difficulty-band filter, save failed trajectories (+ token ids).
2. **[done] teacher construction** — `teacher/build_fork_data.py`: frozen 8B teacher generates
   the suffix from fork point `k` (policies: whole / nofork / fixed_frac / fixed_k), verifies
   recovery, and reports the efficiency metric (teacher gen tokens + generation wall-clock).
3. **[partial] training** — `fork/train.py`: imitation (CE) loss on the teacher suffix. Fully
   covers the **whole-rewrite baseline**; fork's on-policy PREFIX loss (reverse KL) is the next
   addition (needs teacher prefix top-k logprobs pre-saved in step 2). `fork/fork_loss.py` is the
   mixed-loss reference (`fork_k` recovers whole-rewrite / no-fork / fork).
4. **[todo] learned gate** — probe on teacher hidden states; supervised warmup on a cheap
   error-onset proxy, then contextual-bandit refinement.

## Efficiency claim = WALL-CLOCK, not FLOPs

Fork adds one teacher forward pass but does *less autoregressive generation*. In FLOPs fork can
be higher; the win is wall-clock, because autoregressive teacher generation is the bottleneck.
Report **teacher generated tokens (T−k vs T)** and **generation wall-clock under vLLM** — never
FLOPs. Use vLLM generation for every arm so wall-clock is realistic.

Compute runs on an mcli GPU box (launched by the user); this repo is code + git only.
