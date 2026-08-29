#!/usr/bin/env bash
# One-command entrypoint for the WHOLE-REWRITE baseline arm (first end-to-end run).
#
#   git clone git@github.com:mwdgdx/fork-opd.git && cd fork-opd && bash run.sh
#
# Defaults are a fast SMOKE run (50 problems) to validate the pipeline. Scale up with
# env vars, e.g.:  NUM_PROBLEMS=2000 MAX_TOKENS=8192 bash run.sh
# Dataset + models download from HF on first use (the box can reach HF).
set -euo pipefail
cd "$(dirname "$0")"

STUDENT=${STUDENT:-Qwen/Qwen3-1.7B}
TEACHER=${TEACHER:-Qwen/Qwen3-8B}
DATASET=${DATASET:-BytedTsinghua-SIA/DAPO-Math-17k}
NUM_PROBLEMS=${NUM_PROBLEMS:-20}
NUM_SAMPLES=${NUM_SAMPLES:-8}
MAX_TOKENS=${MAX_TOKENS:-4096}
TP=${TP:-1}                 # tensor-parallel size for the 8B teacher
OUT=${OUT:-out}

[ "${SKIP_SETUP:-0}" = "1" ] || bash setup.sh

echo "== [1/3] student rollouts -> failed trajectories =="
python data_gen/generate_rollouts.py \
  --model "$STUDENT" --dataset "$DATASET" \
  --num-problems "$NUM_PROBLEMS" --num-samples "$NUM_SAMPLES" \
  --max-tokens "$MAX_TOKENS" --out "$OUT/failures.jsonl"

echo "== [2/3] whole-rewrite data (teacher rewrites, verify, measure wall-clock) =="
python teacher/build_fork_data.py \
  --failures "$OUT/failures.jsonl" --teacher "$TEACHER" \
  --fork-policy whole --max-new-tokens "$MAX_TOKENS" --tensor-parallel-size "$TP" \
  --out "$OUT/fork_whole.jsonl"

echo "== [3/3] train the whole-rewrite student =="
python fork/train.py \
  --data "$OUT/fork_whole.jsonl" --student "$STUDENT" \
  --epochs 1 --out "$OUT/ckpt_whole"

echo "[run.sh] done -> failures, whole-rewrite data (+ efficiency print), checkpoint in $OUT/"
