#!/bin/bash
#
# End-to-end pipeline: Synthetic Data Generation → VLA Training → Sim Evaluation
#
# Robot:  Franka Panda 7-DOF
# Task:   Pick up the red cube
# Model:  SmolVLA (Vision-Language-Action)
# Sim:    Genesis physics engine
#
# Prerequisites:
#   - genesis-world, lerobot, torch, transformers, accelerate
#   - GPU with ≥4GB VRAM (tested on NV 4090, AMD MI300)
#   - Xvfb or physical display for rendering
#
# Usage:
#   bash run_pipeline.sh                 # default: 100ep, 2000 steps
#   bash run_pipeline.sh 50 1000         # custom: 50ep, 1000 steps
#   N_EPISODES=200 bash run_pipeline.sh  # env override
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

N_EPISODES="${1:-${N_EPISODES:-100}}"
N_STEPS="${2:-${N_STEPS:-2000}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
EVAL_SEED="${EVAL_SEED:-99}"
OUTPUT="${OUTPUT:-/output}"

REPO_ID="local/franka-workshop-${N_EPISODES}ep"
SAVE_DIR="${OUTPUT}/outputs/workshop_smolvla"

echo "============================================================"
echo "  Robot Synthetic Data Generation Workshop Pipeline"
echo "============================================================"
echo "  episodes:    ${N_EPISODES}"
echo "  train steps: ${N_STEPS}"
echo "  batch_size:  ${BATCH_SIZE}"
echo "  repo_id:     ${REPO_ID}"
echo "  output:      ${OUTPUT}"
echo "============================================================"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 1: Synthetic Data Generation
# ──────────────────────────────────────────────────────────────
echo "=== STEP 1/4: DATA GENERATION (${N_EPISODES} episodes) ==="

python scripts/01_gen_data.py \
  --n-episodes "${N_EPISODES}" \
  --repo-id "${REPO_ID}" \
  --save "${OUTPUT}" \
  --seed "${SEED}" \
  --task "Pick up the red cube."

echo ""

# ──────────────────────────────────────────────────────────────
# Step 2: SmolVLA Post-Training
# ──────────────────────────────────────────────────────────────
echo "=== STEP 2/4: SmolVLA TRAINING (${N_STEPS} steps) ==="

python scripts/02_train_vla.py \
  --dataset-id "${REPO_ID}" \
  --n-steps "${N_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --save-dir "${SAVE_DIR}"

echo ""

# ──────────────────────────────────────────────────────────────
# Step 3: Simulation Evaluation — unseen positions
# ──────────────────────────────────────────────────────────────
echo "=== STEP 3/4: EVAL — unseen positions (seed=${EVAL_SEED}) ==="

python scripts/03_eval.py \
  --policy-type smolvla \
  --checkpoint "${SAVE_DIR}/final" \
  --dataset-id "${REPO_ID}" \
  --n-episodes 10 --max-steps 150 \
  --seed "${EVAL_SEED}" \
  --record-video \
  --save "${OUTPUT}/eval_unseen"

echo ""

# ──────────────────────────────────────────────────────────────
# Step 4: Simulation Evaluation — training positions
# ──────────────────────────────────────────────────────────────
echo "=== STEP 4/4: EVAL — training positions (seed=${SEED}) ==="

python scripts/03_eval.py \
  --policy-type smolvla \
  --checkpoint "${SAVE_DIR}/final" \
  --dataset-id "${REPO_ID}" \
  --n-episodes 10 --max-steps 150 \
  --seed "${SEED}" \
  --record-video \
  --save "${OUTPUT}/eval_train"

echo ""

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Pipeline Complete!"
echo "============================================================"
echo ""
echo "  Outputs:"
echo "    Dataset:      ~/.cache/huggingface/lerobot/${REPO_ID}"
echo "    Checkpoint:   ${SAVE_DIR}/final/"
echo "    Train log:    ${SAVE_DIR}/train_summary.json"
echo "    Eval unseen:  ${OUTPUT}/eval_unseen/eval_summary.json"
echo "    Eval train:   ${OUTPUT}/eval_train/eval_summary.json"
echo "    Videos:       ${OUTPUT}/eval_unseen/videos/"
echo "                  ${OUTPUT}/eval_train/videos/"
echo ""

if [ -f "${OUTPUT}/eval_unseen/eval_summary.json" ]; then
  echo "  Unseen eval result:"
  python -c "
import json, sys
s = json.load(open('${OUTPUT}/eval_unseen/eval_summary.json'))
print(f'    {s[\"n_success\"]}/{s[\"n_episodes\"]} = {s[\"success_rate\"]:.0%}')
"
fi
if [ -f "${OUTPUT}/eval_train/eval_summary.json" ]; then
  echo "  Training eval result:"
  python -c "
import json, sys
s = json.load(open('${OUTPUT}/eval_train/eval_summary.json'))
print(f'    {s[\"n_success\"]}/{s[\"n_episodes\"]} = {s[\"success_rate\"]:.0%}')
"
fi

echo ""
echo "  Optional: plot training loss curve"
echo "    python scripts/04_plot_loss.py \\"
echo "      --metrics ${SAVE_DIR}/train_metrics.json \\"
echo "      --summary ${SAVE_DIR}/train_summary.json \\"
echo "      --out ${OUTPUT}/loss_curve.png"
echo ""
