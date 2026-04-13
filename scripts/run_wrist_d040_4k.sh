#!/bin/bash
# R9f E2E: up_wrist D040 corrected params, 100ep, 4k steps, 5 seeds
# Tests "representation bottleneck" hypothesis: longer training -> lower variance
# Key change vs R9e: STEPS=4000, SAVE_EVERY=2000 (disk-friendly)
set -e
cd /workspace/workshop

DATASET="local/franka-pick-up_wrist-d040-100ep"
STEPS=4000
SAVE_EVERY=2000
BATCH=4
WORKERS=4
EVAL_EPS=50
EVAL_SEED=99

echo "============================================"
echo "R9f: up_wrist D040 E2E (5 seeds × ${STEPS} steps)"
echo "dataset: ${DATASET}"
echo "save-every: ${SAVE_EVERY}"
echo "============================================"

for s in 0 1 2 3 4; do
    OUTDIR="/output/outputs/wrist_d040_seed${s}_4k"
    echo ""
    echo "=== TRAIN seed=$s → $OUTDIR ==="
    TRAIN_SEED=$s python scripts/train_with_seed.py \
        --dataset-id "$DATASET" \
        --n-steps $STEPS --batch-size $BATCH --num-workers $WORKERS \
        --save-every $SAVE_EVERY \
        --save-dir "$OUTDIR" 2>&1 | tail -5

    echo "=== EVAL seed=$s (${EVAL_EPS} trials) ==="
    python scripts/03_eval.py \
        --policy-type smolvla \
        --checkpoint "${OUTDIR}/final" \
        --dataset-id "$DATASET" \
        --n-episodes $EVAL_EPS --max-steps 150 --seed $EVAL_SEED \
        --no-bbox-detection --camera-layout up_wrist \
        --save "${OUTDIR}/eval_unseen_50" 2>&1 | grep -E "RESULT|success"
    echo ""
done

echo "============================================"
echo "=== ALL DONE - SUMMARY ==="
echo "============================================"
for s in 0 1 2 3 4; do
    OUTDIR="/output/outputs/wrist_d040_seed${s}_4k"
    if [ -f "${OUTDIR}/eval_unseen_50/eval_summary.json" ]; then
        SR=$(python3 -c "import json; d=json.load(open('${OUTDIR}/eval_unseen_50/eval_summary.json')); print(f'seed={s} success={d.get(\"success_rate\", \"N/A\")}')")
        echo "  $SR"
    else
        echo "  seed=$s: eval not found"
    fi
done
