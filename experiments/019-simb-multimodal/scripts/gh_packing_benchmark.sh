#!/usr/bin/env bash
# Measure the cost of PACKING two training runs onto one GPU, for the 019 expression
# campaign. Solo on GPU 0, then two concurrent on GPU 1, same config, same checkpoint,
# same epoch count. Steady-state s/epoch is read from the Lightning progress bar.
set -u

ROOT="$HOME/Documents/projects/torchcell.worktrees/multimodal-phenotype-retrospective"
CK="/scratch/projects/torchcell-scratch/models/checkpoints/gilahyper-1445_447aa95e06414b55667966ee1f9a489200ccd9d2a8782fc216a61a74c9baaafb/1445-last.ckpt"
SP="$(dirname "$0")/pack"
mkdir -p "$SP"
cd "$ROOT/experiments" || exit 1

PY="$HOME/miniconda3/envs/torchcell/bin/python"
COMMON=(019-simb-multimodal/scripts/train_cgt_multitask.py
        --config-name cgt_expr_v9_mask
        "trainer.resume_ckpt_path=$CK"
        "trainer.max_epochs=10010"
        "trainer.train_eval_every=0")

echo "[$(date +%T)] SOLO on GPU 0"
env PYTHONPATH="$ROOT" WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
  "$PY" "${COMMON[@]}" > "$SP/solo.log" 2>&1
echo "[$(date +%T)] solo done"

echo "[$(date +%T)] PACKED: two concurrent on GPU 1"
for i in 1 2; do
  env PYTHONPATH="$ROOT" WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 \
    "$PY" "${COMMON[@]}" > "$SP/packed_$i.log" 2>&1 &
done
# Sample GPU 1 memory while both are resident, so the VRAM claim is measured too.
sleep 150
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader > "$SP/gpu1_mem.txt" 2>&1
wait
echo "[$(date +%T)] packed done"
echo ALLDONE > "$SP/status"
