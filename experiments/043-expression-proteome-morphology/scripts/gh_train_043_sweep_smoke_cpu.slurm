#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/gh_train_043_sweep_smoke_cpu.slurm
# [[experiments.043-expression-proteome-morphology.scripts.gh_train_043_sweep_smoke_cpu]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/gh_train_043_sweep_smoke_cpu.slurm
#
# CPU smoke of the 2026-10-24 GilaHyper sweep knobs on the real store (prep_v3; prep_v4 for
# the profile case). No GPU: the cards are the overnight round's. Every case is arm all,
# batch 4, every mark, three optimizer steps, a tiny validation pass, W&B offline, and
# attention_query_chunk 1024 (memory only on CPU, the attention is the same). CASES picks
# the cases, run in order:
#   smss smmss ssss mmmm msmsms : trunk layer patterns (smss = the default trunk, by group)
#   dims    : hidden 288, trunk FFN 2x, head_hidden 256
#   mle binq: loss mle | binned_quantile
#   qn none : target_transform quantile_normal | none (with mse)
#   profile : target statistics and weighting by ontology phenotype profile (prep_v4)
#   cos     : cosine schedule
#   ftlegacy: fine-tune alone_kemmeren (lr 3e-4, warmup 30) from a COPY of the overnight
#             stage-1 last.pt of job 6308, i.e. what the pending 6311 runs, on this code
#   chain   : stage 1 binned_quantile + quantile_normal, then its Kemmeren fine-tune
#
#   sbatch --export=ALL,CASES="smmss ssss" experiments/043-expression-proteome-morphology/scripts/gh_train_043_sweep_smoke_cpu.slurm
#SBATCH -p main
#SBATCH --mem=96g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=043-sweep-smoke-cpu
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="$PWD"
export DATA_ROOT=/scratch/projects/torchcell-scratch
export EXPERIMENT_ROOT="$PWD/experiments"
export WANDB_MODE=offline
CKPT="$DATA_ROOT/experiments/043-expression-proteome-morphology/checkpoints"
SMOKE_DIR="$DATA_ROOT/experiments/043-expression-proteome-morphology/sweep_smoke/${SLURM_JOB_ID}"
mkdir -p "$SMOKE_DIR"
COMMON=(
  data.store_version=v2 arm=all loss=hlgauss weighting=equal seed=0 device=cpu
  data.batch_size=4 data.num_workers=4 data.sampler=records data.mark_cap=null
  data.val_max_records_per_dataset=8 data.traineval_max_records_per_dataset=4
  budget.max_epochs=1 +budget.max_steps_per_epoch=3 budget.probe_steps=2
  attention_query_chunk=1024 wandb.mode=offline wandb.label_knobs=true
  "wandb.tags=[smoke,ghyper_sweep_2026_10_24]" wandb.round=ghyper_sweep_smoke_cpu
  "paths.checkpoint_root=$SMOKE_DIR/ckpt"
)
TRAIN=experiments/043-expression-proteome-morphology/scripts/train_043.py
FT=(arm=alone_kemmeren optim.lr=3e-4 optim.warmup_steps=30)
run() { echo "===== case $1  $(date)"; shift; python "$TRAIN" "${COMMON[@]}" "$@"; echo "===== ok $(date)"; }
echo "host $(hostname)  commit $(git rev-parse --short HEAD)  cases ${CASES:?set CASES}  $(date)"
for c in $CASES; do
  case "$c" in
    smss) run smss +model=pattern_smss ;;
    smmss) run smmss +model=pattern_smmss ;;
    ssss) run ssss "model.layer_pattern=[S,S,S,S]" ;;
    mmmm) run mmmm +model=pattern_mmmm ;;
    msmsms) run msmsms +model=pattern_msmsms ;;
    dims) run dims model.hidden_dim=288 model.trunk_ffn_mult=2 model.head_hidden=256 ;;
    mle) run mle loss=mle ;;
    binq) run binq loss=binned_quantile ;;
    qn) run qn data.target_transform=quantile_normal ;;
    none) run none data.target_transform=none loss=mse ;;
    profile) run profile data.prep_version=prep_v4 data.target_stats_by=profile weighting.by=profile ;;
    cos) run cos optim.schedule=cosine optim.warmup_steps=1 ;;
    ftlegacy)
      src="$CKPT/all_hlgauss_equal_s0/6308"
      mkdir -p "$SMOKE_DIR/stage1_6308"
      cp "$src/standardization.json" "$SMOKE_DIR/stage1_6308/"
      cp "$src/last.pt" "$SMOKE_DIR/stage1_6308/last.pt"
      python -c "import torch; torch.load('$SMOKE_DIR/stage1_6308/last.pt', map_location='cpu', weights_only=False)"
      run ftlegacy "${FT[@]}" "init_from=$SMOKE_DIR/stage1_6308/last.pt" ;;
    chain)
      run chain-stage1 loss=binned_quantile data.target_transform=quantile_normal
      s1=$(ls -d "$SMOKE_DIR"/ckpt/all_binned_quantile_equal_tqn_s0/"$SLURM_JOB_ID")
      run chain-ft loss=binned_quantile data.target_transform=quantile_normal "${FT[@]}" "init_from=$s1/last.pt" ;;
    *) echo "unknown case $c" >&2; exit 2 ;;
  esac
done
echo "done $(date)"
