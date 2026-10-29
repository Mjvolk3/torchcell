#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/igb_cabbi_train_043.slurm
# [[experiments.043-expression-proteome-morphology.scripts.igb_cabbi_train_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/igb_cabbi_train_043.slurm
#
# One GPU cell of the 043 trainer on an IGB cabbi card (compute-3-3, RTX 6000 Ada 48 GB), the
# counterpart of gh_train_043_sweep_v2.slurm with the same env knobs. Conventions follow
# igb_cabbi_train_046.slurm (the 046 launcher running on the same node): the code is the
# rsynced tree TREE (no .git; .synced_commit names the commit), the env python is called
# directly (no container), W&B runs OFFLINE under $WANDB_DIR (IGB compute nodes have no
# internet) and is synced later from GilaHyper, since wandb 0.22.3 on the IGB login node
# uploads nothing for an unfinished run.
#
# glibc on IGB is 2.28 and the env's torch_scatter wheels need 2.32: the 043 import chain
# reaches no torch_scatter (torchcell.models scatters through torch_geometric.utils.scatter
# since 9a166da10; tests/torchcell/models/test_scatter_pyg_equivalence.py pins the two ops
# equal), and PyG disables the extensions with a UserWarning at import.
#
# Data: the trainer reads only the prepared store (train_043_prep/<prep>_routeA), the four
# vocab_<encoder>.npy of representation_043_v2 and the four alleles/<encoder>.h5 tables under
# DATA_ROOT; the raw LMDB (raw_043_v2) is read only by the prepare step, which stays on
# GilaHyper. Loader workers: 2 (GPFS mmap lock contention under many workers, memory
# igb-gpfs-mmaplock-stalls-lmdb-loaders; the trainer itself opens no LMDB).
#
# Fixed epochs, as on GilaHyper: the trainer runs exactly MAX_EPOCHS and a `timeout` at
# STOP_HOURS stops it instead (exit 124 = PARTIAL; last.pt is the last finished epoch).
# RESUME_ID names the checkpoint directory AND the W&B run id; a resubmission resumes.
#
# Knobs (env; --export cannot carry a comma inside one value): RESUME_ID (required) ARM (all)
#   LOSS (pinball) WEIGHTING (equal) SEED (0) SAMPLER (records) MAX_EPOCHS (50)
#   STOP_HOURS (118, inside the 4-23:00 wall) WINDOW_START_FRAC (0.8) BATCH (16)
#   PREP_VERSION (prep_v4) CONDITIONING (ontology) INJECT (operator) LR WARMUP_STEPS
#   LAYER_PATTERN HIDDEN FFN_MULT HEAD_HIDDEN DROPOUT TOKEN_DROPOUT WEIGHT_DECAY SIGMA_BINS
#   TRANSFORM STATS_BY WEIGHT_BY SCHEDULE BALANCE_POWER MAX_VISITS ROUND (igb_cabbi)
#   EXTRA_OVERRIDES (space-separated Hydra overrides)
#   INIT_FROM_ID: stage 2 (e.g. ARM=alone_kemmeren); the stage-1 RESUME_ID whose
#     <checkpoints>/*/<INIT_FROM_ID>/last.pt must exist on IGB. The GilaHyper launcher also
#     requires the stage-1 W&B run to hold val/kemmeren/loss; with offline W&B here that
#     check cannot run, so this launcher requires a finished stage-1 checkpoint dir instead
#     (last.pt present) and the driver checks the stage-1 W&B history after syncing.
#
#   cd /home/a-m/mjvolk3/scratch/torchcell-043
#   sbatch --export=ALL,RESUME_ID=igb_smoke_043_20261028,MAX_EPOCHS=1,STOP_HOURS=1.8 \
#     --time=02:00:00 --job-name=043-igb-smoke \
#     experiments/043-expression-proteome-morphology/scripts/igb_cabbi_train_043.slurm
#SBATCH -p cabbi
#SBATCH --gres=gpu:1
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=043-igb
#SBATCH --time=4-23:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell-043/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell-043/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail
TREE=${TREE:-/home/a-m/mjvolk3/scratch/torchcell-043}
PY=/home/a-m/mjvolk3/miniconda3/envs/torchcell/bin/python
cd "$TREE"
export PYTHONPATH="$TREE"
export DATA_ROOT=/home/a-m/mjvolk3/scratch/torchcell
export EXPERIMENT_ROOT="$TREE/experiments"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_DIR="$DATA_ROOT/experiments/043-expression-proteome-morphology/wandb-igb"
mkdir -p "$WANDB_DIR"
CKPT="$DATA_ROOT/experiments/043-expression-proteome-morphology/checkpoints"
# train_043 stamps `git rev-parse --short HEAD`; the tree has no .git, so a job-local shim
# answers that one call from .synced_commit (written by the rsync step), as the ALCF
# launchers do (alcf/aurora_env_043.sh)
SYNCED=$(cut -c1-9 "$TREE/.synced_commit")
SHIM=$(mktemp -d)
cat > "$SHIM/git" <<EOF
#!/bin/bash
if [ "\$*" = "rev-parse --short HEAD" ]; then echo "$SYNCED"; exit 0; fi
echo "git shim: only 'rev-parse --short HEAD' is answered on IGB (got: \$*)" >&2; exit 1
EOF
chmod +x "$SHIM/git"
export PATH="$SHIM:$PATH"

: "${RESUME_ID:?set RESUME_ID (checkpoint dir and W&B run id)}"
ARM="${ARM:-all}"
LOSS="${LOSS:-pinball}"
WEIGHTING="${WEIGHTING:-equal}"
SEED="${SEED:-0}"
SAMPLER="${SAMPLER:-records}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
STOP_HOURS="${STOP_HOURS:-118}"
BATCH="${BATCH:-16}"
ROUND="${ROUND:-igb_cabbi}"
TAGS="v2_store,igb_cabbi"
EXTRA=("budget.registered_window.start_frac=${WINDOW_START_FRAC:-0.8}")
EXTRA+=("data.prep_version=${PREP_VERSION:-prep_v4}")
EXTRA+=("model.conditioning.source=${CONDITIONING:-ontology}")
EXTRA+=("model.conditioning.inject=${INJECT:-operator}")
if [ -n "${LAYER_PATTERN:-}" ]; then
  EXTRA+=("model.layer_pattern=[$(echo "$LAYER_PATTERN" | sed 's/./&,/g; s/,$//')]")
fi
if [ -n "${HIDDEN:-}" ]; then EXTRA+=("model.hidden_dim=${HIDDEN}"); fi
if [ -n "${FFN_MULT:-}" ]; then EXTRA+=("model.trunk_ffn_mult=${FFN_MULT}"); fi
if [ -n "${HEAD_HIDDEN:-}" ]; then EXTRA+=("model.head_hidden=${HEAD_HIDDEN}"); fi
if [ -n "${DROPOUT:-}" ]; then EXTRA+=("model.dropout=${DROPOUT}"); fi
if [ -n "${BALANCE_POWER:-}" ]; then EXTRA+=("data.balance_power=${BALANCE_POWER}"); fi
if [ -n "${MAX_VISITS:-}" ]; then EXTRA+=("data.max_visits_per_record_per_epoch=${MAX_VISITS}"); fi
if [ -n "${TOKEN_DROPOUT:-}" ]; then EXTRA+=("model.token_dropout=${TOKEN_DROPOUT}"); fi
if [ -n "${WEIGHT_DECAY:-}" ]; then EXTRA+=("optim.weight_decay=${WEIGHT_DECAY}"); fi
if [ -n "${SIGMA_BINS:-}" ]; then EXTRA+=("loss.hl_gauss.sigma_bins=${SIGMA_BINS}"); fi
if [ -n "${TRANSFORM:-}" ]; then EXTRA+=("data.target_transform=${TRANSFORM}"); fi
if [ -n "${STATS_BY:-}" ]; then EXTRA+=("data.target_stats_by=${STATS_BY}"); fi
if [ -n "${WEIGHT_BY:-}" ]; then EXTRA+=("weighting.by=${WEIGHT_BY}"); fi
if [ -n "${SCHEDULE:-}" ]; then EXTRA+=("optim.schedule=${SCHEDULE}"); fi
if [ -n "${LR:-}" ]; then EXTRA+=("optim.lr=${LR}"); fi
if [ -n "${WARMUP_STEPS:-}" ]; then EXTRA+=("optim.warmup_steps=${WARMUP_STEPS}"); fi
if [ -n "${EXTRA_OVERRIDES:-}" ]; then read -r -a more <<< "${EXTRA_OVERRIDES}"; EXTRA+=("${more[@]}"); fi
if [ -n "${INIT_FROM_ID:-}" ]; then
  mapfile -t found < <(ls -d "$CKPT"/*/"$INIT_FROM_ID" 2>/dev/null)
  if [ "${#found[@]}" -ne 1 ] || [ ! -f "${found[0]}/last.pt" ]; then
    echo "stage-1 checkpoint for ${INIT_FROM_ID}: ${found[*]:-none}" >&2; exit 3
  fi
  EXTRA+=("init_from=${found[0]}/last.pt")
fi
STOP=(timeout --signal=TERM --kill-after=120 "$(awk "BEGIN{print int(${STOP_HOURS} * 3600)}")")
echo "host $(hostname)  tree $TREE  commit $SYNCED  $(date)"
echo "resume_id $RESUME_ID  arm $ARM  loss $LOSS  weighting $WEIGHTING  seed $SEED  epochs $MAX_EPOCHS  stop ${STOP_HOURS} h  batch $BATCH  workers 2  extra ${EXTRA[*]}  tags $TAGS"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
rc=0
"${STOP[@]}" "$PY" experiments/043-expression-proteome-morphology/scripts/train_043.py \
  data.store_version=v2 resume_id="$RESUME_ID" \
  arm="$ARM" loss="$LOSS" weighting="$WEIGHTING" seed="$SEED" \
  data.batch_size="$BATCH" data.num_workers=2 data.sampler="$SAMPLER" data.mark_cap=null \
  budget.max_epochs="$MAX_EPOCHS" budget.time_budget_hours=null \
  "${EXTRA[@]}" wandb.mode=offline "wandb.tags=[${TAGS}]" wandb.round="$ROUND" \
  wandb.label_knobs=true || rc=$?
echo "done $(date)  exit ${rc} (124 = stopped at STOP_HOURS: PARTIAL)"
exit "$rc"
