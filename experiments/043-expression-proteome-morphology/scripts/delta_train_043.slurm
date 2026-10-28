#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/delta_train_043.slurm
# [[experiments.043-expression-proteome-morphology.scripts.delta_train_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/delta_train_043.slurm
#
# ONE 043 TRAINING CELL ON ONE DELTA A40. The Delta twin of gh_train_043_sweep_v2.slurm:
# the same env knobs and the same trainer call (store v2, fixed epochs, a `timeout` at
# STOP_HOURS, RESUME_ID naming the checkpoint directory and the W&B run). What differs is the
# machine:
#   * the code tree is a plain rsync of the 043 worktree on /scratch (no .git; the synced
#     commit is in <tree>/.synced_commit),
#   * DATA_ROOT is the Delta scratch tree, holding ONLY what a stage-1 or fine-tune run opens:
#     train_043_prep/prep_v4_route{A,B}, representation_043_v2 (vocab_<enc>.npy),
#     ontology_043_v2, and data/scerevisiae/isolate_embeddings/peter2018_1011/alleles/*.h5
#     (raw_043_v2 is read by --stage prepare only, never at train time, so it is not on Delta),
#   * the interpreter is the torchcell043 env, packed as /scratch/bbub/mjvolk3/envs/torchcell043.tar
#     and unpacked to node-local /tmp per job, and the code + inputs are staged there too
#     (STAGE=1, default); outputs stay under DATA_ROOT on Lustre,
#   * W&B settings come from <tree>/.env (online; auth from ~/.netrc).
#
#   ssh delta
#   cd /scratch/bbub/mjvolk3/torchcell-043
#   sbatch --job-name=043-d-<cell> \
#     --export=ALL,RESUME_ID=<id>,LOSS=pinball,PREP_VERSION=prep_v4,CONDITIONING=ontology,INJECT=operator \
#     experiments/043-expression-proteome-morphology/scripts/delta_train_043.slurm
#
# Smoke (one epoch, 40 min, interactive partition):
#   sbatch -p gpuA40x4-interactive --time=01:00:00 --job-name=043-d-smoke \
#     --export=ALL,RESUME_ID=delta_smoke_043_<date>,MAX_EPOCHS=1,STOP_HOURS=0.8,LOSS=pinball,PREP_VERSION=prep_v4,CONDITIONING=ontology,INJECT=operator,ROUND=delta_smoke \
#     experiments/043-expression-proteome-morphology/scripts/delta_train_043.slurm
#
# Environment knobs (--export cannot carry a comma inside one value), as the GilaHyper twin:
#   RESUME_ID (required)  ARM (all)  LOSS (hlgauss)  WEIGHTING (equal)  SEED (0)
#   SAMPLER (records)  MAX_EPOCHS (50)  STOP_HOURS (23.5)  WINDOW_START_FRAC (0.8)  BATCH (16)
#   LAYER_PATTERN  HIDDEN  FFN_MULT  HEAD_HIDDEN  TRANSFORM  STATS_BY  WEIGHT_BY  SCHEDULE
#   LR  WARMUP_STEPS  PREP_VERSION  CONDITIONING  DROPOUT  TOKEN_DROPOUT  WEIGHT_DECAY
#   SIGMA_BINS  BALANCE_POWER  MAX_VISITS  INJECT  EXTRA_OVERRIDES (space-separated hydra)
#   INIT_FROM_ID: stage 2; the stage-1 RESUME_ID, whose W&B run must have logged
#     val/kemmeren/loss and whose last.pt must be under the Delta checkpoint root (a stage 1
#     trained on GilaHyper has to be rsynced here first).
#   ROUND (wandb.round, default delta_043_2026_10_28)  TAGS (comma-free extra tag)
#   NUM_WORKERS (4): the trainer's loader uses the default fork context, so a worker does
#     NOT re-import torch from Lustre (the 019 spawn cost in delta_grid_common.sh does not
#     apply); its workers are rebuilt each epoch (persistent_workers=False in train_043.py).
#     Set NUM_WORKERS=0 if the first-epoch timing in the log shows a loader stall.
#
# ACCOUNT bfjt-delta-gpu (override with -A). NEVER a -preempt partition. slurm does not create
# the --output directory: it is created in the tree at setup and must exist before submitting.
#SBATCH --account=bfjt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#SBATCH --job-name=043-delta
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/scratch/bbub/mjvolk3/torchcell-043/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
#SBATCH --error=/scratch/bbub/mjvolk3/torchcell-043/experiments/043-expression-proteome-morphology/slurm/output/%x_%j.out
set -euo pipefail

# Absolute, never SLURM_SUBMIT_DIR: `ssh delta 'sbatch ...'` submits from $HOME.
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/bbub/mjvolk3/torchcell-043}"
# The env is ONE tarball, unpacked to node-local /tmp per job (the 040 torchcell-r11.tar
# pattern): Lustre metadata ran ~60 ms per small-file create on 2026-10-28 and an import
# chain off Lustre has cost 30 min before (040). DELTA_PY set at submit time skips the unpack.
ENV_TARBALL="${ENV_TARBALL:-/scratch/bbub/mjvolk3/envs/torchcell043.tar}"
EXP_NAME=043-expression-proteome-morphology

cd "$PROJECT_ROOT"
[[ -f "$PROJECT_ROOT/.env" ]] || { echo "ERROR: missing $PROJECT_ROOT/.env" >&2; exit 1; }
set -a; . "$PROJECT_ROOT/.env"; set +a
export DATA_ROOT="${DELTA_DATA_ROOT:-/scratch/bbub/mjvolk3/torchcell}"
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export PYTHONPATH="$PROJECT_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT=600
export WANDB_ENTITY="${WANDB_ENTITY:-zhao-group}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$DATA_ROOT/wandb-experiments/$EXP_NAME"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# nothing under the (full) Delta home: caches go to the tree
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
mkdir -p "$WANDB_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
# The tree has no .git, and train_043.py records `git rev-parse --short HEAD` (config_dump,
# check=True). A shim answers exactly that from .synced_commit and refuses anything else.
SHIM="$PROJECT_ROOT/.delta_bin"
mkdir -p "$SHIM"
cat > "$SHIM/git" <<EOS
#!/bin/bash
c=\$(cat "$PROJECT_ROOT/.synced_commit")
case "\$*" in
  "rev-parse --short HEAD") echo "\${c:0:9}" ;;
  "rev-parse HEAD") echo "\$c" ;;
  *) echo "git shim ($PROJECT_ROOT has no .git): only rev-parse [--short] HEAD" >&2; exit 2 ;;
esac
EOS
chmod +x "$SHIM/git"
export PATH="$SHIM:$PATH"

PROJECT=torchcell_043-expression-proteome-morphology
CKPT="$DATA_ROOT/experiments/$EXP_NAME/checkpoints"
: "${RESUME_ID:?set RESUME_ID (checkpoint dir and W&B run id)}"
ARM="${ARM:-all}"
LOSS="${LOSS:-hlgauss}"
WEIGHTING="${WEIGHTING:-equal}"
SEED="${SEED:-0}"
SAMPLER="${SAMPLER:-records}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
STOP_HOURS="${STOP_HOURS:-23.5}"
BATCH="${BATCH:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ROUND="${ROUND:-delta_043_2026_10_28}"
TAGS="v2_store,delta,${ROUND}${TAGS:+,${TAGS}}"
PREP="${PREP_VERSION:-prep_v3}"
ROUTE=A
EXTRA=("budget.registered_window.start_frac=${WINDOW_START_FRAC:-0.8}")
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
if [ -n "${PREP_VERSION:-}" ]; then EXTRA+=("data.prep_version=${PREP_VERSION}"); fi
if [ -n "${CONDITIONING:-}" ]; then EXTRA+=("model.conditioning.source=${CONDITIONING}"); fi
if [ -n "${INJECT:-}" ]; then EXTRA+=("model.conditioning.inject=${INJECT}"); fi
if [ -n "${EXTRA_OVERRIDES:-}" ]; then
  read -r -a more <<< "${EXTRA_OVERRIDES}"
  EXTRA+=("${more[@]}")
  for o in "${more[@]}"; do
    if [[ "$o" == data.route=* ]]; then ROUTE="${o#data.route=}"; fi
  done
fi

echo "=========================================================="
echo "host        : $(hostname)   job ${SLURM_JOB_ID:-none}   $(date -Is)"
echo "project     : $PROJECT_ROOT   synced commit $(git rev-parse HEAD)"
echo "DATA_ROOT   : $DATA_ROOT"
echo "env         : ${DELTA_PY:-$ENV_TARBALL (unpacked to node-local /tmp)}"
echo "wandb       : mode $WANDB_MODE  entity $WANDB_ENTITY  dir $WANDB_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "=========================================================="

# ---- PREFLIGHT: the tree, the env, and every input the trainer opens. ----
if [ -z "${DELTA_PY:-}" ]; then
  [[ -f "$ENV_TARBALL" ]] || { echo "ERROR: env tarball missing: $ENV_TARBALL" >&2; exit 1; }
fi
[[ -f "$PROJECT_ROOT/.synced_commit" ]] || { echo "ERROR: $PROJECT_ROOT/.synced_commit missing" >&2; exit 1; }
E043="$DATA_ROOT/data/torchcell/experiments/$EXP_NAME"
PREP_DIR="$E043/train_043_prep/${PREP}_route${ROUTE}"
REQUIRED=(
  "$PREP_DIR/meta.json"
  "$PREP_DIR/store.npz"
  "$PROJECT_ROOT/experiments/$EXP_NAME/conf/splits/global_split_v2.json"
)
for enc in fudt_species_upstream fudt_species_downstream prot_t5_xl_uniref50 calm; do
  REQUIRED+=("$E043/representation_043_v2/vocab_${enc}.npy")
  REQUIRED+=("$DATA_ROOT/data/scerevisiae/isolate_embeddings/peter2018_1011/alleles/${enc}.h5")
done
for f in "${REQUIRED[@]}"; do
  [[ -f "$f" ]] || { echo "ERROR: required file missing: $f" >&2; exit 1; }
done

# ---- STAGE to node-local /tmp: the env, the code the trainer imports, and its inputs. ----
# Outputs (checkpoints, hydra, W&B) stay under DATA_ROOT on Lustre; inputs are read through
# the trainer's paths.* overrides, so DATA_ROOT itself is unchanged. STAGE=0 reads the code
# and the inputs from Lustre in place (the env still comes from the tarball unless DELTA_PY).
STAGE="${STAGE:-1}"
LOCAL="/tmp/043_${SLURM_JOB_ID:-stage}"
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
t0=$(date +%s)
if [ -z "${DELTA_PY:-}" ]; then
  mkdir -p "$LOCAL/env" && tar xf "$ENV_TARBALL" -C "$LOCAL/env"
  DELTA_PY="$LOCAL/env/torchcell043/bin/python"
  echo "staged env in $(( $(date +%s) - t0 )) s"
fi
[[ -x "$DELTA_PY" ]] || { echo "ERROR: interpreter not found: $DELTA_PY" >&2; exit 1; }
RUN_ROOT="$PROJECT_ROOT"
if [ "$STAGE" = "1" ]; then
  t1=$(date +%s)
  mkdir -p "$LOCAL/tree" "$LOCAL/in/prep" "$LOCAL/in/representation" "$LOCAL/in/alleles"
  tar cf - -C "$PROJECT_ROOT" --exclude=__pycache__ torchcell scripts .synced_commit \
    "experiments/$EXP_NAME/scripts" "experiments/$EXP_NAME/conf" | tar xf - -C "$LOCAL/tree"
  cp "$PREP_DIR"/* "$LOCAL/in/prep/"
  cp "$E043"/representation_043_v2/vocab_*.npy "$LOCAL/in/representation/"
  cp "$DATA_ROOT"/data/scerevisiae/isolate_embeddings/peter2018_1011/alleles/*.h5 "$LOCAL/in/alleles/"
  RUN_ROOT="$LOCAL/tree"
  EXTRA+=("paths.prep_dir=$LOCAL/in/prep" "paths.representation_root=$LOCAL/in/representation"
          "paths.alleles_root=$LOCAL/in/alleles")
  echo "staged code + inputs in $(( $(date +%s) - t1 )) s ($(du -sh "$LOCAL" | cut -f1) on $(df -h /tmp | tail -1 | awk '{print $1, $4" free"}'))"
fi
cd "$RUN_ROOT"
export PYTHONPATH="$RUN_ROOT"
# The top-level package is not enough: an env with an editable install of another checkout
# answers for any torchcell.X that PYTHONPATH cannot resolve (smoke 22787137, 040 twin).
"$DELTA_PY" -c "import sys, torchcell, torchcell.data, torchcell.models, torchcell.models.cell_model_043, torchcell.losses; mods = [torchcell, torchcell.data, torchcell.models, torchcell.models.cell_model_043, torchcell.losses]; bad = [m.__name__ for m in mods if not (m.__file__ or '').startswith('$RUN_ROOT')]; print('\n'.join(m.__name__ + ' from ' + str(m.__file__) for m in mods)); sys.exit('ERROR: resolved outside the tree: ' + ', '.join(bad)) if bad else None"
"$DELTA_PY" -c "import torch, torch_geometric, torch_scatter; print('torch', torch.__version__, 'pyg', torch_geometric.__version__, 'torch_scatter', torch_scatter.__version__)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

if [ -n "${INIT_FROM_ID:-}" ]; then
  "$DELTA_PY" scripts/check_wandb_run_keys.py "${WANDB_ENTITY}/${PROJECT}/${INIT_FROM_ID}" \
    --require val/kemmeren/loss --min-rows 2
  mapfile -t found < <(ls -d "$CKPT"/*/"$INIT_FROM_ID" 2>/dev/null)
  if [ "${#found[@]}" -ne 1 ] || [ ! -f "${found[0]}/last.pt" ]; then
    echo "stage-1 checkpoint for ${INIT_FROM_ID}: ${found[*]:-none}" >&2; exit 3
  fi
  EXTRA+=("init_from=${found[0]}/last.pt")
fi

STOP=(timeout --signal=TERM --kill-after=120 "$("$DELTA_PY" -c "print(int(${STOP_HOURS} * 3600))")")
echo "resume_id $RESUME_ID  arm $ARM  loss $LOSS  weighting $WEIGHTING  seed $SEED  epochs $MAX_EPOCHS  stop ${STOP_HOURS} h  batch $BATCH  workers $NUM_WORKERS  extra ${EXTRA[*]}  tags $TAGS"
rc=0
"${STOP[@]}" "$DELTA_PY" "experiments/$EXP_NAME/scripts/train_043.py" \
  data.store_version=v2 resume_id="$RESUME_ID" \
  arm="$ARM" loss="$LOSS" weighting="$WEIGHTING" seed="$SEED" \
  data.batch_size="$BATCH" data.num_workers="$NUM_WORKERS" data.sampler="$SAMPLER" data.mark_cap=null \
  budget.max_epochs="$MAX_EPOCHS" budget.time_budget_hours=null \
  "${EXTRA[@]}" "wandb.tags=[${TAGS}]" wandb.round="$ROUND" wandb.mode="$WANDB_MODE" \
  wandb.label_knobs=true || rc=$?
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true
echo "done $(date -Is)  exit ${rc} (124 = stopped at STOP_HOURS: PARTIAL)"
exit "$rc"
