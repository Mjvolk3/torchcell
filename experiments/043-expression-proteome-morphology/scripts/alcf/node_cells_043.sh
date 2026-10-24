#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/alcf/node_cells_043.sh
# [[experiments.043-expression-proteome-morphology.scripts.alcf.node_cells_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/node_cells_043.sh
#
# One Polaris node of round ontology_2026_10_23: rows [4 * PER_CARD * SLOT, +4 * PER_CARD) of
# ontology_grid_043.tsv, row r of the slot on card r % 4, each through run_cell_043.sh.
# Run by polaris_train_043_ontology_grid.pbs (preemptable, one node) and, once per node
# through mpiexec, by polaris_train_043_ontology_capacity.pbs (capacity, four nodes). Slot k
# of either route holds the same cells with the same W&B run ids; run_cell_043.sh skips a
# stage whose run is live elsewhere, so a cell never trains twice.
#
#   node_cells_043.sh <slot> [per_card]
set -euo pipefail
SLOT=$1
PER_CARD="${2:-1}"
TC_PREFIX=/eagle/GenesisHackathonOct26/mjvolk3
TC_TREE=$TC_PREFIX/torchcell
export PY=$TC_PREFIX/miniforge3/envs/torchcell/bin/python
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,.alcf.anl.gov
cd "$TC_TREE"
set -a; source .env; set +a
export DATA_ROOT=$TC_PREFIX/torchcell-data/043
export EXPERIMENT_ROOT=$TC_TREE/experiments
export PYTHONUNBUFFERED=1 PYTHONPATH="$TC_TREE"
export WANDB_MODE=online
SHIM=$(mktemp -d)
REAL_GIT=$(command -v git)
SYNCED=$(cut -c1-9 "$TC_TREE/.synced_commit")
cat > "$SHIM/git" <<SH
#!/bin/bash
if [ "\$*" = "rev-parse --short HEAD" ]; then echo "$SYNCED"; exit 0; fi
exec "$REAL_GIT" "\$@"
SH
chmod +x "$SHIM/git"
export PATH="$SHIM:$PATH"

ALCF=experiments/043-expression-proteome-morphology/scripts/alcf
OUT=$TC_TREE/experiments/043-expression-proteome-morphology/slurm/output/onto_grid
mkdir -p "$OUT"
echo "slot $SLOT per_card $PER_CARD on $(hostname) at $(date); job ${PBS_JOBID:-?}; tree $SYNCED"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
mapfile -t ROWS < <(grep -v '^#' "$ALCF/ontology_grid_043.tsv")
per_node=$((4 * PER_CARD))
first=$((SLOT * per_node))
pids=()
for ((i = 0; i < per_node; i++)); do
  r=$((first + i))
  [ "$r" -lt "${#ROWS[@]}" ] || break
  IFS=$'\t' read -r cell finetune overrides <<< "${ROWS[$r]}"
  card=$((i % 4))
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$card bash "$ALCF/run_cell_043.sh" "$cell" "$finetune" $overrides \
    >> "$OUT/$cell.log" 2>&1 &
  pids+=($!)
  echo "card $card: $cell (fine-tune $finetune) pid ${pids[-1]}"
done
( while sleep 300; do date; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; done ) \
  >> "$OUT/nvidia_smi_slot${SLOT}_$(hostname).log" 2>&1 &
MON=$!
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
kill "$MON" 2>/dev/null || true
echo "slot $SLOT done $(date) rc $rc"
exit "$rc"
