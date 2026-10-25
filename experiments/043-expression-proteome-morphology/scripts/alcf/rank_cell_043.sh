#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/alcf/rank_cell_043.sh
# [[experiments.043-expression-proteome-morphology.scripts.alcf.rank_cell_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/rank_cell_043.sh
#
# One MPI rank of a prod chunk (polaris_train_043_ontology_prod.pbs): rank r runs row r of
# ontology_grid_043.tsv through run_cell_043.sh on card PALS_LOCAL_RANKID of its node, with
# the same W&B run id as its preemptable and capacity twins.
set -euo pipefail
: "${PALS_RANKID:?run under mpiexec (Cray PALS)}" "${PALS_LOCAL_RANKID:?}"
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
mapfile -t ROWS < <(grep -v '^#' "$ALCF/ontology_grid_043.tsv")
r=$PALS_RANKID
[ "$r" -lt "${#ROWS[@]}" ] || { echo "rank $r: no row"; exit 0; }
IFS=$'\t' read -r cell finetune overrides <<< "${ROWS[$r]}"
export CUDA_VISIBLE_DEVICES=$PALS_LOCAL_RANKID
echo "rank $r -> $cell on $(hostname) card $CUDA_VISIBLE_DEVICES job ${PBS_JOBID:-?} at $(date)" >> "$OUT/$cell.log"
# shellcheck disable=SC2086
exec bash "$ALCF/run_cell_043.sh" "$cell" "$finetune" $overrides >> "$OUT/$cell.log" 2>&1
