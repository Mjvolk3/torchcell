#!/bin/bash
# experiments/043-expression-proteome-morphology/scripts/alcf/aurora_relaunch_cell_043.sh
# [[experiments.043-expression-proteome-morphology.scripts.alcf.aurora_relaunch_cell_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/aurora_relaunch_cell_043.sh
#
# Restart one cell of aurora_grid_043.tsv on its tile inside the node of its running grid
# job, after the cell's process died (the other cells of the node keep running). Run ON the
# compute node (ssh <exec_host> from an Aurora login node):
#
#   nohup bash aurora_relaunch_cell_043.sh <cell id> > /dev/null 2>&1 &
#
# It resumes from the cell's last.pt when one exists (same resume_id); the output appends
# to the cell's grid log.
set -uo pipefail
cell=$1
HERE=/flare/GenesisHackathonOct26/mjvolk3/torchcell/experiments/043-expression-proteome-morphology/scripts/alcf
# shellcheck disable=SC1091
source "$HERE/aurora_env_043.sh"
export PY TRAIN
OUT=$TC_TREE/experiments/043-expression-proteome-morphology/slurm/output/aurora_grid
row=$(grep -v '^#' "$HERE/aurora_grid_043.tsv" | awk -F'\t' -v c="$cell" '$3 == c')
[ -n "$row" ] || { echo "no cell $cell in aurora_grid_043.tsv" >&2; exit 2; }
IFS=$'\t' read -r node tile _ finetune overrides <<< "$row"
echo "[relaunch] $cell on tile $tile of $(hostname) (node slot $node) at $(date); tree $SYNCED" >> "$OUT/$cell.log"
# shellcheck disable=SC2086
ZE_AFFINITY_MASK=$tile bash "$HERE/aurora_run_cell_043.sh" "$cell" "$finetune" $overrides >> "$OUT/$cell.log" 2>&1
