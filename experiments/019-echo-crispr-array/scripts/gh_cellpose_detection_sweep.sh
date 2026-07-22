#!/bin/bash
# experiments/019-echo-crispr-array/scripts/gh_cellpose_detection_sweep.sh
# [[experiments.019-echo-crispr-array.scripts.gh_cellpose_detection_sweep]]
#
# Detection-tuning sweep: fan a small grid of Cellpose recipes across the GPUs, one
# sbatch job per recipe (up to 4 concurrent on the 4x RTX 6000 node). Each recipe
# writes an outline-only QC overlay per condition (run2_cellpose_<tag>_overlay_<g>.png)
# and prints a RECIPE_JSON line with per-plate accepted / multi(M) / neighbour(N) /
# offgrid / mean_size counts.
#
# Targets three failure modes found on visual QC (2026.07.22):
#   * loose boundary "air gap"  -> Otsu size-tightening (now default in cellpose_seg)
#   * missed faint/edge colonies-> lower cellprob, stronger CLAHE, higher node_tol,
#                                  wider edge_margin (keeps row A/P edge colonies)
#   * false red (M) on singles  -> multi_min_frac gate (2nd colony must be real)
#
# Usage (from repo root, on the GilaHyper login node):
#   bash experiments/019-echo-crispr-array/scripts/gh_cellpose_detection_sweep.sh "P1_t50,P2_t72"
# then read the overlays + slurm/output/*.out RECIPE_JSON to pick a winner.

set -euo pipefail
CONDS="${1:-P1_t50,P2_t72}"
WRAP="experiments/019-echo-crispr-array/scripts/gh_cellpose_recipe.slurm"

# recipe = "tag|extra recipe.py args"; tighten + multi_min_frac gate on for all.
RECIPES=(
  "s1_base|--contrast clahe --clahe_clip 0.01 --cellprob -4 --node_tol 0.55 --edge_margin 0.5 --multi_min_frac 0.5"
  "s2_recall|--contrast clahe --clahe_clip 0.02 --cellprob -6 --node_tol 0.65 --edge_margin 0.8 --multi_min_frac 0.5"
  "s3_recallx|--contrast clahe --clahe_clip 0.03 --cellprob -6 --node_tol 0.70 --edge_margin 0.9 --multi_min_frac 0.5"
  "s4_edge|--contrast clahe --clahe_clip 0.02 --cellprob -4 --node_tol 0.65 --edge_margin 0.9 --multi_min_frac 0.5"
  "s5_tightcp|--contrast clahe --clahe_clip 0.02 --cellprob -2 --node_tol 0.65 --edge_margin 0.8 --multi_min_frac 0.6"
  "s6_maxrec|--contrast clahe --clahe_clip 0.03 --cellprob -8 --node_tol 0.70 --edge_margin 0.9 --multi_min_frac 0.5"
)

for r in "${RECIPES[@]}"; do
  tag="${r%%|*}"
  args="${r#*|}"
  echo "sbatch $tag: $args --conditions $CONDS"
  sbatch --job-name="019-sweep-${tag}" "$WRAP" \
    $args --conditions "$CONDS" --tag "sweep_${tag}"
done
echo "launched ${#RECIPES[@]} recipes on conditions=$CONDS"
