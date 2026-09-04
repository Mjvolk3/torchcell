#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/delta_bxfx_canary.slurm
# [[experiments.027-betaxanthin-metabolic]]
#
# RUN THIS FIRST. ONE GPU, ONE HOUR, AND IT IS NOT OPTIONAL.
#
# The 9-node grid's wall clock is extrapolated from a GilaHyper measurement (51.4 s/epoch,
# median over 98 completed 026 cells, num_workers=3 on an RTX 6000 Ada). Delta is a different
# card AND runs num_workers=0 off a parallel filesystem, and that second difference is exactly
# the one that has burned this project before: the 2026-07-28 Delta smoke test sat in Sanity
# Checking for 38 minutes because spawn re-imported the whole stack per worker off Lustre.
#
# This job replaces the extrapolation with a number. It trains 10 epochs of `pooled` and 10 of
# `flux_anchored` at one seed and prints seconds/epoch for each. Set the grid's `--time` from
# what it reports; the formula is in README.md, "Node-hour estimate".
#
# It also exercises every path the grid needs: the branch has flux_layer.py, the Open Enzyme
# Database mirror is present and resolves a non-zero experimental kcat fraction, the pinned
# Merzbacher split resolves, the test pass runs, and the per-gene dump writes.
#
#   ssh mjvolk3@login.delta.ncsa.illinois.edu
#   cd /projects/bbub/mjvolk3/torchcell
#   mkdir -p experiments/027-betaxanthin-metabolic/{slurm/output,optuna,results}
#   sbatch experiments/027-betaxanthin-metabolic/scripts/delta_bxfx_canary.slurm
#
# THE mkdir IS NOT OPTIONAL. `--output` points into slurm/output/, which is gitignored and so
# does NOT exist on a fresh clone. Slurm will not create it: the job dies at t=0 with NO LOG
# ANYWHERE, and the mkdir inside the script cannot save you -- by then slurm has already
# failed to open the output file.
#
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfjt-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --job-name=027-bxfx-canary
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#
# ACCOUNT: bfjt-delta-gpu, 12,797 hours as of 2026-09-03. It is in the header rather than
# left to the submit line on purpose -- every older Delta launcher in this repo says
# `sbatch --account=bbtp-delta-gpu`, and bbtp NO LONGER EXISTS on this allocation. Override
# at submit time with `sbatch --account=bbub-delta-gpu` (3,851 h) if bfjt is needed elsewhere.

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-/projects/bbub/mjvolk3/torchcell}/experiments/027-betaxanthin-metabolic/scripts/delta_bxfx_common.sh"

bxfx_env
bxfx_preflight

CANARY_EPOCHS="${CANARY_EPOCHS:-10}"
for ARM in pooled flux_anchored; do
  echo "---------- canary: $ARM, $CANARY_EPOCHS epochs ----------"
  START=$(date +%s)
  "$DELTA_PY" -u "$TRAINER" \
    --arm "$ARM" --seed "$GRID_SEED_BASE" --epochs "$CANARY_EPOCHS" \
    --out "canary_${ARM}.json"
  END=$(date +%s)
  # Wall/epoch INCLUDES the one-time dataset open and the test pass, so it OVERSTATES the
  # steady-state cost. The per-cell `wall_time_s` inside the results JSON is the loop only
  # and is what the grid budget should use; this line is the quick read in the log.
  echo "CANARY $ARM: $((END - START)) s total for $CANARY_EPOCHS epochs (incl. setup + test)"
done

echo "=========================================================="
echo "Steady-state seconds/epoch, from the results JSON (use THIS for the grid budget):"
"$DELTA_PY" - <<'PY'
import json, os, os.path as osp
d = osp.join(os.environ["EXPERIMENT_ROOT"], "027-betaxanthin-metabolic", "results")
for arm in ("pooled", "flux_anchored"):
    p = osp.join(d, f"canary_{arm}.json")
    r = json.load(open(p))
    n = len(r["history"])
    print(
        f"  {arm:14s} {r['wall_time_s'] / n:7.2f} s/epoch over {n} epochs   "
        f"select_epoch={r['select_epoch']} ({r['select_rule']})   "
        f"pinned test spearman {r['test']['pinned']['spearman']:+.4f} "
        f"(n={r['test']['pinned']['n']})"
    )
    if r["flux_meta"]:
        print(f"                 kcat experimental fraction "
              f"{r['flux_meta']['kcat_experimental_fraction']:.4f} (GilaHyper: 0.0397)")
PY
echo "=========================================================="
echo "GRID BUDGET: seconds_per_epoch * 200 epochs * 1.44 (2 runs/GPU) * cells_per_worker"
echo "  cells_per_worker = GRID_ROUNDS / (GRID_SHARD_COUNT / n_arms) = 24 / 12 = 2"
echo "Round UP to the next hour and set --time on delta_bxfx_grid.slurm."
echo "=========================================================="
