#!/bin/bash
# experiments/020-cachera-betaxanthin/scripts/delta_grid_betaxanthin.slurm
# [[experiments.020-cachera-betaxanthin]]
#
# DELTA EXPERIMENT 1 of 4 -- MAX OUT BETAXANTHIN ON MERZBACHER'S OWN TEST SPLIT.
# One A40x4 node, 48 h, four workers draining one shared work queue.
#
#   cd /projects/bbub/mjvolk3/torchcell
#   mkdir -p experiments/020-cachera-betaxanthin/{slurm/output,optuna,results}
#   sbatch --account=bbtp-delta-gpu \
#     experiments/020-cachera-betaxanthin/scripts/delta_grid_betaxanthin.slurm
#
# THE mkdir IS NOT OPTIONAL. `--output` points into slurm/output/, which is gitignored and so
# does NOT exist on a fresh clone. Slurm will not create it: the job dies at t=0 with NO LOG
# ANYWHERE. The `mkdir -p` inside this script cannot save you -- by the time the script runs,
# slurm has already failed to open the output file.
#
# WHAT RUNS: 8 fixed settings (a 2x2x2 factorial) x as many seeds as the wall clock buys,
# enqueued ROUND-MAJOR so a wall-clock kill leaves every setting with the same seed count.
# The design, and the noise arithmetic that forces it, are in
# `experiments/019-simb-multimodal/scripts/metabolism_grid_runner.py`. In one line: the arm's
# replicate sigma is 0.030 and the top five configs of the last search spanned 0.021, so more
# SEARCH cannot help and only REPLICATION can.
#
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=180g
#SBATCH --time=2-00:00:00
#SBATCH --job-name=020-bx-grid
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/020-cachera-betaxanthin/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/020-cachera-betaxanthin/slurm/output/%x_%j.out

set -euo pipefail

# SLURM_SUBMIT_DIR, not BASH_SOURCE: slurm copies the submitted script into a spool directory,
# so inside a running job BASH_SOURCE does not point at the repo. Submit from the repo root.
source "${SLURM_SUBMIT_DIR:-/projects/bbub/mjvolk3/torchcell}/experiments/019-simb-multimodal/scripts/delta_grid_common.sh"

#            experiment dir            arm(s)        base config              wall seconds
run_grid_job 020-cachera-betaxanthin  betaxanthin   delta_grid_betaxanthin   172800
