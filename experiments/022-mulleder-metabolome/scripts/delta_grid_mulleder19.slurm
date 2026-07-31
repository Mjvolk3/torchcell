#!/bin/bash
# experiments/022-mulleder-metabolome/scripts/delta_grid_mulleder19.slurm
# [[experiments.022-mulleder-metabolome]]
#
# DELTA EXPERIMENT 2 of 4 -- MAX OUT THE 19-AMINO-ACID METABOLOME (Mulleder 2016).
# One A40x4 node, 48 h. Eight long runs (2 per GPU) draining one shared work queue.
#
#   cd /projects/bbub/mjvolk3/torchcell
#   mkdir -p experiments/022-mulleder-metabolome/{slurm/output,optuna,results}
#   sbatch --account=bbtp-delta-gpu \
#     experiments/022-mulleder-metabolome/scripts/delta_grid_mulleder19.slurm
#
# THE mkdir IS NOT OPTIONAL -- slurm cannot create the --output directory, and a missing one
# kills the job at t=0 with no log anywhere.
#
# The design (8 fixed settings x seed rounds until the deadline) and the noise arithmetic
# behind it: `experiments/019-simb-multimodal/scripts/metabolism_grid_runner.py`. This is the
# QUIETEST of the four arms -- F = 19 features, so the metric averages its own noise down --
# which means the seeds here buy resolution on ~0.01 gaps rather than rescuing an unreadable
# maximum.
#
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=180g
#SBATCH --time=2-00:00:00
#SBATCH --job-name=022-m19-grid
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/022-mulleder-metabolome/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/022-mulleder-metabolome/slurm/output/%x_%j.out

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-/projects/bbub/mjvolk3/torchcell}/experiments/019-simb-multimodal/scripts/delta_grid_common.sh"

#            experiment dir             arm(s)       base config             wall seconds
run_grid_job 022-mulleder-metabolome   mulleder19   delta_grid_mulleder19   172800
