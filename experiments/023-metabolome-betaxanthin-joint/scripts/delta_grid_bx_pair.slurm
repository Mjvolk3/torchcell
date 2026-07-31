#!/bin/bash
# experiments/023-metabolome-betaxanthin-joint/scripts/delta_grid_bx_pair.slurm
# [[experiments.023-metabolome-betaxanthin-joint]]
#
# DELTA EXPERIMENT 3 of 4 -- DOES THE METABOLOME IMPROVE BETAXANTHIN PREDICTION?
# One A40x4 node, 48 h. TWO arms, four long runs each (2 per GPU), in ONE job.
#
#   cd /projects/bbub/mjvolk3/torchcell
#   mkdir -p experiments/023-metabolome-betaxanthin-joint/{slurm/output,optuna,results}
#   sbatch --account=bbtp-delta-gpu \
#     experiments/023-metabolome-betaxanthin-joint/scripts/delta_grid_bx_pair.slurm
#
# THE mkdir IS NOT OPTIONAL -- slurm cannot create the --output directory, and a missing one
# kills the job at t=0 with no log anywhere.
#
# ONE JOB, BOTH ARMS, and that is the design rather than a convenience. `bx_ctrl`
# (betaxanthin alone) and `bx_m19` (betaxanthin + the 19-AA metabolome) run on the same node,
# the same software state, the same queue slot, and the same (setting, seed) cells -- so the
# only difference between them is the one under test. Splitting a paired comparison across
# two jobs days apart is a strictly worse comparison for no benefit.
#
# 8 runs per round = 4 settings x 2 arms. The third factor's budget is spent on the pairing,
# which is worth far more: `cfg.seed` drives model init AND the CellDataModule split, so
# within a cell both arms see an identical split and identical initialization and
# `bx_m19 - bx_ctrl` is a PAIRED difference -- removing the split/init variance that, at
# sigma = 0.030, dominates any unpaired comparison at this sample size.
#
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=180g
#SBATCH --time=2-00:00:00
#SBATCH --job-name=023-bxm19-grid
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/023-metabolome-betaxanthin-joint/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/023-metabolome-betaxanthin-joint/slurm/output/%x_%j.out

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-/projects/bbub/mjvolk3/torchcell}/experiments/019-simb-multimodal/scripts/delta_grid_common.sh"

#            experiment dir                        arm(s)             base config           wall
run_grid_job 023-metabolome-betaxanthin-joint  "bx_ctrl bx_m19"  delta_grid_bx_pair  172800
