#!/bin/bash
# experiments/021-ozaydin-beta-carotene/scripts/delta_grid_beta_carotene.slurm
# [[experiments.021-ozaydin-beta-carotene]]
#
# DELTA EXPERIMENT 4 of 4 -- MAX OUT BETA-CAROTENE (Ozaydin 2013 colony-colour visual score).
# One A40x4 node, 48 h, four workers draining one shared work queue.
#
#   cd /projects/bbub/mjvolk3/torchcell
#   mkdir -p experiments/021-ozaydin-beta-carotene/{slurm/output,optuna,results}
#   sbatch --account=bbtp-delta-gpu \
#     experiments/021-ozaydin-beta-carotene/scripts/delta_grid_beta_carotene.slurm
#
# THE mkdir IS NOT OPTIONAL -- slurm cannot create the --output directory, and a missing one
# kills the job at t=0 with no log anywhere.
#
# THIS IS THE ARM WHERE REPLICATION MATTERS MOST. `_002` measured sigma = 0.025 from an
# identical-config, identical-seed pair (0.1993 / 0.1494) while its reported best of 0.2231
# came from a configuration contradicting every other arm's winner; `_003` then had the CONTROL
# embedding top its screen. Both are what an unreplicated search produces on a noisy arm. The
# deliverable here is a number WITH an error bar, which this arm has never had.
#
# RANKED ON SPEARMAN, not Pearson: the readout is a subjective ordinal on -5..+5, so Pearson
# would assert an interval scale the measurement does not have. Ceiling ~0.54 rank agreement
# -- report against that, never against betaxanthin's 0.914.
#
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=180g
#SBATCH --time=2-00:00:00
#SBATCH --job-name=021-bc-grid
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/projects/bbub/mjvolk3/torchcell/experiments/021-ozaydin-beta-carotene/slurm/output/%x_%j.out
#SBATCH --error=/projects/bbub/mjvolk3/torchcell/experiments/021-ozaydin-beta-carotene/slurm/output/%x_%j.out

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-/projects/bbub/mjvolk3/torchcell}/experiments/019-simb-multimodal/scripts/delta_grid_common.sh"

#            experiment dir               arm(s)          base config                wall
run_grid_job 021-ozaydin-beta-carotene   beta_carotene   delta_grid_beta_carotene   172800
