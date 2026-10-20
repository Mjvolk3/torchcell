#!/bin/bash
# experiments/040-inhibitor-synergy-wetlab/conf/mixture/r9_submit_delta.sh
#
# ROUND 9 SUBMISSION, written by scripts/r3_sweep.py. One GPU and one
# process per job, no dependencies and no -preempt partition, so every job
# is eligible the moment it is submitted.
#
# THE SECOND WAVE. The learning-curve arms (r3w2_*_n5, _n10) run a code path
# added after the first wave was rsynced (n_fit_compounds under fit_on: train),
# so re-rsync the tree before this script: anchored excludes, NO delete flags.
#
#   ssh delta
#   cd /scratch/bbub/mjvolk3/torchcell-040
#   mkdir -p experiments/040-inhibitor-synergy-wetlab/slurm/output
#   bash experiments/040-inhibitor-synergy-wetlab/conf/mixture/r9_submit_delta.sh
#
# An account at its limit: rerun with ACCOUNT=<other>-delta-gpu, which sets
# the matching QOS as well (an account change alone leaves the old QOS and
# the job is rejected as InvalidQOS).
set -eu
ACCOUNT="${ACCOUNT:-bfjt-delta-gpu}"
cd /scratch/bbub/mjvolk3/torchcell-040
mkdir -p experiments/040-inhibitor-synergy-wetlab/slurm/output

sbatch --job-name=040-r9_van_e150_cell_sum --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r9_van_e150_cell_sum.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r9_van_e150_cell_bliss --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r9_van_e150_cell_bliss.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r9_van_e150_cell_anchors_bliss --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r9_van_e150_cell_anchors_bliss.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r9_van_e150_cell_bliss_film --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r9_van_e150_cell_bliss_film.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r9_vanhophet_e150_cell_bliss --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r9_vanhophet_e150_cell_bliss.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r9_van_e150_cell_lh0p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r9_van_e150_cell_lh0p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r9_van_e150_cell_s345_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r9_van_e150_cell_s345_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r9_van_e150_cell_s345_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r9_van_e150_cell_s345_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

