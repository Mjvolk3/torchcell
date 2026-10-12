#!/bin/bash
# experiments/040-inhibitor-synergy-wetlab/conf/mixture/r5_submit_delta.sh
#
# ROUND 5 SUBMISSION, written by scripts/r3_sweep.py. One GPU and one
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
#   bash experiments/040-inhibitor-synergy-wetlab/conf/mixture/r5_submit_delta.sh
#
# An account at its limit: rerun with ACCOUNT=<other>-delta-gpu, which sets
# the matching QOS as well (an account change alone leaves the old QOS and
# the job is rejected as InvalidQOS).
set -eu
ACCOUNT="${ACCOUNT:-bfjt-delta-gpu}"
cd /scratch/bbub/mjvolk3/torchcell-040
mkdir -p experiments/040-inhibitor-synergy-wetlab/slurm/output

sbatch --job-name=040-r5_van_e150_fs1_s12 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_van_e150_fs1_s12.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_van_e150_fs2_s12 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_van_e150_fs2_s12.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhop_e150_fs1_s12_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhop_e150_fs1_s12_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhop_e150_fs1_s12_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhop_e150_fs1_s12_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhop_e150_fs2_s12_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhop_e150_fs2_s12_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhop_e150_fs2_s12_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhop_e150_fs2_s12_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhophet_e150_fs1_s12_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhophet_e150_fs1_s12_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhophet_e150_fs1_s12_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhophet_e150_fs1_s12_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhophet_e150_fs1_s12_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhophet_e150_fs1_s12_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhophet_e150_fs2_s12_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhophet_e150_fs2_s12_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhophet_e150_fs2_s12_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhophet_e150_fs2_s12_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhophet_e150_fs2_s12_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhophet_e150_fs2_s12_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_van_e150_do0p4 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_van_e150_do0p4.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_van_e150_wd1e3 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_van_e150_wd1e3.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_van_e150_do0p4_wd1e3 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_van_e150_do0p4_wd1e3.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_van_e150_lr5e4 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_van_e150_lr5e4.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_van_e150_lr2e3 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_van_e150_lr2e3.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_van_e150_dim32 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_van_e150_dim32.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_van_e150_nval8 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_van_e150_nval8.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_van_e50_last --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_van_e50_last.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhop_e150_aw3_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhop_e150_aw3_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhop_e150_aw3_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhop_e150_aw3_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhop_e150_aw3_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhop_e150_aw3_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhop_e150_aw3_cb96_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhop_e150_aw3_cb96_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhop_e150_aw3_cb96_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhop_e150_aw3_cb96_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r5_vanhop_e150_aw3_cb96_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r5_vanhop_e150_aw3_cb96_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

