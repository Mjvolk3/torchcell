#!/bin/bash
# experiments/040-inhibitor-synergy-wetlab/conf/mixture/r3_submit_delta.sh
#
# ROUND 3 SUBMISSION, written by scripts/r3_sweep.py. One GPU and one
# process per job, no dependencies and no -preempt partition, so every job
# is eligible the moment it is submitted.
#
#   ssh delta
#   cd /scratch/bbub/mjvolk3/torchcell-040
#   mkdir -p experiments/040-inhibitor-synergy-wetlab/slurm/output
#   bash experiments/040-inhibitor-synergy-wetlab/conf/mixture/r3_submit_delta.sh
#
# An account at its limit: rerun with ACCOUNT=<other>-delta-gpu, which sets
# the matching QOS as well (an account change alone leaves the old QOS and
# the job is rejected as InvalidQOS).
set -eu
ACCOUNT="${ACCOUNT:-bfjt-delta-gpu}"
cd /scratch/bbub/mjvolk3/torchcell-040
mkdir -p experiments/040-inhibitor-synergy-wetlab/slurm/output

sbatch --job-name=040-r3_van_e150 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_van_e150.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhop_e150 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhop_e150.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophetwil_e150_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophetwil_e150_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophetwil_e150_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophetwil_e150_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanwil_e150 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanwil_e150.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e300_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e300_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e300_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e300_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e300_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e300_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhop_e50 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhop_e50.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150_attr --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150_attr.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150_head --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150_head.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150_attr_head --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150_attr_head.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150_aw0p3 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150_aw0p3.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150_cb96_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150_cb96_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150_cb96_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150_cb96_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhop_e150_film_dose --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhop_e150_film_dose.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150_film_dose --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150_film_dose.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhop_e150_film --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhop_e150_film.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_van_e150_gene --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_van_e150_gene.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhop_e150_gene --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhop_e150_gene.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150_gene --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150_gene.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150_L2_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150_L2_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150_L2_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150_L2_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150_cgt2_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150_cgt2_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150_cgt2_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150_cgt2_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150_wide_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150_wide_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhophet_e150_wide_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhophet_e150_wide_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_van_e150_fs1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_van_e150_fs1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_van_e150_fs2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_van_e150_fs2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhop_e150_fs1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhop_e150_fs1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r3_vanhop_e150_fs2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r3_vanhop_e150_fs2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

