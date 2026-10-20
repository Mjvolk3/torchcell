#!/bin/bash
# experiments/040-inhibitor-synergy-wetlab/conf/mixture/r11_submit_delta.sh
#
# ROUND 11 SUBMISSION, written by scripts/r3_sweep.py. One GPU and one
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
#   bash experiments/040-inhibitor-synergy-wetlab/conf/mixture/r11_submit_delta.sh
#
# An account at its limit: rerun with ACCOUNT=<other>-delta-gpu, which sets
# the matching QOS as well (an account change alone leaves the old QOS and
# the job is rejected as InvalidQOS).
set -eu
ACCOUNT="${ACCOUNT:-bfjt-delta-gpu}"
cd /scratch/bbub/mjvolk3/torchcell-040
mkdir -p experiments/040-inhibitor-synergy-wetlab/slurm/output

sbatch --job-name=040-r11_van_e150_cell_bliss_attend --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_attend.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_attend_doff --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_attend_doff.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_attend_co --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_attend_co.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_displace --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_displace.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_displace_doff --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_displace_doff.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_displace_co --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_displace_co.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_displace_co_doff --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_displace_co_doff.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_attend_L2_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_attend_L2_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_attend_L2_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_attend_L2_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_head_hsa --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_head_hsa.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_head_hsa_anchors --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_head_hsa_anchors.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_hill --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_hill.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_hill_anchors --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_hill_anchors.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_hill_loewe --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_hill_loewe.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_hill_loewe_anchors --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_hill_loewe_anchors.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_hill_hsa --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_hill_hsa.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_hill_hsa_anchors --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_hill_hsa_anchors.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_hill_doff --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_hill_doff.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_hill_loewe_doff --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_hill_loewe_doff.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_displace_co_hill_loewe --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_displace_co_hill_loewe.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_dev --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_dev.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_dev_anchors --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_dev_anchors.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_dev_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_dev_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_dev_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_dev_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_dev_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_dev_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_dev_anchors_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_dev_anchors_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_dev_anchors_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_dev_anchors_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_dev_anchors_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_dev_anchors_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophet_e150_cell_bliss_dev_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophet_e150_cell_bliss_dev_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophet_e150_cell_bliss_dev_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophet_e150_cell_bliss_dev_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophet_e150_cell_bliss_dev_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophet_e150_cell_bliss_dev_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophet_e150_cell_bliss_dev_anchors_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophet_e150_cell_bliss_dev_anchors_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophet_e150_cell_bliss_dev_anchors_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophet_e150_cell_bliss_dev_anchors_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophet_e150_cell_bliss_dev_anchors_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophet_e150_cell_bliss_dev_anchors_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_hill_loewe_dev --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_hill_loewe_dev.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_hill_loewe_dev_anchors --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_hill_loewe_dev_anchors.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_hill_loewe_dev_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_hill_loewe_dev_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_hill_loewe_dev_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_hill_loewe_dev_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_hill_loewe_dev_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_hill_loewe_dev_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_hill_loewe_dev_anchors_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_hill_loewe_dev_anchors_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_hill_loewe_dev_anchors_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_hill_loewe_dev_anchors_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_hill_loewe_dev_anchors_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_hill_loewe_dev_anchors_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_doff_heldout_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_doff_heldout_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_doff_heldout_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_doff_heldout_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_doff_heldout_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_doff_heldout_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_hill_doff_heldout_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_hill_doff_heldout_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_hill_doff_heldout_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_hill_doff_heldout_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_hill_doff_heldout_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_hill_doff_heldout_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_doffset_heldout_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_doffset_heldout_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_doffset_heldout_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_doffset_heldout_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_doffset_heldout_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_doffset_heldout_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_hill_doffset_heldout_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_hill_doffset_heldout_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_hill_doffset_heldout_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_hill_doffset_heldout_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_hill_doffset_heldout_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_hill_doffset_heldout_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_dfilm_heldout_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_dfilm_heldout_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_dfilm_heldout_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_dfilm_heldout_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_dfilm_heldout_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_dfilm_heldout_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_hill_dfilm_heldout_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_hill_dfilm_heldout_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_hill_dfilm_heldout_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_hill_dfilm_heldout_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_hill_dfilm_heldout_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_hill_dfilm_heldout_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_hill_ft --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_hill_ft.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_cgt0p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_cgt0p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_cgt1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_cgt1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_cgt2_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_cgt2_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_cgt2_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_cgt2_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_wide_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_wide_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_wide_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_wide_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_do0p4 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_do0p4.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_wd1e3 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_wd1e3.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_lr5e4 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_lr5e4.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_shead_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_shead_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_shead_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_shead_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_shead_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_shead_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_hill_loewe_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_hill_loewe_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_hill_loewe_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_hill_loewe_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_bliss_hill_loewe_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_bliss_hill_loewe_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_hill_loewe_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_hill_loewe_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_hill_loewe_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_hill_loewe_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhet_e150_cell_bliss_hill_loewe_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhet_e150_cell_bliss_hill_loewe_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanwil_e150_cell_bliss_hill_loewe_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanwil_e150_cell_bliss_hill_loewe_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanwil_e150_cell_bliss_hill_loewe_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanwil_e150_cell_bliss_hill_loewe_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanwil_e150_cell_bliss_hill_loewe_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanwil_e150_cell_bliss_hill_loewe_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophet_e150_cell_bliss_hill_loewe_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophet_e150_cell_bliss_hill_loewe_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophet_e150_cell_bliss_hill_loewe_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophet_e150_cell_bliss_hill_loewe_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophet_e150_cell_bliss_hill_loewe_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophet_e150_cell_bliss_hill_loewe_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophetwil_e150_cell_bliss_hill_loewe_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophetwil_e150_cell_bliss_hill_loewe_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophetwil_e150_cell_bliss_hill_loewe_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophetwil_e150_cell_bliss_hill_loewe_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophetwil_e150_cell_bliss_hill_loewe_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophetwil_e150_cell_bliss_hill_loewe_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_anchors_bliss_s345 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_anchors_bliss_s345.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_anchors_bliss_s345_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_anchors_bliss_s345_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_anchors_bliss_s345_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_anchors_bliss_s345_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhop_e150_cell_anchors_bliss_s345_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhop_e150_cell_anchors_bliss_s345_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophet_e150_cell_anchors_bliss_s345_p0 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophet_e150_cell_anchors_bliss_s345_p0.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophet_e150_cell_anchors_bliss_s345_p1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophet_e150_cell_anchors_bliss_s345_p1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_vanhophet_e150_cell_anchors_bliss_s345_p2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_vanhophet_e150_cell_anchors_bliss_s345_p2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_attend_gene --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_attend_gene.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_attend_dev_gene --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_attend_dev_gene.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_displace_gene --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_displace_gene.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_displace_dev_gene --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_displace_dev_gene.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_displace_co_fs1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_displace_co_fs1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_displace_co_fs2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_displace_co_fs2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_hill_loewe_fs1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_hill_loewe_fs1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_hill_loewe_fs2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_hill_loewe_fs2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_dev_fs1 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_dev_fs1.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

sbatch --job-name=040-r11_van_e150_cell_bliss_dev_fs2 --account="$ACCOUNT" --qos="$ACCOUNT" --time=12:00:00 --export=ALL,SWEEP=conf/mixture/r11_van_e150_cell_bliss_dev_fs2.yaml,PARALLEL=1 \
  experiments/040-inhibitor-synergy-wetlab/scripts/delta_train_mixture.slurm

