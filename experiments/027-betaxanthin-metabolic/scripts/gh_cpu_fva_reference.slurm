#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/gh_cpu_fva_reference.slurm
# [[experiments.027-betaxanthin-metabolic.cpu-baselines-2026-09-14]]
#
# E2 of the metabolic review: the wild-type FVA reference, recomputed on the medium and the
# model the 027 cells actually live in.
#
# THE DEFECT THIS FIXES. `fva_reference.py` optimized `YeastGEM().model` with no medium
# applied and no cassette, giving a wild-type growth of 0.085844 /h; YPD-approx with the
# Cachera cassette grows at 0.5352 /h. Every `width_FVA - width_posterior` number on record
# was computed against that unnamed default.
#
# CONTROL. The `default` condition must reproduce `026-metabolism-flux/results/fva_census.json`
# field for field, or the script raises. A difference between conditions is then the flag's,
# not a drifted model's.
#
# PRE-REGISTERED READ. The width census under (default, no cassette) / (YPD, no cassette) /
# (YPD, cassette) side by side.
#   SUCCESS: the censuses differ by more than 5 %. Restate every sampler-versus-FVA
#            comparison against the YPD + cassette reference and correct the note.
#   KILL:    they agree within 5 %. The medium does not matter for the licensed set and the
#            note gains a sentence saying so.
#SBATCH -p main
#SBATCH --nice=1000
#SBATCH --mem=16g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=027-fva-ref
#SBATCH --time=02:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/027-betaxanthin-metabolic/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
echo "host $(hostname); DATA_ROOT=$DATA_ROOT"

python experiments/026-metabolism-flux/scripts/fva_reference.py
