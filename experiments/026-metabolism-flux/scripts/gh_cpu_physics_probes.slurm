#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_cpu_physics_probes.slurm
# [[experiments.026-metabolism-flux.physics-gates-2026-09-14]]
#
# RERUN EVERY PHYSICS PROBE OF THE 2026-09-14 REVIEW FROM COMMITTED SCRIPTS.
#
# The review that motivated wave 9 measured seven things about the flux layer's geometry
# and about yeast-GEM 9.0.2 itself. Under the repo's artifact rule none of those numbers
# may enter a note or the paper until they come from a committed script, so the probes were
# promoted into experiments/026-metabolism-flux/scripts/ and this cell is what regenerates
# their JSON together.
#
#   probe_fba_rule.py       the balance rules calibrated on FBA / pFBA / a random vertex
#   probe_dirbox.py         growth and throughput inside the standard-state directional box
#   probe_tfa_lp.py         thermodynamic feasibility of a real flux, 5 routes x 2 windows,
#                           with the named offenders (the merged union + offenders probe)
#   probe_projrate.py       alternating vs Dykstra, K = 20 .. 5000
#   probe_warmstart.py      the balance rule as a function of proposal quality
#   probe_dual.py           the exact metric projection by dual ascent, float32
#   probe_massbal.py        elemental and charge balance of the GEM
#
# CPU only, and deliberately modest: the database rebuild owns this node tonight, so this
# runs at --nice=1000 on 8 cores with a two-hour bound and yields to every rb_*, verify_*
# and kg_* job.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_cpu_physics_probes.slurm
#
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nice=1000
#SBATCH --job-name=026-physics-probes
#SBATCH --time=02:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/026-metabolism-flux/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/026-metabolism-flux/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

PY=~/miniconda3/envs/torchcell/bin/python
echo "host $(hostname); DATA_ROOT=$DATA_ROOT; probes from $PROJECT_ROOT"

set -e
$PY experiments/026-metabolism-flux/scripts/probe_fba_rule.py
$PY experiments/026-metabolism-flux/scripts/probe_massbal.py
$PY experiments/026-metabolism-flux/scripts/probe_dirbox.py
$PY experiments/026-metabolism-flux/scripts/probe_tfa_lp.py --route all
$PY experiments/026-metabolism-flux/scripts/probe_warmstart.py
$PY experiments/026-metabolism-flux/scripts/probe_projrate.py
$PY experiments/026-metabolism-flux/scripts/probe_dual.py
echo "all probes done"
