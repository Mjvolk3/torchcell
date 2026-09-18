#!/bin/bash
# experiments/028-gene-essentiality/scripts/gh_solve_tolerance_transfer.slurm
# [[experiments.028-gene-essentiality.scripts.solve_tolerance_transfer]]
#
# The solve-tolerance transfer read: every labeled gene's round-27 program under the
# dual simplex, the interior point without crossover at KKT 1e-4/1e-6/1e-8 and PDLP at
# the same three tolerances; a probe fitted on one solver's points and scored on
# another's; the PDLP stores at 1e-6 and 1e-8. CPU only. PDLP is 12 to 30 s a program on
# one thread (the 1e-4 read: 1,121 programs at eight workers in 36 min for five solvers),
# so at twenty workers the three PDLP settings are about an hour and a half; the 120 s
# per-solve cap bounds a setting that never certifies at 1121 x 120 / 20 s.
#
# Needs the standalone HiGHS 1.15 (scipy's 1.8 has no PDLP): pass its directory as
# TC_HIGHSPY_PATH at submission.
#
#   TC_HIGHSPY_PATH=<dir holding highspy/> ESS_MEM=24g ESS_CPUS=20 \
#     bash experiments/028-gene-essentiality/scripts/freeze_and_submit.sh \
#       experiments/028-gene-essentiality/scripts/gh_solve_tolerance_transfer.slurm
#
#SBATCH -p main
#SBATCH --mem=24g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=028-solve-transfer
#SBATCH --time=08:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
: "${TC_HIGHSPY_PATH:?set TC_HIGHSPY_PATH to the directory holding the standalone highspy}"
export PYTHONPATH="$TC_HIGHSPY_PATH:$PROJECT_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export TC_LP_WORKERS="${TC_LP_WORKERS:-$SLURM_CPUS_PER_TASK}"
echo "host $(hostname); workers $TC_LP_WORKERS; highspy from $TC_HIGHSPY_PATH"
python -c "import highspy; print('highspy', highspy.Highs().version())"
python experiments/028-gene-essentiality/scripts/solve_tolerance_transfer.py
