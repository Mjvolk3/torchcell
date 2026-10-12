#!/bin/bash
# experiments/028-gene-essentiality/scripts/gh_dcell_style.slurm
# [[experiments.028-gene-essentiality.scripts.dcell_style_baseline]]
#
# One or more reads of dcell_style_baseline.py on CPU. The script trains its (arm, split,
# seed) cells over a process pool of 8 workers x 2 threads and uses no card, so the job
# asks for 16 cores and no GPU.
#
#   READS="released resplits" DCELL_ARGS="--arms dcell_style_shared --width 8 \
#       --order tanh_bn --weight-decay 0.001 --seeds 2700 2701 2702 --tag _go006" \
#   ESS_GO_DIR=$DATA_ROOT/data/scerevisiae/go_term_hot_dcell006/processed \
#   ESS_MEM=32g ESS_CPUS=16 bash experiments/028-gene-essentiality/scripts/freeze_and_submit.sh \
#       experiments/028-gene-essentiality/scripts/gh_dcell_style.slurm
#
#SBATCH -p main
#SBATCH --mem=32g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=028-dcell
#SBATCH --time=06:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/slurm/output/%x_%j.out

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
echo "host $(hostname); reads ${READS:?set READS}; args ${DCELL_ARGS:-}"
# READS: a space-separated list of --read values, run one after the other in this job.
# DCELL_ARGS: further flags, the same for every read (arms, width, order, weight decay,
# seeds, tag).
# ESS_GO_DIR: a GO store directory (the experiment 006 GO written by go006_store.py), passed
# as --go-dir; the script then insists on a --tag.
GO_ARGS=""
if [ -n "${ESS_GO_DIR:-}" ]; then
    GO_ARGS="--go-dir $ESS_GO_DIR"
    echo "GO store $ESS_GO_DIR"
fi
for read in $READS; do
    echo "=== read $read"
    python experiments/028-gene-essentiality/scripts/dcell_style_baseline.py --read "$read" $GO_ARGS ${DCELL_ARGS:-}
done
