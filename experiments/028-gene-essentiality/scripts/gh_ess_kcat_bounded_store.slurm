#!/bin/bash
# experiments/028-gene-essentiality/scripts/gh_ess_kcat_bounded_store.slurm
# [[experiments.028-gene-essentiality.scripts.ess_kcat_bounded_store]]
#
# Round 113: the five offline deletion-flux stores of ess_kcat_bounded_store.py (point,
# sigma1, sigma2, sampled, moma), each followed by its derived _pfbagpr store through
# isozyme_affine_store.py --store-tag kcat<variant>. One job, 32 CPUs, the variants in
# sequence so the pool is the only parallelism. Submit from the worktree root:
#
#   sbatch experiments/028-gene-essentiality/scripts/gh_ess_kcat_bounded_store.slurm
#
#SBATCH -p main
#SBATCH --job-name=028-kcat-store
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=experiments/028-gene-essentiality/slurm/output/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export PYTHONPATH="$SLURM_SUBMIT_DIR"
PY=~/miniconda3/envs/torchcell/bin/python

echo "worktree $SLURM_SUBMIT_DIR, job $SLURM_JOB_ID, $(date)"
for variant in point sigma1 sigma2 sampled moma; do
    echo "===== variant $variant, $(date)"
    $PY experiments/028-gene-essentiality/scripts/ess_kcat_bounded_store.py \
        --variant "$variant" --processes 32
    echo "===== derived store kcat$variant, $(date)"
    $PY experiments/028-gene-essentiality/scripts/isozyme_affine_store.py \
        --store-tag "kcat$variant"
done
echo "done $(date)"
