#!/bin/bash
# experiments/028-gene-essentiality/scripts/gh_ess_gecko_store.slurm
# [[experiments.028-gene-essentiality.scripts.ess_gecko_store]]
#
# Round 114: the offline deletion-flux stores of ess_gecko_store.py (the GECKO pools
# pool125, pool250, pool125s1, the direction-rule pools pool125sign and pool125sign30, the
# minimal-enzyme-usage pool125minenz, and the enzyme-free plainfba, pfbalambda3 and
# pfbalambda4), each followed by its derived _pfbagpr store through
# isozyme_affine_store.py --store-tag gecko<variant> when the store was written (a variant
# whose wild type is infeasible writes only its summary). One job, 24 CPUs, the variants in
# sequence so the pool is the only parallelism. Submit from the worktree root:
#
#   sbatch experiments/028-gene-essentiality/scripts/gh_ess_gecko_store.slurm
#
#SBATCH -p main
#SBATCH --job-name=028-gecko-store
#SBATCH --cpus-per-task=24
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
DATA_ROOT=$($PY -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.environ['DATA_ROOT'])")

echo "worktree $SLURM_SUBMIT_DIR, job $SLURM_JOB_ID, $(date)"
for variant in ${GECKO_VARIANTS:-pool125 pool250 pool125s1 pool125minenz plainfba pfbalambda3 pfbalambda4 pool125sign pool125sign30}; do
    echo "===== variant $variant, $(date)"
    $PY experiments/028-gene-essentiality/scripts/ess_gecko_store.py \
        --variant "$variant" --processes 24
    store="$DATA_ROOT/data/torchcell/reference_flux/ess_solve_encodings_YPD_gecko$variant.npz"
    if [ -f "$store" ]; then
        echo "===== derived store gecko$variant, $(date)"
        $PY experiments/028-gene-essentiality/scripts/isozyme_affine_store.py \
            --store-tag "gecko$variant"
    else
        echo "===== no store for $variant (wild type infeasible); derived step skipped"
    fi
done
echo "done $(date)"
