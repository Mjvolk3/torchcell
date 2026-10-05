#!/bin/bash
# experiments/019-simb-multimodal/scripts/igb_expression_baselines_split.slurm
# [[experiments.019-simb-multimodal.scripts.igb_expression_baselines_split]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/igb_expression_baselines_split.slurm
#
# The split baselines (expression_baselines_split.py: B0 to B3, every gene representation)
# on the IGB Biocluster CPU partition, one split seed per array task. The GilaHyper
# launcher gh_expression_baselines_split.slurm is the original; this one exists because the
# builds, the split caches for seeds 0 to 11 and every embedding are also on IGB, and a
# baseline is a job there, never a login-node process.
#
# Task N is split seed N. The store, the label, the embedding set and the both-label
# restriction travel as environment variables:
#
#   BASELINES_TAG            fig3_proteome (default) | fig3_core
#   BASELINES_LABEL          expression_log2_ratio | protein_abundance
#   BASELINES_EMBEDDING_SET  full (default) | gate
#   BASELINES_REQUIRE        space-separated labels every record must carry; set it to
#                            "protein_abundance expression_log2_ratio" for the strains of
#                            the joint and conditioned rounds (v19, v20)
#
# Submit from a pinned worktree, every task eligible at once, no dependency:
#
#   sbatch -J 019-bl-expr-both -p normal --cpus-per-task=16 --mem=64g --time=1-00:00:00 \
#       --array=0-11 --export=ALL,BASELINES_LABEL=expression_log2_ratio,\
#   BASELINES_REQUIRE="protein_abundance expression_log2_ratio" \
#       experiments/019-simb-multimodal/scripts/igb_expression_baselines_split.slurm
#
# --time and --mem are NOT yet measured on IGB (the GilaHyper header calls the four-entry
# gate "minutes" at 48g; the full set is 28 representations). Read Elapsed and MaxRSS from
# the first array and tighten them.
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --job-name=019-baselines-split
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out
#SBATCH --error=/home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output/%x_%A_%a.out

set -euo pipefail
mkdir -p /home/a-m/mjvolk3/scratch/torchcell/experiments/019-simb-multimodal/slurm/output

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/torchcell" ]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  echo "FATAL: submit from inside a torchcell checkout" >&2
  exit 1
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torchcell
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

TASK="${SLURM_ARRAY_TASK_ID:?submit with --array}"
ARGS=(--split-seed "$TASK"
      --dataset-tag "${BASELINES_TAG:-fig3_proteome}"
      --label "${BASELINES_LABEL:?set BASELINES_LABEL}"
      --embedding-set "${BASELINES_EMBEDDING_SET:-full}")
if [ -n "${BASELINES_REQUIRE:-}" ]; then
  # shellcheck disable=SC2206
  REQ=(${BASELINES_REQUIRE})
  ARGS+=(--require-labels "${REQ[@]}")
fi

echo "=========================================================="
echo "host    : $(hostname)"
echo "job     : ${SLURM_ARRAY_JOB_ID:-none} task=${TASK}"
echo "root    : $PROJECT_ROOT"
echo "source  : $(git -C "$PROJECT_ROOT" rev-parse --short HEAD) + working tree"
echo "args    : ${ARGS[*]}"
echo "=========================================================="
cd "$PROJECT_ROOT"
exec python experiments/019-simb-multimodal/scripts/expression_baselines_split.py "${ARGS[@]}"
