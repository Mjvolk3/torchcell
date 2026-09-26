#!/bin/bash
# experiments/030-solid-growth-multi/scripts/gh_smoke_dataset_token.slurm
# [[experiments.030-solid-growth-multi.scripts.gh_smoke_dataset_token]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/gh_smoke_dataset_token.slurm
#
# WHY. The per-entry dataset token has to be shown to work on the real 030 build before
# three 4-day mmli jobs run on it (plan.030-per-entry-dataset-token, decision 10). One
# GilaHyper GPU, three steps in one job:
#
#   1. warm   warm_split_cache_030.py writes <build>/data_module_cache/index_seed_{0,1,2}
#             under the campaign's FINAL pool (S3 minus the essentiality holdout), pin and
#             unpinned-to-train rule. CPU only. These are the files sync_igb_030_build.sh
#             mirrors and igb_mmli_cgt_030.slurm's preflight requires.
#   2. tok    cgt_030_smoke_tok_000: every trigenic entry cloned under a synthetic twin
#             of its source, +0.3 normalized; 3 epochs x 500 batches; measures
#             pred(twin) - pred(own) after every validation epoch and at the end.
#   3. report smoke_report_030.py prints PASS/FAIL per criterion and exits non-zero on FAIL.
#
# No control run. Jobs 2863 and 2866 ran one (clones under the original token) and the
# criterion "control loss at least delta^2/4 = 0.0225 above the token run's" was not
# resolvable: the two runs' validation point losses differed by 0.06 from initialization
# alone. Jobs 2863 and 2866 also used ONE synthetic token over every source and measured
# 0.79 to 0.93 for the 0.3 offset: a single token learns one global shift, and on the
# validation triples the difference from the own token carries the trigenic sources'
# residual gap on top of the offset. One twin per source removes that.
#
# The job's own exit code is the report's, so `sacct` shows the verdict. Per-step logs
# stay in the one output file, separated by banners.
#
# MEMORY AND CPUS. One rank opens the 13.5M-record build (025 measured 5.5 GB RSS per
# rank) plus 6 spawned workers that re-import the stack; the cache warm holds the split-
# index JSONs of the build at once (the 025 note: most of the RSS). 32 GB covers that,
# and it is also the most 8 CPUs can ask for: GilaHyper's MaxMemPerCPU is 4.1 GB, so a
# --mem above cpus x 4.1 GB silently RAISES the CPU request (96 GB became 24 CPUs and the
# job sat on "Resources" behind three 1-GPU jobs while a GPU and 8 CPUs were free, job
# 2860, 2026-09-26). Size --mem from the CPUs, not the other way round.
#
#   mkdir -p experiments/030-solid-growth-multi/slurm/output
#   sbatch experiments/030-solid-growth-multi/scripts/gh_smoke_dataset_token.slurm
#SBATCH -p main
#SBATCH --mem=48g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --job-name=030-r1w0-smoke
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/experiments/030-solid-growth-multi/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/030-per-entry-dataset-token/experiments/030-solid-growth-multi/slurm/output/%x_%j.out

PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
EXP="$PROJECT_ROOT/experiments/030-solid-growth-multi"
echo "job=$SLURM_JOB_ID name=$SLURM_JOB_NAME host=$(hostname) started=$(date)"
echo "project=$PROJECT_ROOT ($(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null))"

# conda.sh, not bin/activate (which would consume $1) and not ~/.zshrc (this is bash).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; source "$PROJECT_ROOT/.env"; set +a
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"
export WANDB__SERVICE_WAIT=600
export NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 2))
echo "DATA_ROOT=$DATA_ROOT torchcell=$(python -c 'import torchcell; print(torchcell.__file__)')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "$PROJECT_ROOT/experiments" || exit 1

banner() { echo; echo "================ $1  $(date) ================"; }

banner "1/3 warm split cache (seeds 0 1 2)"
python "$EXP/scripts/warm_split_cache_030.py" \
  --config-name cgt_030_s3_r_tok_embfit_001 '+warm.seeds=[0,1,2]' || { echo "WARM FAILED"; exit 2; }
ls -la "$(readlink -f "$DATA_ROOT/data/torchcell/experiments/030-solid-growth-multi/001-multi-build")/data_module_cache"

banner "2/3 token run cgt_030_smoke_tok_000"
python "$EXP/scripts/equivariant_cell_graph_transformer.py" \
  --config-name cgt_030_smoke_tok_000 +seed=0 data_module.num_workers=$NUM_WORKERS \
  || { echo "TOKEN RUN FAILED"; exit 3; }

banner "3/3 report"
python "$EXP/scripts/smoke_report_030.py" \
  --tok "$EXP/results/smoke/cgt_030_smoke_tok_000_seed0.json"
STATUS=$?
echo "finished=$(date) report_exit=$STATUS"
exit $STATUS
