#!/bin/bash
# experiments/028-gene-essentiality/scripts/gh_train_essentiality_pack.slurm
# [[experiments.028-gene-essentiality.scripts.train_essentiality]]
#
# SEVERAL CELLS ON ONE GPU. Slurm on GilaHyper hands out whole GPUs (gres type gpu, no
# shards), so gh_train_essentiality.slurm's one-cell-per-job design caps the machine at four
# concurrent cells while each width-12 cell holds 3.6 GB of a 46 GB card at about 10 percent
# utilization (measured 2026-09-16, round 3). This launcher takes one GPU and runs every
# cell of a PACK FILE concurrently on it, then waits for all of them.
#
#   ESS_PACK   path, relative to the worktree, of a pack file: one cell per line, each line
#              the ESS_* assignments of that cell exactly as they would prefix a single
#              submission (values with spaces single-quoted); blank lines and # comments
#              are skipped. The knobs are documented in gh_train_essentiality.slurm.
#   ESS_MEM    memory for the job; default 36g, sized for three cells at a measured 5.4 GB
#              peak each with headroom (raise it for a pack of more or wider cells).
#
#   ESS_PACK=experiments/028-gene-essentiality/conf/packs/w15_gate_s0.pack \
#     bash experiments/028-gene-essentiality/scripts/freeze_and_submit.sh \
#          experiments/028-gene-essentiality/scripts/gh_train_essentiality_pack.slurm
#
# Each cell writes its own log, slurm/output/028-pack_<jobid>_<ESS_OUT>.out, so a pack's
# cells read like single jobs; the job log holds only the roster and the exit codes. The
# job fails if any cell fails, so a half-finished pack is visible in sacct.
#
#SBATCH -p main
#SBATCH --mem=36g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --job-name=028-pack
#SBATCH --time=08:00:00
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/028-gene-essentiality/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torchcell
set -a; . "$PROJECT_ROOT/.env"; set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export WANDB_DIR=/scratch/projects/torchcell-scratch/experiments/028-gene-essentiality/wandb
export WANDB_PROJECT=torchcell_028_gene_essentiality
export NUM_WORKERS="${ESS_WORKERS:-0}"
export COLLAPSE_PATIENCE_EPOCHS=200
export TC_TEST_PRED_DIR=$DATA_ROOT/test-predictions-028
export TC_CHECKPOINT_DIR=$DATA_ROOT/checkpoints-028
mkdir -p "$WANDB_DIR" "$TC_TEST_PRED_DIR" "$TC_CHECKPOINT_DIR"
echo "host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}; DATA_ROOT=$DATA_ROOT; EXPERIMENT_ROOT=$EXPERIMENT_ROOT"
echo "snapshot head: $(cat SNAPSHOT_GIT_HEAD 2>/dev/null || echo unfrozen)"

PACK="${ESS_PACK:?set ESS_PACK to a pack file relative to the worktree}"
[ -f "$PACK" ] || { echo "no such pack file: $PACK" >&2; exit 2; }
LOG_DIR="${ESS_PACK_LOG_DIR:-$PROJECT_ROOT/experiments/028-gene-essentiality/slurm/output}"
JOB="${SLURM_JOB_ID:-local}"
CELL_SH=experiments/028-gene-essentiality/scripts/ess_cell.sh

pids=()
names=()
while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    # The stem names the log; it is read from the line the same way the cell reads it.
    stem="$(eval "export $line"; echo "${ESS_OUT:-unnamed_cell}")"
    echo "launch $stem"
    ( eval "export $line"; exec bash "$CELL_SH" ) > "$LOG_DIR/028-pack_${JOB}_${stem}.out" 2>&1 &
    pids+=($!)
    names+=("$stem")
    # STAGGER THE LAUNCHES. Twelve or more cells started at once read the same
    # processed embedding file (sgd_gene_graph_hot/processed/*.pt) in the same
    # second, and one of them died in torch.load with "PytorchStreamReader failed
    # reading file data/2 ... internal miniz error" while its siblings read the same
    # bytes and went on (jobs 2406, 2414 and 2416 on 2026-09-18, one cell each).
    # A few seconds between launches puts the readers out of step; the cells still
    # share the card for the whole run.
    sleep "${ESS_STAGGER_S:-5}"
done < "$PACK"

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "done ${names[$i]}"
    else
        echo "FAILED ${names[$i]} (exit $?)"
        status=1
    fi
done
exit $status
