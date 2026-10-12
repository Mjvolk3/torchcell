#!/bin/bash
# experiments/026-metabolism-flux/scripts/gh_pka_selection_cpu_predictors.slurm
# MolGpKa and pKasolver-lite over the method-selection validation set, sharded across CPU
# cores. No GPU: both tools are CPU-only, and the node's four cards are running training.
#
# One process costs ~39 s per compound for MolGpKa and ~4 s for pKasolver-lite on the
# 300-compound set, so a 2,400-compound set is 26 h and 2.7 h single threaded. Both walk a
# protonation ladder per compound with no shared state, so the set is cut round-robin
# (`--shard i --n-shards N`) and the shards are merged afterwards by
# `external_pka_benchmark.py merge-shards`.
#
# Each shard is pinned to ONE thread. torch defaults to a thread per core, and 64
# processes each spawning 128 threads is 8,192 runnable threads on 128 cores, which is
# slower than the single-process run it replaces.
#
#   cd <worktree-root> && sbatch experiments/026-metabolism-flux/scripts/gh_pka_selection_cpu_predictors.slurm
#
#SBATCH -p main
#SBATCH --mem=200g
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=026-pka-selection-cpu
#SBATCH --time=12:00:00
#SBATCH --mail-user=mjvolk3@illinois.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/026-metabolism-flux/slurm/output/%x_%j.out
#SBATCH --error=/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets/experiments/026-metabolism-flux/slurm/output/%x_%j.out

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
mkdir -p "$PROJECT_ROOT/experiments/026-metabolism-flux/slurm/output"

ENV_FILE="$PROJECT_ROOT/.env"
[ -f "$ENV_FILE" ] || ENV_FILE="$HOME/Documents/projects/torchcell/.env"
set -a
. "$ENV_FILE"
set +a
export EXPERIMENT_ROOT="$PROJECT_ROOT/experiments"

# The selection set and its prediction filenames, so the 300-compound benchmark's files
# are never touched by this run.
export EXTERNAL_PKA_SAMPLE="pka_selection_validation_set.json"
export EXTERNAL_PKA_PRED_TEMPLATE="pka_selection_predictions_{tool}.json"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT" || exit 1
SCRIPT="$PROJECT_ROOT/experiments/026-metabolism-flux/scripts/external_pka_benchmark.py"
MOLGPKA_REPO="$DATA_ROOT/pka/MolGpKa"
LOGS="$PROJECT_ROOT/experiments/026-metabolism-flux/slurm/output"

N_MOLGPKA=${N_MOLGPKA:-48}
N_PKASOLVER=${N_PKASOLVER:-16}

echo "molgpka shards: $N_MOLGPKA; pkasolver shards: $N_PKASOLVER"
echo "molgpka repo: $MOLGPKA_REPO"

for i in $(seq 0 $((N_MOLGPKA - 1))); do
    [ "$N_MOLGPKA" -gt 0 ] || break
    "$HOME/miniconda3/envs/pka-tools/bin/python" "$SCRIPT" predict-molgpka \
        --repo "$MOLGPKA_REPO" \
        --shard "$i" --n-shards "$N_MOLGPKA" \
        > "$LOGS/molgpka_shard_${i}.log" 2>&1 &
done

# pKasolver shells out to its own `call_dimorphite_dl.py` through a bare `python`, so the
# env's bin has to be first on PATH or every compound fails with "No module named
# 'pkasolver'" inside that subprocess while the parent reports a clean error record.
export PATH="$HOME/miniconda3/envs/pkasolver-env/bin:$PATH"

# AND EVERY pKasolver SHARD NEEDS ITS OWN WORKING DIRECTORY. That subprocess ends with
# `pickle.dump(mol, open("test.pkl", "wb"))` -- a FIXED filename in the current directory,
# which the parent then reads back. Two shards in one directory overwrite each other's
# handoff file, and the loser sees `EOFError: Ran out of input` or an AssertionError on a
# half-written pickle. Measured: 14 shards in one directory failed 89 of 172 compounds,
# against 1 of 300 for the single-process run. The failure looks like a tool limitation and
# is not one.
SHARD_ROOT="${TMPDIR:-/tmp}/pkasolver_shards_${SLURM_JOB_ID:-manual}"

for i in $(seq 0 $((N_PKASOLVER - 1))); do
    mkdir -p "$SHARD_ROOT/$i"
    (
        cd "$SHARD_ROOT/$i" || exit 1
        "$HOME/miniconda3/envs/pkasolver-env/bin/python" "$SCRIPT" predict-pkasolver \
            --shard "$i" --n-shards "$N_PKASOLVER"
    ) > "$LOGS/pkasolver_shard_${i}.log" 2>&1 &
done

wait

if [ "$N_MOLGPKA" -gt 0 ]; then
    "$HOME/miniconda3/envs/pka-tools/bin/python" "$SCRIPT" merge-shards \
        --tool molgpka --n-shards "$N_MOLGPKA"
fi
if [ "$N_PKASOLVER" -gt 0 ]; then
    "$HOME/miniconda3/envs/pka-tools/bin/python" "$SCRIPT" merge-shards \
        --tool pkasolver --n-shards "$N_PKASOLVER"
fi
echo "done"
