#!/bin/bash
# experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh
# [[experiments.027-betaxanthin-metabolic.scripts.freeze_and_submit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh
#
# Submit a slurm script from a FROZEN copy of the worktree's code, so that edits made to the
# worktree while the job runs cannot reach it.
#
# Why this exists (measured 2026-09-11, jobs 1681 and 1684). train_bx.py runs its DataLoaders
# with spawned workers, and a spawned worker re-imports the main script and the torchcell
# package every time it starts, which is every epoch. Those two cells died mid-run with
#
#     File ".../torchcell/metabolism/flux_layer.py", line 528, in <module>
#         _PROJECTION_PATTERN = re.compile(r"^projection(?:_k(\d+))?$")
#     NameError: name 're' is not defined
#
# in a worker, because another agent's edit to flux_layer.py was between two Edit calls at
# the moment the worker spawned. The main process then waited on the dead worker forever
# while slurm reported RUNNING. "A running process has already imported the module" is only
# true for the main process.
#
# What it does. rsync the worktree's CODE (no .git, notes, paper, results, slurm output,
# wandb) into $DATA_ROOT/job-snapshots/<stamp>-<script>/, symlink every experiment's
# results/, slurm/ and optuna/ directory back to the real worktree so outputs land where the
# analysis scripts read them, verify the snapshot imports, record the git HEAD and the
# sha256 of the files that matter, then `cd` into the snapshot and sbatch. The 027 scripts
# set PROJECT_ROOT from SLURM_SUBMIT_DIR, so the job's PYTHONPATH, EXPERIMENT_ROOT, conf/
# and both training scripts resolve inside the snapshot.
#
# Usage, from anywhere:
#   bash experiments/027-betaxanthin-metabolic/scripts/freeze_and_submit.sh \
#       experiments/027-betaxanthin-metabolic/scripts/gh_wave6_parity_pooled_s2704.slurm
#
# The relative script path must be given relative to the worktree root. Extra arguments are
# passed to sbatch after the script.
set -euo pipefail

WT="${TC_WORKTREE:-/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/kinetics-equilibrator-datasets}"
SCRIPT_REL="$1"; shift
[ -f "$WT/$SCRIPT_REL" ] || { echo "no such script: $WT/$SCRIPT_REL" >&2; exit 2; }

set -a; . "$WT/.env"; set +a
SNAPROOT="${TC_SNAPSHOT_ROOT:-$DATA_ROOT/job-snapshots}"
STAMP="$(date +%Y%m%d-%H%M%S)"
NAME="$(basename "$SCRIPT_REL" .slurm)"
SNAP="$SNAPROOT/$STAMP-$NAME"
mkdir -p "$SNAP"

rsync -a --quiet \
    --exclude '.git' --exclude 'notes' --exclude 'notes-tex' --exclude 'paper' \
    --exclude '__pycache__' --exclude '*.pyc' --exclude 'wandb' --exclude '.mypy_cache' \
    --exclude '.ruff_cache' --exclude '.pytest_cache' --exclude 'test.pkl' \
    --exclude 'experiments/*/results' --exclude 'experiments/*/slurm' \
    --exclude 'experiments/*/optuna' --exclude 'experiments/*/profiling_results' \
    "$WT/" "$SNAP/"

for exp in "$WT"/experiments/*/; do
    e="$(basename "$exp")"
    for d in results slurm optuna; do
        if [ -e "$exp/$d" ]; then
            mkdir -p "$SNAP/experiments/$e"
            ln -s "$exp/$d" "$SNAP/experiments/$e/$d"
        fi
    done
done

# The snapshot must import and compile before it is allowed to run, otherwise we would
# freeze exactly the broken state this script exists to avoid.
(
    cd "$SNAP"
    PYTHONPATH="$SNAP" ~/miniconda3/envs/torchcell/bin/python -c \
        "import torchcell.metabolism.flux_layer, torchcell.metabolism.parameters, torchcell.metabolism.constraints, torchcell.metabolism.transport"
    ~/miniconda3/envs/torchcell/bin/python -m py_compile \
        experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
        experiments/026-metabolism-flux/scripts/train_flux.py
)

# AND IT MUST BE ABLE TO BUILD A CELL. py_compile checks syntax, which a half-plumbed
# option passes: adding a BxCell field, a CLI flag and the `main()` call-site argument
# without the matching `build_cell` parameter compiles cleanly and then dies at run time on
#
#     TypeError: build_cell() got an unexpected keyword argument 'flux_balance_active_weight'
#
# Job 1788 died that way 7 seconds into a GPU allocation on 2026-09-12, having waited in the
# queue first. This check is the same 7 seconds spent before the job is submitted at all.
# It also asserts that every BxCell field is REACHABLE from build_cell, which is the
# invariant the half-plumbed option broke: a field the launcher cannot set is a knob whose
# recorded value is silently always its default.
(
    cd "$SNAP"
    set -a; . "$WT/.env"; set +a
    EXPERIMENT_ROOT="$SNAP/experiments" PYTHONPATH="$SNAP" \
        ~/miniconda3/envs/torchcell/bin/python -c '
import importlib.util, inspect, sys

spec = importlib.util.spec_from_file_location(
    "train_bx_preflight", "experiments/027-betaxanthin-metabolic/scripts/train_bx.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# Fields build_cell is not expected to take: the ones it derives, reads from the config, or
# takes from the arm file rather than from a launcher flag.
derived = {
    "arm", "base_arm", "seed", "permute_train_targets", "hidden", "attention_heads",
    "batch_size", "num_workers", "lr", "weight_decay", "split_seed",
    "select_smoothing_window", "target_log10_molar", "flux_parameterization",
}
taken = set(inspect.signature(module.build_cell).parameters)
missing = sorted(f for f in module.BxCell.model_fields if f not in taken | derived)
if missing:
    sys.exit(
        f"build_cell cannot set these BxCell fields: {missing}. A cell field with no way "
        "in is a knob that always records its default."
    )
module.build_cell("flux_projection", 1, module.load_config(), epochs=1)
print("[preflight] build_cell constructs a cell and reaches every field")
'
)

git -C "$WT" rev-parse HEAD > "$SNAP/SNAPSHOT_GIT_HEAD"
(
    cd "$SNAP"
    sha256sum experiments/027-betaxanthin-metabolic/scripts/train_bx.py \
        experiments/026-metabolism-flux/scripts/train_flux.py \
        torchcell/metabolism/flux_layer.py torchcell/metabolism/constraints.py \
        torchcell/metabolism/parameters.py torchcell/metabolism/transport.py \
        "$SCRIPT_REL"
) > "$SNAP/SNAPSHOT_SHA256"

cd "$SNAP"
echo "snapshot $SNAP"
sbatch "$SCRIPT_REL" "$@"
