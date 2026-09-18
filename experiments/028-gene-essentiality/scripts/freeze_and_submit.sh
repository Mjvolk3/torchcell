#!/bin/bash
# experiments/028-gene-essentiality/scripts/freeze_and_submit.sh
# [[experiments.028-gene-essentiality.scripts.freeze_and_submit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/freeze_and_submit.sh
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
# sha256 of the files that matter, then `cd` into the snapshot and sbatch. The 028 scripts
# set PROJECT_ROOT from SLURM_SUBMIT_DIR, so the job's PYTHONPATH, EXPERIMENT_ROOT, conf/
# and both training scripts resolve inside the snapshot.
#
# Usage, from anywhere:
#   bash experiments/028-gene-essentiality/scripts/freeze_and_submit.sh \
#       experiments/028-gene-essentiality/scripts/gh_wave6_parity_pooled_s2704.slurm
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
        experiments/028-gene-essentiality/scripts/train_essentiality.py \
        experiments/026-metabolism-flux/scripts/train_flux.py
)

# AND IT MUST BE ABLE TO BUILD A CELL. py_compile checks syntax, which a half-plumbed
# option passes: adding a BxCell field, a CLI flag and the `main()` call-site argument
# without the matching `build_cell` parameter compiles cleanly and then dies at run time on
#
#     TypeError: build_cell() got an unexpected keyword argument 'flux_balance_active_weight'
#
# Job 1788 died that way 7 seconds into a GPU allocation on 2026-09-12, having waited in the
# queue first. This check is the same 6 seconds spent before the job is submitted at all.
#
# THE SECOND CHECK READS THE CALL SITE, NOT A LIST OF FIELDS. An earlier version asserted
# that every BxCell field was reachable from `build_cell` and exempted the ones sourced
# elsewhere by name. That list went stale the same day: `flux_balance_active_row_set` comes
# from the ARM FILE, exactly as `flux_parameterization` does, and the check refused a
# perfectly good submission. A hand-maintained exemption list in a preflight is a false
# positive waiting to happen, and a preflight that cries wolf gets bypassed.
#
# What is checked instead is the exact thing that broke: every keyword `main()` passes to
# `build_cell` must exist in `build_cell`'s signature. It is read from the AST, so it needs
# no list, cannot go stale, and cannot fire on a field that is legitimately sourced from
# somewhere else.
(
    cd "$SNAP"
    set -a; . "$WT/.env"; set +a
    EXPERIMENT_ROOT="$SNAP/experiments" PYTHONPATH="$SNAP" \
        ~/miniconda3/envs/torchcell/bin/python -c '
import ast, importlib.util, inspect, sys

source = "experiments/028-gene-essentiality/scripts/train_essentiality.py"
tree = ast.parse(open(source).read())
passed: set[str] = set()
for node in ast.walk(tree):
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_cell"
    ):
        passed |= {kw.arg for kw in node.keywords if kw.arg is not None}

spec = importlib.util.spec_from_file_location("train_essentiality_preflight", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
accepted = set(inspect.signature(module.build_cell).parameters)
unknown = sorted(passed - accepted)
if unknown:
    sys.exit(
        f"build_cell is called with keywords it does not accept: {unknown}. The job would "
        "die on a TypeError after the queue wait and the allocation."
    )
module.build_cell("pooled", 1, module.load_config(), epochs=1)
print(
    f"[preflight] build_cell constructs a cell; all {len(passed)} call-site keywords accepted"
)
'
)

git -C "$WT" rev-parse HEAD > "$SNAP/SNAPSHOT_GIT_HEAD"
(
    cd "$SNAP"
    sha256sum experiments/028-gene-essentiality/scripts/train_essentiality.py \
        experiments/026-metabolism-flux/scripts/train_flux.py \
        torchcell/metabolism/flux_layer.py torchcell/metabolism/constraints.py \
        torchcell/metabolism/parameters.py torchcell/metabolism/transport.py \
        "$SCRIPT_REL"
) > "$SNAP/SNAPSHOT_SHA256"

cd "$SNAP"
echo "snapshot $SNAP"
# ESS_MEM sizes the job's memory; the launcher's own #SBATCH line is the 36g default and
# a command-line --mem overrides it. Until 2026-09-18 nothing passed it and every pack ran
# at 36g (three packs died out of memory); confirm with `sacct -j <id> -o ReqMem`.
# ESS_CPUS sizes the job's cores the same way (the launcher's #SBATCH line is 24); pooled
# cells with no loader workers and no LP workers need far fewer, and a 24-core request
# pends behind a CPU-heavy job on the node while a card sits idle (2403 on 2026-09-18).
sbatch --mem="${ESS_MEM:-36g}" --cpus-per-task="${ESS_CPUS:-24}" "$SCRIPT_REL" "$@"
