# experiments/027-betaxanthin-metabolic/scripts/audit_cells.py
# [[experiments.027-betaxanthin-metabolic.wave7-parity-one-table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/audit_cells.py

r"""Re-audit finished 027 cells, each against the code tree it was actually trained with.

THE PROBLEM THIS SOLVES, measured. ``torchcell/metabolism/flux_layer.py`` and
``train_bx.py`` are edited by other work streams while cells run: on 2026-09-11 the flux
layer changed at 16:55, 17:01 and 17:03, and ``train_bx.py`` gained a ``conc_lower_m``
config field at 17:10 that the 02:05 flux layer does not have. Running
``audit_top_strain_feasibility.py`` out of the live worktree therefore rebuilds a checkpoint
against a layer the cell never saw: for the 02:05 projection cells it dies with
``"FluxLayerConfig" object has no field "conc_lower_m"``, and where it does not die the
numbers would silently describe a different model.

CONTENT-ADDRESSED RESOLUTION, not a guess. Every result file records the sha256 of the
``train_bx.py`` and ``flux_layer.py`` its process imported. This script searches the trees
that exist on disk (the worktree, the wave-6 and wave-7 freezes, every job snapshot) for the
pair that matches BOTH hashes, and runs the audit with ``PYTHONPATH`` pointed there. No tree
matching the hashes means the cell cannot be re-audited and the script says so by name
rather than auditing it against something else.

WHERE THE AUDIT SCRIPT RUNS FROM. ``audit_top_strain_feasibility.py`` puts its own directory
first on ``sys.path``, so running the worktree's copy would import the worktree's
``train_bx``. It is therefore copied into a neutral run directory under
``$DATA_ROOT/audit-runs/<stamp>/`` and executed there, with the cell's tree on
``PYTHONPATH``. Job snapshots are read, never written.

Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/audit_cells.py \
        wave6_proj_flux_projection_s2702 wave7_parity_nocstr_s2700

``--stochastic`` and ``--draws`` are passed through to every audit in the run. Each audit
writes ``results/audit_<stem>.json``; this script writes nothing of its own.
"""

import argparse
import glob
import hashlib
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
WORKTREE = osp.dirname(osp.abspath(EXPERIMENT_ROOT.rstrip("/")))
RESULTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
AUDIT_SCRIPT = osp.join(
    EXPERIMENT_ROOT,
    "027-betaxanthin-metabolic",
    "scripts",
    "audit_top_strain_feasibility.py",
)
TRAIN_BX_REL = "experiments/027-betaxanthin-metabolic/scripts/train_bx.py"
FLUX_LAYER_REL = "torchcell/metabolism/flux_layer.py"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_trees() -> list[str]:
    """Every tree on disk that could hold a version of the two files, newest first."""
    trees = [WORKTREE]
    trees += sorted(glob.glob(osp.join(DATA_ROOT, "frozen-027-*")), reverse=True)
    trees += sorted(glob.glob(osp.join(DATA_ROOT, "job-snapshots", "*")), reverse=True)
    return trees


def resolve_tree(stem: str, trees: list[str]) -> tuple[str, str, dict[str, str]]:
    """The tree whose train_bx.py and flux_layer.py match the ones the cell recorded.

    Returns ``(train_bx_root, torchcell_root, provenance)``. The two roots are allowed to
    differ because the wave-6 cells ran ``train_bx.py`` out of a job snapshot with
    ``PYTHONPATH`` pinned at the separate 2026-09-10 23:05 freeze.
    """
    with open(osp.join(RESULTS, f"{stem}.json")) as fh:
        provenance = json.load(fh).get("code_provenance") or {}
    want_bx = provenance.get("train_bx.py")
    want_layer = provenance.get("flux_layer.py")
    if not want_bx or not want_layer:
        raise SystemExit(
            f"{stem} carries no code_provenance, so the tree it ran on is unknown. Cells "
            "from before the provenance block cannot be re-audited; re-run them."
        )
    bx_root = layer_root = None
    for tree in trees:
        bx_path = osp.join(tree, TRAIN_BX_REL)
        if bx_root is None and osp.isfile(bx_path) and _sha256(bx_path) == want_bx:
            bx_root = tree
        layer_path = osp.join(tree, FLUX_LAYER_REL)
        if (
            layer_root is None
            and osp.isfile(layer_path)
            and _sha256(layer_path) == want_layer
        ):
            layer_root = tree
        if bx_root and layer_root:
            break
    if bx_root is None or layer_root is None:
        missing = "train_bx.py" if bx_root is None else "flux_layer.py"
        raise SystemExit(
            f"{stem}: no tree on disk carries the {missing} it recorded "
            f"({want_bx[:12] if bx_root is None else want_layer[:12]}). Searched "
            f"{len(trees)} trees. The code that produced this cell is gone; re-run it "
            "rather than auditing it against a different layer."
        )
    return bx_root, layer_root, provenance


def flux_scripts_dir(torchcell_root: str) -> str:
    """Where 026's ``train_flux.py`` lives in that tree.

    The 2026-09-10 freeze holds it under ``026-scripts/`` because that is how it was made;
    every job snapshot is a copy of the worktree and holds it in place.
    """
    frozen = osp.join(torchcell_root, "026-scripts", "train_flux.py")
    if osp.isfile(frozen):
        return osp.dirname(frozen)
    return osp.join(torchcell_root, "experiments", "026-metabolism-flux", "scripts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stems", nargs="+", help="Result file names without '.json'.")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Where the audit script is copied to and run from; default $DATA_ROOT/audit-runs.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or osp.join(DATA_ROOT, "audit-runs", "current")
    os.makedirs(run_dir, exist_ok=True)
    runner = osp.join(run_dir, "audit_top_strain_feasibility.py")
    shutil.copy2(AUDIT_SCRIPT, runner)

    trees = candidate_trees()
    failures: dict[str, str] = {}
    for stem in args.stems:
        bx_root, layer_root, provenance = resolve_tree(stem, trees)
        env: dict[str, Any] = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [layer_root, osp.join(bx_root, osp.dirname(TRAIN_BX_REL))]
        )
        env["TC_FLUX_SCRIPTS_DIR"] = flux_scripts_dir(layer_root)
        env["PYTHONWARNINGS"] = "ignore"
        command = [sys.executable, runner, "--result", f"{stem}.json"]
        if args.stochastic:
            command += ["--stochastic", "--draws", str(args.draws)]
        print(
            f"=== {stem}\n"
            f"    train_bx  {provenance['train_bx.py'][:12]} from {bx_root}\n"
            f"    fluxlayer {provenance['flux_layer.py'][:12]} from {layer_root}",
            flush=True,
        )
        completed = subprocess.run(command, env=env, cwd=run_dir, check=False)
        if completed.returncode != 0:
            failures[stem] = f"exit {completed.returncode}"
    if failures:
        print("\nFAILED:")
        for stem, why in failures.items():
            print(f"  {stem}: {why}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
