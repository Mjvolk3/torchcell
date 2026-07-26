"""Checkout-relative output paths, so a script run from a git worktree writes to THAT
worktree rather than to the primary checkout.

# torchcell/utils/paths.py

THE PROBLEM. `EXPERIMENT_ROOT` and `ASSET_IMAGES_DIR` come from a single `.env` at the
primary checkout, so they are ABSOLUTE paths into the primary tree. Every script that
writes results or figures through those variables therefore writes to the primary tree no
matter which worktree it was launched from. Three separate ways that bit in one session:
a results JSON committed from the wrong tree (stale contents), a figure whose relative
`![](./assets/images/...)` link resolved to nothing inside the branch, and a primary tree
left dirty while the branch was the thing under review.

Inputs were never affected -- they are read from `DATA_ROOT`, which is genuinely shared.
It is only OUTPUTS that must follow the code.

THE FIX. Resolve the checkout root by walking up from a file inside it (normally the
calling script's `__file__`) until a repo marker is found, then build the output paths
under that root. Falls back to the environment variable when the caller is not inside a
checkout at all (e.g. an installed package), so behaviour outside a worktree is unchanged.

Usage in an experiment script:

    from torchcell.utils.paths import asset_images_dir, experiment_results_dir

    results = experiment_results_dir("019-simb-multimodal", __file__)
    images = asset_images_dir(__file__, subdir="019-simb-multimodal")
"""

from __future__ import annotations

import os
import os.path as osp

# A directory is the checkout root if it holds all of these. `.git` alone is not enough:
# in a worktree `.git` is a FILE pointing into the primary tree's object store, which is
# exactly the case that has to keep working.
_ROOT_MARKERS = ("pyproject.toml", "torchcell", "experiments")


def repo_root(from_path: str) -> str | None:
    """Walk up from ``from_path`` to the enclosing checkout root, or None if outside one."""
    current = osp.abspath(from_path)
    if osp.isfile(current):
        current = osp.dirname(current)
    while True:
        if all(osp.exists(osp.join(current, marker)) for marker in _ROOT_MARKERS):
            return current
        parent = osp.dirname(current)
        if parent == current:
            return None
        current = parent


def experiment_root(from_path: str) -> str:
    """`<checkout>/experiments`, or $EXPERIMENT_ROOT when outside a checkout."""
    root = repo_root(from_path)
    if root is None:
        return os.environ["EXPERIMENT_ROOT"]
    return osp.join(root, "experiments")


def asset_images_dir(from_path: str, subdir: str | None = None) -> str:
    """`<checkout>/notes/assets/images[/subdir]`, created if missing.

    Falls back to $ASSET_IMAGES_DIR outside a checkout. Creating the directory here is
    deliberate: every caller needs it to exist before `savefig`, and doing it once removes
    a `makedirs` line from each plotting script.
    """
    root = repo_root(from_path)
    base = (
        os.environ["ASSET_IMAGES_DIR"]
        if root is None
        else osp.join(root, "notes", "assets", "images")
    )
    path = base if subdir is None else osp.join(base, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def experiment_results_dir(experiment: str, from_path: str) -> str:
    """`<checkout>/experiments/<experiment>/results`, created if missing."""
    path = osp.join(experiment_root(from_path), experiment, "results")
    os.makedirs(path, exist_ok=True)
    return path
