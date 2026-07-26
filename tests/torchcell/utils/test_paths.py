"""Tests for checkout-relative output paths.

The bug these guard against is silent: a script launched from a git worktree writes its
results and figures into the PRIMARY checkout, because $EXPERIMENT_ROOT / $ASSET_IMAGES_DIR
are absolute paths baked into one .env. Nothing errors -- the wrong tree just gets the
output, and the branch under review is missing it.
"""

from __future__ import annotations

import os
import os.path as osp
from pathlib import Path

import pytest

from torchcell.utils.paths import (
    asset_images_dir,
    experiment_results_dir,
    experiment_root,
    repo_root,
)


def _make_checkout(base: str) -> str:
    """Minimal tree carrying every marker repo_root() looks for."""
    os.makedirs(osp.join(base, "torchcell"), exist_ok=True)
    os.makedirs(osp.join(base, "experiments"), exist_ok=True)
    with open(osp.join(base, "pyproject.toml"), "w") as f:
        f.write("[project]\nname = 'x'\n")
    return base


def test_repo_root_finds_enclosing_checkout(tmp_path: Path) -> None:
    root = _make_checkout(str(tmp_path / "checkout"))
    script = osp.join(root, "experiments", "019-x", "scripts", "run.py")
    os.makedirs(osp.dirname(script), exist_ok=True)
    open(script, "w").close()
    assert repo_root(script) == root


def test_repo_root_returns_none_outside_a_checkout(tmp_path: Path) -> None:
    stray = tmp_path / "nowhere" / "file.py"
    stray.parent.mkdir(parents=True)
    stray.write_text("")
    assert repo_root(str(stray)) is None


def test_two_checkouts_resolve_independently(tmp_path: Path) -> None:
    """The actual regression: a worktree copy must NOT resolve to the primary tree."""
    primary = _make_checkout(str(tmp_path / "torchcell"))
    worktree = _make_checkout(str(tmp_path / "torchcell.worktrees" / "feat-x"))
    rel = osp.join("experiments", "019-x", "scripts", "run.py")
    for root in (primary, worktree):
        os.makedirs(osp.dirname(osp.join(root, rel)), exist_ok=True)
        open(osp.join(root, rel), "w").close()

    assert repo_root(osp.join(primary, rel)) == primary
    assert repo_root(osp.join(worktree, rel)) == worktree
    assert experiment_results_dir("019-x", osp.join(worktree, rel)).startswith(worktree)
    assert asset_images_dir(osp.join(worktree, rel)).startswith(worktree)


def test_output_dirs_are_created(tmp_path: Path) -> None:
    root = _make_checkout(str(tmp_path / "checkout"))
    script = osp.join(root, "experiments", "019-x", "scripts", "run.py")
    os.makedirs(osp.dirname(script), exist_ok=True)
    open(script, "w").close()

    results = experiment_results_dir("019-x", script)
    images = asset_images_dir(script, subdir="019-x")
    assert osp.isdir(results)
    assert osp.isdir(images)
    assert images.endswith(osp.join("notes", "assets", "images", "019-x"))


def test_env_fallback_outside_a_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stray = tmp_path / "nowhere" / "file.py"
    stray.parent.mkdir(parents=True)
    stray.write_text("")
    monkeypatch.setenv("EXPERIMENT_ROOT", "/fallback/experiments")
    monkeypatch.setenv("ASSET_IMAGES_DIR", str(tmp_path / "fallback_images"))
    assert experiment_root(str(stray)) == "/fallback/experiments"
    assert asset_images_dir(str(stray)) == str(tmp_path / "fallback_images")


def test_git_file_alone_is_not_a_checkout_marker(tmp_path: Path) -> None:
    """In a worktree `.git` is a FILE pointing elsewhere, so it cannot be the marker."""
    fake = tmp_path / "not_a_checkout"
    fake.mkdir()
    (fake / ".git").write_text("gitdir: /somewhere/else\n")
    script = fake / "run.py"
    script.write_text("")
    assert repo_root(str(script)) is None


@pytest.mark.parametrize("subdir", [None, "019-simb-multimodal"])
def test_asset_images_dir_subdir(tmp_path: Path, subdir: str | None) -> None:
    root = _make_checkout(str(tmp_path / "checkout"))
    script = osp.join(root, "torchcell", "x.py")
    open(script, "w").close()
    path = asset_images_dir(script, subdir=subdir)
    expected = osp.join(root, "notes", "assets", "images")
    assert path == (expected if subdir is None else osp.join(expected, subdir))
