# tests/torchcell/scripts/test_wt_cleanup.py
"""The worktree sweep on a throwaway repository with a bare origin.

Four worktrees and two branch-only refs cover every verdict: a landed clean tree
is removed with its local and remote branch, a landed tree with an untracked file
is kept and reported, a tree with a commit not on origin/main is kept, a detached
tree is kept unless asked for, and merged branch-only refs (local and remote) are
deleted. ``gh`` is never called (``use_gh=False``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import wt_cleanup  # type: ignore[import-not-found]  # noqa: E402


def _git(cwd: Path, *args: str) -> str:
    res = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    assert res.returncode == 0, f"git {' '.join(args)}: {res.stderr}"
    return res.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, Path]:
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin))
    main = tmp_path / "main"
    _git(tmp_path, "clone", "-q", str(origin), str(main))
    _git(main, "config", "user.email", "t@t")
    _git(main, "config", "user.name", "t")
    (main / "README").write_text("base\n")
    _git(main, "add", "README")
    _git(main, "commit", "-q", "-m", "base")
    _git(main, "push", "-q", "origin", "main")
    return main, tmp_path


def _worktree(main: Path, name: str, *, branch: bool = True) -> Path:
    wt = main.parent / "wts" / name
    if branch:
        _git(main, "worktree", "add", "-q", str(wt), "-b", name, "origin/main")
    else:
        _git(main, "worktree", "add", "-q", "--detach", str(wt), "origin/main")
    return wt


def test_sweep_classifies_and_removes(repo: tuple[Path, Path]) -> None:
    main, root = repo
    landed = _worktree(main, "landed")
    _git(main, "push", "-q", "origin", "landed")  # a remote ref to delete too
    dirty = _worktree(main, "landed-dirty")
    (dirty / "scratch.txt").write_text("uncommitted\n")
    unlanded = _worktree(main, "unlanded")
    (unlanded / "new.txt").write_text("x\n")
    _git(unlanded, "add", "new.txt")
    _git(
        unlanded,
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "-m",
        "ahead",
    )
    detached = _worktree(main, "detached", branch=False)
    _git(main, "branch", "local-only", "origin/main")
    _git(main, "push", "-q", "origin", "origin/main:refs/heads/remote-only")

    report = wt_cleanup.sweep(
        main, dry_run=False, include_detached=False, use_gh=False, cwd=root
    )

    by = {w.label: w for w in report.worktrees}
    assert by["landed"].verdict == "landed" and by["landed"].removed
    assert (
        by["landed-dirty"].verdict == "landed-dirty" and not by["landed-dirty"].removed
    )
    assert by["landed-dirty"].dirty == ["?? scratch.txt"]
    assert by["unlanded"].verdict == "unlanded" and by["unlanded"].ahead == 1
    assert [w for w in report.worktrees if w.verdict == "detached"]
    assert (
        not landed.exists()
        and dirty.exists()
        and unlanded.exists()
        and detached.exists()
    )
    assert "landed" not in _git(main, "branch", "--format=%(refname:short)").split()
    assert "landed" not in _git(main, "ls-remote", "--heads", "origin")

    branch_only = {(b.where, b.name): b.deleted for b in report.branches}
    assert branch_only == {
        ("local", "local-only"): True,
        ("remote", "remote-only"): True,
    }
    assert "remote-only" not in _git(main, "ls-remote", "--heads", "origin")
    assert report.color == "yellow"  # the dirty tree needs a decision
    assert "1 LANDED TREE(S)" in wt_cleanup.banner(report)


def test_dry_run_removes_nothing(repo: tuple[Path, Path]) -> None:
    main, root = repo
    landed = _worktree(main, "landed")
    report = wt_cleanup.sweep(
        main, dry_run=True, include_detached=False, use_gh=False, cwd=root
    )
    assert report.worktrees[0].verdict == "landed"
    assert not report.worktrees[0].removed and landed.exists()
    assert report.color == "green"
    assert "(dry run)" in wt_cleanup.banner(report)


def test_detached_removed_only_when_asked(repo: tuple[Path, Path]) -> None:
    main, root = repo
    detached = _worktree(main, "detached", branch=False)
    kept = wt_cleanup.sweep(
        main, dry_run=False, include_detached=False, use_gh=False, cwd=root
    )
    assert kept.worktrees[0].verdict == "detached" and detached.exists()
    swept = wt_cleanup.sweep(
        main, dry_run=False, include_detached=True, use_gh=False, cwd=root
    )
    assert swept.worktrees[0].removed and not detached.exists()


def test_cwd_worktree_is_never_removed(repo: tuple[Path, Path]) -> None:
    main, _ = repo
    here = _worktree(main, "here")
    report = wt_cleanup.sweep(
        main, dry_run=False, include_detached=False, use_gh=False, cwd=here.resolve()
    )
    assert report.worktrees[0].verdict == "cwd" and here.exists()
