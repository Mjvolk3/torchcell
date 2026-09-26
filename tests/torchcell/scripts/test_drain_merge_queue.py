# tests/torchcell/scripts/test_drain_merge_queue.py
# [[tests.torchcell.scripts.test_drain_merge_queue]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/scripts/test_drain_merge_queue.py
"""``scripts/drain_merge_queue.py`` landing branches into a bare origin under ``tmp_path``.

The drainer once force-pushed a test's temp history to the real ``main`` through inherited
``GIT_DIR`` (Gotcha 9 of [[plan.test-suite-buildout.2026.09.25]]), so this file is built
to make that impossible: ``_cleanup_remote`` is replaced by a recorder (it would call
``gh``), ``shutil.which`` returns None, the Slack webhook and the four ``GIT_*`` variables
are deleted, the working directory is the temp main, ``HOME`` and the global git config
point away from the developer's, and an autouse teardown asserts every repository under
``tmp_path`` has its ``origin`` under ``tmp_path``.

Layout mirrors production: ``<tmp>/torchcell`` is the main clone and
``<tmp>/torchcell.worktrees/<branch>`` the worktree ``_worktree_for`` derives.
"""

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import drain_merge_queue  # type: ignore[import-not-found]  # noqa: E402
import merge_queue  # type: ignore[import-not-found]  # noqa: E402


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True
    )
    assert result.returncode == 0, f"git {' '.join(args)}: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture(autouse=True)
def _hermetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """No gh, no Slack, no inherited git env, cwd in tmp; every origin must be under tmp."""
    for var in (
        "SLACK_CLAUDE_WEBHOOK",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_PREFIX",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    remote_calls: list[Any] = []
    monkeypatch.setattr(
        drain_merge_queue, "_cleanup_remote", lambda *a: remote_calls.append(a)
    )
    monkeypatch.setattr(drain_merge_queue.shutil, "which", lambda name: None)
    (tmp_path / "torchcell").mkdir()
    monkeypatch.chdir(tmp_path / "torchcell")
    yield remote_calls
    for git_marker in tmp_path.rglob(".git"):
        repo = git_marker.parent
        url = subprocess.run(
            ["git", "-C", str(repo), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert url == "" or url.startswith(str(tmp_path)), f"{repo} points at {url}"


@pytest.fixture
def repos(tmp_path: Path) -> tuple[Path, Path, Path]:
    """(origin, main clone, worktree on branch plan/feat-1 with one commit ahead)."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin))
    main = tmp_path / "torchcell"
    _git(tmp_path, "clone", "-q", str(origin), str(main))
    _git(main, "config", "user.email", "t@t")
    _git(main, "config", "user.name", "t")
    (main / "README").write_text("base\n")
    _git(main, "add", "README")
    _git(main, "commit", "-q", "-m", "base")
    _git(main, "push", "-q", "origin", "main")
    wt = tmp_path / "torchcell.worktrees" / "plan" / "feat-1"
    _git(main, "worktree", "add", "-q", str(wt), "-b", "plan/feat-1", "origin/main")
    (wt / "feature.py").write_text("x = 1\n")
    _git(wt, "add", "feature.py")
    _git(wt, "commit", "-q", "-m", "feat: feat-1")
    return origin, main, wt


def test_drain_lands_the_branch_and_cleans_up(
    tmp_path: Path, repos: tuple[Path, Path, Path], _hermetic: list[Any]
) -> None:
    """origin/main advances to the branch tip, the row is landed with that sha, the worktree
    and local branch are gone, and the remote cleanup was asked for exactly once.
    """
    origin, main, wt = repos
    tip = _git(wt, "rev-parse", "HEAD")
    db = tmp_path / "queue" / "merge_queue.db"
    merge_queue.add_entry(db, "plan/feat-1", worktree=str(wt))

    summary = drain_merge_queue.drain(main, db)

    assert summary == {
        "sweep": "no free notes to sweep",
        "landed": ["plan/feat-1"],
        "blocked": [],
    }
    assert _git(origin, "rev-parse", "main") == tip
    row = merge_queue.branch_row(db, "plan/feat-1")
    assert row is not None and (row["status"], row["landed_sha"]) == ("landed", tip)
    assert not wt.exists()
    assert _git(main, "branch", "--list", "plan/feat-1") == ""
    assert _hermetic == [(main, "plan/feat-1", tip)]
    assert (db.parent / "loop.heartbeat").is_file()


def test_drain_blocks_a_missing_or_dirty_worktree(
    tmp_path: Path, repos: tuple[Path, Path, Path]
) -> None:
    """A recorded worktree that is absent, or one with uncommitted changes, blocks with the reason."""
    origin, main, wt = repos
    before = _git(origin, "rev-parse", "main")
    db = tmp_path / "queue" / "merge_queue.db"
    merge_queue.add_entry(db, "plan/ghost", worktree=str(tmp_path / "nope"))
    (wt / "scratch.txt").write_text("dirty\n")
    merge_queue.add_entry(db, "plan/feat-1", worktree=str(wt))

    summary = drain_merge_queue.drain(main, db)

    assert summary["landed"] == []
    assert summary["blocked"] == [
        f"plan/ghost (worktree missing: {tmp_path / 'nope'})",
        "plan/feat-1 (uncommitted changes in worktree)",
    ]
    assert _git(origin, "rev-parse", "main") == before
    ghost = merge_queue.branch_row(db, "plan/ghost")
    assert (
        ghost is not None
        and ghost["last_error"] == f"worktree missing: {tmp_path / 'nope'}"
    )
    dirty = merge_queue.branch_row(db, "plan/feat-1")
    assert (
        dirty is not None and dirty["last_error"] == "uncommitted changes in worktree"
    )
    assert wt.exists()


def test_drain_blocks_a_rebase_conflict_and_leaves_the_worktree_clean(
    tmp_path: Path, repos: tuple[Path, Path, Path]
) -> None:
    """A conflicting change on origin/main blocks the branch; the aborted rebase leaves no state."""
    origin, main, wt = repos
    (main / "feature.py").write_text("x = 2\n")
    _git(main, "add", "feature.py")
    _git(main, "commit", "-q", "-m", "conflicting main")
    _git(main, "push", "-q", "origin", "main")
    main_tip = _git(origin, "rev-parse", "main")
    db = tmp_path / "queue" / "merge_queue.db"
    merge_queue.add_entry(db, "plan/feat-1", worktree=str(wt))

    summary = drain_merge_queue.drain(main, db)

    assert summary["blocked"] == ["plan/feat-1 (rebase conflict)"]
    row = merge_queue.branch_row(db, "plan/feat-1")
    assert row is not None and row["last_error"] == "rebase conflict onto origin/main"
    assert _git(origin, "rev-parse", "main") == main_tip
    assert _git(wt, "status", "--short") == ""
    assert _git(wt, "rev-parse", "--abbrev-ref", "HEAD") == "plan/feat-1"
    assert (
        merge_queue.classify_watch("blocked", row["last_error"], 0.0, 570.0)
        == "resolve_conflict"
    )


def _track_notes_dir(main: Path) -> None:
    """Commit and push a placeholder so ``notes/`` is a tracked directory, as in production."""
    (main / "notes").mkdir()
    (main / "notes" / "assets").mkdir()
    (main / "notes" / "assets" / "keep.md").write_text("assets\n")
    _git(main, "add", "notes/assets/keep.md")
    _git(main, "commit", "-q", "-m", "notes dir")
    _git(main, "push", "-q", "origin", "main")


def test_sweep_commits_and_pushes_free_notes_only(
    tmp_path: Path, repos: tuple[Path, Path, Path]
) -> None:
    """An untracked weekly note is swept to origin/main; a paired note is not a free note."""
    origin, main, _ = repos
    _track_notes_dir(main)
    (main / "notes" / "user.weekly.md").write_text("- [ ] task\n")
    (main / "notes" / "torchcell.models.dcell.md").write_text("paired\n")
    assert drain_merge_queue.sweep_free_notes(main) == "swept 1 free note(s) to main"
    assert (
        _git(origin, "log", "-1", "--format=%s", "main")
        == "notes: sweep free notes (merge-queue drainer)"
    )
    assert _git(origin, "ls-tree", "--name-only", "-r", "main", "notes/") == (
        "notes/assets/keep.md\nnotes/user.weekly.md"
    )
    assert drain_merge_queue.sweep_free_notes(main) == "no free notes to sweep"
    assert (main / "notes" / "torchcell.models.dcell.md").exists()


def test_sweep_misses_a_note_inside_an_untracked_directory(
    repos: tuple[Path, Path, Path],
) -> None:
    """``git status --porcelain`` collapses an untracked directory to one ``?? dir/`` line, so
    a free note inside a directory git has never seen is not swept. Pinned as behavior: a
    brand-new ``notes/`` or ``.claude/rules/`` tree needs a first commit by hand.
    """
    origin, main, _ = repos
    (main / "notes").mkdir()
    (main / "notes" / "user.weekly.md").write_text("- [ ] task\n")
    assert drain_merge_queue._dirty_free_notes(main) == []
    assert drain_merge_queue.sweep_free_notes(main) == "no free notes to sweep"
    assert _git(origin, "log", "-1", "--format=%s", "main") == "base"


def test_sweep_refuses_a_diverged_local_main(
    tmp_path: Path, repos: tuple[Path, Path, Path]
) -> None:
    """An unpushed commit on local main makes the sweep skip rather than push it."""
    origin, main, _ = repos
    (main / "local.txt").write_text("local\n")
    _git(main, "add", "local.txt")
    _git(main, "commit", "-q", "-m", "local only")
    (main / "notes").mkdir()
    (main / "notes" / "user.weekly.md").write_text("x\n")
    # The divergence check runs before the note scan, so the untracked directory is moot here.
    assert (
        drain_merge_queue.sweep_free_notes(main)
        == "sweep skipped: local main diverged (1 ahead)"
    )
    assert _git(origin, "log", "-1", "--format=%s", "main") == "base"


@pytest.mark.parametrize(
    ("path", "free"),
    [
        ("notes/user.weekly.md", True),
        ("notes/plan.thing.2026.09.25.md", True),
        (".claude/rules/data.md", True),
        ("notes/torchcell.models.dcell.md", False),
        ("notes/scratch.idea.md", False),
        ("notes/experiments.010.md", False),
        ("notes/scripts.ops.md", False),
        ("notes/assets/images/x.md", False),
        ("torchcell/x.py", False),
        ("notes/user.weekly.txt", False),
    ],
)
def test_is_free_note(path: str, free: bool) -> None:
    """Free notes are notes/*.md outside assets and the paired/scratch prefixes, plus .claude/rules."""
    assert drain_merge_queue._is_free_note(path) is free


def test_landing_lock_is_exclusive_and_released_on_close(tmp_path: Path) -> None:
    """A second non-blocking acquisition fails while the first handle is open, succeeds after."""
    lock = tmp_path / "queue" / "landing.lock"
    first = drain_merge_queue._acquire_lock(lock)
    assert first is not None
    assert drain_merge_queue._acquire_lock(lock) is None
    first.close()
    third = drain_merge_queue._acquire_lock(lock)
    assert third is not None
    third.close()


def test_main_strips_inherited_git_env_and_reports_an_idle_drain(
    tmp_path: Path,
    repos: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """GIT_DIR and friends are dropped before any git call; an empty queue prints idle.

    The environment check is the discriminating assertion: with GIT_DIR left in place
    every git call fails silently and the idle line is printed all the same.
    """
    _, main, _ = repos
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "evil.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    db = tmp_path / "queue" / "merge_queue.db"
    assert drain_merge_queue.main(["--db", str(db), "--main", str(main)]) == 0
    assert (
        capsys.readouterr().out.strip()
        == "merge-queue drain: idle (no free notes to sweep)"
    )
    assert "GIT_DIR" not in os.environ and "GIT_WORK_TREE" not in os.environ
    assert (db.parent / "loop.heartbeat").is_file()
