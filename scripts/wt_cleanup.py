#!/usr/bin/env python
# scripts/wt_cleanup.py
# [[scripts.wt_cleanup]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/wt_cleanup
"""Sweep worktrees and branches whose work has landed on ``origin/main``.

The merge-queue drainer removes the worktree, the local branch and the remote
branch of every branch *it* lands. Everything that lands by another route, or
never lands, piles up: a plan worktree whose note went in with a sibling, a
detached build worktree, a branch pushed for a PR that was closed by hand. This
script is the sweep that runs after ``/enqueue-merge`` (and on demand as
``/wt-cleanup``) and ends with its own broom banner, the way the drainer ends
with the dove.

Rules, all judged against ``origin/main`` after ``git fetch --prune`` (local
``main`` lags every landing, see [[vanished-worktree-check-origin-main]]):

- **landed**: every commit of the worktree is an ancestor of ``origin/main`` and
  ``git status --short`` is empty. Removed: worktree, local branch, remote
  branch, and any PR still open on it is closed with a comment.
- **landed-dirty**: landed, but the tree holds uncommitted or untracked files.
  Kept, and listed under NEED YOU: the files are the only copy of that work.
- **unlanded**: commits not on ``origin/main``. Kept; in-flight work.
- **detached**: no branch (build and bisect trees). Kept unless ``--detached``,
  and then only when landed and clean.
- Branches with no worktree, local or remote, that are ancestors of
  ``origin/main`` are deleted too; ``main`` and the checkout's own branch never.

The landing flock is taken for the whole sweep, so it cannot interleave with a
drainer removing the same worktree.

    python scripts/wt_cleanup.py            # sweep, table, banner
    python scripts/wt_cleanup.py --dry-run  # classify only, nothing removed
    python scripts/wt_cleanup.py --json     # machine-readable report

Exit codes: 0 swept clean (green), 3 landed-dirty trees need a decision
(yellow), 2 a removal failed (red), 4 the landing flock stayed busy.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal, TextIO

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_queue  # noqa: E402  -- sibling script, needs sys.path mutation

REPO_SLUG = "Mjvolk3/torchcell"
PROTECTED_BRANCHES = frozenset({"main", "master", "HEAD"})
DIRTY_PREVIEW = 3

Verdict = Literal["landed", "landed-dirty", "unlanded", "detached", "cwd"]


class WorktreeState(BaseModel):
    """One ``git worktree list`` row after classification and action."""

    path: Path
    branch: str | None
    head: str
    ahead: int
    dirty: list[str] = Field(default_factory=list)
    remote: bool = False
    pr: int | None = None
    verdict: Verdict
    removed: bool = False
    error: str | None = None

    @property
    def label(self) -> str:
        """Branch name, or the short head for a detached tree."""
        return self.branch or f"(detached {self.head[:9]})"


class BranchState(BaseModel):
    """A local or remote branch with no worktree, already on ``origin/main``."""

    name: str
    where: Literal["local", "remote"]
    deleted: bool = False
    error: str | None = None


class SweepReport(BaseModel):
    """One sweep: every worktree and branch-only ref, with what happened to it."""

    main: Path
    origin_main: str
    dry_run: bool
    worktrees: list[WorktreeState]
    branches: list[BranchState]

    @property
    def removed(self) -> list[WorktreeState]:
        """Worktrees removed in this pass."""
        return [w for w in self.worktrees if w.removed]

    @property
    def dirty(self) -> list[WorktreeState]:
        """Landed worktrees kept because they hold uncommitted work."""
        return [w for w in self.worktrees if w.verdict == "landed-dirty"]

    @property
    def kept(self) -> list[WorktreeState]:
        """Worktrees kept as in flight: unlanded, detached, or the shell's own."""
        return [
            w for w in self.worktrees if w.verdict in ("unlanded", "detached", "cwd")
        ]

    @property
    def failed(self) -> list[str]:
        """Human-readable removal failures, worktrees then branch-only refs."""
        out = [f"{w.label}: {w.error}" for w in self.worktrees if w.error]
        out += [f"{b.where} {b.name}: {b.error}" for b in self.branches if b.error]
        return out

    @property
    def color(self) -> Literal["green", "yellow", "red"]:
        """Banner color: red on any failure, yellow on dirty landed trees, else green."""
        if self.failed:
            return "red"
        if self.dirty:
            return "yellow"
        return "green"


# ── git plumbing ───────────────────────────────────────────────────────────


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a command capturing text. Never raises on non-zero exit."""
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True)


def _git(main: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", "-C", str(main), *args], cwd=main)


def resolve_main(start: Path) -> Path:
    """The primary checkout that owns the shared ``.git``, from any worktree."""
    common = _git(start, "rev-parse", "--git-common-dir")
    if common.returncode != 0:
        raise SystemExit(f"not a git repository: {start}")
    return (start / common.stdout.strip()).resolve().parent


def list_worktrees(main: Path) -> list[tuple[Path, str, str | None]]:
    """``(path, head, branch)`` per worktree, primary first, from porcelain."""
    out = _git(main, "worktree", "list", "--porcelain").stdout
    rows: list[tuple[Path, str, str | None]] = []
    path: Path | None = None
    head = ""
    branch: str | None = None
    for line in out.splitlines() + [""]:
        if line.startswith("worktree "):
            path = Path(line[len("worktree ") :])
        elif line.startswith("HEAD "):
            head = line[len("HEAD ") :]
        elif line.startswith("branch "):
            branch = line[len("branch ") :].removeprefix("refs/heads/")
        elif line == "" and path is not None:
            rows.append((path, head, branch))
            path, head, branch = None, "", None
    return rows


def ahead_of_origin_main(main: Path, head: str) -> int:
    """Commits reachable from ``head`` that are not on ``origin/main``."""
    res = _git(main, "rev-list", "--count", f"origin/main..{head}")
    if res.returncode != 0:
        raise SystemExit(f"rev-list failed for {head}: {res.stderr.strip()}")
    return int(res.stdout.strip())


def dirty_paths(wt: Path) -> list[str]:
    """``git status --short`` lines: modified, staged and untracked, never ignored."""
    res = _run(["git", "status", "--short", "--untracked-files=all"], cwd=wt)
    return [line for line in res.stdout.splitlines() if line.strip()]


def remote_branch_exists(main: Path, branch: str) -> bool:
    """Whether ``origin/<branch>`` exists locally after the pruning fetch."""
    res = _git(main, "show-ref", "--verify", "--quiet", f"refs/remotes/origin/{branch}")
    return res.returncode == 0


def open_pr_number(branch: str, use_gh: bool) -> int | None:
    """The open PR number for ``branch`` via ``gh``, or None when unknown or skipped."""
    if not use_gh or shutil.which("gh") is None:
        return None
    res = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            REPO_SLUG,
            "--head",
            branch,
            "--state",
            "open",
            "--json",
            "number",
            "--jq",
            ".[0].number",
        ],
        capture_output=True,
        text=True,
    )
    text = res.stdout.strip()
    return int(text) if res.returncode == 0 and text else None


def merged_branches(main: Path, remote: bool) -> list[str]:
    """Branch names (without ``origin/``) that are ancestors of ``origin/main``."""
    args = ["branch", "--format=%(refname:short)", "--merged", "origin/main"]
    if remote:
        args.insert(1, "-r")
    names = []
    for raw in _git(main, *args).stdout.splitlines():
        name = raw.strip()
        if remote:
            if not name.startswith("origin/") or name == "origin":
                continue
            name = name.removeprefix("origin/")
        if name and name not in PROTECTED_BRANCHES:
            names.append(name)
    return names


# ── classification + actions ───────────────────────────────────────────────


def classify(
    main: Path, path: Path, head: str, branch: str | None, cwd: Path, use_gh: bool
) -> WorktreeState:
    """Assign one worktree its verdict (see the module docstring for the rules)."""
    ahead = ahead_of_origin_main(main, head)
    dirty = dirty_paths(path)
    remote = remote_branch_exists(main, branch) if branch else False
    pr = open_pr_number(branch, use_gh) if branch else None
    verdict: Verdict
    if path == cwd:
        verdict = "cwd"
    elif branch is None:
        verdict = "detached"
    elif ahead > 0:
        verdict = "unlanded"
    elif dirty:
        verdict = "landed-dirty"
    else:
        verdict = "landed"
    return WorktreeState(
        path=path,
        branch=branch,
        head=head,
        ahead=ahead,
        dirty=dirty,
        remote=remote,
        pr=pr,
        verdict=verdict,
    )


def _delete_remote(main: Path, branch: str) -> str | None:
    res = _git(main, "push", "origin", "--delete", branch)
    return None if res.returncode == 0 else res.stderr.strip()


def _close_pr(pr: int, use_gh: bool) -> str | None:
    if not use_gh or shutil.which("gh") is None:
        return None
    res = subprocess.run(
        [
            "gh",
            "pr",
            "close",
            str(pr),
            "--repo",
            REPO_SLUG,
            "--comment",
            "Every commit is already on `main`; the worktree sweep (`scripts/wt_cleanup.py`) "
            "is removing the branch. Closing.",
        ],
        capture_output=True,
        text=True,
    )
    return None if res.returncode == 0 else res.stderr.strip()


def remove_worktree(main: Path, state: WorktreeState, use_gh: bool) -> None:
    """Remove a landed worktree and its branch everywhere. Records the first error."""
    res = _git(main, "worktree", "remove", "--force", str(state.path))
    if res.returncode != 0:
        state.error = f"worktree remove: {res.stderr.strip()}"
        return
    if state.branch is not None:
        res = _git(main, "branch", "-D", state.branch)
        if res.returncode != 0:
            state.error = f"branch -D: {res.stderr.strip()}"
            return
        if state.pr is not None:
            err = _close_pr(state.pr, use_gh)
            if err:
                state.error = f"pr close #{state.pr}: {err}"
                return
        if state.remote:
            err = _delete_remote(main, state.branch)
            if err:
                state.error = f"remote delete: {err}"
                return
    state.removed = True


def sweep(
    main: Path, *, dry_run: bool, include_detached: bool, use_gh: bool, cwd: Path
) -> SweepReport:
    """Fetch, classify every worktree and branch-only ref, then act unless ``dry_run``."""
    fetched = _git(main, "fetch", "origin", "--prune", "--quiet")
    if fetched.returncode != 0:
        raise SystemExit(f"git fetch origin --prune failed: {fetched.stderr.strip()}")
    origin_main = _git(main, "rev-parse", "origin/main").stdout.strip()

    rows = list_worktrees(main)
    states: list[WorktreeState] = []
    for path, head, branch in rows[1:]:  # rows[0] is the primary checkout
        states.append(classify(main, path, head, branch, cwd, use_gh))

    # Local branches with no worktree, and remote branches, already on origin/main.
    with_worktree = {s.branch for s in states if s.branch}
    branches: list[BranchState] = []
    for name in merged_branches(main, remote=False):
        if name not in with_worktree:
            branches.append(BranchState(name=name, where="local"))
    for name in merged_branches(main, remote=True):
        if name not in with_worktree:
            branches.append(BranchState(name=name, where="remote"))

    if dry_run:
        return SweepReport(
            main=main,
            origin_main=origin_main,
            dry_run=True,
            worktrees=states,
            branches=branches,
        )

    for state in states:
        removable = state.verdict == "landed" or (
            include_detached
            and state.verdict == "detached"
            and state.ahead == 0
            and not state.dirty
        )
        if removable:
            remove_worktree(main, state, use_gh)
    for b in branches:
        if b.where == "local":
            res = _git(main, "branch", "-D", b.name)
            b.deleted = res.returncode == 0
            b.error = None if b.deleted else res.stderr.strip()
        else:
            err = _delete_remote(main, b.name)
            b.deleted = err is None
            b.error = err
    return SweepReport(
        main=main,
        origin_main=origin_main,
        dry_run=False,
        worktrees=states,
        branches=branches,
    )


# ── output ─────────────────────────────────────────────────────────────────


def table(report: SweepReport) -> str:
    """One row per worktree, then one per branch-only ref."""
    lines = [
        f"{'VERDICT':<13} {'AHEAD':>5} {'DIRTY':>5} {'REMOTE':<6} {'PR':<5} WORKTREE"
    ]
    for w in report.worktrees:
        lines.append(
            f"{w.verdict:<13} {w.ahead:>5} {len(w.dirty):>5} "
            f"{'yes' if w.remote else '-':<6} {w.pr or '-':<5} {w.label}"
        )
    for b in report.branches:
        what = (
            "deleted" if b.deleted else ("would delete" if report.dry_run else "FAILED")
        )
        lines.append(
            f"{'branch-only':<13} {'0':>5} {'-':>5} {b.where:<6} {'-':<5} {b.name}  [{what}]"
        )
    return "\n".join(lines)


_BROOM = "\U0001f9f9"
_GREEN, _RED, _YELLOW = "\U0001f7e2", "\U0001f534", "\U0001f7e1"
_CHECK, _STOP, _PAUSE = "✅", "\U0001f6d1", "⏸️"


def _bell(count: int) -> None:
    sys.stdout.write("\a" * count)
    sys.stdout.flush()


def banner(report: SweepReport) -> str:
    """The broom banner: green swept clean, yellow needs you, red a removal failed."""
    verb = "WOULD REMOVE" if report.dry_run else "removed"
    removed = [
        w.label
        for w in (
            report.removed
            if not report.dry_run
            else [w for w in report.worktrees if w.verdict == "landed"]
        )
    ]
    deleted = [
        f"{b.where}:{b.name}" for b in report.branches if b.deleted or report.dry_run
    ]
    body = [
        f"   {verb:<9} {len(removed)} worktree(s)"
        + (f": {', '.join(removed)}" if removed else ""),
        f"   branches  {len(deleted)} branch-only ref(s)"
        + (f": {', '.join(deleted)}" if deleted else ""),
        f"   kept      {len(report.kept)} in flight (unlanded / detached / cwd)",
    ]
    for w in report.dirty:
        preview = ", ".join(p.strip() for p in w.dirty[:DIRTY_PREVIEW])
        more = (
            f" (+{len(w.dirty) - DIRTY_PREVIEW})"
            if len(w.dirty) > DIRTY_PREVIEW
            else ""
        )
        body.append(f"   dirty     {w.label}: {len(w.dirty)} path(s) {preview}{more}")
    for f in report.failed:
        body.append(f"   failed    {f}")

    if report.color == "red":
        bar = _RED * 3
        head = f"{_BROOM} {bar}  {_STOP} SWEEP FAILED - NEEDS YOU  {bar} {_BROOM}"
    elif report.color == "yellow":
        bar = _YELLOW * 3
        head = (
            f"{_BROOM} {bar}  {_PAUSE} SWEPT - {len(report.dirty)} LANDED TREE(S) "
            f"HOLD UNCOMMITTED WORK  {bar} {_BROOM}"
        )
        body.append(
            "   commit or move that work, then re-run the sweep; nothing dirty is ever removed."
        )
    else:
        bar = _GREEN * 3
        head = f"{_BROOM} {bar}  {_CHECK} SWEPT - NOTHING LEFT BEHIND  {bar} {_BROOM}"
    if report.dry_run:
        head += "  (dry run)"
    return "\n".join([head, *body])


# ── flock ──────────────────────────────────────────────────────────────────


def acquire_lock(lock_path: Path, wait_s: float) -> TextIO | None:
    """Take the landing flock, polling up to ``wait_s``. None if still busy."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w")
    deadline = time.monotonic() + wait_s
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            if time.monotonic() >= deadline:
                handle.close()
                return None
            time.sleep(2)


# ── cli ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """CLI entry: take the flock, sweep, print the table and the banner last."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--dry-run", action="store_true", help="classify and report; remove nothing"
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="print the report as JSON instead of the table",
    )
    ap.add_argument(
        "--detached",
        action="store_true",
        help="also remove detached worktrees that are landed and clean",
    )
    ap.add_argument(
        "--no-gh", action="store_true", help="skip PR lookups and PR closes"
    )
    ap.add_argument(
        "--lock-wait",
        type=float,
        default=120.0,
        help="seconds to wait for the landing flock (default 120)",
    )
    ap.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="any checkout of the repository (default: this script's)",
    )
    args = ap.parse_args(argv)

    main_path = resolve_main(args.repo)
    cwd = Path(os.getcwd()).resolve()
    handle = acquire_lock(
        merge_queue._lock_path(merge_queue.DEFAULT_DB_PATH), args.lock_wait
    )
    if handle is None:
        print(
            f"{_BROOM} {_YELLOW * 3}  {_PAUSE} SWEEP SKIPPED - LANDING FLOCK BUSY  {_YELLOW * 3} {_BROOM}"
        )
        print(
            "   a drainer or manual landing holds the lock; re-run after it finishes."
        )
        return 4
    try:
        report = sweep(
            main_path,
            dry_run=args.dry_run,
            include_detached=args.detached,
            use_gh=not args.no_gh,
            cwd=cwd,
        )
    finally:
        handle.close()

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(table(report))
        print()
        print(banner(report))
    if report.color == "red":
        _bell(3)
        return 2
    if report.color == "yellow":
        return 3
    _bell(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
