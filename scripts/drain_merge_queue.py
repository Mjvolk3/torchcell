"""
scripts/drain_merge_queue
[[scripts.drain_merge_queue]]
https://github.com/Mjvolk3/torchcell/tree/main/scripts/drain_merge_queue.py

Deterministic, model-free drainer for the single-writer merge queue
([[scripts.merge_queue]]).

Why this exists: the landing work -- rebase onto origin/main, ff-push the tip,
close the PR, remove the worktree -- is entirely deterministic. Running it as a
model loop would burn tokens to do nothing on idle ticks. This script is pure
git/gh orchestration, so it costs zero tokens and can be triggered two ways:

- **Event-driven** -- ``/enqueue-merge`` runs it right after adding a branch, so
  a landing fires the instant work arrives.
- **Cheap cron safety-net** -- a plain ``*/2`` crontab entry runs it to pick up
  any branch orphaned by a session that died mid-drain (the OS releases its
  flock on death, but its un-landed rows still need a drainer).

Either way it self-guards with a **non-blocking flock**: at most one drainer
runs at a time, so two concurrent triggers cannot race the shared ``.git``. The
holder drains the whole queue (claims in a loop), so a trigger that loses the
lock is fine -- the active drainer picks up the freshly-added branch.

Landings go **worktree -> origin** (``push HEAD:main``), never via local
``main``, so a stale/diverged local ``main`` cannot block them. This also rides
out the semantic-release bump commit CI pushes to ``main`` after each landing:
the push retry re-fetches origin/main and re-rebases before pushing again.

Each drain first **sweeps free notes** edited on ``main`` to origin (weekly task
notes and other standalone notes -- not paired module/experiment/script notes,
which travel with their source through a worktree). The sweep is divergence-aware
and isolated from landings, mirroring iBioFoundry-AI's drainer.
"""

# scripts/drain_merge_queue.py
# [[scripts.drain_merge_queue]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/drain_merge_queue.py

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_queue  # noqa: E402  -- sibling script, needs sys.path mutation

# The repo that contains this script -- robust to where it is invoked from
# (cron, an enqueuing session, a test). parents[1] of scripts/x.py is the repo.
DEFAULT_MAIN = Path(__file__).resolve().parents[1]

REPO_SLUG = "Mjvolk3/torchcell"

# A "free note" is swept to main; scratch and paired module/experiment/script
# notes are not -- those travel with their source through a worktree. We
# classify explicit dirty paths in Python and `git add` exactly those, rather
# than rely on `git add <pathspec>` (which aborts the whole command if any
# single pathspec matches nothing).
_PAIRED_OR_SCRATCH_PREFIXES = ("scratch.", "torchcell.", "experiments.", "scripts.")


def _is_free_note(path: str) -> bool:
    """True for sweepable notes: notes/*.md minus scratch/paired/assets, and
    .claude/rules/*.md."""
    if path.endswith(".md") and path.startswith(".claude/rules/"):
        return True
    if not (path.startswith("notes/") and path.endswith(".md")):
        return False
    rest = path[len("notes/") :]
    if rest.startswith("assets/"):
        return False
    return not rest.startswith(_PAIRED_OR_SCRATCH_PREFIXES)


def _dirty_free_notes(main: Path) -> list[str]:
    """Changed/untracked free-note paths under notes/ and .claude/rules/."""
    out = _run(
        ["git", "-C", str(main), "status", "--porcelain", "--", "notes", ".claude/rules"]
    ).stdout
    paths: list[str] = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:  # rename entry: take the destination path
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"')
        if _is_free_note(path):
            paths.append(path)
    return paths


def _run(
    args: list[str], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing text output. Never raises on non-zero exit.

    Every call targets its repo explicitly (`git -C`, absolute paths), so the
    process cwd is irrelevant to behavior -- but the cwd must still *exist* for
    the fork. Default to DEFAULT_MAIN (always present) so a caller running with
    a since-deleted cwd (e.g. a test worker after a sibling cleaned its tmp dir)
    cannot crash the fork.
    """
    return subprocess.run(
        args, cwd=str(cwd) if cwd else str(DEFAULT_MAIN),
        capture_output=True, text=True,
    )


def _slack(message: str) -> None:
    """Post to SLACK_CLAUDE_WEBHOOK if set. Best-effort -- never raises.

    Block alerts go here. When the webhook env var is absent the block is still
    recorded in the queue and the log, so Slack is a bonus, not a dependency.
    """
    webhook = os.environ.get("SLACK_CLAUDE_WEBHOOK")
    if not webhook:
        return
    payload = json.dumps({"text": message}).encode()
    request = urllib.request.Request(
        webhook, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(request, timeout=10).read()
    except OSError as exc:  # network/HTTP failure must not break a landing
        print(f"slack post failed (non-fatal): {exc}", file=sys.stderr)


def _acquire_lock(lock_path: Path) -> TextIO | None:
    """Take the landing flock non-blocking. Return the held fd, or None.

    Returning None means another drainer holds it -- the caller exits, and that
    active drainer will claim whatever we just enqueued. The lock releases when
    the fd is closed or the process dies (so a crashed drainer never wedges).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _worktree_for(main: Path, branch: str, recorded: str | None) -> Path:
    """Resolve the worktree path for a branch (recorded value wins)."""
    if recorded:
        return Path(recorded)
    return main.parent / "torchcell.worktrees" / branch


def sweep_free_notes(main: Path) -> str:
    """Commit + push free notes to main. Divergence-aware. Returns a status line.

    Best-effort and isolated from landings: a sweep failure logs and returns,
    never aborting the drain. Landings do not touch `main`'s working tree, so a
    stale sweep cannot block them.
    """
    _run(["git", "-C", str(main), "fetch", "origin"])
    counts = _run(
        ["git", "-C", str(main), "rev-list", "--left-right", "--count", "main...origin/main"]
    )
    ahead_behind = counts.stdout.split()
    ahead = int(ahead_behind[0]) if len(ahead_behind) == 2 else 0
    behind = int(ahead_behind[1]) if len(ahead_behind) == 2 else 0
    if ahead > 0:
        # Local main carries commits not on origin (a session committed straight
        # to local main). Do NOT rebase shared local main (sibling-orphan
        # hazard) -- skip the sweep; the landing path is unaffected.
        return f"sweep skipped: local main diverged ({ahead} ahead)"
    if behind > 0:
        ff = _run(["git", "-C", str(main), "merge", "--ff-only", "origin/main"])
        if ff.returncode != 0:
            return "sweep skipped: ff-sync of local main failed"
    notes = _dirty_free_notes(main)
    if not notes:
        return "no free notes to sweep"
    _run(["git", "-C", str(main), "add", "--", *notes])
    for attempt in (1, 2):  # one retry: a hook may auto-fix a note and abort
        commit = _run(
            ["git", "-C", str(main), "commit", "-m", "notes: sweep free notes (merge-queue drainer)"]
        )
        if commit.returncode == 0:
            break
        _run(["git", "-C", str(main), "add", "--", *notes])
        if attempt == 2:
            return "sweep commit failed after auto-fix retry (notes left staged)"
    push = _run(["git", "-C", str(main), "push", "origin", "main"])
    if push.returncode != 0:
        return "sweep commit landed locally but push failed"
    return f"swept {len(notes)} free note(s) to main"


def land_branch(
    main: Path, db: Path, entry: merge_queue.QueueEntry
) -> tuple[bool, str]:
    """Land one claimed branch worktree->origin. Returns (landed, detail).

    On failure the entry is marked blocked here and (False, reason) returned;
    on success it is marked landed and cleaned up.
    """
    wt = _worktree_for(main, entry.branch, entry.worktree)
    if not wt.exists():
        merge_queue.mark_blocked(db, entry.branch, f"worktree missing: {wt}")
        return False, f"worktree missing: {wt}"
    dirty = _run(["git", "-C", str(wt), "status", "--short"])
    if dirty.stdout.strip():
        merge_queue.mark_blocked(db, entry.branch, "uncommitted changes in worktree")
        return False, "uncommitted changes in worktree"

    _run(["git", "-C", str(wt), "fetch", "origin"])
    rebase = _run(["git", "-C", str(wt), "rebase", "origin/main"])
    if rebase.returncode != 0:
        _run(["git", "-C", str(wt), "rebase", "--abort"])
        merge_queue.mark_blocked(db, entry.branch, "rebase conflict onto origin/main")
        return False, "rebase conflict"

    push = _run(["git", "-C", str(wt), "push", "origin", "HEAD:main"])
    if push.returncode != 0:
        # One retry covers a remote that moved between fetch and push (e.g. the
        # semantic-release bump commit CI pushed after the previous landing).
        _run(["git", "-C", str(wt), "fetch", "origin"])
        _run(["git", "-C", str(wt), "rebase", "origin/main"])
        push = _run(["git", "-C", str(wt), "push", "origin", "HEAD:main"])
        if push.returncode != 0:
            tail = (push.stderr or "").strip().splitlines()[-1:] or [""]
            merge_queue.mark_blocked(db, entry.branch, f"push failed: {tail[0]}")
            return False, "push failed"

    sha = _run(["git", "-C", str(wt), "rev-parse", "HEAD"]).stdout.strip()
    merge_queue.mark_landed(db, entry.branch, sha)
    _cleanup_local(main, entry.branch, wt)
    _cleanup_remote(main, entry.branch, sha)
    return True, f"landed at {sha[:10]}"


def _cleanup_local(main: Path, branch: str, wt: Path) -> None:
    """Remove the landed worktree + local branch. Best-effort (already landed)."""
    _run(["git", "-C", str(main), "worktree", "remove", "--force", str(wt)])
    _run(["git", "-C", str(main), "branch", "-D", branch])


def _cleanup_remote(main: Path, branch: str, sha: str) -> None:
    """Close the PR and delete the remote branch. Best-effort.

    `main` already advanced on the remote. The `gh` calls are guarded by
    `shutil.which("gh")` so a host without `gh` degrades to a log line instead
    of a crash (and tests stub this whole function out so they never touch the
    real GitHub API).
    """
    if shutil.which("gh"):
        pr = _run(
            ["gh", "pr", "list", "--repo", REPO_SLUG, "--head", branch,
             "--state", "open", "--json", "number", "--jq", ".[0].number"]
        )
        pr_number = pr.stdout.strip()
        if pr.returncode == 0 and pr_number:
            _run(
                ["gh", "pr", "close", pr_number, "--repo", REPO_SLUG, "--comment",
                 "Landed on `main` via rebase + ff-only by the merge-queue drainer "
                 "(linear history). Not merged through GitHub -- closing."]
            )
        _run(["gh", "api", f"repos/{REPO_SLUG}/git/refs/heads/{branch}",
              "--method", "DELETE"])
    else:
        print("cleanup: gh not on PATH; skipped PR close + remote branch delete")


def drain(main: Path, db: Path) -> dict[str, object]:
    """Run one drain pass: heartbeat -> (optional) sweep -> claim/land until empty.

    Assumes the caller holds the landing flock. Returns a structured summary.
    """
    merge_queue._heartbeat_path(db).parent.mkdir(parents=True, exist_ok=True)
    merge_queue._heartbeat_path(db).touch()
    sweep = sweep_free_notes(main)
    landed: list[str] = []
    blocked: list[str] = []
    while True:
        merge_queue._heartbeat_path(db).touch()  # keep fresh across slow landings
        entry = merge_queue.claim_one(db)
        if entry is None:
            break
        ok, detail = land_branch(main, db, entry)
        if ok:
            landed.append(entry.branch)
        else:
            blocked.append(f"{entry.branch} ({detail})")
            _slack(f":warning: merge-queue blocked *{entry.branch}*: {detail}")
    return {"sweep": sweep, "landed": landed, "blocked": blocked}


def cmd_run(args: argparse.Namespace) -> int:
    """Acquire the flock (or bail), run one drain pass, print a summary."""
    handle = _acquire_lock(merge_queue._lock_path(args.db))
    if handle is None:
        print("drain: another drainer holds the lock; nothing to do")
        return 0
    try:
        summary = drain(args.main, args.db)
    finally:
        handle.close()  # releases the flock
    landed = summary["landed"]
    blocked = summary["blocked"]
    if not landed and not blocked:
        print(f"merge-queue drain: idle ({summary['sweep']})")
    else:
        print(
            f"merge-queue drain: {summary['sweep']}; "
            f"landed={landed or '-'}; blocked={blocked or '-'}"
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drain_merge_queue",
        description="Deterministic single-writer merge-queue drainer (no model).",
    )
    parser.add_argument(
        "--db", type=Path, default=merge_queue.DEFAULT_DB_PATH,
        help="Queue DB path (default: $DATA_ROOT/dev/merge_queue/merge_queue.db).",
    )
    parser.add_argument(
        "--main", type=Path, default=DEFAULT_MAIN,
        help="Main repo path (default: the repo containing this script).",
    )
    parser.set_defaults(func=cmd_run)
    return parser


def _strip_inherited_git_env() -> None:
    """Drop git env a parent process exports to its hooks.

    A `git commit` exports GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE / GIT_PREFIX
    to every hook it runs. The drainer always targets repos via `git -C <abs
    path>`, and GIT_DIR *overrides* that discovery -- so an inherited value
    would silently redirect every command at whatever repo the parent was
    committing in. That is exactly how a polluted test once force-pushed its
    temp history to the real `main`. The drainer never wants inherited git env,
    so strip it unconditionally at startup (defense-in-depth beyond callers).
    """
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX"):
        os.environ.pop(var, None)


def main(argv: list[str] | None = None) -> int:
    """Entry point -- parse args and run one drain pass."""
    _strip_inherited_git_env()
    args = _build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
