"""
scripts/merge_queue
[[scripts.merge_queue]]
https://github.com/Mjvolk3/torchcell/tree/main/scripts/merge_queue.py

SQLite-backed merge queue CLI. A single draining worker
(``scripts/drain_merge_queue.py``, triggered by the ``/enqueue-merge`` skill and
a ``*/2`` cron) is the *only* automated writer to ``main``; every concurrent
session that finishes a worktree branch ``add``-s it here instead of running
``/merge-worktree`` inline. That makes landings serialized by construction -- no
two landings race the shared ``.git`` or push to the same ref at once (the root
cause of blocked/conflicting merges when several worktrees finish at once).

Why a queue and not just a lock: we want an *editable, reorderable* backlog --
swap order, drop entries, clear it -- not a blocking mutex. So the order is an
explicit ``position`` column the ``mv`` command rewrites, and the worker always
claims the lowest-position ``queued`` row.

Stdlib only (``sqlite3``, ``argparse``, ``time``) so it runs from any worktree,
a cron shell, or a stripped container without extra deps. Connection shape:
``PRAGMA journal_mode=DELETE`` + ``synchronous=2`` (DELETE-mode is safe on every
filesystem including NFS, where WAL silently corrupts) and
``isolation_level=None`` so we can issue ``BEGIN IMMEDIATE`` directly for the
atomic claim.
"""

# scripts/merge_queue.py
# [[scripts.merge_queue]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/merge_queue.py

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def _resolve_data_root() -> Path | None:
    """Resolve ``DATA_ROOT`` (where the queue DB lives) without python-dotenv.

    Order: the ``DATA_ROOT`` env var first (set by ``.env`` in a normal shell);
    then the repo ``.env`` parsed by hand (a ``*/2`` cron shell has no
    ``DATA_ROOT`` exported, but the file is always present). Stdlib-only so the
    module keeps its zero-dependency portability contract.
    """
    root = os.environ.get("DATA_ROOT")
    if root:
        return Path(root)
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("DATA_ROOT="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return Path(value)
    return None


# The DB lives under DATA_ROOT (torchcell's canonical large/shared data root,
# e.g. /scratch/projects/torchcell-scratch). Falls back to a per-user cache dir
# only if DATA_ROOT cannot be resolved at all (never expected in practice).
_DATA_ROOT = _resolve_data_root()
DEFAULT_DB_PATH = (
    (_DATA_ROOT / "dev" / "merge_queue" / "merge_queue.db")
    if _DATA_ROOT is not None
    else Path.home() / ".cache" / "torchcell" / "merge_queue" / "merge_queue.db"
)

# Orderable / actionable statuses. ``queued`` rows are reorderable and
# claimable; ``claimed`` is in-flight (the worker is landing it right now);
# ``blocked`` needs a human fix + ``requeue``; ``landed`` is terminal history.
ACTIVE_STATUSES = ("queued", "claimed", "blocked")

# Default freshness window (seconds) for ``loop-status`` -- a heartbeat newer
# than this means the drain path is live. The cron ticks every 2 min, so 3
# minutes of slack distinguishes "alive" from "stopped".
DEFAULT_LOOP_WINDOW_S = 180

# ``watch`` defaults. A single ``watch`` call runs as a foreground Bash command,
# which the Claude Code harness hard-caps at 600s -- so the internal block MUST
# stay strictly under 600 or the harness kills the call mid-watch (the model
# then sees a timeout error instead of the clean exit-4 "still queued" code).
# 570 sits just under the cap (a 30s margin for python startup + the final
# print/bell). To watch longer than one window across a deep queue, the enqueue
# skill re-invokes ``watch`` on exit 4 (auto-re-watch), so total watch time is
# unbounded without any single call exceeding the cap.
WATCH_TIMEOUT_S = 570
WATCH_INTERVAL_S = 15


@dataclass(frozen=True)
class QueueEntry:
    """One branch awaiting landing.

    Stdlib ``dataclass`` (not Pydantic) to preserve this module's stdlib-only
    portability contract. ``frozen=True`` because an entry is a snapshot read
    out of the DB; mutation happens via SQL ``UPDATE``, never by reassigning the
    object.
    """

    id: int  # queue row id
    branch: str  # the worktree branch to land
    worktree: str | None  # absolute worktree path (derivable from branch)
    position: int  # landing order; lower lands first
    status: str  # one of queued / claimed / blocked / landed
    attempts: int  # times this entry has been claimed for a landing attempt
    enqueued_by: str | None  # session / actor that added it (provenance)
    note: str | None  # free-text note
    last_error: str | None  # populated when status == blocked

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> QueueEntry:
        """Build an entry from a ``sqlite3.Row`` (column names match fields)."""
        return cls(
            id=row["id"],
            branch=row["branch"],
            worktree=row["worktree"],
            position=row["position"],
            status=row["status"],
            attempts=row["attempts"],
            enqueued_by=row["enqueued_by"],
            note=row["note"],
            last_error=row["last_error"],
        )


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open the queue DB in autocommit-off-mode with NFS-safe pragmas.

    ``isolation_level=None`` is load-bearing: it lets us issue
    ``BEGIN IMMEDIATE`` directly so the atomic claim grabs the reserved-write
    lock before any concurrent caller can.
    """
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=2")
    # 30s busy-timeout so a concurrent enqueue waits on the reserved-write lock
    # instead of raising ``OperationalError: database is locked``.
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create the table + indexes if missing -- safe to call repeatedly."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS queue (
          id          INTEGER PRIMARY KEY AUTOINCREMENT,
          branch      TEXT NOT NULL,
          worktree    TEXT,
          position    INTEGER NOT NULL,
          status      TEXT NOT NULL DEFAULT 'queued'
                      CHECK(status IN ('queued','claimed','blocked','landed')),
          attempts    INTEGER NOT NULL DEFAULT 0,
          enqueued_by TEXT,
          note        TEXT,
          last_error  TEXT,
          landed_sha  TEXT,
          created_at  TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
          claimed_at  TEXT,
          landed_at   TEXT
        )
        """
    )
    # At most one *active* row per branch (a branch may reappear after landing,
    # e.g. a re-opened follow-up). Partial unique index -- terminal ``landed``
    # rows are exempt so history accumulates.
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_queue_active_branch"
        " ON queue(branch) WHERE status IN ('queued','claimed','blocked')"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_queue_status_position"
        " ON queue(status, position)"
    )


def _open(db_path: Path) -> sqlite3.Connection:
    """Resolve parent dir + open the DB. Caller is responsible for ``close()``."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return _connect(db_path)


def _next_position(conn: sqlite3.Connection) -> int:
    """Return one past the current max position (append to the tail)."""
    row = conn.execute("SELECT COALESCE(MAX(position), -1) AS m FROM queue").fetchone()
    return int(row["m"]) + 1


def _heartbeat_path(db_path: Path) -> Path:
    """Loop liveness marker, co-located with the DB."""
    return db_path.parent / "loop.heartbeat"


def _lock_path(db_path: Path) -> Path:
    """The landing flock file, co-located with the DB.

    Both ``/drain-merge-queue`` and ``/merge-worktree`` wrap their
    push-to-main critical section with ``flock`` on this path, so a manual
    landing can never overlap the drainer's landing even though the queue would
    not have ordered it.
    """
    return db_path.parent / "landing.lock"


# ── Library entrypoints (the drain skill / tests call these directly) ──────


def add_entry(
    db_path: Path,
    branch: str,
    worktree: str | None = None,
    note: str | None = None,
    enqueued_by: str | None = None,
) -> int | None:
    """Append ``branch`` to the queue tail. Idempotent on an active duplicate.

    Returns the new row id, or ``None`` when an active (queued/claimed/blocked)
    entry for the branch already exists -- enqueuing twice is a no-op, not an
    error, so a session can safely re-run its enqueue step.
    """
    conn = _open(db_path)
    _init_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    existing = conn.execute(
        "SELECT id FROM queue WHERE branch=? AND status IN ('queued','claimed','blocked')",
        (branch,),
    ).fetchone()
    if existing is not None:
        conn.execute("COMMIT")
        conn.close()
        return None
    position = _next_position(conn)
    cursor = conn.execute(
        """
        INSERT INTO queue (branch, worktree, position, enqueued_by, note)
        VALUES (?, ?, ?, ?, ?)
        """,
        (branch, worktree, position, enqueued_by, note),
    )
    new_id = cursor.lastrowid
    conn.execute("COMMIT")
    conn.close()
    return new_id


def claim_one(db_path: Path) -> QueueEntry | None:
    """Atomically claim the lowest-position ``queued`` entry, or ``None``.

    ``BEGIN IMMEDIATE`` takes the reserved-write lock so two ticks of the drain
    (or a stray second worker) cannot grab the same row. ``RETURNING`` hands
    back the claimed fields so the caller lands without a second query.
    """
    conn = _open(db_path)
    _init_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    row: sqlite3.Row | None = conn.execute(
        """
        UPDATE queue
        SET status='claimed', attempts=attempts+1,
            claimed_at=datetime('now'), updated_at=datetime('now')
        WHERE id = (
          SELECT id FROM queue
          WHERE status='queued'
          ORDER BY position ASC, created_at ASC LIMIT 1
        )
        RETURNING id, branch, worktree, position, status, attempts,
                  enqueued_by, note, last_error
        """
    ).fetchone()
    conn.execute("COMMIT")
    conn.close()
    return QueueEntry.from_row(row) if row is not None else None


def peek_one(db_path: Path) -> QueueEntry | None:
    """Return the head ``queued`` entry without claiming it."""
    conn = _open(db_path)
    _init_schema(conn)
    row = conn.execute(
        "SELECT * FROM queue WHERE status='queued'"
        " ORDER BY position ASC, created_at ASC LIMIT 1"
    ).fetchone()
    conn.close()
    return QueueEntry.from_row(row) if row is not None else None


def _resolve_active(conn: sqlite3.Connection, branch: str) -> sqlite3.Row | None:
    """Find the single active row for ``branch`` (claimed > queued > blocked).

    Prefers the in-flight ``claimed`` row so ``done``/``block`` target the
    attempt actually underway when several states could in theory coexist
    (they cannot, given the partial unique index -- this is belt-and-suspenders
    ordering).
    """
    row: sqlite3.Row | None = conn.execute(
        """
        SELECT * FROM queue WHERE branch=? AND status IN ('queued','claimed','blocked')
        ORDER BY CASE status WHEN 'claimed' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END
        LIMIT 1
        """,
        (branch,),
    ).fetchone()
    return row


def mark_landed(db_path: Path, branch: str, sha: str | None) -> bool:
    """Set the active entry for ``branch`` to ``landed``. Returns hit/miss."""
    conn = _open(db_path)
    _init_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    row = _resolve_active(conn, branch)
    if row is None:
        conn.execute("COMMIT")
        conn.close()
        return False
    conn.execute(
        """
        UPDATE queue SET status='landed', landed_at=datetime('now'),
                         landed_sha=?, updated_at=datetime('now')
        WHERE id=?
        """,
        (sha, row["id"]),
    )
    conn.execute("COMMIT")
    conn.close()
    return True


def mark_blocked(db_path: Path, branch: str, reason: str | None) -> bool:
    """Set the active entry for ``branch`` to ``blocked`` with a reason."""
    conn = _open(db_path)
    _init_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    row = _resolve_active(conn, branch)
    if row is None:
        conn.execute("COMMIT")
        conn.close()
        return False
    conn.execute(
        """
        UPDATE queue SET status='blocked', last_error=?, updated_at=datetime('now')
        WHERE id=?
        """,
        (reason, row["id"]),
    )
    conn.execute("COMMIT")
    conn.close()
    return True


# ── watch (the enqueue skill blocks on this until its branch is terminal) ───


def branch_row(db_path: Path, branch: str) -> sqlite3.Row | None:
    """Return the single most-relevant row for ``branch``, or ``None``.

    Prefers an *active* row (claimed > queued > blocked); falls back to the most
    recently updated terminal ``landed`` row. After a successful landing the same
    row flips to ``landed`` (and its worktree is cleaned up), so ``watch`` still
    finds it here to print the success banner.
    """
    conn = _open(db_path)
    _init_schema(conn)
    row: sqlite3.Row | None = conn.execute(
        """
        SELECT * FROM queue WHERE branch=?
        ORDER BY CASE status WHEN 'claimed' THEN 0 WHEN 'queued' THEN 1
                             WHEN 'blocked' THEN 2 ELSE 3 END,
                 updated_at DESC
        LIMIT 1
        """,
        (branch,),
    ).fetchone()
    conn.close()
    return row


def classify_watch(
    status: str, last_error: str | None, elapsed: float, timeout: float
) -> str:
    """Pure decision for a watched row -- the testable core of ``watch``.

    Returns one of:

    - ``landed``           -- terminal success; green banner, exit 0.
    - ``resolve_conflict`` -- blocked on a rebase conflict the *owning session*
      must resolve in its worktree, then ``requeue``; exit 3.
    - ``needs_human``      -- blocked for a non-conflict reason (push failed,
      dirty/missing worktree); red banner, exit 2.
    - ``timeout``          -- still queued/claimed past ``timeout``; yellow
      banner, exit 4.
    - ``wait``             -- not yet terminal and within ``timeout``; keep
      polling.
    """
    if status == "landed":
        return "landed"
    if status == "blocked":
        if last_error and "rebase conflict" in last_error.lower():
            return "resolve_conflict"
        return "needs_human"
    if elapsed >= timeout:
        return "timeout"
    return "wait"


# ---------------------------------------------------------------------------
# Foreign-commit detection (the enqueue-time tangle guard)
#
# A "tangle" is a branch that carries a sibling feature's ``plan:`` note commit,
# inherited because the worktree was cut from a stale local ``main`` that still
# held un-pushed plan commits. Such commits touch only notes and do NOT conflict
# on rebase, so they land silently under a green banner -- the watch/resolve flow
# (which only fires on conflicts) can never catch them. Worktree-first planning
# stops NEW tangles forming; this detector is the enqueue-time net for legacy /
# regression cases. The enqueuing agent runs ``foreign`` and strips before
# queueing.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommitMeta:
    """One commit in ``base..branch`` -- the minimum needed to judge foreignness."""

    sha: str
    subject: str
    note_only: bool


@dataclass(frozen=True)
class ForeignScan:
    """Result of scanning a branch's commits for inherited sibling plan notes."""

    foreign: list[str]  # SHAs flagged foreign, oldest-first
    contiguous: bool  # True iff they form a base prefix (one rebase --onto strips them)
    strip_point: str | None  # newest foreign SHA when contiguous, else None


def branch_slug(branch: str) -> str:
    """Slug = the branch name minus a single leading kind prefix.

    ``plan/protocol-tier2-...`` -> ``protocol-tier2-...``; a bare name is its own
    slug. Worktree-first plan commits are ``plan: <slug>``, so the own plan
    commit's subject contains exactly this.
    """
    return branch.split("/", 1)[-1]


def branch_issue(branch: str) -> str | None:
    """Trailing ``-<digits>`` issue number in the slug (e.g. ``...-826`` -> 826)."""
    match = re.search(r"-(\d+)$", branch_slug(branch))
    return match.group(1) if match else None


def is_foreign_commit(meta: CommitMeta, slug: str, issue: str | None) -> bool:
    """True when ``meta`` is a sibling feature's inherited ``plan:`` note commit.

    Deliberately tight so an auto-strip is safe -- ALL of:
    - note-only (touches only ``notes/``); a commit touching code is never foreign;
    - subject starts ``plan:`` (the only thing /plan-4.8 commits to local main);
    - the branch's own slug is absent from the subject;
    - the branch's issue number (``#<n>``) is absent from the subject.
    A branch's own plan commit is ``plan: <slug>`` -> slug present -> kept.
    """
    if not meta.note_only:
        return False
    if not meta.subject.startswith("plan:"):
        return False
    if slug.lower() in meta.subject.lower():
        return False
    if issue is not None and f"#{issue}" in meta.subject:
        return False
    return True


def scan_foreign(
    commits: list[CommitMeta], slug: str, issue: str | None
) -> ForeignScan:
    """Classify ``commits`` (OLDEST-first) and decide if one rebase strips them.

    Foreign commits are contiguous iff every foreign commit precedes every own
    commit -- the tangle shape, since inherited commits sit at the branch base.
    When contiguous, ``rebase --onto <base> <strip_point>`` drops exactly them.
    """
    flags = [is_foreign_commit(c, slug, issue) for c in commits]
    foreign = [c.sha for c, flag in zip(commits, flags) if flag]
    if not foreign:
        return ForeignScan([], True, None)
    last_foreign = max(i for i, flag in enumerate(flags) if flag)
    first_own = next((i for i, flag in enumerate(flags) if not flag), len(flags))
    contiguous = last_foreign < first_own
    strip_point = commits[last_foreign].sha if contiguous else None
    return ForeignScan(foreign, contiguous, strip_point)


def _git_lines(args: list[str]) -> list[str]:
    """Run ``git <args>`` and return stdout lines (raises on non-zero)."""
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def commits_in_range(base: str, branch: str) -> list[CommitMeta]:
    """Commits in ``base..branch`` OLDEST-first, each with note-only + subject.

    Worktrees share one ``.git``, so both refs resolve from any cwd in the repo.
    """
    metas: list[CommitMeta] = []
    for line in _git_lines(
        ["log", "--reverse", "--format=%H%x00%s", f"{base}..{branch}"]
    ):
        if not line:
            continue
        sha, _, subject = line.partition("\x00")
        files = _git_lines(["diff-tree", "--no-commit-id", "--name-only", "-r", sha])
        note_only = bool(files) and all(f.startswith("notes/") for f in files)
        metas.append(CommitMeta(sha, subject, note_only))
    return metas


def _bell(count: int) -> None:
    """Emit ``count`` terminal bells -- the audible half of the close signal."""
    sys.stdout.write("\a" * count)
    sys.stdout.flush()


# Emoji glyphs for the in-pane close signal, named once so the three banners
# stay readable: dove (the merge-queue marker) + stoplight dots + check/stop/
# pause. The dove/pause carry a U+FE0F variation selector for emoji rendering.
_DOVE = "\U0001f54a️"
_GREEN, _RED, _YELLOW = "\U0001f7e2", "\U0001f534", "\U0001f7e1"
_CHECK, _STOP, _PAUSE = "✅", "\U0001f6d1", "⏸️"


def banner_landed(branch: str, sha: str | None) -> str:
    """Green 'safe to close' banner -- dove + stoplight green."""
    bar = _GREEN * 3
    return (
        f"{_DOVE} {bar}  {_CHECK} LANDED - SAFE TO CLOSE  {bar} {_DOVE}\n"
        f"   branch    {branch}\n"
        f"   main      {(sha or '')[:10]}\n"
        "   cleanup   worktree + local & remote branch removed | PR closed"
    )


def banner_needs_human(branch: str, reason: str | None) -> str:
    """Red 'needs you' banner -- a non-conflict block a human must clear."""
    bar = _RED * 3
    return (
        f"{_DOVE} {bar}  {_STOP} BLOCKED - NEEDS YOU  {bar} {_DOVE}\n"
        f"   branch    {branch}\n"
        f"   reason    {reason or 'unknown'}\n"
        "   still queued as `blocked`; fix the worktree, then it re-lands."
    )


def banner_queued(branch: str, position: int) -> str:
    """Yellow 'leave open' banner -- watch timed out, branch still queued."""
    bar = _YELLOW * 3
    return (
        f"{_DOVE} {bar}  {_PAUSE} QUEUED - LEAVE OPEN  {bar} {_DOVE}\n"
        f"   branch    {branch}\n"
        f"   still at position {position} | cron will land it (no auto-resolve)\n"
        "   re-run /enqueue-merge to resume watching."
    )


# ── argparse command handlers ──────────────────────────────────────────────


def cmd_init(args: argparse.Namespace) -> int:
    """Create the DB file and schema. Idempotent -- safe to re-run."""
    conn = _open(args.db)
    _init_schema(conn)
    conn.close()
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    """Append a branch to the queue. No-ops (exit 0) on an active duplicate."""
    new_id = add_entry(args.db, args.branch, args.worktree, args.note, args.by)
    if new_id is None:
        print(f"add: {args.branch!r} already queued/claimed/blocked; no-op")
        return 0
    print(new_id)
    return 0


def cmd_ls(args: argparse.Namespace) -> int:
    """Print queue entries. Default: active rows; ``--all`` includes landed."""
    conn = _open(args.db)
    _init_schema(conn)
    if args.all:
        rows = conn.execute(
            "SELECT * FROM queue ORDER BY status, position ASC, created_at ASC"
        ).fetchall()
    elif args.status:
        rows = conn.execute(
            "SELECT * FROM queue WHERE status=? ORDER BY position ASC, created_at ASC",
            (args.status,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM queue WHERE status IN ('queued','claimed','blocked')"
            " ORDER BY position ASC, created_at ASC"
        ).fetchall()
    conn.close()
    if args.json:
        print(json.dumps([dict(row) for row in rows], indent=2))
    else:
        for row in rows:
            err = f"  err={row['last_error']}" if row["last_error"] else ""
            print(
                f"{row['position']:>3}  {row['status']:<8} {row['branch']}"
                f"\tattempts={row['attempts']}"
                f"\tby={row['enqueued_by'] or '-'}{err}"
            )
    return 0


def cmd_claim(args: argparse.Namespace) -> int:
    """Atomically claim the head; print 'branch<TAB>worktree' or stay silent."""
    entry = claim_one(args.db)
    if entry is None:
        return 0
    if args.json:
        print(json.dumps(entry.__dict__, indent=2))
    else:
        print(f"{entry.branch}\t{entry.worktree or '-'}")
    return 0


def cmd_peek(args: argparse.Namespace) -> int:
    """Show the head queued entry without claiming it."""
    entry = peek_one(args.db)
    if entry is None:
        return 0
    if args.json:
        print(json.dumps(entry.__dict__, indent=2))
    else:
        print(f"{entry.branch}\t{entry.worktree or '-'}")
    return 0


def cmd_done(args: argparse.Namespace) -> int:
    """Mark the active entry for a branch ``landed``."""
    hit = mark_landed(args.db, args.branch, args.sha)
    print(f"done: {args.branch} -> landed" if hit else f"done: no active {args.branch}")
    return 0


def cmd_block(args: argparse.Namespace) -> int:
    """Mark the active entry for a branch ``blocked`` with a reason."""
    hit = mark_blocked(args.db, args.branch, args.reason)
    print(
        f"block: {args.branch} -> blocked" if hit else f"block: no active {args.branch}"
    )
    return 0


def cmd_requeue(args: argparse.Namespace) -> int:
    """Move a ``blocked`` branch back to ``queued`` at the tail (re-land it)."""
    conn = _open(args.db)
    _init_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute(
        "SELECT id FROM queue WHERE branch=? AND status='blocked'", (args.branch,)
    ).fetchone()
    if row is None:
        conn.execute("COMMIT")
        conn.close()
        print(f"requeue: no blocked entry for {args.branch!r}")
        return 0
    position = _next_position(conn)
    conn.execute(
        "UPDATE queue SET status='queued', position=?, updated_at=datetime('now')"
        " WHERE id=?",
        (position, row["id"]),
    )
    conn.execute("COMMIT")
    conn.close()
    print(f"requeue: {args.branch} -> queued")
    return 0


def cmd_rm(args: argparse.Namespace) -> int:
    """Remove the active entry for a branch from the queue."""
    conn = _open(args.db)
    _init_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    cursor = conn.execute(
        "DELETE FROM queue WHERE branch=? AND status IN ('queued','claimed','blocked')",
        (args.branch,),
    )
    conn.execute("COMMIT")
    removed = cursor.rowcount
    conn.close()
    print(f"rm: removed {removed} entry/entries for {args.branch!r}")
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    """Clear the actionable backlog. Default: ``queued`` + ``blocked``.

    ``claimed`` (in-flight) and ``landed`` (history) are spared by default --
    use ``--status`` to target exactly one status if you really mean it.
    """
    conn = _open(args.db)
    _init_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    if args.status:
        cursor = conn.execute("DELETE FROM queue WHERE status=?", (args.status,))
    else:
        cursor = conn.execute("DELETE FROM queue WHERE status IN ('queued','blocked')")
    conn.execute("COMMIT")
    removed = cursor.rowcount
    conn.close()
    print(f"clear: removed {removed} entry/entries")
    return 0


def cmd_mv(args: argparse.Namespace) -> int:
    """Reorder a queued branch to 0-based ``index`` among queued entries.

    Renumbers ``position`` densely over the ``queued`` set so the worker's
    ``ORDER BY position`` reflects the new order. ``claimed``/``blocked`` rows
    are not part of the orderable set.
    """
    conn = _open(args.db)
    _init_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    rows = conn.execute(
        "SELECT id, branch FROM queue WHERE status='queued'"
        " ORDER BY position ASC, created_at ASC"
    ).fetchall()
    ids = [row["id"] for row in rows]
    branches = [row["branch"] for row in rows]
    if args.branch not in branches:
        conn.execute("COMMIT")
        conn.close()
        print(f"mv: {args.branch!r} is not a queued entry")
        return 1
    cur_index = branches.index(args.branch)
    moved_id = ids.pop(cur_index)
    target = max(0, min(args.index, len(ids)))
    ids.insert(target, moved_id)
    for new_pos, row_id in enumerate(ids):
        conn.execute(
            "UPDATE queue SET position=?, updated_at=datetime('now') WHERE id=?",
            (new_pos, row_id),
        )
    conn.execute("COMMIT")
    conn.close()
    print(f"mv: {args.branch} -> position {target}")
    return 0


def cmd_lock_path(args: argparse.Namespace) -> int:
    """Print the canonical landing-lock path (for ``flock`` in the skills)."""
    print(_lock_path(args.db))
    return 0


def cmd_heartbeat(args: argparse.Namespace) -> int:
    """Touch the loop heartbeat -- the drainer calls this each pass."""
    path = _heartbeat_path(args.db)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return 0


def cmd_loop_status(args: argparse.Namespace) -> int:
    """Report whether the drain path is live (heartbeat fresher than window).

    ``/merge-worktree`` calls this to decide whether to nudge the user toward
    ``merge_queue add`` instead of landing inline. Prints ``live`` / ``stopped``
    and the heartbeat age; exit code 0 = live, 1 = stopped (script-friendly).
    """
    path = _heartbeat_path(args.db)
    if not path.exists():
        if args.json:
            print(json.dumps({"state": "stopped", "age_s": None}))
        else:
            print("stopped (no heartbeat)")
        return 1
    age = time.time() - path.stat().st_mtime
    live = age < args.window
    if args.json:
        print(json.dumps({"state": "live" if live else "stopped", "age_s": round(age)}))
    else:
        print(f"{'live' if live else 'stopped'} (heartbeat {round(age)}s ago)")
    return 0 if live else 1


def cmd_watch(args: argparse.Namespace) -> int:
    """Block until ``branch`` reaches a terminal/actionable state, then signal.

    The enqueue skill calls this after triggering a drain. It polls the row
    every ``--interval`` seconds up to ``--timeout``; on a terminal state it
    prints a single *terse* control-flow line and returns a status-specific exit
    code the skill branches on:

    - 0  landed (green, safe to close)
    - 2  blocked, non-conflict -- needs a human (red)
    - 3  blocked on a rebase conflict -- the session resolves + requeues
    - 4  still queued past the timeout (yellow, leave open)
    - 5  no queue entry for the branch (nothing to watch)

    ``watch`` deliberately does NOT print the dove close banner -- that is the
    final, bottom-most signal and belongs to the ``banner`` subcommand, which
    the skill runs *last*, after any wrap-up (issue close, summary). Keeping the
    banner out of ``watch`` is what stops it from scrolling off-screen above the
    wrap-up output.
    """
    start = time.monotonic()
    while True:
        row = branch_row(args.db, args.branch)
        if row is None:
            print(f"watch: no queue entry for {args.branch!r}")
            return 5
        action = classify_watch(
            row["status"], row["last_error"], time.monotonic() - start, args.timeout
        )
        if action == "wait":
            time.sleep(args.interval)
            continue
        if action == "landed":
            print(f"watch: landed {row['branch']} @ {(row['landed_sha'] or '')[:10]}")
            return 0
        if action == "resolve_conflict":
            print(
                f"watch: rebase conflict on {row['branch']} -- the owning session "
                "resolves it in the worktree, then requeue + re-drain."
            )
            return 3
        if action == "needs_human":
            print(f"watch: blocked {row['branch']} -- {row['last_error'] or 'unknown'}")
            return 2
        # action == "timeout"
        print(f"watch: still queued {row['branch']} @ position {row['position']}")
        return 4


def cmd_banner(args: argparse.Namespace) -> int:
    """Print the terminal-state close banner for a branch -- the FINAL signal.

    The enqueue skill calls this *last*, after ``watch`` returns and after any
    wrap-up (closing the GitHub issue, a one-line summary), so the dove banner
    is the bottom-most thing in the pane -- a glance-and-close signal that never
    scrolls away. It reads the branch's current row and renders the banner that
    matches its terminal status, ringing the matching bell:

    - landed  -> green "safe to close" banner + 1 bell
    - blocked -> red "needs you" banner + 3 bells
    - queued/claimed -> yellow "leave open" banner (no bell)
    """
    row = branch_row(args.db, args.branch)
    if row is None:
        print(f"banner: no queue entry for {args.branch!r}")
        return 5
    if row["status"] == "landed":
        print(banner_landed(row["branch"], row["landed_sha"]))
        _bell(1)
    elif row["status"] == "blocked":
        print(banner_needs_human(row["branch"], row["last_error"]))
        _bell(3)
    else:
        print(banner_queued(row["branch"], row["position"]))
    return 0


def cmd_foreign(args: argparse.Namespace) -> int:
    """Flag sibling ``plan:`` commits a branch inherited from a stale local main.

    The enqueue-time tangle guard. The enqueuing agent runs this *before* it
    queues the branch; on exit 1 it runs the printed ``rebase --onto`` to strip
    them, then enqueues. Exit codes (machine-readable for the skill):

    - 0  clean -- every commit in ``base..branch`` is the branch's own work
    - 1  foreign commits found AND strippable with one ``rebase --onto`` (printed)
    - 2  foreign commits found but NOT a contiguous base prefix -- needs a human
    """
    commits = commits_in_range(args.base, args.branch)
    scan = scan_foreign(commits, branch_slug(args.branch), branch_issue(args.branch))
    if not scan.foreign:
        print(
            f"foreign: clean -- every commit in {args.base}..{args.branch} "
            "is this branch's own work"
        )
        return 0
    subjects = {c.sha: c.subject for c in commits}
    print(
        f"foreign: {len(scan.foreign)} inherited sibling plan commit(s) in "
        f"{args.base}..{args.branch} (would land as residue):"
    )
    for sha in scan.foreign:
        print(f"  {sha[:10]}  {subjects[sha]}")
    if scan.contiguous and scan.strip_point is not None:
        print("strip them with (then requeue):")
        print(
            f"  git -C <worktree> rebase --onto {args.base} "
            f"{scan.strip_point[:10]} {args.branch}"
        )
        return 1
    print(
        "foreign commits are NOT a contiguous base prefix -- strip by hand "
        "(interactive rebase) before enqueuing."
    )
    return 2


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree -- subcommands match the plan's CLI surface."""
    parser = argparse.ArgumentParser(
        prog="merge_queue",
        description="SQLite-backed single-writer merge queue for serialized landings.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite DB (default: $DATA_ROOT/dev/merge_queue/merge_queue.db).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create DB + schema (idempotent).")
    p_init.set_defaults(func=cmd_init)

    p_add = sub.add_parser("add", help="Append a branch to the queue tail.")
    p_add.add_argument("branch", help="Worktree branch to land.")
    p_add.add_argument(
        "--worktree", default=None, help="Absolute worktree path (optional)."
    )
    p_add.add_argument("--note", default=None, help="Free-text note column.")
    p_add.add_argument(
        "--by", default=None, help="Who enqueued it (session id / actor)."
    )
    p_add.set_defaults(func=cmd_add)

    p_ls = sub.add_parser("ls", help="List queue entries; --json for machine output.")
    p_ls.add_argument(
        "--status", default=None, help="Filter to one status (queued/claimed/...)."
    )
    p_ls.add_argument(
        "--all", action="store_true", help="Include terminal 'landed' rows."
    )
    p_ls.add_argument("--json", action="store_true", help="Emit JSON.")
    p_ls.set_defaults(func=cmd_ls)

    p_claim = sub.add_parser(
        "claim",
        help="Atomic pop -- claim the head queued entry; print branch+worktree.",
    )
    p_claim.add_argument("--json", action="store_true", help="Emit JSON.")
    p_claim.set_defaults(func=cmd_claim)

    p_peek = sub.add_parser("peek", help="Show the head queued entry without claiming.")
    p_peek.add_argument("--json", action="store_true", help="Emit JSON.")
    p_peek.set_defaults(func=cmd_peek)

    p_done = sub.add_parser("done", help="Mark a branch landed (worker success).")
    p_done.add_argument("branch", help="Branch that landed.")
    p_done.add_argument("--sha", default=None, help="main HEAD SHA after the ff-merge.")
    p_done.set_defaults(func=cmd_done)

    p_block = sub.add_parser("block", help="Mark a branch blocked (worker failure).")
    p_block.add_argument("branch", help="Branch that could not land.")
    p_block.add_argument(
        "--reason", default=None, help="Why it blocked (stored on last_error)."
    )
    p_block.set_defaults(func=cmd_block)

    p_req = sub.add_parser("requeue", help="Move a blocked branch back to queued.")
    p_req.add_argument("branch", help="Blocked branch to re-land.")
    p_req.set_defaults(func=cmd_requeue)

    p_rm = sub.add_parser("rm", help="Remove a branch's active entry from the queue.")
    p_rm.add_argument("branch", help="Branch to drop.")
    p_rm.set_defaults(func=cmd_rm)

    p_clear = sub.add_parser(
        "clear", help="Clear queued+blocked (default); --status targets one status."
    )
    p_clear.add_argument(
        "--status", default=None, help="Clear exactly this status instead."
    )
    p_clear.set_defaults(func=cmd_clear)

    p_mv = sub.add_parser("mv", help="Reorder a queued branch to a 0-based index.")
    p_mv.add_argument("branch", help="Queued branch to move.")
    p_mv.add_argument("index", type=int, help="0-based target index among queued.")
    p_mv.set_defaults(func=cmd_mv)

    p_lock = sub.add_parser("lock-path", help="Print the landing flock path.")
    p_lock.set_defaults(func=cmd_lock_path)

    p_hb = sub.add_parser("heartbeat", help="Touch the loop heartbeat (drainer).")
    p_hb.set_defaults(func=cmd_heartbeat)

    p_loop = sub.add_parser(
        "loop-status", help="Report whether the drain path is live (exit 0=live)."
    )
    p_loop.add_argument(
        "--window",
        type=int,
        default=DEFAULT_LOOP_WINDOW_S,
        help=f"Freshness window in seconds (default {DEFAULT_LOOP_WINDOW_S}).",
    )
    p_loop.add_argument("--json", action="store_true", help="Emit JSON.")
    p_loop.set_defaults(func=cmd_loop_status)

    p_watch = sub.add_parser(
        "watch",
        help="Block until a branch lands/blocks; print the close signal "
        "(exit 0=landed, 2=needs-human, 3=conflict, 4=queued, 5=absent).",
    )
    p_watch.add_argument("branch", help="Branch to watch to a terminal state.")
    p_watch.add_argument(
        "--timeout",
        type=float,
        default=WATCH_TIMEOUT_S,
        help=f"Max seconds to block before the queued signal (default {WATCH_TIMEOUT_S}).",
    )
    p_watch.add_argument(
        "--interval",
        type=float,
        default=WATCH_INTERVAL_S,
        help=f"Poll interval in seconds (default {WATCH_INTERVAL_S}).",
    )
    p_watch.set_defaults(func=cmd_watch)

    p_banner = sub.add_parser(
        "banner",
        help="Print a branch's terminal-state close banner (dove + stoplight) "
        "-- the final, bottom-most signal; run last, after any wrap-up.",
    )
    p_banner.add_argument("branch", help="Branch whose close banner to print.")
    p_banner.set_defaults(func=cmd_banner)

    p_foreign = sub.add_parser(
        "foreign",
        help="Flag sibling plan commits a branch inherited from a stale local "
        "main (enqueue-time tangle guard; exit 0=clean, 1=strippable, 2=manual).",
    )
    p_foreign.add_argument("branch", help="Branch to scan for foreign commits.")
    p_foreign.add_argument(
        "--base",
        default="origin/main",
        help="Base ref the branch lands onto (default: origin/main).",
    )
    p_foreign.set_defaults(func=cmd_foreign)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Top-level entry point -- parse args and dispatch to the subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
