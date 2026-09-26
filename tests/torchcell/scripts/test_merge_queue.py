# tests/torchcell/scripts/test_merge_queue.py
# [[tests.torchcell.scripts.test_merge_queue]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/scripts/test_merge_queue.py
"""``scripts/merge_queue.py`` as a subprocess against a queue database under ``tmp_path``.

The CLI is run with ``--db`` (so ``$DATA_ROOT`` never enters), a ``PATH`` holding only
``/usr/bin:/bin`` (no ``gh``), and ``HOME`` under ``tmp_path``. Every assertion is an exact
exit code, an exact stdout line, or an exact row, computed from the command sequence.
The pure helpers (``classify_watch``, the foreign-commit scan, the banners) are imported
and tested directly; the ``foreign`` command runs against a bare-origin plus clone
repository whose branches carry known commits. Phase 4 of
[[plan.test-suite-buildout.2026.09.25]].
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import merge_queue  # type: ignore[import-not-found]  # noqa: E402


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "DATA_ROOT": str(tmp_path / "data-root"),
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }


def _cli(
    tmp_path: Path, db: Path, *args: str, cwd: Path | None = None
) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "merge_queue.py"), "--db", str(db), *args],
        capture_output=True,
        text=True,
        env=_env(tmp_path),
        cwd=str(cwd or tmp_path),
    )
    return result.returncode, result.stdout.strip("\n")


def _rows(tmp_path: Path, db: Path, *args: str) -> list[dict[str, object]]:
    code, out = _cli(tmp_path, db, "ls", "--json", *args)
    assert code == 0
    return list(json.loads(out))


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "queue" / "merge_queue.db"


def test_add_returns_row_ids_and_is_idempotent_on_an_active_branch(
    tmp_path: Path, db: Path
) -> None:
    """First add prints id 1, a repeat is a stated no-op, the next branch gets id 2."""
    assert _cli(tmp_path, db, "add", "plan/a") == (0, "1")
    assert _cli(tmp_path, db, "add", "plan/a") == (
        0,
        "add: 'plan/a' already queued/claimed/blocked; no-op",
    )
    assert _cli(tmp_path, db, "add", "plan/b", "--note", "n") == (0, "2")
    rows = _rows(tmp_path, db)
    assert [(r["branch"], r["position"], r["status"], r["note"]) for r in rows] == [
        ("plan/a", 0, "queued", None),
        ("plan/b", 1, "queued", "n"),
    ]


def test_mv_and_rm_reorder_the_queue(tmp_path: Path, db: Path) -> None:
    """Mv c 0 puts c first; rm b removes one row; ls prints the surviving order."""
    for branch in ("plan/a", "plan/b", "plan/c"):
        _cli(tmp_path, db, "add", branch, "--worktree", f"/wt/{branch}")
    assert _cli(tmp_path, db, "mv", "plan/c", "0") == (0, "mv: plan/c -> position 0")
    assert [r["branch"] for r in _rows(tmp_path, db)] == ["plan/c", "plan/a", "plan/b"]
    assert _cli(tmp_path, db, "rm", "plan/b") == (
        0,
        "rm: removed 1 entry/entries for 'plan/b'",
    )
    code, out = _cli(tmp_path, db, "ls")
    assert code == 0
    assert out.splitlines() == [
        "  0  queued   plan/c\tattempts=0\tby=-",
        "  1  queued   plan/a\tattempts=0\tby=-",
    ]
    assert _cli(tmp_path, db, "mv", "plan/zzz", "0") == (
        1,
        "mv: 'plan/zzz' is not a queued entry",
    )


def test_claim_done_watch_and_banner_on_a_landing(tmp_path: Path, db: Path) -> None:
    """Claim takes the head, done marks it landed, watch exits 0, the green banner prints."""
    _cli(tmp_path, db, "add", "plan/a", "--by", "session-1")
    assert _cli(tmp_path, db, "peek") == (0, "plan/a\t-")
    assert _cli(tmp_path, db, "claim") == (0, "plan/a\t-")
    (row,) = _rows(tmp_path, db, "--status", "claimed")
    assert (row["attempts"], row["enqueued_by"]) == (1, "session-1")
    assert _cli(tmp_path, db, "claim") == (0, "")  # nothing left to claim
    assert _cli(tmp_path, db, "done", "plan/a", "--sha", "abcdef1234567890") == (
        0,
        "done: plan/a -> landed",
    )
    assert _cli(
        tmp_path, db, "watch", "plan/a", "--timeout", "1", "--interval", "0.05"
    ) == (0, "watch: landed plan/a @ abcdef1234")
    code, banner = _cli(tmp_path, db, "banner", "plan/a")
    assert code == 0
    assert "LANDED - SAFE TO CLOSE" in banner.splitlines()[0]
    assert "   main      abcdef1234" in banner.splitlines()
    assert _rows(tmp_path, db) == []
    assert [r["status"] for r in _rows(tmp_path, db, "--all")] == ["landed"]
    assert _cli(tmp_path, db, "done", "plan/a") == (0, "done: no active plan/a")


def test_block_requeue_and_the_watch_exit_codes(tmp_path: Path, db: Path) -> None:
    """A conflict block exits 3, requeue re-tails the branch (exit 4 on timeout, yellow banner),
    a non-conflict block exits 2 with the red banner, an unknown branch exits 5.
    """
    _cli(tmp_path, db, "add", "plan/a")
    _cli(tmp_path, db, "add", "plan/b")
    _cli(tmp_path, db, "claim")
    assert _cli(
        tmp_path, db, "block", "plan/a", "--reason", "rebase conflict onto origin/main"
    ) == (0, "block: plan/a -> blocked")
    code, out = _cli(
        tmp_path, db, "watch", "plan/a", "--timeout", "1", "--interval", "0.05"
    )
    assert code == 3
    assert out == (
        "watch: rebase conflict on plan/a -- the owning session resolves it in the "
        "worktree, then requeue + re-drain."
    )
    assert _cli(tmp_path, db, "requeue", "plan/a") == (0, "requeue: plan/a -> queued")
    assert [(r["branch"], r["position"]) for r in _rows(tmp_path, db)] == [
        ("plan/b", 1),
        ("plan/a", 2),
    ]
    assert _cli(
        tmp_path, db, "watch", "plan/a", "--timeout", "0.2", "--interval", "0.05"
    ) == (4, "watch: still queued plan/a @ position 2")
    code, banner = _cli(tmp_path, db, "banner", "plan/a")
    assert code == 0 and "QUEUED - LEAVE OPEN" in banner and "position 2" in banner
    _cli(tmp_path, db, "claim")  # claims plan/b (head)
    _cli(tmp_path, db, "block", "plan/b", "--reason", "push failed: rejected")
    assert _cli(
        tmp_path, db, "watch", "plan/b", "--timeout", "1", "--interval", "0.05"
    ) == (2, "watch: blocked plan/b -- push failed: rejected")
    code, banner = _cli(tmp_path, db, "banner", "plan/b")
    assert code == 0 and "BLOCKED - NEEDS YOU" in banner
    assert "   reason    push failed: rejected" in banner.splitlines()
    assert _cli(tmp_path, db, "watch", "plan/zzz", "--timeout", "0.1") == (
        5,
        "watch: no queue entry for 'plan/zzz'",
    )
    assert _cli(tmp_path, db, "requeue", "plan/zzz") == (
        0,
        "requeue: no blocked entry for 'plan/zzz'",
    )


def test_loop_status_follows_the_heartbeat_file(tmp_path: Path, db: Path) -> None:
    """No heartbeat: stopped, exit 1; after heartbeat: live, exit 0; a zero window: stopped."""
    _cli(tmp_path, db, "init")
    assert _cli(tmp_path, db, "loop-status") == (1, "stopped (no heartbeat)")
    assert _cli(tmp_path, db, "heartbeat")[0] == 0
    assert (db.parent / "loop.heartbeat").is_file()
    code, out = _cli(tmp_path, db, "loop-status", "--json")
    assert code == 0 and json.loads(out) == {"state": "live", "age_s": 0}
    assert _cli(tmp_path, db, "loop-status", "--window", "0") == (
        1,
        "stopped (heartbeat 0s ago)",
    )
    assert _cli(tmp_path, db, "lock-path") == (0, str(db.parent / "landing.lock"))


@pytest.mark.parametrize(
    ("status", "error", "elapsed", "expected"),
    [
        ("landed", None, 0.0, "landed"),
        ("blocked", "rebase conflict onto origin/main", 0.0, "resolve_conflict"),
        ("blocked", "Rebase Conflict", 0.0, "resolve_conflict"),
        ("blocked", "push failed: x", 0.0, "needs_human"),
        ("blocked", None, 0.0, "needs_human"),
        ("queued", None, 10.0, "wait"),
        ("claimed", None, 570.0, "timeout"),
        ("queued", None, 600.0, "timeout"),
    ],
)
def test_classify_watch(
    status: str, error: str | None, elapsed: float, expected: str
) -> None:
    """The five watch states from status, last_error and elapsed time against a 570 s timeout."""
    assert merge_queue.classify_watch(status, error, elapsed, 570.0) == expected


def test_foreign_commit_rule_requires_all_four_conditions() -> None:
    """Note-only, plan: subject, no own slug, no own issue number; any miss keeps the commit."""
    slug, issue = (
        merge_queue.branch_slug("plan/my-slug-12"),
        merge_queue.branch_issue("plan/my-slug-12"),
    )
    assert (slug, issue) == ("my-slug-12", "12")
    assert merge_queue.branch_issue("plan/no-issue") is None
    assert merge_queue.branch_slug("bare") == "bare"
    meta = merge_queue.CommitMeta
    assert merge_queue.is_foreign_commit(
        meta("a", "plan: other feature", True), slug, issue
    )
    assert not merge_queue.is_foreign_commit(
        meta("a", "plan: other feature", False), slug, issue
    )
    assert not merge_queue.is_foreign_commit(
        meta("a", "feat: other", True), slug, issue
    )
    assert not merge_queue.is_foreign_commit(
        meta("a", "plan: My-Slug-12 note", True), slug, issue
    )
    assert not merge_queue.is_foreign_commit(
        meta("a", "plan: fixes #12", True), slug, issue
    )


def test_scan_foreign_is_contiguous_only_when_foreign_commits_form_a_base_prefix() -> (
    None
):
    """[foreign, foreign, own] strips at the second sha; [own, foreign] is not contiguous."""
    m = merge_queue.CommitMeta
    prefix = merge_queue.scan_foreign(
        [
            m("f1", "plan: x", True),
            m("f2", "plan: y", True),
            m("o1", "feat: mine", False),
        ],
        "s",
        None,
    )
    assert (prefix.foreign, prefix.contiguous, prefix.strip_point) == (
        ["f1", "f2"],
        True,
        "f2",
    )
    tangled = merge_queue.scan_foreign(
        [m("o1", "feat: mine", False), m("f1", "plan: x", True)], "s", None
    )
    assert (tangled.foreign, tangled.contiguous, tangled.strip_point) == (
        ["f1"],
        False,
        None,
    )
    clean = merge_queue.scan_foreign([m("o1", "feat: mine", False)], "s", None)
    assert (clean.foreign, clean.contiguous, clean.strip_point) == ([], True, None)


def test_banners_name_the_branch_and_state() -> None:
    """Each banner's first line carries its state and the branch line follows."""
    landed = merge_queue.banner_landed("plan/a", "abcdef1234567890").splitlines()
    assert "LANDED - SAFE TO CLOSE" in landed[0] and landed[1:3] == [
        "   branch    plan/a",
        "   main      abcdef1234",
    ]
    human = merge_queue.banner_needs_human("plan/a", None).splitlines()
    assert "BLOCKED - NEEDS YOU" in human[0] and human[2] == "   reason    unknown"
    queued = merge_queue.banner_queued("plan/a", 3).splitlines()
    assert "QUEUED - LEAVE OPEN" in queued[0] and "position 3" in queued[2]


def _git(cwd: Path, *args: str, env: dict[str, str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, f"git {' '.join(args)}: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A bare origin with main pushed, and a clone with git identity set."""
    env = _env(tmp_path)
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin), env=env)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone), env=env)
    _git(clone, "config", "user.email", "t@t", env=env)
    _git(clone, "config", "user.name", "t", env=env)
    (clone / "README").write_text("base\n")
    _git(clone, "add", "README", env=env)
    _git(clone, "commit", "-q", "-m", "base", env=env)
    _git(clone, "push", "-q", "origin", "main", env=env)
    return clone, env


def _commit(clone: Path, env: dict[str, str], path: str, subject: str) -> str:
    (clone / path).parent.mkdir(parents=True, exist_ok=True)
    (clone / path).write_text(subject + "\n")
    _git(clone, "add", path, env=env)
    _git(clone, "commit", "-q", "-m", subject, env=env)
    return _git(clone, "rev-parse", "HEAD", env=env)


def test_foreign_cli_flags_an_inherited_plan_note_and_names_the_strip_command(
    tmp_path: Path, db: Path, repo: tuple[Path, dict[str, str]]
) -> None:
    """A note-only `plan:` commit at the branch base exits 1 with the rebase --onto recipe."""
    clone, env = repo
    _git(clone, "checkout", "-q", "-b", "plan/my-slug-7", env=env)
    foreign = _commit(clone, env, "notes/plan.other.md", "plan: other feature")
    _commit(clone, env, "code.py", "feat: my-slug-7 work")
    code, out = _cli(tmp_path, db, "foreign", "plan/my-slug-7", cwd=clone)
    assert code == 1
    assert f"  {foreign[:10]}  plan: other feature" in out.splitlines()
    assert (
        f"  git -C <worktree> rebase --onto origin/main {foreign[:10]} plan/my-slug-7"
        in out
    )


def test_foreign_cli_is_clean_for_own_commits_and_manual_for_a_tangle(
    tmp_path: Path, db: Path, repo: tuple[Path, dict[str, str]]
) -> None:
    """Own commits only: exit 0; a foreign note after an own commit: exit 2, no strip recipe."""
    clone, env = repo
    _git(clone, "checkout", "-q", "-b", "plan/clean-1", env=env)
    _commit(clone, env, "code.py", "feat: clean-1 work")
    assert _cli(tmp_path, db, "foreign", "plan/clean-1", cwd=clone) == (
        0,
        "foreign: clean -- every commit in origin/main..plan/clean-1 is this branch's own work",
    )
    _git(clone, "checkout", "-q", "-b", "plan/tangle-2", env=env)
    _commit(clone, env, "notes/plan.sibling.md", "plan: sibling feature")
    code, out = _cli(tmp_path, db, "foreign", "plan/tangle-2", cwd=clone)
    assert code == 2
    assert "strip them with" not in out
    assert out.splitlines()[-1] == (
        "foreign commits are NOT a contiguous base prefix -- strip by hand "
        "(interactive rebase) before enqueuing."
    )
