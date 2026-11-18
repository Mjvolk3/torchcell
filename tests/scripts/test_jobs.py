# tests/scripts/test_jobs.py
# [[tests.scripts.test_jobs]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/scripts/test_jobs.py
"""``scripts/jobs.py`` (``make jobs``) on fixture data, and the registry against crontab.

The registry test is the single-source-of-truth check: every job line of
``scripts/crontab.txt`` must be a ``CRON_JOBS`` entry with the same schedule and
command, and every entry must have exactly one line. The record round trip runs the
real ``scripts/cron_run.sh`` into ``tmp_path`` and reads the records back with the
panel's readers. Everything else is a pure function on fixture JSON with a fixed
``NOW``; the local zone is pinned to UTC so rendered timestamps are exact.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)


def _load() -> ModuleType:
    path = REPO / "scripts" / "jobs.py"
    spec = importlib.util.spec_from_file_location("tc_jobs_panel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["tc_jobs_panel"] = module
    spec.loader.exec_module(module)
    return module


jobs = _load()
PLAIN = jobs.palette(False)


@pytest.fixture
def utc_zone(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the process's local zone to UTC so ``fmt_when`` output is fixed."""
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _record(
    started: str, finished: str | None, exit_code: int | None, job: str = "demo"
) -> str:
    return json.dumps(
        {
            "job": job,
            "started_at": started,
            "finished_at": finished,
            "exit_code": exit_code,
            "host": "gilahyper",
            "log": f"/tmp/torchcell-{job}.log",
        }
    )


# --- registry and crontab ------------------------------------------------------------


def test_cron_period_shapes() -> None:
    assert jobs.cron_period("*/2 * * * *") == timedelta(minutes=2)
    assert jobs.cron_period("15 * * * *") == timedelta(hours=1)
    assert jobs.cron_period("30 3 * * *") == timedelta(days=1)
    assert jobs.cron_period("0 2 * * 0") == timedelta(days=7)


@pytest.mark.parametrize(
    "schedule", ["0 4 */14 * *", "0 4 1 * *", "0 */6 * * *", "0 4 * *"]
)
def test_cron_period_refuses_unknown_shapes(schedule: str) -> None:
    with pytest.raises(ValueError, match="cron schedule"):
        jobs.cron_period(schedule)


def test_registry_names_periods_and_sources() -> None:
    assert [(j.name, j.period, j.status_source) for j in jobs.CRON_JOBS] == [
        ("drain-merge-queue", timedelta(minutes=2), "heartbeat"),
        ("lit-sync", timedelta(days=1), "wrapper"),
        ("lit-annotations", timedelta(days=1), "wrapper"),
        ("lit-bib-store", timedelta(days=1), "wrapper"),
        ("purge-deprecated", timedelta(days=1), "wrapper"),
        ("backup-mirrors", timedelta(days=7), "wrapper"),
        ("deps-weekly", timedelta(days=7), "script"),
    ]


def test_parse_crontab_fixture() -> None:
    text = (
        "# a comment\n"
        "MAILTO=nobody\n"
        "\n"
        "*/2 * * * * bash -lc 'cd /r && py scripts/drain.py >> /tmp/d.log 2>&1'\n"
        "30 3 * * * bash -lc 'cd /r && bash scripts/cron_run.sh lit-sync -- py a.py"
        " --x 1; bash scripts/cron_run.sh lit-annotations -- py b.py'\n"
        "0 2 * * 0 bash -lc 'cd /r && bash scripts/cron_run.sh backup -- bash c.sh'\n"
    )
    assert [e.model_dump() for e in jobs.parse_crontab(text)] == [
        {
            "schedule": "*/2 * * * *",
            "job": None,
            "command": "bash -lc 'cd /r && py scripts/drain.py >> /tmp/d.log 2>&1'",
        },
        {"schedule": "30 3 * * *", "job": "lit-sync", "command": "py a.py --x 1"},
        {"schedule": "30 3 * * *", "job": "lit-annotations", "command": "py b.py"},
        {"schedule": "0 2 * * 0", "job": "backup", "command": "bash c.sh"},
    ]


def test_crontab_txt_matches_registry_both_ways() -> None:
    entries = jobs.parse_crontab((REPO / "scripts" / "crontab.txt").read_text())
    registry = {j.name: j for j in jobs.CRON_JOBS}
    wrapped = [e for e in entries if e.job is not None]
    unwrapped = [e for e in entries if e.job is None]
    # every wrapped line is a registry wrapper job with the same schedule and command
    for entry in wrapped:
        assert entry.job in registry, f"{entry.job} is in crontab.txt, not CRON_JOBS"
        job = registry[entry.job]
        assert (entry.schedule, entry.command, job.status_source) == (
            job.schedule,
            job.command,
            "wrapper",
        )
    # every unwrapped line is a heartbeat or script job's command on its schedule
    heartbeat = [j for j in jobs.CRON_JOBS if j.status_source != "wrapper"]
    assert [
        (e.schedule, [j.name for j in heartbeat if j.command in e.command])
        for e in unwrapped
    ] == [("*/2 * * * *", ["drain-merge-queue"]), ("0 5 * * 0", ["deps-weekly"])]
    # and every registry job has exactly one line
    names = sorted(e.job for e in wrapped) + [j.name for j in heartbeat]
    assert sorted(names) == sorted(registry)


def test_installed_wrapped() -> None:
    sync, drain = jobs.CRON_JOBS[1], jobs.CRON_JOBS[0]
    old = "30 3 * * * bash -lc 'cd /r && py scripts/lit_sync.py >> /tmp/x 2>&1'"
    new = "30 3 * * * bash -lc 'cd /r && bash scripts/cron_run.sh lit-sync -- py'"
    assert (jobs.installed_wrapped(sync, old), jobs.installed_wrapped(sync, new)) == (
        False,
        True,
    )
    assert jobs.installed_wrapped(drain, f"*/2 * * * * x '{drain.command} >> l'")


# --- run records and verdicts ------------------------------------------------------------


def test_record_round_trip_through_the_real_wrapper(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "TORCHCELL_CRON_LOG_DIR": str(tmp_path),
        "TORCHCELL_CRON_HISTORY_MAX": "2",
    }
    wrapper = str(REPO / "scripts" / "cron_run.sh")
    codes = [
        subprocess.run(
            ["bash", wrapper, "demo", "--", *cmd], env=env, check=False
        ).returncode
        for cmd in (["true"], ["bash", "-c", "echo hi"], ["bash", "-c", "exit 3"])
    ]
    assert codes == [0, 0, 3]
    directory = tmp_path / "state" / "torchcell" / "cron"
    latest = jobs.read_latest(directory, "demo")
    assert latest is not None
    assert (latest.job, latest.exit_code, latest.log) == (
        "demo",
        3,
        str(tmp_path / "torchcell-demo.log"),
    )
    assert latest.finished_at is not None and latest.finished_at >= latest.started_at
    history = (directory / "demo.history.jsonl").read_text().splitlines()
    assert [json.loads(line)["exit_code"] for line in history] == [0, 3]
    success = jobs.read_last_success(directory, "demo")
    assert success == jobs.RunRecord.model_validate_json(history[0]).finished_at
    log = (tmp_path / "torchcell-demo.log").read_text().splitlines()
    assert [ln.split(" ")[3] for ln in log if ln.startswith("=== ")] == [
        "start",
        "end",
        "start",
        "end",
        "start",
        "end",
    ]
    assert "hi" in log


def test_wrapper_refuses_a_bad_name_or_missing_separator(tmp_path: Path) -> None:
    env = {**os.environ, "XDG_STATE_HOME": str(tmp_path)}
    wrapper = str(REPO / "scripts" / "cron_run.sh")
    bad = subprocess.run(["bash", wrapper, "Bad", "--", "true"], env=env, check=False)
    sep = subprocess.run(["bash", wrapper, "demo", "true"], env=env, check=False)
    assert (bad.returncode, sep.returncode) == (2, 2)
    assert list(tmp_path.iterdir()) == []


def test_record_model_round_trip() -> None:
    line = _record("2026-11-17T03:30:00-06:00", "2026-11-17T03:41:05-06:00", 0)
    rec = jobs.RunRecord.model_validate_json(line)
    assert rec.finished_at is not None
    assert rec.finished_at - rec.started_at == timedelta(minutes=11, seconds=5)
    assert rec.started_at == datetime(2026, 11, 17, 9, 30, tzinfo=UTC)
    assert json.loads(rec.model_dump_json()) == {
        "job": "demo",
        "started_at": "2026-11-17T03:30:00-06:00",
        "finished_at": "2026-11-17T03:41:05-06:00",
        "exit_code": 0,
        "host": "gilahyper",
        "log": "/tmp/torchcell-demo.log",
    }


def _rec(start_h_ago: float, dur_min: float | None, code: int | None) -> object:
    start = NOW - timedelta(hours=start_h_ago)
    finish = None if dur_min is None else start + timedelta(minutes=dur_min)
    return jobs.RunRecord(
        job="demo",
        started_at=start,
        finished_at=finish,
        exit_code=code,
        host="h",
        log="/tmp/l",
    )


def test_judge_run_verdicts() -> None:
    day = timedelta(days=1)
    ok = _rec(10, 5, 0)
    assert jobs.judge_run(None, None, day, NOW) == "never ran"
    assert jobs.judge_run(_rec(1, None, None), None, day, NOW) == "running"
    assert jobs.judge_run(_rec(10, 5, 1), NOW - timedelta(days=1), day, NOW) == "FAILED"
    assert jobs.judge_run(ok, NOW - timedelta(hours=10), day, NOW) == "ok"
    # 1.5 periods is the edge: 36 h old is still ok, a second more is stale
    assert jobs.judge_run(ok, NOW - timedelta(hours=36), day, NOW) == "ok"
    assert jobs.judge_run(ok, NOW - timedelta(hours=36, seconds=1), day, NOW) == "STALE"
    # a run stuck "running" past the window with no recent success is stale
    assert (
        jobs.judge_run(_rec(40, None, None), NOW - timedelta(days=3), day, NOW)
        == "STALE"
    )
    assert jobs.judge_run(_rec(40, None, None), None, day, NOW) == "STALE"


def test_judge_heartbeat() -> None:
    period = timedelta(minutes=2)
    assert jobs.judge_heartbeat(None, period) == "never ran"
    assert jobs.judge_heartbeat(timedelta(seconds=180), period) == "ok"
    assert jobs.judge_heartbeat(timedelta(seconds=181), period) == "STALE"


# --- formatting ----------------------------------------------------------------------------


def test_fmt_span() -> None:
    spans = [0, 45, 60, 720, 3600, 3660, 11100, 86400, 90000, 604800, -5]
    assert [jobs.fmt_span(timedelta(seconds=s)) for s in spans] == [
        "0s",
        "45s",
        "1m",
        "12m",
        "1h",
        "1h01m",
        "3h05m",
        "1d",
        "1d01h",
        "7d",
        "0s",
    ]


def test_truncate() -> None:
    assert (jobs.truncate("abcdef", 6), jobs.truncate("abcdefg", 6)) == (
        "abcdef",
        "abcde…",
    )


def test_format_table_aligns_and_leaves_last_column_unpadded() -> None:
    lines = jobs.format_table(
        ["A", "LONGER", "LAST"], [["xyz", "1", "end"], ["q", "22", "e"]], ["✓", "✗"]
    )
    assert lines == [
        "     A    LONGER  LAST",
        "  ✓  xyz  1       end",
        "  ✗  q    22      e",
    ]
    assert jobs.format_table(["A", "B"], [], []) == ["     A  B"]


def test_cron_rows_from_fixture_records(tmp_path: Path, utc_zone: None) -> None:
    (tmp_path / "lit-sync.json").write_text(
        _record("2026-11-17T03:30:00+00:00", "2026-11-17T03:41:05+00:00", 0, "lit-sync")
    )
    (tmp_path / "lit-sync.history.jsonl").write_text(
        _record("2026-11-17T03:30:00+00:00", "2026-11-17T03:41:05+00:00", 0, "lit-sync")
        + "\n"
    )
    (tmp_path / "lit-bib-store.json").write_text(
        _record(
            "2026-11-17T03:45:00+00:00", "2026-11-17T03:45:30+00:00", 1, "lit-bib-store"
        )
    )
    (tmp_path / "purge-deprecated.json").write_text(
        _record("2026-11-17T11:00:00+00:00", None, None, "purge-deprecated")
    )
    (tmp_path / "backup-mirrors.json").write_text(
        _record(
            "2026-11-01T02:00:00+00:00",
            "2026-11-01T02:20:00+00:00",
            0,
            "backup-mirrors",
        )
    )
    (tmp_path / "backup-mirrors.history.jsonl").write_text(
        _record(
            "2026-11-01T02:00:00+00:00",
            "2026-11-01T02:20:00+00:00",
            0,
            "backup-mirrors",
        )
        + "\n"
    )
    live = "30 3 * * * bash -lc 'cd /r && bash scripts/cron_run.sh lit-sync -- py'\n"
    lines = jobs.cron_rows(
        jobs.CRON_JOBS, tmp_path, live, timedelta(seconds=40), NOW, PLAIN
    )
    assert lines == [
        "     JOB                SCHEDULE     LAST START        DURATION  EXIT  AGE / PERIOD  VERDICT",
        "  ✓  drain-merge-queue  */2 * * * *  2026-11-17 11:59  -         -     40s / 2m      ok (heartbeat)",
        "  ✓  lit-sync           30 3 * * *   2026-11-17 03:30  11m       0     8h30m / 1d    ok",
        "  !  lit-annotations    30 3 * * *   -                 -         -     - / 1d        never ran (wrapper not installed yet)",
        "  ✗  lit-bib-store      45 3 * * *   2026-11-17 03:45  30s       1     8h15m / 1d    FAILED",
        "         log: /tmp/torchcell-lit-bib-store.log",
        "  ✓  purge-deprecated   0 4 * * *    2026-11-17 11:00  1h+       -     1h / 1d       running",
        "  ✗  backup-mirrors     0 2 * * 0    2026-11-01 02:00  20m       0     16d10h / 7d   STALE",
        "  !  deps-weekly        0 5 * * 0    -                 -         -     - / 7d        never ran (wrapper not installed yet)",
    ]


# --- continuous jobs -----------------------------------------------------------------------


def test_parse_elapsed() -> None:
    assert [
        jobs.parse_elapsed(t) for t in ("0:00", "5:07", "2:25:01", "3-04:05:06")
    ] == [
        timedelta(0),
        timedelta(minutes=5, seconds=7),
        timedelta(hours=2, minutes=25, seconds=1),
        timedelta(days=3, hours=4, minutes=5, seconds=6),
    ]


def test_slurm_lines_from_squeue_fixture() -> None:
    text = (
        "7541|031-r7-w814|RUNNING|2:25:00|gilahyper|gilahyper\n"
        "7600|long-one|RUNNING|1-02:00:00|gilahyper|gilahyper\n"
        "7555|031-r7-w816|PENDING|0:00||(Dependency)\n"
        "7556|031-r7-w817|PENDING|0:00||(Dependency)\n"
        "7557|x|PENDING|0:00||(Priority)\n"
        "7558|y|COMPLETING|0:10|gilahyper|gilahyper\n"
    )
    assert jobs.slurm_lines(jobs.parse_squeue(text), PLAIN) == [
        "  ✓  slurm: 2 running, 3 pending, 1 other",
        "     pending by reason: Dependency 2, Priority 1",
        "     JOBID  NAME         STATE    ELAPSED  NODE",
        "  ✓  7600   long-one     RUNNING  1d02h    gilahyper",
        "  ✓  7541   031-r7-w814  RUNNING  2h25m    gilahyper",
    ]


def test_docker_lines_from_fixture() -> None:
    text = (
        "tc-lit-endpoint|running|Up 2 months (healthy)|2 months ago\n"
        "bad|running|Up 1 hour (unhealthy)|3 days ago\n"
        "tc-neo4j|exited|Exited (137) 2 months ago|2 months ago\n"
    )
    assert jobs.docker_lines(jobs.parse_docker(text), PLAIN) == [
        "  ✓  docker: 2 running, 1 not running",
        "     CONTAINER        STATUS                 CREATED",
        "  ✓  tc-lit-endpoint  Up 2 months (healthy)  2 months ago",
        "  ✗  bad              Up 1 hour (unhealthy)  3 days ago",
        "     not running: tc-neo4j (exited (137) 2 months ago)",
    ]


# --- GitHub sections -------------------------------------------------------------------------


def _run(
    name: str, status: str, conclusion: str, sha: str, hours_ago: int
) -> dict[str, str]:
    return {
        "workflowName": name,
        "status": status,
        "conclusion": conclusion,
        "headSha": sha,
        "createdAt": (NOW - timedelta(hours=hours_ago))
        .isoformat()
        .replace("+00:00", "Z"),
        "url": f"https://github.com/o/r/actions/runs/{sha}",
    }


def test_workflow_lines_from_fixture() -> None:
    raw = [
        _run("Lint (ruff)", "completed", "success", "aaaaaaaaaa", 1),
        _run("Lint (ruff)", "completed", "failure", "bbbbbbbbbb", 5),
        _run("Pytest", "in_progress", "", "cccccccccc", 0),
        _run("Pytest", "completed", "failure", "dddddddddd", 3),
        _run("Docs", "completed", "cancelled", "eeeeeeeeee", 30),
        _run("New", "queued", "", "ffffffffff", 0),
    ]
    runs = [jobs.WorkflowRun.model_validate(r) for r in raw]
    assert jobs.workflow_lines(runs, NOW, PLAIN) == [
        "     WORKFLOW     RESULT                       COMMIT    AGE",
        "  !  Docs         cancelled                    eeeeeeee  1d06h",
        "  ✓  Lint (ruff)  success                      aaaaaaaa  1h",
        "  !  New          queued (no completed run)    ffffffff  0s",
        "  ✗  Pytest       in_progress (last: failure)  cccccccc  0s",
        "  ✗  failed on main:",
        "https://github.com/o/r/actions/runs/dddddddddd",
    ]


def test_rollup_state() -> None:
    def item(**kw: str) -> object:
        return jobs.CheckItem.model_validate(kw)

    ok = item(__typename="CheckRun", status="COMPLETED", conclusion="SUCCESS")
    skip = item(__typename="CheckRun", status="COMPLETED", conclusion="SKIPPED")
    running = item(__typename="CheckRun", status="IN_PROGRESS", conclusion="")
    bad = item(__typename="CheckRun", status="COMPLETED", conclusion="FAILURE")
    ctx_pending = item(__typename="StatusContext", state="PENDING")
    ctx_error = item(__typename="StatusContext", state="ERROR")
    assert [
        jobs.rollup_state(c)
        for c in (
            [],
            [ok, skip],
            [ok, running],
            [ok, ctx_pending],
            [running, bad],
            [ok, ctx_error],
        )
    ] == ["none", "success", "pending", "pending", "failure", "failure"]


def _iso(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def test_pr_lines_from_fixture() -> None:
    ok = {"__typename": "CheckRun", "status": "COMPLETED", "conclusion": "SUCCESS"}
    bad = {"__typename": "CheckRun", "status": "COMPLETED", "conclusion": "FAILURE"}
    raw = [
        {
            "number": 7,
            "title": "old and failing",
            "headRefName": "fix/old",
            "createdAt": _iso(40),
            "updatedAt": _iso(20),
            "isDraft": False,
            "statusCheckRollup": [ok, bad],
        },
        {
            "number": 12,
            "title": "x" * 70,
            "headRefName": "feat/new",
            "createdAt": _iso(2),
            "updatedAt": _iso(0.5),
            "isDraft": True,
            "statusCheckRollup": [ok],
        },
        {
            "number": 9,
            "title": "no checks",
            "headRefName": "docs/none",
            "createdAt": _iso(15),
            "updatedAt": _iso(14),
            "isDraft": False,
            "statusCheckRollup": [],
        },
    ]
    prs = [jobs.PullRequest.model_validate(p) for p in raw]
    assert jobs.pr_lines(prs, NOW, PLAIN) == [
        "     open: 3, stale (no update in 14 days): 1",
        "     PR   AGE(d)  UPDATED  CI       FLAGS  BRANCH     TITLE",
        "  ✓  #12  2       12h ago  success  draft  feat/new   " + "x" * 59 + "…",
        "  !  #9   15      14d ago  none     -      docs/none  no checks",
        "  ✗  #7   40      20d ago  failure  stale  fix/old    old and failing",
    ]
    assert jobs.pr_lines([], NOW, PLAIN) == [
        "     open: 0, stale (no update in 14 days): 0"
    ]


def test_issue_lines_from_fixture() -> None:
    def issue(n: int, labels: list[str], days_ago: float) -> object:
        return jobs.Issue.model_validate(
            {
                "number": n,
                "title": f"issue {n}",
                "labels": [{"name": lb} for lb in labels],
                "updatedAt": _iso(days_ago),
            }
        )

    issues = [
        issue(1, ["bug", "priority:high"], 40),
        issue(2, ["dataset"], 1),
        issue(3, [], 31),
        issue(4, ["dataset", "bug"], 2),
    ]
    assert jobs.issue_lines(issues, NOW, PLAIN) == [
        "     open: 4, no activity in 30 days: 2",
        "     by label: bug 2, dataset 2, priority:high 1, (none) 1",
        "  ✗  priority:high: 1",
        "     ISSUE  UPDATED  LABELS             TITLE",
        "  ✗  #1     40d ago  bug,priority:high  issue 1",
        "     most recently updated (4):",
        "     ISSUE  UPDATED  LABELS             TITLE",
        "     #2     1d ago   dataset            issue 2",
        "     #4     2d ago   dataset,bug        issue 4",
        "     #3     31d ago  -                  issue 3",
        "     #1     40d ago  bug,priority:high  issue 1",
    ]


def test_gh_failure_is_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd="gh", timeout=3)

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    assert jobs.section_prs(3.0, NOW, PLAIN) == [
        "== pull requests ==",
        "  ✗  GitHub unreachable: no answer within 3s",
    ]
