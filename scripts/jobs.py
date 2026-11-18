#!/usr/bin/env python
# scripts/jobs.py
# [[scripts.jobs]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/jobs.py
"""Status panel for torchcell's recurring and continuous jobs, run from GilaHyper.

    make jobs          every section below
    make jobs-fast     the local sections only (no GitHub calls)

Sections, in order:

1. Recurring jobs (cron). One row per job in ``CRON_JOBS``, the registry that
   ``scripts/crontab.txt`` is checked against (``tests/scripts/test_jobs.py``). A
   wrapped job's last run comes from the records ``scripts/cron_run.sh`` writes under
   ``${XDG_STATE_HOME:-~/.local/state}/torchcell/cron/``; the merge-queue drainer's
   from its heartbeat (``merge_queue.py loop-status --json``). Verdicts: ``ok``,
   ``running``, ``FAILED`` (the last run exited nonzero), ``STALE`` (no successful run
   within ``STALE_FACTOR`` expected periods) and ``never ran``, which also says when
   the live crontab does not run the job through the wrapper yet.
2. Continuous jobs: the user's slurm queue (running and pending counts, the longest
   running few) and the docker containers. Long-lived services are ``make ops``'s.
3. GitHub Actions: the latest run on ``main`` of each workflow seen in the newest
   ``RUN_LIMIT`` main runs; a failed one gets its run URL on its own line.
4. Open pull requests, newest update first, with their CI rollup; stale after
   ``PR_STALE_DAYS`` days without an update.
5. Open issues: count, count by label, the ``priority:high`` ones, the most recently
   updated, and how many have been quiet for ``ISSUE_QUIET_DAYS`` days.

Every external command goes through ``run_cmd``, the one boundary that turns a timeout,
a missing binary or a nonzero exit into a one-line "unreachable" for its section. Three
``gh`` calls in all. Read-only; exits 0.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
CRONTAB_TXT = REPO_ROOT / "scripts" / "crontab.txt"
ENV_PYTHON = "/home/michaelvolk/miniconda3/envs/torchcell/bin/python"

STALE_FACTOR = 1.5
PR_STALE_DAYS = 14
ISSUE_QUIET_DAYS = 30
ISSUE_LIMIT = 500
PR_LIMIT = 200
RUN_LIMIT = 200
RECENT_ISSUES = 10
LONGEST_SLURM = 5
TITLE_WIDTH = 60
LOCAL_TIMEOUT_S = 10.0

# --- registry ---------------------------------------------------------------------


class CronJob(BaseModel):
    """One recurring job: its crontab schedule and the command it runs from the repo."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    schedule: str
    command: str
    status_source: Literal["wrapper", "heartbeat"]

    @property
    def period(self) -> timedelta:
        """Expected time between two runs, derived from the schedule."""
        return cron_period(self.schedule)


# The single registry. Each entry's line in scripts/crontab.txt runs `command` (for a
# wrapper job, as `bash scripts/cron_run.sh <name> -- <command>`) on `schedule`;
# tests/scripts/test_jobs.py fails when the two disagree in either direction.
# Extension point: the weekly dependency-update job `deps-weekly` (Sunday) is
# registered here, with its wrapped crontab line, by its own PR.
CRON_JOBS: tuple[CronJob, ...] = (
    CronJob(
        name="drain-merge-queue",
        schedule="*/2 * * * *",
        command=f"{ENV_PYTHON} scripts/drain_merge_queue.py",
        status_source="heartbeat",
    ),
    CronJob(
        name="lit-sync",
        schedule="30 3 * * *",
        command=(
            f"{ENV_PYTHON} scripts/lit_sync.py --limit 10"
            " --personal-root torchcell --personal-root thesis"
        ),
        status_source="wrapper",
    ),
    CronJob(
        name="lit-annotations",
        schedule="30 3 * * *",
        command=f"{ENV_PYTHON} scripts/lit_annotations.py",
        status_source="wrapper",
    ),
    CronJob(
        name="lit-bib-store",
        schedule="45 3 * * *",
        command=f"{ENV_PYTHON} scripts/lit_bib_store.py",
        status_source="wrapper",
    ),
    CronJob(
        name="purge-deprecated",
        schedule="0 4 * * *",
        command="bash scripts/purge_deprecated.sh",
        status_source="wrapper",
    ),
    CronJob(
        name="backup-mirrors",
        schedule="0 2 * * 0",
        command="bash scripts/backup_mirrors_to_bulk.sh",
        status_source="wrapper",
    ),
)


def cron_period(schedule: str) -> timedelta:
    """Expected period of the schedule shapes the registry uses.

    ``*/N * * * *`` is N minutes, ``M * * * *`` an hour, ``M H * * *`` a day and
    ``M H * * D`` a week. Any other shape raises, so a new job with a new shape adds
    its rule here instead of being judged against a wrong period.
    """
    fields = schedule.split()
    if len(fields) != 5:
        raise ValueError(f"cron schedule needs 5 fields: {schedule!r}")
    minute, hour, dom, month, dow = fields
    if dom != "*" or month != "*":
        raise ValueError(f"unsupported cron schedule {schedule!r}")
    step = re.fullmatch(r"\*/(\d+)", minute)
    if step and hour == "*" and dow == "*":
        return timedelta(minutes=int(step.group(1)))
    if not minute.isdigit():
        raise ValueError(f"unsupported cron schedule {schedule!r}")
    if hour == "*" and dow == "*":
        return timedelta(hours=1)
    if hour.isdigit() and dow == "*":
        return timedelta(days=1)
    if hour.isdigit() and dow.isdigit():
        return timedelta(days=7)
    raise ValueError(f"unsupported cron schedule {schedule!r}")


# --- crontab parsing -----------------------------------------------------------------


class CrontabEntry(BaseModel):
    """One command a crontab line runs; a line with two wrapped jobs gives two."""

    schedule: str
    job: str | None  # the cron_run.sh job name; None for an unwrapped line
    command: str  # the wrapped command, or the whole line body when unwrapped


_WRAPPED = re.compile(
    r"bash scripts/cron_run\.sh (?P<job>\S+) -- (?P<cmd>.+?)"
    r"(?=; bash scripts/cron_run\.sh |'$)"
)


def parse_crontab(text: str) -> list[CrontabEntry]:
    """The job entries of a crontab text; comments, blanks and VAR=value lines skipped."""
    entries: list[CrontabEntry] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" in line.split(maxsplit=1)[0]:
            continue
        fields = line.split(maxsplit=5)
        schedule, body = " ".join(fields[:5]), fields[5]
        wrapped = list(_WRAPPED.finditer(body))
        if wrapped:
            entries.extend(
                CrontabEntry(schedule=schedule, job=m["job"], command=m["cmd"])
                for m in wrapped
            )
        else:
            entries.append(CrontabEntry(schedule=schedule, job=None, command=body))
    return entries


def installed_wrapped(job: CronJob, live_crontab: str) -> bool:
    """Whether the live crontab runs this job the way the registry expects."""
    if job.status_source == "heartbeat":
        return job.command in live_crontab
    return f"cron_run.sh {job.name} -- " in live_crontab


# --- run records -----------------------------------------------------------------------


class RunRecord(BaseModel):
    """What ``scripts/cron_run.sh`` writes for one run (finished_at None while running)."""

    job: str
    started_at: datetime
    finished_at: datetime | None
    exit_code: int | None
    host: str
    log: str


def state_dir() -> Path:
    """The wrapper's record directory, resolved by the same rule as cron_run.sh."""
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "torchcell" / "cron"


def read_latest(directory: Path, job: str) -> RunRecord | None:
    """The job's latest record, None when the wrapper has never run it."""
    path = directory / f"{job}.json"
    if not path.exists():
        return None
    return RunRecord.model_validate_json(path.read_text())


def read_last_success(directory: Path, job: str) -> datetime | None:
    """Finish time of the newest exit-0 run in the job's history, None if none."""
    path = directory / f"{job}.history.jsonl"
    if not path.exists():
        return None
    successes = [
        rec.finished_at
        for rec in (
            RunRecord.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        )
        if rec.exit_code == 0 and rec.finished_at is not None
    ]
    return max(successes) if successes else None


Verdict = Literal["ok", "running", "FAILED", "STALE", "never ran"]


def judge_run(
    last: RunRecord | None,
    last_success: datetime | None,
    period: timedelta,
    now: datetime,
) -> Verdict:
    """Verdict for a wrapped job.

    In order: no record is ``never ran``; a run started less than ``STALE_FACTOR``
    periods ago and not finished is ``running``; a last run that exited nonzero is
    ``FAILED``; no successful finish within ``STALE_FACTOR`` periods is ``STALE``
    (this also catches a run that has been "running" too long); else ``ok``.
    """
    window = period * STALE_FACTOR
    if last is None:
        return "never ran"
    if last.finished_at is None and now - last.started_at < window:
        return "running"
    if last.exit_code is not None and last.exit_code != 0:
        return "FAILED"
    if last_success is None or now - last_success > window:
        return "STALE"
    return "ok"


def judge_heartbeat(age: timedelta | None, period: timedelta) -> Verdict:
    """Verdict for a job that only touches a heartbeat: fresh within the window or not."""
    if age is None:
        return "never ran"
    return "ok" if age <= period * STALE_FACTOR else "STALE"


# --- formatting --------------------------------------------------------------------------


def fmt_span(delta: timedelta) -> str:
    """A compact non-negative span: 45s, 12m, 3h05m, 2d, 2d04h."""
    s = max(int(delta.total_seconds()), 0)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        hours, minutes = divmod(s // 60, 60)
        return f"{hours}h{minutes:02d}m" if minutes else f"{hours}h"
    days, hours = divmod(s // 3600, 24)
    return f"{days}d{hours:02d}h" if hours else f"{days}d"


def fmt_when(moment: datetime) -> str:
    """A timestamp in the local zone, to the minute."""
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


def truncate(text: str, width: int) -> str:
    """Cut to ``width`` characters, the last one an ellipsis when cut."""
    return text if len(text) <= width else text[: width - 1] + "…"


def format_table(
    header: Sequence[str], rows: Sequence[Sequence[str]], marks: Sequence[str]
) -> list[str]:
    """Aligned lines in the ``make ops`` shape: two spaces, a mark, two spaces, cells.

    ``marks`` holds one mark per row (it may carry color codes: it sits outside the
    padded cells, so it never shifts a column). The last column is not padded.
    """
    widths = [
        max([len(header[i]), *(len(row[i]) for row in rows)])
        for i in range(len(header))
    ]

    def cells(values: Sequence[str]) -> str:
        padded = [v.ljust(w) for v, w in zip(values[:-1], widths[:-1], strict=True)]
        return "  ".join([*padded, values[-1]])

    return ["     " + cells(header)] + [
        f"  {mark}  {cells(row)}" for mark, row in zip(marks, rows, strict=True)
    ]


class Palette(BaseModel):
    """Check, bang and cross marks, colored on a terminal like ``make ops``."""

    ok: str
    warn: str
    bad: str


def palette(color: bool) -> Palette:
    """The three marks, with ANSI color when ``color``."""
    if not color:
        return Palette(ok="✓", warn="!", bad="✗")
    return Palette(
        ok="\033[0;32m✓\033[0m", warn="\033[0;33m!\033[0m", bad="\033[0;31m✗\033[0m"
    )


def verdict_mark(verdict: Verdict, marks: Palette) -> str:
    """✓ for ok/running, ! for never ran, ✗ for FAILED/STALE."""
    if verdict in ("ok", "running"):
        return marks.ok
    if verdict == "never ran":
        return marks.warn
    return marks.bad


# --- the one boundary to external commands ------------------------------------------


class CmdResult(BaseModel):
    """A command's stdout, or a one-line reason it gave none."""

    ok: bool
    stdout: str
    error: str


def run_cmd(
    argv: Sequence[str], timeout: float, ok_codes: tuple[int, ...] = (0,)
) -> CmdResult:
    """Run ``argv`` from the repo root; a timeout, missing binary or bad exit is an error line."""
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=REPO_ROOT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CmdResult(ok=False, stdout="", error=f"no answer within {timeout:g}s")
    except FileNotFoundError:
        return CmdResult(ok=False, stdout="", error=f"{argv[0]} not found")
    if proc.returncode not in ok_codes:
        lines = [ln.strip() for ln in proc.stderr.splitlines() if ln.strip()]
        reason = lines[0] if lines else f"exit {proc.returncode}"
        return CmdResult(ok=False, stdout=proc.stdout, error=reason)
    return CmdResult(ok=True, stdout=proc.stdout, error="")


# --- section 1: recurring jobs ----------------------------------------------------------


class LoopStatus(BaseModel):
    """``merge_queue.py loop-status --json``."""

    state: str
    age_s: int | None


def cron_rows(
    jobs: Sequence[CronJob],
    directory: Path,
    live_crontab: str | None,
    heartbeat_age: timedelta | None,
    now: datetime,
    marks: Palette,
) -> list[str]:
    """The recurring-jobs table, plus a ``log:`` line under each FAILED row."""
    rows: list[list[str]] = []
    row_marks: list[str] = []
    notes: dict[int, str] = {}
    for job in jobs:
        period = fmt_span(job.period)
        if job.status_source == "heartbeat":
            verdict = judge_heartbeat(heartbeat_age, job.period)
            start = "-" if heartbeat_age is None else fmt_when(now - heartbeat_age)
            age = "-" if heartbeat_age is None else fmt_span(heartbeat_age)
            row = [job.name, job.schedule, start, "-", "-", f"{age} / {period}"]
            label = f"{verdict} (heartbeat)"
        else:
            last = read_latest(directory, job.name)
            success = read_last_success(directory, job.name)
            verdict = judge_run(last, success, job.period, now)
            if last is None:
                row = [job.name, job.schedule, "-", "-", "-", f"- / {period}"]
            else:
                duration = (
                    fmt_span(last.finished_at - last.started_at)
                    if last.finished_at is not None
                    else f"{fmt_span(now - last.started_at)}+"
                )
                exit_code = "-" if last.exit_code is None else str(last.exit_code)
                age = fmt_span(now - last.started_at)
                row = [
                    job.name,
                    job.schedule,
                    fmt_when(last.started_at),
                    duration,
                    exit_code,
                    f"{age} / {period}",
                ]
                if verdict == "FAILED":
                    notes[len(rows)] = f"log: {last.log}"
            label = verdict
        if (
            verdict == "never ran"
            and live_crontab is not None
            and not installed_wrapped(job, live_crontab)
        ):
            label = "never ran (wrapper not installed yet)"
        rows.append([*row, label])
        row_marks.append(verdict_mark(verdict, marks))
    header = ["JOB", "SCHEDULE", "LAST START", "DURATION", "EXIT", "AGE / PERIOD"]
    lines = format_table([*header, "VERDICT"], rows, row_marks)
    out = [lines[0]]
    for i, line in enumerate(lines[1:]):
        out.append(line)
        if i in notes:
            out.append(f"         {notes[i]}")
    return out


def section_cron(now: datetime, marks: Palette) -> list[str]:
    """Section 1, reading the live crontab and the drainer heartbeat."""
    out = ["== recurring jobs (cron) =="]
    crontab = run_cmd(["crontab", "-l"], LOCAL_TIMEOUT_S)
    live = crontab.stdout if crontab.ok else None
    if not crontab.ok:
        out.append(f"  {marks.warn}  live crontab unreadable: {crontab.error}")
    loop = run_cmd(
        [sys.executable, "scripts/merge_queue.py", "loop-status", "--json"],
        LOCAL_TIMEOUT_S,
        ok_codes=(0, 1),
    )
    heartbeat_age: timedelta | None = None
    if loop.ok:
        status = LoopStatus.model_validate_json(loop.stdout)
        if status.age_s is not None:
            heartbeat_age = timedelta(seconds=status.age_s)
    else:
        out.append(f"  {marks.bad}  merge-queue heartbeat unreadable: {loop.error}")
    out += cron_rows(CRON_JOBS, state_dir(), live, heartbeat_age, now, marks)
    out.append(f"     records: {state_dir()}")
    return out


# --- section 2: continuous jobs ---------------------------------------------------------


class SlurmJob(BaseModel):
    """One ``squeue`` row."""

    job_id: str
    name: str
    state: str
    elapsed: timedelta
    node: str
    reason: str


SQUEUE_FORMAT = "%i|%j|%T|%M|%N|%R"


def parse_elapsed(text: str) -> timedelta:
    """Slurm's ``[D-][H:]M:SS`` elapsed time."""
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def parse_squeue(text: str) -> list[SlurmJob]:
    """Rows of ``squeue -h -o SQUEUE_FORMAT``."""
    jobs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        job_id, name, state, elapsed, node, reason = line.split("|", 5)
        jobs.append(
            SlurmJob(
                job_id=job_id,
                name=name,
                state=state,
                elapsed=parse_elapsed(elapsed),
                node=node,
                reason=reason,
            )
        )
    return jobs


def slurm_lines(jobs: Sequence[SlurmJob], marks: Palette) -> list[str]:
    """Running/pending counts, pending reasons, and the longest-running jobs."""
    running = sorted(
        (j for j in jobs if j.state == "RUNNING"), key=lambda j: j.elapsed, reverse=True
    )
    pending = [j for j in jobs if j.state == "PENDING"]
    other = len(jobs) - len(running) - len(pending)
    summary = f"slurm: {len(running)} running, {len(pending)} pending"
    if other:
        summary += f", {other} other"
    out = [f"  {marks.ok}  {summary}"]
    if pending:
        reasons = Counter(j.reason.strip("()") for j in pending)
        out.append(
            "     pending by reason: "
            + ", ".join(f"{r} {n}" for r, n in reasons.most_common())
        )
    if running:
        top = running[:LONGEST_SLURM]
        out += format_table(
            ["JOBID", "NAME", "STATE", "ELAPSED", "NODE"],
            [[j.job_id, j.name, j.state, fmt_span(j.elapsed), j.node] for j in top],
            [marks.ok] * len(top),
        )
    return out


class Container(BaseModel):
    """One ``docker ps -a`` row."""

    name: str
    state: str
    status: str
    running_for: str


DOCKER_FORMAT = "{{.Names}}|{{.State}}|{{.Status}}|{{.RunningFor}}"


def parse_docker(text: str) -> list[Container]:
    """Rows of ``docker ps -a --format DOCKER_FORMAT``."""
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        name, state, status, running_for = line.split("|", 3)
        out.append(
            Container(name=name, state=state, status=status, running_for=running_for)
        )
    return out


def docker_lines(containers: Sequence[Container], marks: Palette) -> list[str]:
    """Running containers as a table (✗ when unhealthy), stopped ones on one line."""
    running = [c for c in containers if c.state == "running"]
    stopped = [c for c in containers if c.state != "running"]
    out = [f"  {marks.ok}  docker: {len(running)} running, {len(stopped)} not running"]
    if running:
        out += format_table(
            ["CONTAINER", "STATUS", "CREATED"],
            [[c.name, c.status, c.running_for] for c in running],
            [marks.bad if "unhealthy" in c.status else marks.ok for c in running],
        )
    if stopped:
        out.append(
            "     not running: "
            + ", ".join(f"{c.name} ({c.status.lower()})" for c in stopped)
        )
    return out


def section_continuous(marks: Palette) -> list[str]:
    """Section 2: slurm queue, docker containers, and a pointer to ``make ops``."""
    out = ["== continuous jobs =="]
    squeue = run_cmd(
        ["squeue", "-u", getpass.getuser(), "-h", "-o", SQUEUE_FORMAT], LOCAL_TIMEOUT_S
    )
    if squeue.ok:
        out += slurm_lines(parse_squeue(squeue.stdout), marks)
    else:
        out.append(f"  {marks.bad}  slurm unreachable: {squeue.error}")
    docker = run_cmd(["docker", "ps", "-a", "--format", DOCKER_FORMAT], LOCAL_TIMEOUT_S)
    if docker.ok:
        out += docker_lines(parse_docker(docker.stdout), marks)
    else:
        out.append(f"  {marks.warn}  docker unreadable: {docker.error}")
    out.append(
        "     services (neo4j browser, tc-lit, merge-queue loop, disks, radiant):"
        " see `make ops`"
    )
    return out


# --- section 3: GitHub Actions ----------------------------------------------------------

_GH = ConfigDict(populate_by_name=True)
FAILED_CONCLUSIONS = {"failure", "timed_out", "startup_failure", "action_required"}


class WorkflowRun(BaseModel):
    """One row of ``gh run list --json``."""

    model_config = _GH

    workflow_name: str = Field(alias="workflowName")
    status: str
    conclusion: str
    head_sha: str = Field(alias="headSha")
    created_at: datetime = Field(alias="createdAt")
    url: str


RUN_FIELDS = "workflowName,status,conclusion,headSha,createdAt,url"


class WorkflowLatest(BaseModel):
    """A workflow's newest main run and its newest completed one."""

    latest: WorkflowRun
    last_completed: WorkflowRun | None


def latest_per_workflow(runs: Sequence[WorkflowRun]) -> list[WorkflowLatest]:
    """Per workflow, by name: newest run and newest completed run."""
    by_name: dict[str, list[WorkflowRun]] = {}
    for run in sorted(runs, key=lambda r: r.created_at, reverse=True):
        by_name.setdefault(run.workflow_name, []).append(run)
    return [
        WorkflowLatest(
            latest=group[0],
            last_completed=next((r for r in group if r.status == "completed"), None),
        )
        for _, group in sorted(by_name.items())
    ]


def workflow_lines(
    runs: Sequence[WorkflowRun], now: datetime, marks: Palette
) -> list[str]:
    """One row per workflow; the run URL of each failed one on its own line."""
    rows, row_marks, failed_urls = [], [], []
    for item in latest_per_workflow(runs):
        run = item.latest
        judged = item.last_completed
        if run.status == "completed":
            result = run.conclusion
        elif judged is None:
            result = f"{run.status} (no completed run)"
        else:
            result = f"{run.status} (last: {judged.conclusion})"
        conclusion = judged.conclusion if judged is not None else ""
        if conclusion in FAILED_CONCLUSIONS:
            row_marks.append(marks.bad)
            failed_urls.append(judged.url if judged is not None else run.url)
        elif conclusion == "success":
            row_marks.append(marks.ok)
        else:
            row_marks.append(marks.warn)
        rows.append(
            [
                run.workflow_name,
                result,
                run.head_sha[:8],
                fmt_span(now - run.created_at),
            ]
        )
    out = format_table(["WORKFLOW", "RESULT", "COMMIT", "AGE"], rows, row_marks)
    if failed_urls:
        out.append(f"  {marks.bad}  failed on main:")
        out += failed_urls
    return out


def section_actions(timeout: float, now: datetime, marks: Palette) -> list[str]:
    """Section 3."""
    out = [f"== GitHub Actions (latest run on main, of the newest {RUN_LIMIT}) =="]
    res = run_cmd(
        [
            "gh",
            "run",
            "list",
            "--branch",
            "main",
            "--limit",
            str(RUN_LIMIT),
            "--json",
            RUN_FIELDS,
        ],
        timeout,
    )
    if not res.ok:
        return [*out, f"  {marks.bad}  GitHub unreachable: {res.error}"]
    runs = [WorkflowRun.model_validate(r) for r in json.loads(res.stdout)]
    return out + workflow_lines(runs, now, marks)


# --- section 4: pull requests -------------------------------------------------------------


class CheckItem(BaseModel):
    """A ``statusCheckRollup`` entry: a CheckRun (status, conclusion) or a StatusContext (state)."""

    model_config = _GH

    typename: str = Field(alias="__typename")
    status: str | None = None
    conclusion: str | None = None
    state: str | None = None


CHECK_FAILED = {
    "FAILURE",
    "TIMED_OUT",
    "CANCELLED",
    "STARTUP_FAILURE",
    "ACTION_REQUIRED",
    "ERROR",
}


def rollup_state(checks: Sequence[CheckItem]) -> str:
    """``failure`` over ``pending`` over ``success``; ``none`` with no checks."""
    if not checks:
        return "none"
    failed = pending = False
    for check in checks:
        if check.typename == "CheckRun":
            if check.status != "COMPLETED":
                pending = True
            elif check.conclusion in CHECK_FAILED:
                failed = True
        elif check.state in CHECK_FAILED:
            failed = True
        elif check.state in ("PENDING", "EXPECTED"):
            pending = True
    if failed:
        return "failure"
    return "pending" if pending else "success"


class PullRequest(BaseModel):
    """One row of ``gh pr list --json``."""

    model_config = _GH

    number: int
    title: str
    head_ref_name: str = Field(alias="headRefName")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    is_draft: bool = Field(alias="isDraft")
    status_check_rollup: list[CheckItem] = Field(alias="statusCheckRollup")


PR_FIELDS = "number,title,headRefName,createdAt,updatedAt,isDraft,statusCheckRollup"


def pr_lines(prs: Sequence[PullRequest], now: datetime, marks: Palette) -> list[str]:
    """Open PR count and table, newest update first (so the oldest-updated is last)."""
    stale_after = timedelta(days=PR_STALE_DAYS)
    ordered = sorted(prs, key=lambda p: p.updated_at, reverse=True)
    stale = sum(now - p.updated_at > stale_after for p in prs)
    out = [f"     open: {len(prs)}, stale (no update in {PR_STALE_DAYS} days): {stale}"]
    if not ordered:
        return out
    rows, row_marks = [], []
    for pr in ordered:
        ci = rollup_state(pr.status_check_rollup)
        flags = ["draft"] if pr.is_draft else []
        if now - pr.updated_at > stale_after:
            flags.append("stale")
        row_marks.append(
            marks.bad
            if ci == "failure"
            else marks.ok
            if ci == "success"
            else marks.warn
        )
        rows.append(
            [
                f"#{pr.number}",
                str((now - pr.created_at).days),
                f"{fmt_span(now - pr.updated_at)} ago",
                ci,
                ",".join(flags) or "-",
                truncate(pr.head_ref_name, 32),
                truncate(pr.title, TITLE_WIDTH),
            ]
        )
    return out + format_table(
        ["PR", "AGE(d)", "UPDATED", "CI", "FLAGS", "BRANCH", "TITLE"], rows, row_marks
    )


def section_prs(timeout: float, now: datetime, marks: Palette) -> list[str]:
    """Section 4."""
    out = ["== pull requests =="]
    res = run_cmd(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            str(PR_LIMIT),
            "--json",
            PR_FIELDS,
        ],
        timeout,
    )
    if not res.ok:
        return [*out, f"  {marks.bad}  GitHub unreachable: {res.error}"]
    prs = [PullRequest.model_validate(p) for p in json.loads(res.stdout)]
    return out + pr_lines(prs, now, marks)


# --- section 5: issues ----------------------------------------------------------------------


class IssueLabel(BaseModel):
    """A label on an issue."""

    name: str


class Issue(BaseModel):
    """One row of ``gh issue list --json``."""

    model_config = _GH

    number: int
    title: str
    labels: list[IssueLabel]
    updated_at: datetime = Field(alias="updatedAt")


ISSUE_FIELDS = "number,title,labels,updatedAt"
PRIORITY_HIGH = "priority:high"


def issue_table(issues: Sequence[Issue], now: datetime, mark: str) -> list[str]:
    """Issue rows: number, last update, labels, title."""
    return format_table(
        ["ISSUE", "UPDATED", "LABELS", "TITLE"],
        [
            [
                f"#{i.number}",
                f"{fmt_span(now - i.updated_at)} ago",
                truncate(",".join(lb.name for lb in i.labels) or "-", 30),
                truncate(i.title, TITLE_WIDTH),
            ]
            for i in issues
        ],
        [mark] * len(issues),
    )


def issue_lines(issues: Sequence[Issue], now: datetime, marks: Palette) -> list[str]:
    """Counts, labels, the priority:high ones, the recently updated ones."""
    count = (
        f"{len(issues)}+ (list capped)"
        if len(issues) >= ISSUE_LIMIT
        else str(len(issues))
    )
    quiet = sum(now - i.updated_at > timedelta(days=ISSUE_QUIET_DAYS) for i in issues)
    labels = Counter(lb.name for i in issues for lb in i.labels)
    unlabeled = sum(not i.labels for i in issues)
    by_label = sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))
    if unlabeled:
        by_label.append(("(none)", unlabeled))
    out = [
        f"     open: {count}, no activity in {ISSUE_QUIET_DAYS} days: {quiet}",
        "     by label: " + (", ".join(f"{n} {c}" for n, c in by_label) or "-"),
    ]
    high = sorted(
        (i for i in issues if any(lb.name == PRIORITY_HIGH for lb in i.labels)),
        key=lambda i: i.updated_at,
        reverse=True,
    )
    out.append(f"  {marks.bad if high else marks.ok}  {PRIORITY_HIGH}: {len(high)}")
    if high:
        out += issue_table(high, now, marks.bad)
    recent = sorted(issues, key=lambda i: i.updated_at, reverse=True)[:RECENT_ISSUES]
    out.append(f"     most recently updated ({len(recent)}):")
    if recent:
        out += issue_table(recent, now, " ")
    return out


def section_issues(timeout: float, now: datetime, marks: Palette) -> list[str]:
    """Section 5."""
    out = ["== issues =="]
    res = run_cmd(
        [
            "gh",
            "issue",
            "list",
            "--state",
            "open",
            "--limit",
            str(ISSUE_LIMIT),
            "--json",
            ISSUE_FIELDS,
        ],
        timeout,
    )
    if not res.ok:
        return [*out, f"  {marks.bad}  GitHub unreachable: {res.error}"]
    issues = [Issue.model_validate(i) for i in json.loads(res.stdout)]
    return out + issue_lines(issues, now, marks)


# --- main ---------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Print the panel; always 0."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Local sections only: skip the GitHub Actions, PR and issue sections.",
    )
    parser.add_argument(
        "--gh-timeout",
        type=float,
        default=20.0,
        help="Seconds each gh call may take before its section reads unreachable.",
    )
    args = parser.parse_args(argv)
    now = datetime.now(UTC)
    marks = palette(sys.stdout.isatty())
    sections = [section_cron(now, marks), section_continuous(marks)]
    if not args.no_network:
        sections += [
            section_actions(args.gh_timeout, now, marks),
            section_prs(args.gh_timeout, now, marks),
            section_issues(args.gh_timeout, now, marks),
        ]
    print("\n\n".join("\n".join(s) for s in sections))
    return 0


if __name__ == "__main__":
    sys.exit(main())
