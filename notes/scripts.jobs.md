---
id: ucngtdgz8wguillxwlaqo4j
title: Jobs
desc: ''
updated: 1794961607349
created: 1794961607349
---

## 2026.11.17 - The jobs panel

`make jobs` (`scripts/jobs.py`) is the terminal panel for recurring and continuous
jobs, beside [[scripts.ops]]: `make ops` says whether the served stores and services
answer, `make jobs` says whether the scheduled work ran and what GitHub holds open.
`make jobs-fast` (`--no-network`) prints only the local sections. Read-only, exit 0.

Sections:

1. **Recurring jobs (cron).** One row per `CRON_JOBS` entry: name, schedule, last
   start, duration, exit code, age against the expected period, verdict. The records
   come from [[scripts.cron_run]] under `${XDG_STATE_HOME:-~/.local/state}/torchcell/cron/`;
   the merge-queue drainer runs every 2 min and is judged by its heartbeat
   (`merge_queue.py loop-status --json`) instead of a record per pass. Verdicts, in
   order: `never ran` (no record; `(wrapper not installed yet)` when the live crontab
   does not call `cron_run.sh <job> --`), `running` (unfinished and younger than 1.5
   periods), `FAILED` (last exit nonzero, with its `log:` path on the next line),
   `STALE` (no successful finish within 1.5 periods, which also catches a run stuck
   in `running`), else `ok`.
2. **Continuous jobs.** `squeue -u $USER`: running and pending counts, pending by
   reason, the five longest-running jobs. `docker ps -a`: running containers as a
   table (a cross on `unhealthy`), stopped ones on one line. Services point to `make ops`.
3. **GitHub Actions.** Per workflow, the newest run on `main` among the newest 200
   main runs, with the newest completed run's conclusion when the newest is still in
   progress. A failed workflow's run URL goes on its own line.
4. **Pull requests.** Open count and stale count, then a table newest update first
   (so the oldest-updated is last): number, age in days, last update, CI rollup
   (`failure` over `pending` over `success`, `none` with no checks), flags (`draft`,
   `stale` after 14 days without an update), branch, title.
5. **Issues.** Open count, count with no activity in 30 days, count by label,
   the `priority:high` ones, the 10 most recently updated.

Three `gh` calls (`gh run list`, `gh pr list`, `gh issue list --limit 500`), each under
`--gh-timeout` (20 s). Every external command goes through `run_cmd`, the one place a
timeout, a missing binary or a nonzero exit becomes a one-line "unreachable" for its
section.

### The registry is the single source of truth

`CRON_JOBS` holds name, schedule, command, and whether the job is judged by wrapper
records or a heartbeat. `scripts/crontab.txt` is checked against it by
[[tests.scripts.test_jobs]] in both directions: each wrapped line must be a registry
job with the same schedule and command, and each registry job must have exactly one
line. The period comes from the schedule (`cron_period`), which refuses a shape it
does not know, so a new schedule shape adds its rule rather than being judged against
a wrong period.

Adding a job (the weekly `deps-weekly`, Sunday, is the next): add a `CronJob` to
`CRON_JOBS`, add its line to `scripts/crontab.txt` as
`bash scripts/cron_run.sh <name> -- <command>`, install the crontab.

### Research before building

No single service found covers cron, slurm and GitHub issues, PRs and workflows in one
terminal view. Cron monitors (healthchecks.io, Cronitor, Dead Man's Snitch, Uptime
Kuma push monitors) are dead-man's switches with a web UI and alerting; `chronic`
only silences successful runs; `gh status` and `gh-dash` cover GitHub only. A
healthchecks.io-style ping (`runitor` wraps a command and pings on exit) is a later
add-on for alerting, not a replacement for this panel.

### Drift found while building

The live crontab already chained `scripts/lit_annotations.py` after `lit_sync.py` on
the 03:30 line, and `scripts/crontab.txt` did not list it. It is now its own registry
job, `lit-annotations`, wrapped on the same line after `lit-sync`.
