---
id: pfudht5sknxixy2o6wqmvzz
title: Cron_run
desc: ''
updated: 1794961615619
created: 1794961615619
---

## 2026.11.17 - Recording each cron run

`bash scripts/cron_run.sh <job-name> -- <command...>` runs one recurring job so that
[[scripts.jobs]] (`make jobs`) can say whether its last run succeeded.

- Output: the command's stdout and stderr are appended to
  `${TORCHCELL_CRON_LOG_DIR:-/tmp}/torchcell-<job-name>.log`, the same names the
  crontab used before, between `=== cron_run <job> start ... ===` and
  `=== cron_run <job> end ... exit N ===` marker lines.
- Records, in `${XDG_STATE_HOME:-$HOME/.local/state}/torchcell/cron/`:
  `<job>.json` is the latest run, written at start with `finished_at` and
  `exit_code` null (so a run in progress reads as running) and rewritten at the end;
  `<job>.history.jsonl` gains one line per finished run and keeps the newest
  `TORCHCELL_CRON_HISTORY_MAX` (500) lines. Fields: `job`, `started_at`,
  `finished_at`, `exit_code`, `host`, `log`.
- Writes go to a temp file in the same directory and are renamed into place under a
  per-job `flock` (`<job>.lock`), so a reader never sees half a record.
- The job name must match `[a-z0-9][a-z0-9-]*`; it is the registry key in
  `CRON_JOBS`. A log path with a quote or backslash is refused, since the JSON is
  built with `printf`.
- Exit code: the command's own, so cron mail and callers see what they saw before.
