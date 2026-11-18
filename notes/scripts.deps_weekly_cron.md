---
id: s8um2vyywse679sb5y7fg5u
title: Deps_weekly_cron
desc: ''
updated: 1794963685211
created: 1794963685211
---

## 2026.11.17 - Cron entry of the weekly dependency job

Runs [[scripts.deps_weekly]] every Sunday at 05:00. The crontab line (in
`scripts/crontab.txt`) does not use the primary checkout's files, because that checkout
can sit behind `origin/main`:

```cron
0 5 * * 0 bash -lc '( git -C /home/michaelvolk/Documents/projects/torchcell fetch -q origin && git -C /home/michaelvolk/Documents/projects/torchcell show origin/main:scripts/deps_weekly_cron.sh | bash -s -- /home/michaelvolk/Documents/projects/torchcell ) >> /tmp/torchcell-deps-weekly.log 2>&1'
```

The script adds a detached worktree of `origin/main` at
`/scratch/tmp/torchcell-deps-weekly/wt-<YYYY.MM.DD>`, runs
`scripts/deps_weekly.py run --land` in it, and then:

- removes the worktree when it is still detached (no change, a red suite, or a failure
  before the commit made no branch);
- keeps it when the job put it on `deps/weekly-<date>`: an enqueued branch is landed and
  removed by the merge-queue drainer, and a PR left open (checks failed, timed out, loop
  not live) needs the worktree to be landed later by hand.

When `scripts/cron_run.sh` (PR #1168) exists in the fresh worktree, the job runs through
it as job `deps-weekly`, so `make jobs` sees the run; until then it runs directly. Both
log to `/tmp/torchcell-deps-weekly.log`.

The whole body is one `main` function called with `< /dev/null`: bash reads the script
from the crontab's pipe, so the function makes it parse the whole file first, and
`/dev/null` stops a child process from reading the rest of the script as input.
