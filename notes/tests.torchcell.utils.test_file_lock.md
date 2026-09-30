---
id: tm46vihww5x0rvs8wfk7ljz
title: Test_file_lock
desc: ''
updated: 1790765153872
created: 1790765153872
---

## 2026.09.30 - Phase 14: contention, timeouts, stale locks, release on exception

Seven to nineteen tests (21 cases), 81 to 100 percent. A held lock blocks a second caller until release; the timeout path for read, write and update with the exact `Timeout` text and log line, elapsed measured on a monotonic clock within [0.2, 1.5) s; a stale lock file does not block; release on an exception. Nothing is asserted about the lock file after release because CI installs `filelock>=3.20.1`, which deletes it.

Findings: `timeout=0` becomes 60 s (`timeout or default`); `retry_delay` is never passed to filelock; writes stage through `<stem>.tmp`, so `a.json` and `a.yaml` share `a.tmp` under two different locks.
