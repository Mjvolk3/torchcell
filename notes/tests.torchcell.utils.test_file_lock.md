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

## 2026.10.01 - Findings retired: zero timeout, poll interval, staging path (issue #529)

Retired the three Findings. Now asserted: `with_file_lock(timeout=0).timeout == 0`; a read with `timeout=0` on a held lock raises `Timeout` in under 0.5 s and logs `within 0s`; `retry_delay=1.0` makes a reader whose holder releases at 0.15 s take at least 1.0 s (it is `poll_interval`); every method passes `(0, 0)` for explicit zeros and `(60.0, 0.1)` for `None` to `FileLock.acquire`; staging is `a.json.tmp`, so a blocked `a.json` stage leaves `a.yaml` writable and `a.tmp` is never created. 19 to 21 tests.
