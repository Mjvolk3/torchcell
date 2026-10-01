---
id: 1oduj42sja944dis39zollb
title: File_lock
desc: ''
updated: 1790872934821
created: 1790872934821
---

## 2026.10.01 - Zero timeout, retry delay, unique staging path (issue #529)

Previous behavior, pinned as Findings by Phase 14 of the test campaign:

- every method resolved `timeout = timeout or cls.default_timeout`, so `timeout=0` (one non-blocking attempt in `filelock`) became the 60 s default;
- `retry_delay` was resolved and never passed to `filelock`, which polled at its own 0.05 s;
- writes staged through `file_path.with_suffix(".tmp")`, so `a.json` and `a.yaml` shared `a.tmp` while holding two different locks (`a.json.lock`, `a.yaml.lock`).

Fix: `_resolve_timeout` and `_resolve_retry_delay` apply the class defaults only to `None`; `retry_delay` goes to `FileLock.acquire` as `poll_interval` (filelock 3.20.0 signature `acquire(timeout=None, poll_interval=0.05, ...)`); `_get_temp_path` appends `.tmp` to the full name (`a.json.tmp`).

Callers checked (`grep -rn "FileLockHelper"`): `torchcell/data/neo4j_cell.py` passes `timeout=60.0` and no `retry_delay`; `torchcell/data/neo4j_preprocessed_cell.py` passes neither. No caller passes `timeout=0` or `retry_delay`, so none changes behavior; nothing globs `*.tmp`.

Evidence: `test_zero_timeout_is_kept_and_only_none_takes_the_default`, `test_zero_timeout_fails_at_once_on_a_held_lock`, `test_retry_delay_is_the_polling_interval`, `test_every_method_forwards_timeout_and_poll_interval`, `test_staging_file_is_the_full_name_tmp_and_unique_per_target` in [[tests.torchcell.utils.test_file_lock]].
