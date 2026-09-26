---
id: rpe2mu2yeclah8ym3ifsr1z
title: Test_merge_queue
desc: ''
updated: 1790416734126
created: 1790416734126
---

## 2026.09.26 - The queue CLI as a subprocess against a tmp database

`scripts/merge_queue.py` runs as a subprocess with `--db` under `tmp_path` (so `$DATA_ROOT` never enters), `PATH=/usr/bin:/bin` (no `gh`), `HOME` in `tmp_path` and the global git config at `/dev/null`. Every assertion is an exact exit code, an exact stdout line or an exact row computed from the command sequence: `add` is idempotent for an active branch, `ls`/`ls --json` print the position-ordered rows (`ls` right-aligns the position to width 3, so the helper strips only newlines from stdout), `mv` and `rm` reorder and remove, `block --reason` and `requeue` move a row through its states, `watch` classifies a row (landed / blocked needs-human / conflict / queued / absent, exit codes 0, 2, 3, 4, 5) and `banner` renders the matching stoplight. The pure helpers (`classify_watch`, the foreign-commit scan, the banner text) are imported and called directly. `foreign` runs against a bare origin plus a clone: exit 0 for a branch carrying only its own commits, 1 when a stale-`main` commit is a strippable base prefix (the printed `rebase --onto` command is asserted), 2 when a foreign commit sits between the branch's own. Phase 4 of [[plan.test-suite-buildout.2026.09.25]]; the drainer is covered in [[tests.torchcell.scripts.test_drain_merge_queue]].
