---
id: bjorukn1hej7qxkraafx350
title: Test_build_manifest
desc: ''
updated: 1790759650606
created: 1790759650606
---

## 2026.09.30 - Phase 12: every field, fresh and stale verdicts, the CLI

Six to eighteen tests, 74 to 100 percent. Every manifest field; a Media edit drifts Media alone; a change outside a loader's closure leaves it fresh; the scan counts only directories with `processed/lmdb` and reports the name inside the manifest; a malformed manifest is refused; `_git_info` on a throwaway repo (clean, dirty) and outside a repo; the CLI's exact output with exit 0 or 1, the `$DATA_ROOT` default, `KeyError('DATA_ROOT')`.

Finding: the CLI exits 0 when the only problem is stores with no manifest (line 281).
