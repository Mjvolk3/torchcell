---
id: 1py1lprf2ou95qfrzypo5i7
title: Test_deps_weekly
desc: ''
updated: 1794963693371
created: 1794963693371
---

## 2026.11.17 - Pure-function tests of the weekly dependency job

Tests of [[scripts.deps_weekly]] with exact expected values and no network or
subprocess: requirement-line parsing (extras, inline comments, rejected forms),
byte-for-byte rendering of every line of the real `env/requirements.txt`, floor raising
(never lower, local labels dropped, bounds kept, pins untouched, `[hold]` skipped,
`torch_geometric` and `GEOparse` matched by PEP 503 name), cap stripping and lifting
with and without a `comment_marker`, the policy table removal, the freeze parser, lock
rendering and diff, the torch pick from fixture PyPI JSON and PyG pages, the commit
subject, the land decision, cap verdicts, the pytest summary line, exit codes, and that
every bounded line of `env/requirements.txt` has a `[caps]` entry in
`env/deps_policy.toml` and the reverse.
