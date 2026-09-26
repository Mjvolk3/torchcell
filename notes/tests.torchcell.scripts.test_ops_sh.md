---
id: 5v552ydm4t4ooig437c8bhr
title: Test_ops_sh
desc: ''
updated: 1790416764010
created: 1790416764010
---

## 2026.09.26 - ops.sh dispatch only

`status`, `releases` and `health` probe the served Neo4j stores, tc-lit, slurm and the disks, so they belong to a `--network` run and are not tested here. The hermetic contract is the dispatch: an unknown action exits 2 and prints `usage: <script> {status|releases|health}` to stderr with nothing on stdout. The script sources `$REPO_ROOT/.env` before dispatching, so the test runs with the repository root as its working directory. Phase 4 of [[plan.test-suite-buildout.2026.09.25]].
