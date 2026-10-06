---
id: ps8a2d0i1695i7rxedn9e04
title: Test_directory_setup
desc: ''
updated: 1791270313609
created: 1791270313609
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): `dotenv.load_dotenv` is replaced before a fresh import (the module calls it at import), then `DATA_ROOT` and `WORKSPACE_DIR` point at `tmp_path`. Pinned: the exact directory tree, `gh_neo4j.conf` copied as `conf/neo4j.conf`, the env file (default or `--env-file`) copied as `database/.env`, `biocypher/` replaced (a stale file is gone), the success line, and a missing env file failing after the tree and conf exist.

### Audit 2 notes applied

- Audit 2: the fixture also restores the `torchcell.database.directory_setup` package attribute after the fresh import.
