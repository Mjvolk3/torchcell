---
id: dsgy8ksqtzu3fg4n0u353x7
title: Test_db_connection
desc: ''
updated: 1791270280852
created: 1791270280852
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): the per-thread connection manager with a recording stand-in database class. Pinned: lazy creation, reuse, reopen after `close_connection`, the no-op close, the missing-file refusal, one connection per thread, `__getstate__` / `__setstate__`, and a pickle round trip that carries configuration but no connection.

Finding: `__reduce__` returns `db_kwargs` as the third tuple element, which pickle hands to `__setstate__` as STATE, so after a round trip `db_kwargs` is `{}`, each keyword becomes an attribute of the manager, and the worker's connection opens without it (db_connection.py:116-120). Latent: live callers pass no keyword.

Left uncovered: the `DatabaseProtocol` stub bodies (`...`).
