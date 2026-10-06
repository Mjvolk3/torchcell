---
id: g9v6254j2s4k5kbqpzzwef4
title: Test_tcdb
desc: ''
updated: 1791270305940
created: 1791270305940
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): the `tcdb` cliff app registers `build`, `complete` and `help`; `main(["build", "--mode", "regular"])` runs the stubbed build end to end and returns its code (7). Root logger handlers are restored after each test because `App.run` configures logging.
