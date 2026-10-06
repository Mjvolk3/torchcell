---
id: eu2qttqb3268c2dk53gzzm8
title: Test_head_ontology
desc: ''
updated: 1791270321262
created: 1791270321262
---

## 2026.10.06 - Phase 21 lane D tests

Phase 21 (lane D): `build_mirror_record` with and without the re-check, and the `record` and `verify` CLI on a tmp ontology file. The re-check runs the real `check_source` and `direct_url` retriever over an `httpx.MockTransport` serving either the stored bytes (`matches: true`) or drifted bytes (`matches: false`, stored sha256 unchanged), with `datetime` frozen at 2026-10-06T12:00:00Z. The written record is the model JSON (indent 2, trailing newline) and the printed lines are exact.
