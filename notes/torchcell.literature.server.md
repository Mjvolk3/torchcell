---
id: e7s6i7a7yc8prriopvfkq5j
title: Server
desc: ''
updated: 1790814997572
created: 1790814997572
---

## 2026.09.30 - Service directories, truncated flag, --port 0

Issue #525 and the `--port 0` item of #534. Previously `/keys/_sync_reports/files` listed and streamed the sync report and `/keys/_bib/files` answered 500 (its store manifest read as a paper `Manifest`); `truncated` was true when hits exactly equaled the cap; `--port 0` fell back to `TC_LIT_PORT` (`args.port or config.port`).

Fix: `_key_dir` refuses an underscore-prefixed name with the same 404 `unknown citation key` as an absent key (`/bib` is unaffected, it has its own route); search sets `truncated` only when a hit beyond the cap is found and dropped; `--port` is used whenever given (`config.port if args.port is None else args.port`), so 0 binds an ephemeral port. No route path or response schema changed.

Evidence: [[tests.torchcell.literature.test_server]], tests `test_service_directories_answer_as_unknown_citation_keys`, `test_search_is_not_truncated_when_hits_exactly_fill_the_cap`, `test_main_runs_uvicorn_with_config_or_override_host_and_port`.
