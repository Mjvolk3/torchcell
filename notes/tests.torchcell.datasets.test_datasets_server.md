---
id: uh26rgflg0tfhvla1fk7sa3
title: Test_datasets_server
desc: ''
updated: 1790769092007
created: 1790769092007
---

## 2026.09.30 - Phase 15: ranges, header sources, the error contract, the CLI

Fifteen to twenty-six tests, 86 to 100 percent. A 206 slice on a raw file still carries the whole-file hash; suffix ranges and 416 on an archive; `X-Artifact-SHA256` comes from the index or the manifest and the server never hashes bytes; a listed file missing on disk is 404, a manifest entry escaping its key directory is 400, `..` as a citation key is 404; a corrupt manifest is 500 on all three routes and a corrupt `index.json` is 500 even on `/health`, while a missing manifest stays a clean 404; `from_env` refuses a missing raw mirror; `create_app_from_env`; `--gen-key`; `main` host and port overrides with `--port 0` falling back to the config.

## 2026.09.30 - --port 0 Finding retired (issue #534)

`test_main_runs_uvicorn_with_config_or_override_host_and_port` now asserts the parsed ports `[None, 9200, 0]` and that `--port 0` reaches `uvicorn.run` and the log line as `0.0.0.0:0`.

## 2026.09.30 - Underscore key hole closed (issue #564)

New `test_underscore_directory_is_an_unknown_key_even_with_a_manifest`: a `_sync_reports` directory holding a valid manifest and its listed file is absent from `/raw` and answers 404 `unknown citation key` on `/manifest`, `/files` and `/artifact/data/table.csv`, while the same manifest under `fakeKey2020` still serves the file.
