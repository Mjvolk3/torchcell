---
id: 3s4y71fn440h2t8clq0i1a5
title: Test_server
desc: ''
updated: 1790762206131
created: 1790762206131
---

## 2026.09.30 - Phase 13: the keyed endpoint rewritten against a hand-built mirror

Ten to thirty-five tests (50 cases), 66 to 100 percent; the file was rewritten and every original test replaced by a stricter one (path traversal now requires 400). Fixture: three citation keys (a manifest, no manifest, a corrupt manifest), a `_bib` store with one present and one missing file, `_sync_reports`, a plain root file. Pinned: the exact `/health` response; 401 with the exact message on all seven keyed routes for no, empty and wrong keys, both named keys accepted; keys from the file and the inline form with the file winning; the exact list of eight routes; `/keys` hiding service directories and seeing a new directory without restart; `/files` verbatim from the manifest or listed live without sha256; exact 404 messages; traversal refused with 400 in two encodings and through a symlink; `X-Artifact-SHA256` equal to the manifest digest, absent for an unlisted file, and the manifest's value rather than a fresh hash after the file changes; 206 range slices with `content-range`; the 404-versus-500 contract (a missing file under a corrupt manifest is a clean 404, `/files`, `/manifest` and a present file are 500 with a pydantic `ValidationError` behind them); search case-insensitive with the result cap; `/bib` errors; `create_app_from_env` with `load_dotenv` patched out so the repo `.env` cannot inject keys; `--gen-key` printing five lines with the JSON line holding the sha256 of the printed key; `main` passing host and port to a faked `uvicorn.run`.

Findings: underscore directories are hidden from `/keys` but still answer as citation keys (lines 152-158); search reports `truncated: true` when the hit count equals the cap (323-326); `--port 0` is ignored (381).
