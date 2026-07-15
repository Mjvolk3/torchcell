---
id: inrpfruw6n460k5my9tc9bv
title: Literature Keyed Endpoint + Manifest Backfill
desc: ''
updated: 1784088918308
created: 1784088918308
---

## 2026.07.14 - Plan: keyed endpoint for the OCR literature mirror + manifest backfill

Related: [[paper.literature-ocr-ingestion]] · [[literature-provenance-subsystem]] · issue #20.
Worktree: `feat/literature-keyed-endpoint`. Land via `/enqueue-merge` (rebase + ff), never `gh pr merge`.

### Goal

Two coupled deliverables, both under `torchcell/literature/`:

1. **Backfill** `manifest.json` for the ~18 OCR'd citation keys that lack one, so the
   mirror is 38/38 provenance-indexed (sha256 on every byte).
2. **Serve** the mirror over a **key-authenticated read-only HTTP endpoint** on GilaHyper
   so the M1 Mac (and collaborators) can list + pull artifacts on demand, with per-file
   sha256 exposed for client-side integrity. This realizes the `radiant_endpoint`
   `RetrievalMethod` slot (issue #20).

Mirror root: `$DATA_ROOT/torchcell-library/` (`DATA_ROOT=/scratch/projects/torchcell-scratch`),
~5.4 GB, 38 keys. **The mirror is NOT in git** — backfill writes `manifest.json` into the
live data dir; that is the product, verified by count, not a repo diff.

### Key decisions (from 3-scout deliberation)

- **Host = GilaHyper**, where the bytes physically live. Issue #20 says "Radiant VM" but
  the enum value names the *method*, not the machine; serving from where the 5.4 GB
  already sits avoids a copy. Document this choice in the note.
- **FastAPI + uvicorn** (NOT stdlib). Neither is installed today; add to
  `env/requirements.txt` and `pip install` into the `torchcell` env. FastAPI reuses the
  `Manifest` pydantic models directly and `httpx` (already present, 0.28.1) gives
  `TestClient` for free. Both are typed (`py.typed`) → no new mypy overrides.
- **Config = frozen pydantic `BaseModel.from_env()`**, copying the `ZoteroConfig` idiom
  (`zotero.py:43`). API keys as `SecretStr`. **No `pydantic-settings`/`BaseSettings`** —
  not installed, not the repo idiom.
- **Backfill uses the existing `build_manifest`** (`manifest.py:181`), which by design
  leaves `retrieval`/`processing` = null on every file (matches all existing manifests —
  only SI-data files ever carry `retrieval`, via the separate `retrieve.py` path). No
  parallel schema.
- **No a-priori DOI needed.** The repo has no citation_key→DOI table (`.bib` files don't
  cover these keys). Match each dir name against `zotero._resolve_citation_key` over one
  `everything(items())` scan to enrich (doi/title/library_id/zotero_item_key/collections
  - per-attachment md5/source). Keys not in Zotero → offline manifest, provenance null.
- **Honest marker, not fabrication.** Add `provenance_complete: bool = True` to `Manifest`;
  backfill sets it `False` when it could not source retrieval provenance (offline / not in
  Zotero). Never invent a source_url, sha256 origin, or MinerU version.
- **Fix a real `_role_for` gap** (see below) so data-only keys classify correctly instead
  of falling to `"other"`.

### `_role_for` correction (`manifest.py:161`)

Today `_role_for` returns `"other"` for `data/*`, `thesis.*`, and loose non-pdf/md under
`si/` — yet existing captured manifests already use a `raw_data` role for `data/` files
(assigned by the SI-data path, not `_role_for`). Align `_role_for` to that reality:

- Add role constant `ROLE_RAW_DATA = "raw_data"` (currently referenced in data but not a
  constant).
- `rel_path.startswith("data/")` → `raw_data` (xue2025, lopez, messner data/, ...).
- loose `si/<file>` that is not `.pdf`/`.md` and not under `si/si_data/` → `si_data`
  (hoepfner `si/Table_S5.xls`).
- top-level (no `/`) `*.pdf` → `paper_pdf`; top-level `*.txt`/`*.md` born-digital extract
  → `paper_ocr` (lopez `thesis.pdf`/`thesis.txt`).

Pin every branch with unit tests so the shared capture path can't silently drift. Existing
captured dirs are NOT re-run, so no risk to landed manifests.

### File-by-file plan

**A. `torchcell/literature/manifest.py`** (MODIFY)

- Add `ROLE_RAW_DATA` constant; extend `_role_for` per above.
- Add `provenance_complete: bool = True` to `Manifest`.
- No change to `build_manifest`/`write_manifest` signatures.

**B. `torchcell/literature/backfill.py`** (NEW) — reusable, idempotent regularizer.

- `backfill_key(artifact_dir, *, lib=None) -> Manifest`: builds a manifest for one key;
  if `lib` given, look up the Zotero item by citation key (dir name) via a passed-in
  resolved index and enrich; else offline with `provenance_complete=False`. Skip if
  `manifest.json` exists unless `force`.
- `backfill_mirror(root, *, use_zotero=True, force=False) -> BackfillReport`: iterate
  dirs lacking a manifest (dynamic scan — do not hardcode the 18), build one Zotero
  `everything(items())` index once (only if `use_zotero`), regularize each, return a
  pydantic `BackfillReport` (per-key: enriched|offline|skipped, file count, null-field
  list). Fail loud on real errors; no per-key try/except swallow.
- `if __name__ == "__main__"`: `load_dotenv()`, argparse `--root`, `--force`,
  `--no-zotero`, `--dry-run`; print the report. Run from repo root with the env python.

**C. `torchcell/literature/server.py`** (NEW) — FastAPI app.

- `LiteratureServerConfig(BaseModel, frozen=True)`: `mirror_root: Path`, `keys: RadiantKeys`,
  `host: str`, `port: int`; `from_env()` reads `DATA_ROOT` (→ `<root>/torchcell-library`),
  `RADIANT_KEYS_FILE` (or `RADIANT_API_KEYS`), `RADIANT_HOST` (default `0.0.0.0`),
  `RADIANT_PORT` (default `8723`).
- `RadiantKeys`: named keys loaded from a JSON keys file `{name: sha256hex}` (store
  **hashes**, not plaintext). `verify(presented: str) -> str|None` returns the key name
  via `hmac.compare_digest` over `sha256(presented)` against each stored hash
  (constant-time; never log the presented value).
- Auth dependency: read `X-API-Key` header → `RadiantKeys.verify`; 401 if missing/unknown.
- `_safe_artifact_path(citation_key, rel_path)`: join under mirror root, `.resolve()`,
  enforce `is_relative_to(mirror_root)` → 404/400 on traversal. No helper exists in repo;
  write + test it.
- Endpoints (all behind auth except `/health`):
  - `GET /health` → `{status:"ok", n_keys, mirror_root}` (no auth).
  - `GET /keys` → list of citation keys present (dynamic `iterdir`, cheap, reflects new
    papers with no restart).
  - `GET /keys/{ck}/manifest` → the `Manifest` (pydantic response; 404 if none — signals
    "not yet backfilled").
  - `GET /keys/{ck}/files` → per-file `{path, role, bytes, sha256}` (from manifest, or
    computed live if no manifest).
  - `GET /keys/{ck}/artifact/{rel_path:path}` → `FileResponse`/stream with correct
    `media_type` and header `X-Artifact-SHA256: <hex>` (from manifest when present) so the
    client can verify before writing. Path-guarded.
  - `GET /search?q=...` → filename + `paper.md` substring/regex grep across keys
    (convenience only; **semantic search explicitly deferred**). Bounded result count,
    `log()` if truncated.
- `main()`: `load_dotenv()`, build config, `uvicorn.run(app, host, port)`. Module exposes
  `app` for `uvicorn torchcell.literature.server:app`.
- Never log key values. Fail loud if `RADIANT_KEYS_FILE` missing (no unauth fallback).

**D. `torchcell/literature/retrieve.py`** (MODIFY) — close the provenance loop.

- Add one-liner `radiant_endpoint(url: str) -> bytes: return _get(url)` and register it in
  `RETRIEVERS` (`"torchcell.literature.retrieve.radiant_endpoint"`). Bytes are hash-verified
  against the manifest by the existing `provenance.check_source`/`verify_artifact`; the key
  travels in an httpx header set by `_get`'s caller, NOT in `RetrievalRecord.params`. (A
  full client-side `radiant_endpoint` retriever that injects the header can come later; the
  registered fn + enum value make the slot real now. Keep `_get` signature untouched or add
  an optional `headers` kwarg — decide during impl, minimal change preferred.)

**E. `torchcell/literature/__init__.py`** (MODIFY) — re-export `backfill_mirror`,
`LiteratureServerConfig`, `app` factory as appropriate; keep `__all__` tidy.

**F. `pyproject.toml` / `env/requirements.txt`** (MODIFY)

- Add `fastapi` + `uvicorn[standard]` to `env/requirements.txt` (the dynamic dependency
  source). Optionally add a `[project.scripts]` entry `tc-lit-server =
  "torchcell.literature.server:main"`.
- `pip install fastapi "uvicorn[standard]"` into `~/miniconda3/envs/torchcell`.

**G. `.claude/skills/radiant-pull/SKILL.md`** (NEW, optional-but-cheap) — client-pull skill
stub for the Mac: set `RADIANT_URL` + `RADIANT_API_KEY`, `curl -H "X-API-Key: $RADIANT_API_KEY"`
to list keys and pull an artifact, verify `X-Artifact-SHA256`. Minimal SKILL.md (frontmatter
`name`+`description`, H1, step body — matches `setup-worktree`).

**H. Tests** `tests/torchcell/literature/` (NEW files)

- `test_backfill.py`: fixture `tmp_path` mirror with 2 fake keys (one paper-shaped, one
  data-only) → `backfill_mirror(use_zotero=False)` → each `manifest.json` round-trips
  (`Manifest(**json)`), every `ArtifactRecord.sha256` matches `sha256_file` on disk,
  data-only key has `raw_data` roles + `provenance_complete=False`, idempotent (second run
  skips), `force` rewrites. Also unit-test `_role_for` new branches directly.
- `test_server.py`: build app over a `tmp_path` fixture mirror + a keys file; use FastAPI
  `TestClient`. Assert: bad/missing key → 401; good key → 200; `/keys` reflects fixture;
  `/keys/{ck}/manifest` returns the model; artifact download bytes' sha256 == manifest
  sha256 and `X-Artifact-SHA256` header matches; `../` traversal blocked; `/health` no-auth.
  Dependency-inject the config (override) so no `$DATA_ROOT`/Zotero/network dependence.

### `.env` additions (document, do not commit secrets)

```
RADIANT_KEYS_FILE=/home/michaelvolk/.config/torchcell/radiant_keys.json   # {name: sha256hex}
RADIANT_HOST=0.0.0.0
RADIANT_PORT=8723
```

Keys file is created by hand (or a tiny `--add-key NAME` helper in server.py that prints a
fresh random key once and stores only its hash). Mac side sets `RADIANT_URL` +
`RADIANT_API_KEY`.

### Run recipe (GilaHyper) — document in the note, model on `scripts/crontab.txt`

```
bash -lc 'cd /home/michaelvolk/Documents/projects/torchcell && \
  RADIANT_KEYS_FILE=... /home/michaelvolk/miniconda3/envs/torchcell/bin/python \
  -m uvicorn torchcell.literature.server:app --host 0.0.0.0 --port 8723'
```

Local-network exposure note: bind to the LAN interface, key rotation = edit keys file +
restart, no TLS on trusted LAN initially (put behind an SSH tunnel / reverse proxy for
off-LAN). No systemd precedent in repo; a tmux/`bash -lc` recipe is the documented minimum.

### Verification gate (must actually run, report real output)

1. `_role_for` + backfill unit tests: `~/miniconda3/envs/torchcell/bin/python -m pytest tests/torchcell/literature/test_backfill.py tests/torchcell/literature/test_server.py -xvs`.
2. **Real backfill**: run `backfill_mirror` over `$DATA_ROOT/torchcell-library` (Zotero if
   creds present, else `--no-zotero`). Confirm manifest count rises to **38/38** (or
   document any key that legitimately can't — e.g. genuinely empty dir — and why). Spot-check
   one new manifest round-trips and its sha256 matches disk.
3. **Live server**: start uvicorn on a test port with a temp keys file; `curl` (a) `/health`,
   (b) `/keys` with a good key (200) and a bad key (401), (c) download `paper.md` for one key
   and verify the returned bytes' sha256 == `X-Artifact-SHA256` == manifest. Capture output.
4. `mypy` (whole-tree local, not just diff — CI is diff-scoped) + `ruff` on new files.

### Notes/docs updates

- Append `## 2026.07.14 - Delivered: keyed endpoint + backfill` to
  `notes/paper.literature-ocr-ingestion.md`: API surface, auth model, env vars, run
  command, coverage now N/38, and convert "Hosting options → keyed endpoint" to
  **delivered** (host = GilaHyper). Preserve frontmatter (append only).
- Weekly note: append a bullet to `notes/user.Mjvolk3.torchcell.tasks.weekly.2026.28.md`
  (current ISO-week file) linking this plan + the delivered note.

### Risks / watch-items

- Installing FastAPI/uvicorn mutates the shared `torchcell` env (all worktrees share it) —
  additive, low risk, but note it in the PR.
- Zotero enrichment needs `ZOTERO_*` creds in `.env`; if absent, backfill still completes
  offline (`provenance_complete=False`) — that's the honest degrade, explicitly logged, not
  a silent fabrication.
- `provenance_complete` field addition changes the `Manifest` schema — existing manifests
  deserialize fine (defaults `True`); the server must not choke on old manifests without it.
- Don't put the API key into `RetrievalRecord.params` (it's persisted in manifests).
