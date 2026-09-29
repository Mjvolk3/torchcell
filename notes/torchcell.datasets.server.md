---
id: fdtfk11kvgw97ycv3yuc5jb
title: Server
desc: ''
updated: 1790716953987
created: 1790716953987
---

## 2026.09.29 - tc-data: the dataset download endpoint

`torchcell/datasets/server.py` is the FastAPI app `torchcell dataset endpoint` (console script `tc-data-server`, port 8724), built on the `tc-lit` pattern in `torchcell/literature/server.py`: named API keys stored as sha256 hashes (`torchcell/api_keys.py`, shared by both servers), `X-API-Key` compared constant-time, `/health` without a key, Swagger at `/docs`. Routes:

| Route | Returns |
|---|---|
| `GET /health` | status, `n_artifacts`, `n_raw_keys`, the two served roots (no key) |
| `GET /datasets` | the store's `index.json` as an `ArtifactIndex` |
| `GET /datasets/{slug}` | that slug's rows, 404 when none |
| `GET /datasets/{slug}/{archive}` | `FileResponse` with `X-Artifact-SHA256` from the index row, `Accept-Ranges: bytes`, 206 + `Content-Range` on a `Range` header |
| `GET /raw` | citation keys in the raw mirror (underscore dirs excluded) |
| `GET /raw/{ck}/manifest`, `/files`, `/artifact/{path}` | the key's `manifest.json` (literature `Manifest` shape), its file list, one listed file with `X-Artifact-SHA256` |

Config from `TC_DATA_ROOT`, `TC_DATA_RAW_ROOT` (default `$DATA_ROOT/torchcell-raw`), `TC_DATA_HOST`, `TC_DATA_PORT`, `TC_DATA_KEYS_FILE` or `TC_DATA_API_KEYS`; `--gen-key NAME` mints a key and prints the keys-file line.

Decisions:

- The index and the manifest are authoritative. An archive on disk that `index.json` does not list is 404, and a raw file the manifest does not list is 404, so a traversal path never reaches the filesystem (the hermetic tests send `%2e%2e%2findex.json` and `../../etc/passwd` and get 404 with no leak). A key directory without `manifest.json` is 404, not a live listing; a corrupt manifest surfaces as 500, the same honest behavior as `tc-lit`.
- `Range` is served by Starlette 1.3.1's `FileResponse`, which parses the header, answers 206 with `Content-Range`, and 416 past the end; the server adds nothing beyond the sha256 header.
- The index is re-read on every request, so a new packaging on the host shows up with no restart.
- `Dockerfile.tc-data` is `python:3.13-slim` with fastapi, uvicorn, pydantic and python-dotenv; it copies the package and blanks `torchcell/datasets/__init__.py` and `torchcell/literature/__init__.py`, whose eager imports pull torch and pyzotero. Verified by copying the package into a scratch tree with those two files emptied and importing `torchcell.datasets.server`: none of torch, torch_geometric, numpy, pandas, lmdb or pyzotero load. Deployment recipe: [[database.tc-data-endpoint]].
- Tests live in `tests/torchcell/datasets/test_datasets_server.py` (not `test_server.py`, which `tests/torchcell/literature/` already takes under pytest's prepend import mode); the pairing is registered in `[tool.torchcell.test_exceptions.pairs]`.
