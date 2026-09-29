---
id: 0pnv6peuve8sn90ccpumo3l
title: Client
desc: ''
updated: 1790716961469
created: 1790716961469
---

## 2026.09.29 - DatasetClient: select, download, verify, unpack

`torchcell/datasets/client.py` is the library side of `tc-data`. `DatasetClient(url, api_key)` (or `from_env()` reading `TC_DATA_URL` and `TC_DATA_API_KEY`) wraps an `httpx.Client`; `index()`, `artifacts(slug)` and `select(slug, torchcell_version=torchcell.__version__)` mirror the routes, with `select` applying `ArtifactIndex.select` (newest `supported` row on the installed `major.minor`, None otherwise).

`download(artifact, dest, verify=True)` streams to `<dest>.part`; when a partial file exists it sends `Range: bytes=<size>-` and requires a 206 (a 200 to a ranged request raises rather than silently restarting), appends, then hashes the whole file against `artifact.archive_sha256`; the server's `X-Artifact-SHA256` must also agree with the row. A mismatch unlinks the partial file and raises `ArtifactIntegrityError`, so a later retry starts clean instead of resuming from garbage. Only a verified file is renamed to `dest`.

`unpack_artifact(archive, dataset_root)` extracts with `tarfile`'s `filter="data"`, which rejects absolute members, `..` components and links outside the root (the test proves `../escape.txt` raises `OutsideDestinationError`).

Testability: the constructor takes `http=` so a `fastapi.testclient.TestClient` (an `httpx.Client` subclass) stands in for the network; the hermetic tests drive the real app this way, including the resume path (a spy on `stream` sees exactly one `Range: bytes=1000-` request).
