# Downloading datasets

Built torchcell datasets are served as packaged archives by `tc-data`, a keyed HTTP
endpoint with Swagger documentation. Each archive holds a dataset's `processed/`
directory (the records LMDB and its `interned` sibling) and `preprocess/` directory
(`build_manifest.json`, `gene_set.json`, `experiment_reference_index.json`), exactly
as the loader wrote them, so a download replaces the build step. The raw files each
loader consumed are served from the same endpoint, hash-pinned by their `manifest.json`.

The endpoint runs on the database host, `torchcell-database.ncsa.illinois.edu`, port
8724, serving the archive store and the raw mirror from the Taiga project storage. The
port is not yet open to the internet, so until it is, reach it through an ssh tunnel to
that host and use `http://127.0.0.1:8724` as the URL:

```bash
ssh -N -L 8724:127.0.0.1:8724 rocky@torchcell-database.ncsa.illinois.edu
```

Ask the maintainers for a named API key; every request except `/health` carries the
key in the `X-API-Key` header. The store currently holds `amino_acid_cooper2010`,
`amino_acid_mulleder2016` and `betaxanthin_cachera2023`; `GET /datasets` is the
authoritative list.

## Swagger

The interactive documentation lives at `<TC_DATA_URL>/docs` and the OpenAPI schema at
`<TC_DATA_URL>/openapi.json`. The routes:

| Route | Returns |
|---|---|
| `GET /health` | liveness, artifact and raw-key counts (no key) |
| `GET /datasets` | the artifact index: one row per packaged archive |
| `GET /datasets/{slug}` | every packaged version of one dataset |
| `GET /datasets/{slug}/{archive}` | the archive; `X-Artifact-SHA256`, `Accept-Ranges: bytes` |
| `GET /raw` | citation keys in the raw mirror |
| `GET /raw/{citation_key}/manifest` | the key's `manifest.json` |
| `GET /raw/{citation_key}/files` | the files that manifest lists, with sha256 |
| `GET /raw/{citation_key}/artifact/{path}` | one raw file; `X-Artifact-SHA256` |

An index row records the dataset `slug`, the loader `dataset_class`, the
`torchcell_version` and git commit the LMDB was built with, the knowledge-graph
release it was admitted to, `content_sha256` (over the experiment ids, comparable with
the release manifest), the archive name, `archive_sha256`, `archive_bytes`, and
`status` (`supported` or `deprecated`). Pick the newest `supported` row whose
`torchcell_version` shares your installed `major.minor`.

## curl

```bash
export TC_DATA_URL=http://127.0.0.1:8724   # through the tunnel above, or the host once port 8724 is open
export TC_DATA_API_KEY=<your key>

curl -H "X-API-Key: $TC_DATA_API_KEY" "$TC_DATA_URL/datasets" | python -m json.tool
curl -H "X-API-Key: $TC_DATA_API_KEY" "$TC_DATA_URL/datasets/smf_costanzo2016"

# download an archive, resuming if interrupted, then verify it
ARCHIVE=smf_costanzo2016-1.2.1-d8a0f06c.tar.xz
curl -C - -H "X-API-Key: $TC_DATA_API_KEY" \
  -o "$ARCHIVE" "$TC_DATA_URL/datasets/smf_costanzo2016/$ARCHIVE"
sha256sum "$ARCHIVE"   # must equal the row's archive_sha256

# unpack into a dataset root; processed/ and preprocess/ appear beside it
mkdir -p data/torchcell/smf_costanzo2016
tar -xf "$ARCHIVE" -C data/torchcell/smf_costanzo2016

# raw files
curl -H "X-API-Key: $TC_DATA_API_KEY" "$TC_DATA_URL/raw"
curl -H "X-API-Key: $TC_DATA_API_KEY" "$TC_DATA_URL/raw/<citation_key>/files"
```

## Environment variables

| Variable | Where | Meaning |
|---|---|---|
| `TC_DATA_URL` | client | endpoint base URL; when set, loaders download instead of building |
| `TC_DATA_API_KEY` | client | the plaintext key for the `X-API-Key` header |
| `TC_DATA_ROOT` | server | the artifact store (`index.json`, `SHA256SUMS`, `<slug>/<archive>`) |
| `TC_DATA_RAW_ROOT` | server | the raw mirror; defaults to `$DATA_ROOT/torchcell-raw` |
| `TC_DATA_KEYS_FILE` | server | JSON `{name: sha256hex}` of accepted keys (preferred) |
| `TC_DATA_API_KEYS` | server | inline `name:key,name2:key2` (quick start only) |
| `TC_DATA_HOST`, `TC_DATA_PORT` | server | bind address and port (default `0.0.0.0:8724`) |

## From Python

With `TC_DATA_URL` and `TC_DATA_API_KEY` set, any `ExperimentDataset` loader whose
LMDB is absent fetches, verifies and unpacks its artifact instead of downloading the
publisher files and running `process()`. If the endpoint has no artifact compatible
with the installed version the loader raises and names the slug; it never falls back
to the publisher path while the variable is set.

```python
from torchcell.datasets.client import DatasetClient, unpack_artifact

client = DatasetClient.from_env()          # TC_DATA_URL, TC_DATA_API_KEY
artifact = client.select("smf_costanzo2016")  # newest supported row on this major.minor
if artifact is None:
    raise SystemExit("nothing compatible with the installed torchcell")
archive = client.download(artifact, dest=f"artifacts/{artifact.archive}")  # verifies sha256
unpack_artifact(archive, "data/torchcell/smf_costanzo2016")
```

The raw mirror keys and files are one `GET` away with the same header; see the Swagger
page for the exact shapes.
