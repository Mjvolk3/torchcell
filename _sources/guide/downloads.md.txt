# Downloading datasets

Built torchcell datasets are served as packaged archives by `tc-data`, a keyed HTTP
endpoint with Swagger documentation. Each archive holds a dataset's `processed/`
directory (the records LMDB and its `interned` sibling) and `preprocess/` directory
(`build_manifest.json`, `gene_set.json`, `experiment_reference_index.json`), exactly
as the loader wrote them, so a download replaces the build step. The raw files each
loader consumed are served from the same endpoint, hash-pinned by their `manifest.json`,
and so are two more file tiers that graph records point at without storing their bytes:
the genomes tier (reference and isolate assembly sets) and the objects tier (derived
files such as embeddings, matrices and indexed FASTA).

The endpoint is `http://torchcell-database.ncsa.illinois.edu:8724`, on the database
host, serving the archive store and the raw mirror from the Taiga project storage. A
user needs nothing but that URL and a key: no account on the host, no ssh. Ask the
maintainers for a named API key (the Keys section below says how one is issued); every request except `/health`
carries the key in the `X-API-Key` header. The store currently holds `gene_essentiality_sgd`,
`smf_costanzo2016`, `amino_acid_cooper2010`, `amino_acid_mulleder2016` and
`betaxanthin_cachera2023`; `GET /datasets` is the authoritative list.

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
| `GET /genomes` | assembly sets in the genomes tier that carry a manifest |
| `GET /genomes/{assembly_set}/manifest` | the set's genome manifest |
| `GET /genomes/{assembly_set}/files` | the files that manifest lists, with sha256 |
| `GET /genomes/{assembly_set}/artifact/{path}` | one genomes file; `X-Artifact-SHA256`, `Accept-Ranges: bytes` |
| `GET /objects` | object keys in the objects tier that carry a manifest |
| `GET /objects/{object_key}/manifest` | the key's `manifest.json` |
| `GET /objects/{object_key}/files` | the files that manifest lists, with sha256 |
| `GET /objects/{object_key}/artifact/{path}` | one objects file; `X-Artifact-SHA256`, `Accept-Ranges: bytes` |

An index row records the dataset `slug`, the loader `dataset_class`, the
`torchcell_version` and git commit the LMDB was built with, the knowledge-graph
release it was admitted to, `content_sha256` (over the experiment ids, comparable with
the release manifest), the archive name, `archive_sha256`, `archive_bytes`, and
`status` (`supported` or `deprecated`). Pick the newest `supported` row whose
`torchcell_version` shares your installed `major.minor`.

## Raw, genomes and objects tiers

The three file tiers share one serving rule. Each key (a citation key, an assembly set,
or an object key) is a directory with a `manifest.json` listing every file by path,
role, size and sha256; a file is served only if its key's manifest lists it, and the
`X-Artifact-SHA256` header of the response is the manifest's hash, so a client checks
the bytes it received against it. Every file route honors `Range`, which is how an
interrupted download of a large assembly container resumes.

- **Raw** (`/raw`): the source files each dataset loader consumed, keyed by citation
  key; the manifest is the literature `Manifest`.
- **Genomes** (`/genomes`): sequence-level reference data, one assembly set per key
  (for example the SGD S288C R64-4-1 release, or the Peter et al. 2018 isolate
  assemblies); the manifest is a genome manifest naming the organism, strain or
  population, source and release.
- **Objects** (`/objects`): derived files a record points at, keyed by a citation key
  or a named derived set; the manifest is the literature `Manifest`, and each file's
  processing record names the script and inputs that produced it.

`/genomes` and `/objects` list only keys that carry a manifest, and an endpoint without
one of those tiers answers its listing with an empty list.

## curl

```bash
export TC_DATA_URL=http://torchcell-database.ncsa.illinois.edu:8724
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

# genomes and objects tiers: list, read the file rows, fetch one file, verify it
curl -H "X-API-Key: $TC_DATA_API_KEY" "$TC_DATA_URL/genomes"
curl -H "X-API-Key: $TC_DATA_API_KEY" "$TC_DATA_URL/genomes/<assembly_set>/files"
curl -C - -H "X-API-Key: $TC_DATA_API_KEY" -D headers.txt \
  -o "<file>" "$TC_DATA_URL/genomes/<assembly_set>/artifact/<path>"
grep -i x-artifact-sha256 headers.txt; sha256sum "<file>"   # the two must agree
curl -H "X-API-Key: $TC_DATA_API_KEY" "$TC_DATA_URL/objects"
curl -H "X-API-Key: $TC_DATA_API_KEY" "$TC_DATA_URL/objects/<object_key>/files"
```

## Keys

A key is issued per person or group by a maintainer; there is no self-service sign-up.
Each key has a name, the endpoint stores only its sha256, and the plaintext is shown
once, at the moment it is made. To issue one, on the endpoint host:

```bash
python -m torchcell.datasets.server --gen-key <name>
```

It prints the plaintext key, which goes to the user, and a `{"<name>": "<sha256hex>"}`
line, which is added to the JSON file `TC_DATA_KEYS_FILE` points at. The service reads
that file when it starts, so restart it for a new key to be accepted. Deleting an entry
and restarting revokes that key and no other.

## Environment variables

| Variable | Where | Meaning |
|---|---|---|
| `TC_DATA_URL` | client | endpoint base URL; when set, loaders download instead of building |
| `TC_DATA_API_KEY` | client | the plaintext key for the `X-API-Key` header |
| `TC_DATA_ROOT` | server | the artifact store (`index.json`, `SHA256SUMS`, `<slug>/<archive>`) |
| `TC_DATA_RAW_ROOT` | server | the raw mirror; defaults to `$DATA_ROOT/torchcell-raw` |
| `TC_DATA_GENOMES_ROOT` | server | the genomes tier; defaults to `$DATA_ROOT/torchcell-genomes`; may be absent |
| `TC_DATA_OBJECTS_ROOT` | server | the objects tier; defaults to `$DATA_ROOT/torchcell-objects`; may be absent |
| `TC_DATA_KEYS_FILE` | server | JSON `{name: sha256hex}` of accepted keys (preferred) |
| `TC_DATA_API_KEYS` | server | inline `name:key,name2:key2` (quick start only) |
| `TC_DATA_HOST`, `TC_DATA_PORT` | server | bind address and port (default `0.0.0.0:8724`) |

## From Python

With `TC_DATA_URL` and `TC_DATA_API_KEY` set, any `ExperimentDataset` loader whose
LMDB is absent fetches, verifies and unpacks its artifact instead of downloading the
publisher files and running `process()`. If the endpoint has no artifact compatible
with the installed version the loader raises and names the slug; it never falls back
to the publisher path while the variable is set.

The two variables are read from the process environment. Either export them in the
shell, as in the curl section, or put the same two lines, without `export`, in a `.env`
file in your project and load it before constructing a loader:

```python
from dotenv import load_dotenv

load_dotenv()  # reads the .env beside this script or in a parent directory

from torchcell.datasets.scerevisiae.costanzo2016 import SmfCostanzo2016Dataset

dataset = SmfCostanzo2016Dataset(root="data/torchcell/smf_costanzo2016")
```

torchcell does not read `.env` on its own. A `.env` with no `load_dotenv()` call leaves
`TC_DATA_URL` unset, and the loader then downloads the publisher files and builds, with
no error. `load_dotenv()` never overrides a variable that is already exported.

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
page for the exact shapes. All three file tiers have client methods:

```python
from torchcell.datasets.client import DatasetClient

client = DatasetClient.from_env()
client.raw_manifest("<citation_key>")    # literature Manifest
client.raw_files("<citation_key>")       # path, role, bytes, sha256
client.download_raw_file("<citation_key>", "<path>", dest="raw/<path>")
client.genomes()                       # ["peter2018_1011_assemblies", ...]
client.genome_manifest("peter2018_1011_assemblies")   # GenomeManifest
client.genome_files("peter2018_1011_assemblies")      # path, role, bytes, sha256
client.download_genome_file(
    "peter2018_1011_assemblies", "<path>", dest="genomes/<path>"
)
client.objects()
client.object_manifest("<object_key>")  # literature Manifest
client.download_object_file("<object_key>", "<path>", dest="objects/<path>")
```

Each download looks up the file's row in the key's manifest first (a path the manifest
does not list raises `KeyError` before any request), resumes a `<dest>.part` with a
`Range` request, and requires both the server's `X-Artifact-SHA256` and the hash of
the received bytes to equal the manifest row; on a mismatch the partial file is
removed and `ArtifactIntegrityError` is raised.
