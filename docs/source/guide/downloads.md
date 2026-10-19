# Downloading datasets

Built torchcell datasets are served as packaged archives by `tc-data`, a public
read-only HTTP endpoint with Swagger documentation. Each archive holds a dataset's `processed/`
directory (the records LMDB and its `interned` sibling) and `preprocess/` directory
(`build_manifest.json`, `gene_set.json`, `experiment_reference_index.json`), exactly
as the loader wrote them, so a download replaces the build step. The raw files each
loader consumed are served from the same endpoint, hash-pinned by their `manifest.json`,
and so are three more file tiers: two that graph records point at without storing their
bytes, the genomes tier (reference and isolate assembly sets) and the objects tier
(derived files such as embeddings, matrices and indexed FASTA), and the library tier,
which serves each paper's supplementary material (never the paper itself).

The endpoint is `http://torchcell-database.ncsa.illinois.edu:8724`, on the database
host, serving the archive store and the raw mirror from the Taiga project storage. No
key, account or ssh is needed: every route is open for reading. The URL is built into
the package as `torchcell.endpoint_http.DEFAULT_URL`, and `TC_DATA_URL` overrides it
for a mirror or a development server. The store currently holds `gene_essentiality_sgd`,
`smf_costanzo2016`, `amino_acid_cooper2010`, `amino_acid_mulleder2016` and
`betaxanthin_cachera2023`; `GET /datasets` is the authoritative list.

## Swagger

The interactive documentation lives at `<TC_DATA_URL>/docs` and the OpenAPI schema at
`<TC_DATA_URL>/openapi.json`. The routes:

| Route | Returns |
|---|---|
| `GET /health` | liveness, artifact and per-tier key counts |
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
| `GET /library` | citation keys in the library tier that carry a manifest |
| `GET /library/{citation_key}/manifest` | the key's `manifest.json`, without paper, OCR or layout records |
| `GET /library/{citation_key}/files` | the supplementary files that manifest lists, with sha256 |
| `GET /library/{citation_key}/artifact/{path}` | one supplementary file; `X-Artifact-SHA256`, `Accept-Ranges: bytes` |

An index row records the dataset `slug`, the loader `dataset_class`, the
`torchcell_version` and git commit the LMDB was built with, the knowledge-graph
release it was admitted to, `content_sha256` (over the experiment ids, comparable with
the release manifest), the archive name, `archive_sha256`, `archive_bytes`, and
`status` (`supported` or `deprecated`). Pick the newest `supported` row whose
`torchcell_version` shares your installed `major.minor`.

## Raw, genomes, objects and library tiers

The four file tiers share one serving rule. Each key (a citation key, an assembly set,
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
- **Library** (`/library`): a paper's supplementary material, keyed by citation key;
  the manifest is the literature `Manifest`. Only the SI and data roles are served
  (`si_data`, `si_pdf`, `si_ocr`, `si_text`, `raw_data`, `dataset_raw`, `annotations`,
  `other`), for example a plasmid GenBank file a strain record points at. The paper,
  its OCR markdown, OCR images and layout files are never listed or served; they stay
  behind a key on the literature endpoint.

`/genomes`, `/objects` and `/library` list only keys that carry a manifest, and an
endpoint without one of those tiers answers its listing with an empty list.

Paper content is withheld. In the raw, objects and library tiers a manifest record
whose role begins `paper_` (`paper_pdf`, `paper_text`, `paper_ocr`, `paper_fulltext`)
or `ocr_` (`ocr_image`, `ocr_layout`, `ocr_provenance`) is left out of `/manifest` and
`/files`, and a request for its bytes answers 404 `paper content is not served`. SI
files and data files under the same key are served as usual.

## curl

```bash
export TC_DATA_URL=http://torchcell-database.ncsa.illinois.edu:8724

curl "$TC_DATA_URL/datasets" | python -m json.tool
curl "$TC_DATA_URL/datasets/smf_costanzo2016"

# download an archive, resuming if interrupted, then verify it
ARCHIVE=smf_costanzo2016-1.2.1-d8a0f06c.tar.xz
curl -C - -o "$ARCHIVE" "$TC_DATA_URL/datasets/smf_costanzo2016/$ARCHIVE"
sha256sum "$ARCHIVE"   # must equal the row's archive_sha256

# unpack into a dataset root; processed/ and preprocess/ appear beside it
mkdir -p data/torchcell/smf_costanzo2016
tar -xf "$ARCHIVE" -C data/torchcell/smf_costanzo2016

# raw files
curl "$TC_DATA_URL/raw"
curl "$TC_DATA_URL/raw/<citation_key>/files"

# genomes and objects tiers: list, read the file rows, fetch one file, verify it
curl "$TC_DATA_URL/genomes"
curl "$TC_DATA_URL/genomes/<assembly_set>/files"
curl -C - -D headers.txt \
  -o "<file>" "$TC_DATA_URL/genomes/<assembly_set>/artifact/<path>"
grep -i x-artifact-sha256 headers.txt; sha256sum "<file>"   # the two must agree
curl "$TC_DATA_URL/objects"
curl "$TC_DATA_URL/objects/<object_key>/files"

# library tier: a paper's supplementary files (the paper itself is not served)
curl "$TC_DATA_URL/library/<citation_key>/files"
```

## Environment variables

| Variable | Where | Meaning |
|---|---|---|
| `TC_DATA_URL` | client | endpoint base URL, overriding the built-in `DEFAULT_URL`; when set, loaders download instead of building |
| `TC_DATA_ROOT` | server | the artifact store (`index.json`, `SHA256SUMS`, `<slug>/<archive>`) |
| `TC_DATA_RAW_ROOT` | server | the raw mirror; defaults to `$DATA_ROOT/torchcell-raw` |
| `TC_DATA_GENOMES_ROOT` | server | the genomes tier; defaults to `$DATA_ROOT/torchcell-genomes`; may be absent |
| `TC_DATA_OBJECTS_ROOT` | server | the objects tier; defaults to `$DATA_ROOT/torchcell-objects`; may be absent |
| `TC_DATA_LIBRARY_ROOT` | server | the library tier, of which only supplementary material is served; defaults to `$DATA_ROOT/torchcell-library`; may be absent |
| `TC_DATA_HOST`, `TC_DATA_PORT` | server | bind address and port (default `0.0.0.0:8724`) |

## From Python

With `TC_DATA_URL` set, any `ExperimentDataset` loader whose LMDB is absent fetches, verifies and unpacks its artifact instead of downloading the
publisher files and running `process()`. If the endpoint has no artifact compatible
with the installed version the loader raises and names the slug; it never falls back
to the publisher path while the variable is set.

A loader prefers a download only when `TC_DATA_URL` is set explicitly; the built-in
default does not switch loaders over. The variable is read from the process environment.
Either export it in the shell, as in the curl section, or put the same line, without
`export`, in a `.env` file in your project and load it before constructing a loader:

```python
from dotenv import load_dotenv

load_dotenv()  # reads the .env beside this script or in a parent directory

from torchcell.datasets.scerevisiae.costanzo2016 import SmfCostanzo2016Dataset

dataset = SmfCostanzo2016Dataset(root="data/torchcell/smf_costanzo2016")
```

torchcell does not read `.env` on its own. A `.env` with no `load_dotenv()` call leaves
`TC_DATA_URL` unset, and the loader then downloads the publisher files and builds, with
no error. `load_dotenv()` never overrides a variable that is already exported.

The client needs no configuration: `DatasetClient()` with no arguments talks to the
public endpoint, and `DatasetClient.from_env()` uses `TC_DATA_URL` when it is set and
the public endpoint otherwise.

```python
from torchcell.datasets.client import DatasetClient, unpack_artifact

client = DatasetClient.from_env()          # TC_DATA_URL if set, else the public endpoint
artifact = client.select("smf_costanzo2016")  # newest supported row on this major.minor
if artifact is None:
    raise SystemExit("nothing compatible with the installed torchcell")
archive = client.download(artifact, dest=f"artifacts/{artifact.archive}")  # verifies sha256
unpack_artifact(archive, "data/torchcell/smf_costanzo2016")
```

The raw mirror keys and files are one `GET` away; see the Swagger page for the exact
shapes. All four file tiers have client methods:

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
client.download_library_file("<citation_key>", "<path>", dest="library/<path>")
```

Each download looks up the file's row in the key's manifest first (a path the manifest
does not list raises `KeyError` before any request), resumes a `<dest>.part` with a
`Range` request, and requires both the server's `X-Artifact-SHA256` and the hash of
the received bytes to equal the manifest row; on a mismatch the partial file is
removed and `ArtifactIntegrityError` is raised.
