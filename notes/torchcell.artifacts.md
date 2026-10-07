---
id: 9tb53tac9xfnbc2shl896vd
title: Artifacts
desc: ''
updated: 1791363651121
created: 1791363651121
---

## 2026.10.07 - Phase 1: ArtifactRef, the resolver, the objects tier

Phase 1 of the artifact-tier plan (decisions D1 to D3). A graph record points at bytes kept off the graph with one pointer type, `ArtifactRef`, and one resolver turns that pointer into verified bytes on disk. Nothing in the schema or the query pipeline uses it yet; that is phases 3 and 4.

### Package

- `torchcell/artifacts/ref.py`: `ArtifactRef(ModelStrict)` with `tier` (`raw`, `genomes`, `library`, `objects`), `key`, `path`, `member`, `sha256` (of the FILE at `path`), `bytes`, `media_type`. String form `tc://<tier>/<key>/<path>[#<member>]`; `ArtifactRef.parse(text, sha256=...)` takes the sha256 beside the string, because the string carries the location only. The member is everything after the FIRST `#`, so Caudal's member `<gene>.fasta#<token>` survives the round trip; a path or key may not contain `#`. Validators: sha256 is 64 lowercase hex; key is one directory name not starting with `_` (service directories such as the library's `_bib`); path is relative, normalized (`posixpath.normpath(path) == path`), free of `..`; bytes non-negative; a member is non-empty.
- `torchcell/artifacts/tiers.py`: tier roots as functions of `data_root` (`DATA_ROOT` after `load_dotenv()` when None, `KeyError` when unset): `torchcell-raw/<key>`, `torchcell-library/<key>`, `torchcell-objects/<key>` (literature `Manifest`), and the genomes registry for `genomes` (`GenomeManifest`). A literature-shape manifest must name its own key (`citation_key == key`), as the registry already requires for `assembly_set`. `manifest_record(ref, data_root)` locates the record or returns None.
- `torchcell/artifacts/resolve.py`: `resolve`, `check`, `materialize`, `ResolvedArtifact(path, source, verified)`, the errors `ArtifactUnresolvableError(LookupError)` and `ArtifactIntegrityError(RuntimeError)`, the `RemoteSource` protocol (`manifest(tier, key)`, `download(tier, key, path, dest)`, a `RemoteMissError` for an absent key or path), and `TcDataSource` over the `HttpClient` protocol of `torchcell.datasets.client`.
- `torchcell/artifacts/deposit.py`: `deposit(directory, *, tier="objects", key, processing, roles, data_root, allow_change=False)`.
- `torchcell/artifacts/__main__.py`: `python -m torchcell.artifacts [--data-root R] resolve <tc-uri> --sha256 <hex> [--no-materialize]`, `check <tc-uri> --sha256 <hex>` (exit 0 or 1), `deposit <dir> --key <k> [--tier objects|raw] [--processing <json>] [--allow-change]`.

### Resolver order

1. The local tier, through its manifest. A manifest that lists the path with another sha256 (or another size than `ref.bytes`) raises `ArtifactIntegrityError` at once; it is never routed around to the remote. With `materialize=True` the bytes on disk are hashed too. An absent manifest, an unlisted path, or a listed file missing from disk is a miss and the resolver goes on.
2. With `materialize=True` only: the artifact cache. A cached file is re-hashed before it is returned; a corrupted one raises.
3. The remote source: the `client` argument, or tc-data from `TC_DATA_URL` + `TC_DATA_API_KEY` when no client is passed (an unset `TC_DATA_URL` is recorded as an unconfigured source, not silently skipped). Its manifest must list the path with `ref.sha256`; with `materialize=True` the file streams to `<cache file>.part`, is hashed, and only then renamed into place. A wrong hash removes the `.part` (and the sha256 directory once empty) and raises.

`materialize=False` consults manifests only (local, then remote) and downloads and hashes nothing; `verified` is then False. A ref no source holds raises `ArtifactUnresolvableError` whose message names the ref string, its sha256 and each source tried with the reason, numbered in order. `check` returns False only for that error; an integrity disagreement still raises.

### Cache layout

`$DATA_ROOT/artifact-cache/<sha256>/<basename of path>`. The directory is the content hash, so two refs to the same bytes share one cached file, and a file is never trusted by name: it is hashed on every materializing resolve.

### Deposit

`deposit` hashes every file under `directory` except `manifest.json`, copies them to `<tier root>/<key>/` (in place when `directory` already is that key directory), and writes the literature `Manifest`. An existing manifest is updated: records the deposit does not touch are kept verbatim, an unchanged file keeps its record and the manifest keeps its `created_at`, so depositing the same bytes twice writes the same manifest bytes. A listed file whose sha256 changed raises `DepositChangeError` (nothing is copied or written) unless `allow_change=True`; the new record's `source` is then `supersedes sha256:<old>`. Default roles: `object` in the objects tier, `raw_data` in the raw tier; `roles` overrides per path and may not name a path the directory lacks. `processing` attaches to every new or changed record; `provenance_complete` is True only when every record carries a `processing` or `retrieval`. The genomes tier (`registry.deposit_assembly_set`) and the library tier (its capture pipeline) are refused.

### Tests

`tests/torchcell/artifacts/` (one module per source module, prefixed `test_artifact_` to avoid basename collisions; hermetic tiers under `tmp_path`, a fake `RemoteSource` in `_fakes.py`, and `TcDataSource` over the real tc-data app through the ASGI test client for the raw tier). 82 tests; line and branch coverage of `torchcell/artifacts/` 99%, the two missed lines being the `RemoteSource` protocol method bodies (`pytest --cov=torchcell/artifacts --cov-branch`, 2026.10.07).

### What phases 2 to 5 add

- Phase 2 (`feat/tc-data-tier-endpoints`): tc-data serves `/genomes/{key}/manifest|artifact/{path}` and `/objects/{key}/manifest|artifact/{path}` beside `/raw`, with `X-Artifact-SHA256` and range requests; `DatasetClient` gains the matching methods. `TcDataSource` already uses these URL shapes; until phase 2 lands, the genomes and objects tiers miss on tc-data with a 404.
- Phase 3 (`feat/schema-artifact-ref`): `sequence_ref: ArtifactRef | None` replaces `sequence_uri` + `sequence_sha256` on the natural-variation perturbations, and `effector_plasmid_ref` on the CRISPR construct; Caudal and Bloom loaders build the refs (full KG rebuild).
- Phase 4 (`feat/query-artifact-check`): `Neo4jQueryRaw.process` calls `resolve(ref, materialize=False)` once per distinct ref and fails the build on an unresolvable one; the cell dataset exposes `materialize`.
- Phase 5: deposits into the objects tier (isolate ESM2 embeddings, the Peter gene-keyed store as bgzip FASTA + fai, the perturb-seq per-cell matrix as h5ad), then the Taiga ship and the Radiant tc-data roots (phase 6 of the plan).

## 2026.10.07 - ArtifactRef moved onto the schema surface

`ArtifactRef` is now defined in `torchcell/datamodels/schema.py` (with `ArtifactTier`, `ARTIFACT_TIERS`, `ARTIFACT_URI_SCHEME`, `ARTIFACT_MEMBER_SEPARATOR`); `torchcell/artifacts/ref.py` re-exports it under the old names (`Tier`, `TIERS`, `URI_SCHEME`, `MEMBER_SEPARATOR`), so every import of `torchcell.artifacts.ref` is unchanged. Schema records carry the ref (phase 3, [[torchcell.datamodels.schema]] 2026.10.07), and importing it into `schema.py` from `ref.py` was a cycle (`ref.py` -> `torchcell.datamodels.pydant` -> `torchcell/datamodels/__init__.py` -> `schema`). On the surface its contract is fingerprinted by the schema-impact check.

## 2026.10.07 - Phase 4: the walk, the query gate, lazy materialization

- `torchcell/artifacts/walk.py`: `iter_refs(model)` yields every `ArtifactRef` in a pydantic tree (declared fields, `model_extra`, nested models, lists, tuples, sets, dict values; a ref is not descended into; computed fields are not walked), in field order with repeats. `distinct_refs(models)` keys them by `ref_key(ref) = (tier, key, path, sha256)` and keeps the first ref met per key, so refs to one file with different members collapse. Pure, no I/O. Exported from the package.
- `Neo4jQueryRaw.process` resolves every distinct ref of its records once per run (manifests only) and fails the build with `UnresolvableArtifactError` on one that resolves nowhere; `Neo4jQueryRaw.materialize(ref)` fetches lazily. Details in [[torchcell.data.neo4j_query_raw]].
- `Neo4jCellDataset.refs_of(index)` lists the distinct refs of one processed entry (every aggregated record, experiment then reference) by reading the processed LMDB only. `Neo4jCellDataset.materialize(ref)` is `torchcell.artifacts.materialize(ref)` over the environment's `DATA_ROOT` and `TC_DATA_URL`, the sources the raw stage's gate used by default. Models call these when they need bytes.
- Tests: `tests/torchcell/artifacts/test_artifact_walk.py` (6), `tests/torchcell/data/test_neo4j_query_raw_artifacts.py` (12), `tests/torchcell/data/test_neo4j_cell_artifacts.py` (3).

## 2026.10.07 - Figure: a query, then the off-graph bytes

[[torchcell.artifacts.mermaid.query-resolve]] is the sequence diagram of one query: Neo4j answers with records whose perturbations carry `ArtifactRef` pointers, the gate checks each distinct ref against a manifest (local tier, else tc-data) without downloading, and `materialize` later fetches the bytes through the same resolver order with sha256 verification at every step. Rendered to `notes/assets/pdf-output/torchcell.artifacts.mermaid.query-resolve.{pdf,svg,png}`.

## 2026.10.07 - Figure: the user's wiring

[[torchcell.artifacts.mermaid.user-wiring]] is the one-glance version: you query Neo4j, it answers with records plus pointers, and touching a pointer makes the same library fetch the verified bytes from the sequence and transcriptome store. Rendered to `notes/assets/pdf-output/torchcell.artifacts.mermaid.user-wiring.{pdf,svg,png}`.
