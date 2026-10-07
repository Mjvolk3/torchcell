---
id: t910de10597055dc8mceacj
title: '07'
desc: ''
updated: 1791363026902
created: 1791363026902
---

## 2026.10.07 - Plan: the artifact tier and the ArtifactRef pointer

Owner: M. Volk. Driver session: 93b444d6. Umbrella: the database-system document.

## The problem

A query over Caudal 2024 returns each isolate's expression vector inline and its genotype
as perturbations whose sequence is an off-graph pointer (`sequence_uri="<gene>.fasta#<token>"`,
`sequence_sha256=<tarball sha>`). Nothing dereferences those pointers: the query client,
the cell dataset and the graph processors never read `sequence_uri`; the only resolver,
`torchcell.sequence.genome.registry.resolve(assembly_set, filename)`, takes a different
form, runs only against local disk, and is called by nobody downstream; tc-data serves
packaged LMDBs and the raw mirror but not the genomes tier. Isolate ESM2 embeddings and
the perturb-seq per-cell matrix have no pointer at all. Every future pan-genome DNA or RNA
dataset has the same shape: a graph record plus bytes that do not belong in the graph.

## Decisions

D1. **One pointer type, `ArtifactRef`.** A pydantic `ModelStrict` in a new package
`torchcell/artifacts/` (module `ref.py`), later added to the schema surface:
`tier: Literal["raw", "genomes", "library", "objects"]`, `key: str` (citation key,
assembly set, or object namespace), `path: str` (file path relative to the key directory,
no leading slash), `member: str | None` (a sub-member inside the file: a tar member, a
FASTA record token, an h5ad obs name; the file's sha256 still identifies the bytes),
`sha256: str` (64 hex, of the FILE at `path`), `bytes: int | None`, `media_type: str |
None`. String form `tc://<tier>/<key>/<path>[#<member>]` via `__str__` / `parse()`.
Validators: sha256 hex, path normalization, member only when path is set.

D2. **One resolver.** `torchcell.artifacts.resolve(ref, *, materialize=True,
data_root=None, client=None) -> ResolvedArtifact(path: Path, source: Literal["local",
"remote"], verified: bool)`. Ordered sources, no others: (1) the local tier root, read
through that tier's manifest (`raw`: `$DATA_ROOT/torchcell-raw/<key>/manifest.json`,
literature `Manifest` shape; `genomes`: the genomes registry manifest; `library`:
`$DATA_ROOT/torchcell-library/<key>/manifest.json`; `objects`:
`$DATA_ROOT/torchcell-objects/<key>/manifest.json`, same `Manifest` shape), the manifest
sha256 must equal `ref.sha256` and the bytes on disk must match; (2) tc-data over HTTP
(`TC_DATA_URL`, `TC_DATA_API_KEY`) into `$DATA_ROOT/artifact-cache/<sha256>/<basename>`,
verified against `ref.sha256` on arrival. `materialize=False` answers "does this ref
resolve" (manifest lists the path with that sha256, locally or via the remote manifest)
without downloading. A ref that resolves nowhere raises `ArtifactUnresolvableError`; a
sha256 mismatch anywhere raises `ArtifactIntegrityError`. No silent fallback beyond this
list.

D3. **The objects tier.** `$DATA_ROOT/torchcell-objects/<key>/` with a `manifest.json`
in the literature `Manifest` shape; `key` is a citation key for data derived from one
paper's release, or a named derived set (`caudal2024-isolate-esm2`). Files are standard
formats: bgzip FASTA with `.fai`/`.gzi`, bgzip GFF3/VCF with tabix, `h5ad`/zarr for
matrices, parquet for tables, zarr/npy for embeddings. Every `ArtifactRecord` carries a
`ProcessingRecord` naming the script and the input refs. `torchcell.artifacts.deposit`
writes or updates a manifest from a directory of files. The tier joins the Sunday rsync
to /bulk and the Taiga copy like the other three.

D4. **Schema.** `SequenceVariantPerturbation`, `NaturalGenePresencePerturbation`,
`NaturalGeneAbsencePerturbation` replace `sequence_uri` + `sequence_sha256` with
`sequence_ref: ArtifactRef | None` (keep `sequence_source`); the CRISPR construct replaces
`effector_plasmid_uri` + `_sha256` with `effector_plasmid_ref`. This is a schema-surface
change (full rebuild at the next KG build; `TORCHCELL_SCHEMA_ACK=1`). Caudal and Bloom
loaders build the refs: Caudal `tier=genomes, key=peter2018_1011_assemblies,
path=<the allReferenceGenes tarball filename>, member="<gene>.fasta#<token>", sha256=<tar
sha256 from the genomes manifest>`; Bloom likewise from its `tarball::member` form.

D5. **The query pipeline enforces resolvability.** `Neo4jQueryRaw.process` walks each
processed record for `ArtifactRef` instances and calls `resolve(ref, materialize=False)`
once per distinct `(tier, key, path, sha256)`; an unresolvable ref fails the build, the
same rule as interned constants. Materialization is lazy: `torchcell.artifacts.materialize(ref)`
returns the local path, fetching through the resolver on first use; the cell dataset
exposes it, models call it when they need the bytes.

D6. **Serving.** tc-data gains `/genomes` (list sets, manifest, files, artifact) and
`/objects/{key}` (manifest, files, artifact) beside `/raw`, same key auth, same
`X-Artifact-SHA256` header, range requests honored; `DatasetClient` gains the matching
methods and the resolver uses them. Deployment recipe in `notes/database.tc-data-endpoint.md`
gains the two new roots (`TC_DATA_GENOMES_ROOT`, `TC_DATA_OBJECTS_ROOT`).

## Phases (one PR each, landed through the merge queue)

1. `feat/artifact-tier-core`: `torchcell/artifacts/` (`ref.py`, `tiers.py`, `resolve.py`,
   `deposit.py`, `__main__.py` CLI `resolve|check|deposit`), tests, the paired note.
2. `feat/tc-data-tier-endpoints`: server endpoints, client methods, tests, docs page.
3. `feat/schema-artifact-ref` (after 1 lands): D4 schema + Caudal/Bloom loaders + tests;
   dev LMDB rebuild of both under slurm; measurements in the note.
4. `feat/query-artifact-check` (after 1 lands): D5 in `Neo4jQueryRaw` + cell dataset
   `materialize`; tests with a fake resolver.
5. Deposits (after 1 and 2): isolate ESM2 embeddings as an objects set; the Peter gene-keyed
   store re-indexed as bgzip FASTA + fai; the perturb-seq per-cell matrix as h5ad (needs an
   R session to read the Seurat objects; separate go-ahead).

## Not in scope

PostgreSQL or any second database engine; images (none exist); re-fetching FASTQ.

## Phase 6: prove it on GilaHyper, then move it to Radiant with Taiga as the home

The artifact service is tc-data (already live on Radiant:8724, serving packaged LMDBs and
the raw mirror from Taiga over NFS; file streaming over NFS works, only a Neo4j store does
not). The sequence:

1. GilaHyper, dev: run tc-data locally (`torchcell.datasets.server`) with
   `TC_DATA_GENOMES_ROOT` and `TC_DATA_OBJECTS_ROOT` pointing at the local tiers, and
   prove the loop end to end: a Caudal record's `ArtifactRef` resolves locally; with the
   local tier hidden (`data_root` pointed at an empty dir) the same ref resolves through
   HTTP into the cache and verifies; `Neo4jQueryRaw` on a small Caudal query passes the
   resolvability gate; a model-side `materialize` returns the gene FASTA member.
2. Ship the tiers to Taiga beside the raw mirror and tc-data store:
   `/mnt/zhao5/mjvolk3/projects/torchcell/data/torchcell/{torchcell-genomes,torchcell-objects}`
   with `rsync -rlpt` and `sha256sum -c` against each manifest on arrival (a `kg_release.sh`-style
   `ship` for tiers, or one script `scripts/ship_artifact_tiers.sh`).
3. Radiant: add the two roots to `~/projects/tc-data/tc-data.conf`, rebuild the container
   from the landed commit (the recipe in `notes/database.tc-data-endpoint.md`), restart,
   verify `/genomes` and `/objects` answer with the right sha256 headers, then resolve the
   same Caudal ref from a client machine with no local tier (`TC_DATA_URL` set to Radiant).
4. From then on every deposit is: deposit locally, ship, done; the graph never changes for
   a new object, only for a new pointer in a record.

Phase 6 lands as `notes/database.tc-data-endpoint.md` measurements plus the ship script.
