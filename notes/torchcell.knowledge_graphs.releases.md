---
id: a77onzh6kxbwx1hcy6ibszc
title: Releases
desc: ''
updated: 1789840098627
created: 1789840098627
---

## 2026.09.19 - What a release is, and how code names one

A release is one built store. Its id is `<build date>-<commit[:8]>`, the shape the
iBioFoundry deployments print in their `make ops` BUILD column (`2026.09.17-7715ee35`
is the store swapped in on 2026-09-17 from commit 7715ee35). Its version is
`<major>.<minor>`: a full rebuild bumps the major, an incremental admission bumps the
minor. The store carries one `KgRelease` node with both, the commit, the build time, the
CSV archive name, the node count, every dataset's record count and content hash, and
every dataset's schema closure fingerprints as JSON string properties. A dump or backup
loaded anywhere answers `MATCH (r:KgRelease) RETURN r` with what it is. The manifest file
carries the same `version`, `release` and per-dataset `content_sha256` (three new
optional fields on `KgBuildManifest` / `KgDatasetEntry`, None for manifests written
before this).

**Byte identity.** `CellAdapter` builds every `Experiment` node id as the sha256 of the
serialized record, so a dataset's sorted experiment ids fingerprint its content.
`content_sha256(ids)` is the sha256 of those ids sorted and newline-joined with a
trailing newline; `database/scripts/kg_content_hashes.sh` computes the same thing
through cypher-shell (`ORDER BY e.id | sha256sum`) without holding ids in memory, and
`content_hashes_from_csv` computes it from the `ExperimentMemberOf` CSVs. Equal hashes
across two releases mean byte-identical serialized experiment records; `diff(a, b)`
lists unchanged / changed / added / removed datasets by that rule. Reference nodes are
not part of the hash: it is a statement about the experiment records.

**Version selection.** `Neo4jConnectionSettings.version` reads `TORCHCELL_KG_VERSION`
(default `latest`); `Neo4jQueryRaw` gained a `version` attribute and resolves it to a
database name at fetch time through `resolve_database`: `latest` and `pinned` are
database aliases on the served DBMS (created 2026-09-19 on GilaHyper, both at
`torchcell`; a driver session opened on an alias name works, checked with cypher-shell
`-d latest`), a physical database name passes through, a release id or `major.minor`
is looked up in the release nodes of every online database. Anything else raises
`LookupError`, no fallback.

**Source code moving with the store.** `compatibility(release, repo_root)` compares the
release's per-dataset closure fingerprints with `surface_in_worktree(repo_root)` and
names the datasets the checkout would serialize under a different contract; the CLI's
`compat` exits 1 on drift. This is the same drift the admission gate checks before an
increment, turned around to face a client.

**CLI.** `status` (one row per served database: version, release, commit index on
main, commit date, datasets, nodes, aliases, health; `--json`), `datasets`, `diff`,
`compat`, `hashes` (from `--csv-dir` or `--database`), `stamp` (version, release id,
hashes into a manifest; a full build passes every hash, an increment passes the
admitted datasets' and the rest keep theirs), `write-node` (the KgRelease node from a
stamped manifest). `list_databases` probes each online database with a property read,
because a store whose pages fault (the Radiant VM's NFS store on 2026-09-19) still
answers count-store queries; a DBMS whose system database faults comes back as one
`(dbms)` row with the error. Tests: `tests/torchcell/knowledge_graphs/test_releases.py`.

Importing `torchcell.knowledge_graphs` imports BioCypher (a banner on stdout and a
`biocypher-log/` directory in the cwd, gitignored); `scripts/ops.sh` filters the banner.

## 2026.09.29 - The package version a release names (T4 of the data release program)

Plan: [[plan.data-release-program.2026.09.29]], Decisions 1, 2 and 5; issue #467. Until
now a release pinned a git commit and nothing said which package version reads it.

- `KgRelease` gains `torchcell_version` (the stamping checkout's `torchcell.__version__`)
  and `torchcell_tag` (`git describe --tags --exact-match HEAD`; None between package
  releases). Both are recorded at `stamp` (`kg_manifest.checkout_package_version` reads
  `torchcell/__version__.py` of `--repo`, or of the checkout the package was imported
  from, `package_checkout()`), never inferred later, and travel through the manifest,
  `to_properties`/`from_properties`, `write-node` and the snapshot. Nodes written before
  this carry None.
- `status` has a `PKG` column after `RELEASE`: the tag when the build ran from one,
  `1.2.0 (untagged)` otherwise, `-` for a pre-spine node. `scripts/ops.sh` finds the
  release id by regex, not by column, so `make ops` is unchanged.
- `compatibility` is split: `closure_compatibility(release, commit, datasets, closures,
  surface)` runs the check against any `SchemaSurface`, which is what the compatibility
  page uses with the surface at a git tag; `compatibility(release, repo_root)` is the
  working-tree case as before.
- New subcommand `snapshot --manifest M [--n-nodes N] [--built-at T] [--repo-root R]`
  writes `database/releases/<release>.json` and `<release>.closures.json`
  ([[torchcell.knowledge_graphs.release_snapshot]]); both slurm scripts call it right
  after `write-node`. `--torchcell-version/--torchcell-tag/--note` are for a manifest
  stamped before the spine and are refused when the manifest records a version.
- Bootstrapped snapshot for the served release `2026.09.21-ab6d8c5d` (version 1.2, 51
  datasets): `torchcell_version` 1.2.0 (`git show ab6d8c5d:torchcell/__version__.py`),
  `torchcell_tag` None (`git tag --points-at ab6d8c5d` is empty; v1.2.0 is 4309a482a of
  2026.07.01 and v1.2.1 is 6d6c8bb27 of 2026.09.27), `n_nodes` 99,724,909 (the count
  job 2678 passed to `write-node`, `database/slurm/output/2678_increment_kg.out` line
  1602), composite `39e26472ceb731c22f882e80fbd9c10a941637e57823317795283fcfbf36d2dc`.
  The derivation is in the snapshot's last event note.
- Measured with `scripts/kg_compat_page.py`: v1.2.1 is `compatible` with that release
  (all 51 closures match the surface at the tag) and v1.2.0 is `partial (all 51 datasets
  drifted)`; page at [[scripts.kg_compat_page]].

## 2026.09.29 - Supported queries check against the committed snapshots (T5)

Plan: [[plan.data-release-program.2026.09.29]], Decisions 3 and 4; issue #468. The committed
snapshot `database/releases/<release>.json` and its `.closures.json` are also the input of
the supported-query drift check ([[torchcell.knowledge_graphs.supported_queries.check]]):
node labels and relationship types come from the snapshot's `graph_schema` (BioCypher's
Pascal-case label of each class, plus `is_a` ancestors from the checkout's schema config),
`missing_dataset` and the per-query `dataset_composite` from its `datasets`
(`release_snapshot.composite_sha256` over the selected subset), and `contract_changed` from
the closures, the same fingerprints `closure_compatibility` compares. A new snapshot written
by `snapshot` is therefore also the release that the next `validate` runs against, and on
`main` a drifted supported query files a `before-next-kg-build` issue
([[database.supported-queries]]). Nothing in `releases.py` changed.

## 2026.10.06 - Pairing: `require_paired`, `IncompatibleReleaseError`, `retag`

The two October full builds (KG 2.0, 3.0) ran from an untagged `main` and their snapshot commits were tagged `RELEASE(kg)`, which bumps nothing, so no published package was compatible with the served store and nothing refused the mismatch: `Neo4jQueryRaw._connect` resolved the database name and ran the query. The pairing rule is now enforced at three points.

- `ReleaseCompatibility.paired` is stricter than `ok`: no drifted AND no unverified dataset. `require_paired(release, surface, installed_version=, database=)` returns the report or raises `IncompatibleReleaseError` (a `RuntimeError`) whose text names the release, its paired package (`package_label`), the installed version, each failing dataset with its symbols, and the remedy (`pip install torchcell==X.Y.Z` when the release names a tag, the compatibility page otherwise). A store with no `KgRelease` node is refused the same way. `Neo4jQueryRaw._connect` calls it with `schema_deps.load_default_surface()` (the installed package's `schema.py` + `pydant.py`, 0.3 s to parse, 129 symbols) before opening a driver.
- `retag --release <id> --tag vX.Y.Z [--repo-root] [--manifest]` pairs a snapshot built from an untagged commit with a tag cut afterwards, accepted only when `kg_manifest.surface_at_ref(root, tag)` reproduces every served closure (`release_snapshot.pair_package_tag`); it sets `torchcell_version`/`torchcell_tag` from the tag, notes what the build checkout reported in the last event, is idempotent for the same tag, and refuses a second, different tag. With `--manifest` the manifest carries the pairing so `write-node` rewrites the store's node (`scripts/kg_release.sh retag` chains the two under the writable toggle).
- The live rebuild script refuses to start unless `BUILD_COMMIT` carries a `v*` tag and the fingerprint checkout sits on it, so future stamps record the tag and `retag` is never needed again.

Measured 2026-10-06 on `e6367528b` (main after the `DB(kg)` cut): `compat --version latest` 51 compatible, 0 drifted, 0 unchecked; the three committed snapshots were retagged to `v1.2.1`, `v1.6.1`, `v1.6.2`.

## 2026.10.08 - The pointer set on the release node, and `artifacts`

`KgRelease.artifact_refs: dict[str, list[ArtifactPointer]] | None` (dataset -> file-level pointers) goes on the node as `artifact_refs_json`, compact sorted JSON without None fields; `to_properties` writes None when unrecorded (the `SET r += $props` then leaves no property), and `from_properties` reads a node without the key as None. `write-node` fills it from the manifest through `kg_manifest.manifest_artifact_refs`, None when any entry is unrecorded.

`artifacts --host LABEL=URI|USER|PASSWORD --database NAME --tc-data-url URL [--host-timeout S] [--tc-data-timeout S] [--api-key-env TC_DATA_API_KEY] [--json]` is the reporter `scripts/ops.sh` renders. It calls `load_dotenv()` (the repo `.env`, as the loaders do), reads the node with `read_release` bounded by `--host-timeout` (`read_release_bounded`), and classifies (`classify_pointers`, `artifact_probe`): refs deduplicated across datasets by `ref_key`, grouped by `(tier, key)`, one `TcDataSource.manifest` call per key; a file is `listed` (path listed with the pinned sha256), `sha256_mismatch`, or `missing` (path absent, or the key answers 404, which includes a tier tc-data does not serve). One `STATE<TAB>CODE<TAB>DETAIL` line, exit 0 always, `--json` for the full classification. The checks run cheapest first, so a release with nothing to check never needs the key or the endpoint:

| state | code | detail |
|---|---|---|
| `fail` | `n/a` | `host unreachable within S s` (no answer within `--host-timeout`) or `release read failed: <first line>` (refused, auth, a faulting store) |
| `warn` | `n/a` | `no release node` |
| `warn` | `n/a` | `release predates pointer recording` (`artifact_refs` None) |
| `ok` | `0` | `the release points at no artifact file` |
| `fail` | `n/a` | `TC_DATA_API_KEY unset` (the variable named by `--api-key-env`; the key is never printed) |
| `fail` | `n/a` | `tc-data unreachable: <first line>` (a transport error or a non-404 error status, e.g. 401) |
| `ok` | `N/N` | `every pointer listed by tc-data with its sha256` |
| `fail` | `k/N` | `m missing, s sha256 mismatch: tc://... (first three, then ", ...")` |

The line is printed with `flush=True` because ops.sh bounds the process with `timeout`, and an abandoned driver thread can hold interpreter exit past the verdict. The pointer type is `kg_manifest.ArtifactPointer`, a local model rather than `ArtifactRef`, so this module keeps the slim import closure the docs workflow's query-drift job installs (pydantic, PyYAML, python-dotenv; no `torchcell.datamodels`, no `torchcell.artifacts` at module level). The `--host` parsing is shared with `status` (`_host_spec`).

Tests (`tests/torchcell/knowledge_graphs/test_releases.py`): `test_properties_round_trip_the_artifact_refs_as_compact_json`, `test_cli_write_node_carries_the_manifest_pointer_set_or_none_when_half_recorded`, and one per state, each pinning the printed line: `test_cli_artifacts_ok_when_every_pointer_is_listed_with_its_sha256`, `test_cli_artifacts_ok_without_asking_tc_data_when_nothing_is_pointed_at`, `test_cli_artifacts_warns_on_a_store_without_a_release_node`, `test_cli_artifacts_warns_on_a_release_that_predates_pointer_recording`, `test_cli_artifacts_fails_naming_missing_and_mismatched_pointers`, `test_cli_artifacts_fails_when_the_api_key_is_unset`, `test_cli_artifacts_fails_when_tc_data_is_unreachable_or_refuses`, `test_cli_artifacts_fails_when_the_host_does_not_answer_or_the_read_raises`. `test_to_properties_flattens_the_maps_to_compact_sorted_json` now pins `artifact_refs_json: None`.
