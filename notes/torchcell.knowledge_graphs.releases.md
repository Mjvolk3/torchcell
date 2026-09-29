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
