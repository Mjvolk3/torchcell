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
