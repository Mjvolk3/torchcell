---
id: 4pqkzah1yn0090jr93un777
title: Data release program 2026.09.29
desc: 'Versioned source and database releases, public dataset showcase, download API, documentation build-out, and the supported-query drift gate'
updated: 1790714615848
created: 1790714615848
---

## Context

The repository serves a knowledge graph (release `2026.09.21-ab6d8c5d`, version 1.2, 51
datasets) and a Python package (v1.2.1) that are versioned independently: the KG manifest and
the `KgRelease` node pin a git commit, never a package version, and nothing publishes which
package versions can read which KG release. The public documentation at
`https://mjvolk3.github.io/torchcell/` is a PyTorch Geometric template shell with two API
pages, the ontology explorer, and no guide. Datasets are fetched by each loader from
publisher URLs; there is no way for a collaborator to get a built dataset without the
library, and no way for the library to get one from us. The Phase 8 test campaign showed
that served records can carry defects (Kemmeren channel assignment, Sameith strain) that
only a re-read of the paper and GEO catches. This plan turns those five gaps into one
program with a shared versioning spine. Tracking issues: `#464` (umbrella) and the
per-track issues listed under Approach.

Request (2026-09-29, dictated): keep raising coverage; build out and complete all
documentation; cut a source version synced with the database and publish a page of
compatible database versions linked from the README; host a few example datasets with a
page that shows what each experiment is, what the data looks like (tables, graphs, a
debugger-style dump of the source record), and what a query returns, with draw.io
diagrams (later stylized by an image model); version the supported queries so ontology
revisions that touch their component classes file an issue before the next database
update, with supported and deprecated marks; Swagger docs for download so collaborators
get the raw data without the library, and the library downloads through the same API from
zipped LMDBs hosted on Radiant.

## Relevant files

| Path | Action | Purpose | Stance |
|---|---|---|---|
| `torchcell/knowledge_graphs/releases.py` | MODIFY | add `torchcell_version` to `KgRelease`, `stamp`, `write-node`; export a release snapshot | stable (plan.kg-releases.2026.09.19) |
| `torchcell/knowledge_graphs/kg_manifest.py` | MODIFY | manifest gains `torchcell_version`; `admit`/`record` keep it | stable |
| `database/releases/<release>.json` | NEW | committed release snapshots (small; no closures) that CI can read | n/a |
| `scripts/kg_compat_page.py` | NEW | generates `docs/source/database/compatibility.md` from the snapshots | n/a |
| `torchcell/knowledge_graphs/supported_queries/` | NEW | registry (pydantic), shipped `.cql` files, dependency extraction, drift check CLI | n/a |
| `scripts/run-supported-queries.sh` + `.pre-commit-config.yaml` | NEW/MODIFY | pre-commit hook on schema, adapter, query and registry changes | stable |
| `.github/workflows/docs.yaml` | MODIFY | `query-drift` job that opens or updates the before-next-kg-build issue on main | stable |
| `torchcell/datasets/server.py`, `torchcell/datasets/client.py` | NEW | `tc-data` FastAPI app and the library-side client (same key scheme as `torchcell/literature/server.py`) | n/a |
| `scripts/package_dataset_lmdb.py` | NEW | tar+zstd a built dataset with an `index.json` record and `SHA256SUMS` | n/a |
| `torchcell/data/experiment_dataset.py` | MODIFY | `_download` takes the endpoint path when `TC_DATA_URL` is set | stable |
| `docs/source/**` | MODIFY | Sphinx site: conf, index, API pages for 31 packages, guide, database, showcase | provisional (plan.docs-modernization.2026.07.01 called RTD "not a real gate") |
| `README.md` | MODIFY | links to the guide, the database releases page, the downloads page | stable |
| `pyproject.toml` `[tool.semantic_release]` | MODIFY | `FIX` becomes a patch tag; `DOCS`, `TST`, `NOTE` allowed with no bump | stable |
| `experiments/034-showcase-datasets/` | NEW | scripts that generate every table, figure and record dump on the showcase pages | n/a |
| `notes/assets/drawio/showcase-*.drawio` | NEW | the three experiment diagrams | n/a |
| `notes/datasets.showcase-verification.2026.09.29.md` | NEW | the five re-verification reports, consolidated | n/a |

## Key design decisions

1. **One versioning spine: the package version is recorded in the KG release, and the
   compatibility table is generated from committed release snapshots.** `KgRelease` and
   `KgBuildManifest` gain `torchcell_version` (the `torchcell.__version__` string at build
   time, plus `torchcell_tag` when the commit is tagged). Every stamp writes a compact
   snapshot `database/releases/<release>.json` (release, version, commit, package version,
   built_at, neo4j and biocypher versions, per-dataset `content_sha256` and
   `n_experiments`, graph-schema class list, and the composite hash of Decision 3) into the
   repo, so CI and a fresh clone can compute compatibility without GilaHyper's manifest.
   `scripts/kg_compat_page.py` turns the snapshots into the docs page. Rejected: reading
   `/scratch/projects/torchcell/database/kg_manifest.json` from CI (machine local) and a
   hand-maintained table (the memory `paper-table-generator-pattern` rule applies).
2. **Compatibility is defined per (package version, KG release) by the existing closure
   check, not by a date rule.** `releases.compatibility()` already compares each served
   dataset's closure fingerprints with the checkout; the page's status column is its
   verdict: `compatible` when every served dataset's closure matches, `partial` naming
   the drifted datasets, `incompatible` when a supported query's class is gone. A package
   version is "the reader" of a KG release when its tagged commit passes; the generator
   runs the check for each tag since v1.2.0 against each snapshot (closures live in the
   snapshot's companion `<release>.closures.json`, committed too; about 50 KB).
3. **Supported queries are first-class, shipped with the package, and versioned by a
   composite hash.** `SupportedQuery` (pydantic) records id, title, the `.cql` path under
   `torchcell/knowledge_graphs/queries/`, the converter class, the datasets it selects
   (`dataset.id` literals), the node labels, relationship types, property names and
   `graph_level` literals it uses (extracted from the Cypher text by a small parser, not
   typed by hand), `status: supported | deprecated`, `since_kg_version`,
   `deprecated_in`, and `dataset_composite`: sha256 over the sorted `content_sha256` of
   the datasets it selects, taken from the release snapshot it was last validated on.
   Drift is any of: a used class or property missing from the snapshot's graph schema, a
   converter or phenotype class whose contract fingerprint changed against the snapshot's
   closures, or a `dataset_composite` that differs from the current snapshot's. The check
   is `python -m torchcell.knowledge_graphs.supported_queries check`; exit 1 on drift for a
   `supported` query, 0 with a notice for a `deprecated` one.
4. **Drift files an issue, and the issue is the queue for the next KG build.** The pre-commit
   hook reports; the CI job on `main` opens one issue per drifted query (title
   `Before the next KG build: supported query <id> drifts against <release>`, label
   `before-next-kg-build`, body = the check's report) and updates it instead of duplicating
   (search by title). Closing the issue is manual, after re-validation on the new release.
   Rejected: filing from pre-commit (no token, runs on every machine).
5. **Cut the source release before the KG build, not after.** Procedure recorded in the
   database page: land everything, push a `REL:` or `FEAT:` commit so semantic-release tags
   `vX.Y.0`, then run the full build from that tag (`TORCHCELL_SRC` at the tag; `stamp`
   refuses a dirty tree unless acknowledged). The KG release then names a released package
   version. `FIX` joins the patch tags so bug fixes bump a patch instead of silently not
   releasing; `DOCS`, `TST`, `NOTE` become allowed tags with no bump so they stop being
   parser noise.
6. **Dataset downloads are an API with the tc-lit shape, and the artifact is the unit.**
   `tc-data` (FastAPI, `/docs` Swagger on by default, named API keys, `X-API-Key`) serves an
   artifact store `$TC_DATA_ROOT/<slug>/<artifact>.tar.zst` with `index.json`
   (`DatasetArtifact`: slug, loader class, package version, KG release it was admitted to,
   content sha256 of the record ids, archive sha256 and bytes, built_at, `status`). Routes:
   `/datasets` (index), `/datasets/{slug}` (versions), `/datasets/{slug}/{artifact}` (stream
   with `X-Artifact-SHA256`, `Accept-Ranges`), `/raw/{citation_key}/...` mirroring
   `torchcell-raw/`. The library downloads through it when `TC_DATA_URL` is set: the base
   class fetches the artifact whose package version matches the installed major.minor,
   verifies the sha256, unpacks `processed/` and `preprocess/`, and skips `process()`. No
   silent fallback: if the endpoint is set and has no compatible artifact, the loader says
   so and stops; the publisher path runs only when the endpoint is unset.
7. **Hosting: Radiant serves static archives from Taiga NFS now; the Neo4j store waits for
   the block volume.** NFS breaks Neo4j page reads, not file streaming, so the archive
   store can go on the existing Taiga mount today. Showcase artifacts are small (SMF
   Costanzo 190 MB built; the expression and morphology sets are in the same range). The
   full 3.5 TB dev tree is not published; `status` marks what is served, and `deprecated`
   phases an artifact out while its index row stays.
8. **Showcase pages are generated, like paper figures.** Every table, figure, record dump
   and query result on a showcase page comes from a script under
   `experiments/034-showcase-datasets/scripts/`, writing markdown fragments into
   `docs/source/showcase/_generated/` and SVGs into `notes/assets/images/034-showcase-datasets/`.
   Query results against the served graph run under slurm (`--neo4j`), never from a shell,
   and the cached JSON they produce is what the page embeds. The draw.io diagrams follow
   the `drawio-diagram` skill (true-size canvas, palette, font ladder) and export with the
   `Makefile.common` recipe; the image-model stylization is a later pass on the exported
   PNGs and is out of scope here.
9. **Group 2 pages wait for the re-verification.** Five Fable agents re-read Kemmeren 2014,
   Sameith 2015, Messner 2023, Mulleder 2016 and Ohya 2005 against the mirror, GEO and the
   publisher pages and grade every previously recorded claim; the consolidated note is the
   source of what the pages may state. "Proteome Mulleder" in the request is read as the
   Messner 2023 deletion-collection proteome (Mulleder is a coauthor); Mulleder 2016 is the
   amino-acid metabolome and belongs to group 3 with Cachera 2023.
10. **Sphinx on GitHub Pages is the site; the dead Dendron publish workflow is retired.**
    `publish.yml` targets a `pages` branch that does not exist and would collide with the
    Sphinx URL. API pages for every non-legacy package, a guide, a database section and a
    showcase section go under `docs/source/`. Paired dendron notes for the 235 unnoted
    modules are a separate, later fan-out (adapters first) because a note without a design
    decision in it is padding.
11. **Coverage keeps its campaign loop.** Phase 10 (`data/neo4j_cell.py`, `models/dcell.py`,
    `models/hetero_cell_bipartite_dango_gi.py`) runs under `/test-campaign` with Opus
    writers and a Fable audit; the PR-0d ratchet lands once TOTAL clears 50%.

## Approach

Tracks run in parallel in their own worktrees; agents in a shared worktree never run git.

- **T1 Re-verification (running).** Reports land in the session scratchpad, then in
  `notes/datasets.showcase-verification.2026.09.29.md` with one H2 per dataset and a
  verdict table; confirmed defects that are not already on a branch get an issue with the
  `before-next-kg-build` label. Issue `#465`.
- **T2 Documentation (running).** Worktree `docs/build-out`: conf, index, 31 API pages,
  package docstrings, guide (installation, quickstart, data model, datasets, knowledge
  graph, contributing), database landing page with the first compatibility row, README
  links. Built locally with a venv over the conda env (Sphinx 8.2.3) using the CI command.
  Issue `#466`.
- **T3 Coverage Phase 10 (running).** Worktree `plan/test-suite-buildout-p10`; the
  `test-campaign` skill steps 4 to 7 after the writers report. Campaign note gains its
  Phase 10 section.
- **T4 Versioning spine.** `torchcell_version` in releases and manifest, snapshots,
  `kg_compat_page.py`, semantic-release tag map, README link. Issue `#467`.
- **T5 Supported queries.** Registry, parser, check, pre-commit hook, CI issue job; seed
  with the three showcase queries (essentiality + SMF, expression + proteome + morphology,
  amino acids + betaxanthin) marked `supported`, and the 025 solid-growth query. Issue `#468`.
- **T6 Download API.** Packager, `DatasetArtifact`, server, client, base-class path, tests
  with a tmp artifact store, Docker compose like `docker-compose.tc-lit.yml`, deployment
  recipe for Radiant. Issue `#469`.
- **T7 Showcase pages.** Group 1 first (essentiality + SMF: the same raw query, two
  converters; essentiality becomes fitness 0 through `GeneEssentialityToFitnessConverter`),
  group 3, then group 2 after T1. Each page: what the experiment measured (diagram),
  the record dump, the tables, the graphs, the query and what it returns. Issue `#470`.

Order: T1, T2, T3 now; T4 and T5 next (T5 depends on the snapshot format from T4); T6
after T4 (artifact records name the package version and KG release); T7 group 1 after T2's
skeleton, group 2 after T1.

## Gotchas

1. semantic-release pushes a bump commit to `main` after a releasing landing; the drainer
   already tolerates it, but a worktree rebased before the bump needs one more rebase.
2. `kg_manifest.json` lives only on GilaHyper (`/scratch/projects/torchcell/database/`);
   anything CI needs must be a committed snapshot (Decision 1).
3. `stamp` raises when a served dataset lacks a content hash; the snapshot writer must run
   after `stamp`, inside the same slurm script, and every store-touching step is a slurm job.
4. Radiant: 40 GB root, no block volume, root-squashed NFS; a Python service there runs in
   Docker as uid 67392 with the archive dir bind-mounted read-only; nothing is staged on
   the root disk.
5. Kemmeren and Sameith fixes are on PR `#460` (open by decision); their showcase page
   must describe the fixed loader and say the served release predates it, until re-admission.
6. Costanzo SMF carries 8,454 phantom 26 C twins (`#410`); the group 1 page must state which
   temperature it shows and why.
7. `GEOparse.get_GEO` writes a cache; verifiers fetch GEO SOFT text over HTTP instead.
8. Mendeley is not a live dependency (memory `no-mendeley-data-source`); the Mulleder and
   Messner pages cite the mirror and the pin, not the URL.
9. The docs build imports every package under autodoc; a module that touches `DATA_ROOT`
   at import time must be mocked, not "fixed" in the docs PR.
10. CI's typed stubs differ from local (memory `ci-mypy-stubs-differ-from-local`); annotate,
    never cast.
11. Draw.io is not installed; use the extracted `.deb` binary with `xvfb-run` and put the
    input file before the Electron flags.
12. Two docs agents share one worktree: ownership is by path (`conf.py`, `index.rst`,
    `modules/**` versus `guide/**`, `database/**`); the toctree references the guide before it
    exists, and the warning clears when both land.

## Verification

- Docs: the CI build command runs locally with zero warnings from new pages; every API page
  renders members (`grep -c sig-name`); the Pages deploy on `main` shows the guide.
- Versioning: `python -m torchcell.knowledge_graphs.releases status` prints the package
  version column; `scripts/kg_compat_page.py --check` is byte-stable; a unit test builds a
  snapshot from a fake manifest and asserts the generated table rows.
- Supported queries: a test with a synthetic snapshot proves each drift kind (missing class,
  changed fingerprint, changed composite) exits 1 for `supported` and 0 for `deprecated`;
  the pre-commit hook fires on the listed paths only.
- Download API: tests with a tmp artifact store cover index, range streaming, sha256 header,
  key rejection, and a loader that unpacks an artifact and skips `process()`; the Swagger
  page lists every route.
- Showcase: every number on a page traces to a script under `experiments/034-showcase-datasets/`;
  query results were produced by a slurm job whose id is recorded in the page fragment.
- Coverage: Phase 10 diff-cover at or above 80%; the audit recorded in the campaign note.

## Open questions

- Radiant block volume: the request to NCSA is open (weekly 2026.39); until it lands the
  Neo4j store stays on GilaHyper and only the archive store goes to Radiant.
- Whether the four showcase queries should also carry `since_torchcell_version`; the
  snapshot gives it for free, so the registry records it on first validation.
