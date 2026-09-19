---
id: jvl4rn8x4g8ty0evc339vg4
title: Incremental Admission
desc: ''
updated: 1789101186139
created: 1789101186139
---

## 2026.09.10 - Adding a Dataset to the Served Graph Without Rebuilding the Others

Until now the served Neo4j store was remade in one full build whenever anything changed (CLAUDE.md rebuild policy). This note records the incremental path built with Nadal-Ribelles 2025 as the first dataset, the measurements that fixed its design, and the rule that decides when the full rebuild is still required.

### The rule

A dataset is admitted incrementally when nothing already served would change. The served store is a function of three things, and the store's manifest (`torchcell/knowledge_graphs/kg_manifest.py`, written to `$BUILD_ROOT/database/kg_manifest.json`, one file per physical database, never in git) records all three per served dataset:

1. **Schema contract.** The closure of pydantic symbols each served loader depends on, fingerprinted by `torchcell.provenance.schema_deps` (contract hash: fields, validators, bases; docstrings and field order excluded). Any drift for a served dataset means its stored records would serialize differently today. Incremental import cannot update or delete nodes, so the answer is the full rebuild. This is the check the user asked for: a new dataset that forces a change to a class imported elsewhere is caught here, and the report names the datasets and the symbols.
2. **Graph schema.** `biocypher/config/torchcell_schema_config.yaml`. A node class present in the store must keep its exact property set. An edge class may gain source or target labels (a new phenotype family joining `phenotype member of` is additive); it may not lose one. New classes are additive.
3. **Adapter code.** Node ids are sha256 of what `CellAdapter` serializes, so adapter changes can move ids silently. The check is method level: `CellAdapter` methods are fingerprinted individually (docstrings stripped, the method tables in `__init__` stripped because they are compared separately), and a changed method blocks only if a served adapter conf enables it or if it is plumbing outside the method tables. A changed per-dataset adapter file or conf blocks only for a served dataset. A block can be acknowledged with a written reason (`--ack-adapter-drift`), which is recorded in the manifest's event log.

Plus, for the new dataset: it is in `dataset_adapter_map`, its dev-tree LMDB is fresh against the working-tree schema (`preprocess/build_manifest.json`), and every phenotype node method its conf enables is declared in the graph schema (BioCypher drops undeclared classes silently; `tests/torchcell/knowledge_graphs/test_adapter_schema_consistency.py` now enforces this for every mapped dataset).

```bash
python -m torchcell.knowledge_graphs.kg_manifest --manifest $BUILD_ROOT/database/kg_manifest.json \
    admit --dataset NadalRibellesPerturbSeq2025Dataset --data-root $DEV_DATA_ROOT --report admission.json
```

### What the rehearsals measured (Neo4j 5.26.28 Enterprise, `neo4j-admin database import incremental`)

Rehearsed on a toy store in a throwaway container (`scratchpad/toy/rehearse.sh`, not committed) before any production step:

- `--schema=<cypher>` is refused: "Applying schema commands during incremental import is not currently supported". The uniqueness constraint has to exist before the import, created through Cypher on the online database.
- The served database is read-only by server default (`server.databases.default_to_read_only=true`), and a constraint create fails there. The dynamic setting `CALL dbms.setConfigValue('server.databases.writable', 'torchcell')` flips it to read-write with no restart; `SHOW DATABASE` reports `read-write`, the constraint is created, and setting it back to `''` returns `read-only` (a write then fails again). No container restart, no config edit.
- One constraint per label fails: "Multiple different indexes for group global id space". BioCypher writes every node into the single global id space and its edge files mix endpoint labels row by row, so per-label id groups are impossible. The constraint keys a label every node carries: `Entity`, the BioLink root BioCypher stacks onto every `:LABEL` list. The incremental header becomes `id:ID{label:Entity}` with the redundant `id` column `id:IGNORE`; global uniqueness of `id` holds in the served store by construction (the full import ran one global id space with `--skip-duplicate-nodes`).
- With the single `Entity.id` constraint the toy increment imports cleanly. An incoming node whose id already exists is matched, not replaced: the existing node keeps its properties (the toy re-sent `e2` with a different payload and the stored payload was unchanged afterwards), and `import.report` lists each such id as "defined more than once in group 'global id space'", which is informational. New nodes attach to existing ones through relationships in either direction.
- A relationship re-sent between two EXISTING nodes is duplicated (the toy's `e2 -> D1` edge count went from 1 to 2). Incremental import does not dedup relationships, which is why `prepare_incremental_import` refuses an increment containing any relationship whose both endpoints are external.
- A relationship whose endpoint is missing aborts the import with `--skip-bad-relationships=false --strict=true` (the data source and the missing id are named). The full build's `--skip-bad-relationships=true` would have dropped it silently; the increment keeps the abort.
- After a failed incremental import the toy database restarted with its prior content intact, but Neo4j's own warning is that the store "is likely in an unusable state". The `/bulk/biocypher-out` archive holds every full and incremental import ever applied, so the store is regenerable: re-run the last full import call, then each archived increment in order.

### Pipeline (`database/slurm/scripts/gilahyper_increment_kg-slurm_docker.slurm`)

1. ADMIT (above). 2. STAGE the dev-tree LMDB into the uid-7474 build tree via a container (the previous copy is moved aside, not deleted). 3. REFRESH the build tree's `biocypher/` config copy and container `.env` from the checkout (`directory_setup --env-file`, because a worktree lacks the untracked env file). 4. GENERATE this dataset's CSVs in an ephemeral python container (no Neo4j server) with `kg_increment.yaml` (`datasets=[...]`, `import_mode: incremental`); `incremental_import.prepare_incremental_import` rewrites headers, writes the constraint file, the call script and a reference analysis, and refuses an increment with a relationship between two already-existing nodes (incremental import would duplicate it). 5. IMPORT inside the serving container: constraint (writable override), `STOP DATABASE`, `neo4j-admin database import incremental` as the neo4j user, `START DATABASE`. 6. VERIFY the new `Dataset` node's `ExperimentMemberOf` count equals the CSV row count. 7. RECORD in the manifest and archive the CSVs.

The serving container must mount `$SERVE_ROOT/biocypher-out` (the launch in `scripts/migrate_storage_tiers.sh` did not); the runner refuses otherwise.

### Nadal-Ribelles 2025 specifics

- Loader existed (`torchcell/datasets/scerevisiae/nadal_ribelles2025.py`); its dev LMDB was stale (pre `Media.is_synthetic`, 2026-07-14) and was rebuilt with `python -m torchcell.database.build_dataset_lmdb --dataset NadalRibellesPerturbSeq2025Dataset` (slurm job 1688: 6188 records, 630 s, 45.6 G peak RSS), then L0-L4 verified with the RNA-seq family verifier: PASS (6188 records, 3150 strains, 35,211,710 log2 ratios finite, 1.000 gene containment).
- Graph representation added (all additive): `pseudobulk expression phenotype` node class (log2 ratios as JSON, `dispersion` and `n_cells` typed), `environment perturbation` node class + `environment perturbation member of` edge so the 0.4 M NaCl osmostress condition is a queryable node rather than a value inside the environment's `serialized_data`, the adapter methods for both, `NadalRibellesPerturbSeq2025Adapter` + conf, and the `dataset_adapter_map` entry.
- Admission check against the served store (built at commit 513cbfa1, job 1558): ADMISSIBLE. 29 closure symbols, 6 novel (the pseudobulk trio, `EnvironmentPerturbation`, `SmallMoleculePerturbation`, `Solvent`), 23 shared with served datasets and all unchanged; 0 served datasets with schema drift; graph schema 0 changed, 3 added; adapter drift touching served datasets: none.

### Dress rehearsal (2026-09-11, `tc-neo4j-rehearsal`)

A separate container (bolt 7688, store under `/db/rehearsal`) was full-imported from the archived 2026-07-22 CSVs (35 datasets, 7,180,465 nodes, 21,045,597 relationships, import 3 min) and its manifest bootstrapped; the runner was then executed against it with the worktree code (`TORCHCELL_PIP_REF=src`). Five attempts were needed; each failure was a runner defect, fixed and folded back: pip cannot build from a read-only bind mount (the worktree is now handed over as a host-built wheel), the target database name lives in the BioCypher config rather than a hydra key, `csv` refuses the ~1 MB `serialized_data` fields at its 128 KiB default limit (the analysis now streams rows with the limit lifted), containers launched with `--entrypoint bash` run as root rather than the neo4j user (ownership is now set explicitly), and `neo4j-admin database import incremental` takes the database name FIRST because `--relationships=<files>...` swallows a trailing positional ("File 'torchcell' doesn't exist").

The passing run also exposed the one defect the toy could not: the increment re-emits the shared context nodes its records use (the S288C genome, the YPD medium, the 30 C temperature, and the control environment that other datasets had already produced) together with THEIR relationships, so `MediaMemberOf` and `TemperatureMemberOf` into the shared control environment were created a second time (1 to 2 each), while the node rows were matched. `filter-existing-edges` (`torchcell.knowledge_graphs.incremental_import`) now runs in the runner after the constraint exists and before the database is stopped: every relationship row is looked up through the `Entity.id` index (40,291 rows in 8 s on the rehearsal store) and the edge part files are rewritten without the rows that already exist (originals under `unfiltered/`; a sibling suffix would be picked up by neo4j-admin's `part.*` regex). Re-running the whole admission against a store that already holds the dataset then drops every relationship row and adds no node, which is the idempotency check recorded below. That rerun found one more limit: under `--skip-duplicate-nodes` every id the store already holds counts as a "bad entry", and neo4j-admin's default `--bad-tolerance` of 1000 aborted the import at 1008 duplicates ("Too many bad entries"); the call now lifts the tolerance, since an increment may legitimately share thousands of nodes with the store.

Idempotency rerun (manifest entry removed, dataset re-admitted to the store that already held it): the filter dropped all 40,291 relationship rows, the import added 0 nodes, relationships stayed at 21,085,888, and 6188 experiments still verified.

Result of the passing run: 12 node labels / 12 edge types, 18,688 distinct node ids, 0 external endpoints, 0 relationships between existing nodes; `Entity.id` constraint created and online within the poll interval on 7.18 M nodes; incremental import 5 s; nodes 7,180,465 to 7,199,149 (+18,684: the S288C genome, the YPD liquid medium, the 30 C temperature and the control environment already existed and were matched, the S288C genome node now spans 18 datasets); 6188 of 6188 Nadal experiments attached to the new Dataset node; the NaCl `environment perturbation` node connects to one environment carrying 3097 experiments, the control environment carries 3091; a stored experiment and its reference round-trip through `EXPERIMENT_TYPE_MAP["pseudobulk_expression"]`; all 6190 phenotype nodes carry `dispersion` (mean 1.09) and `n_cells` (max 1894). The manifest recorded the admission and the CSVs were archived.

### Production admission (2026-09-11, slurm job 1722, 24 min 40 s wall)

Prerequisite done once: `tc-neo4j-readonly` was relaunched with `/db/database/biocypher-out` mounted (`scripts/migrate_storage_tiers.sh` now includes the mount); the outage was the JVM restart, about 30 s, node count 83,048,247 before and after. The production manifest was bootstrapped from the live store at the full build's commit (513cbfa1, job 1558) into `/scratch/projects/torchcell/database/kg_manifest.json`; admission at commit 6d4c0446: ADMISSIBLE.

The run, driven from the worktree (`TORCHCELL_PIP_REF=src`): admission passed; the dev LMDB was staged; 18,688 node ids and 40,291 relationship rows generated; the `Entity.id` uniqueness constraint was created on the 83 M-node store through the writable override and came online within the first poll; the existing-edge filter dropped exactly 2 rows (`MediaMemberOf` and `TemperatureMemberOf` into the control environment that da Silveira 2014 had already produced); `STOP DATABASE`, incremental import (build 5.7 s, merge 11.4 s), `START DATABASE`. Verified live: nodes 83,048,247 to 83,066,931 (+18,684), relationships 310,922,919 to 310,963,208 (+40,289), 36 datasets, 6188 of 6188 Nadal experiments, one relationship of each type into the shared control environment (no duplicates), the S288C genome node spanning 18 datasets, the NaCl environment perturbation node on 3097 experiments, a stored record round-tripping through the pydantic classes, and `dispersion` on all 6190 pseudobulk phenotype nodes. The manifest now lists 36 datasets (Nadal `incremental`, `2026-09-11_07-02-12`) and the CSVs are archived at `/bulk/biocypher-out/2026-09-11_07-02-12` (2.3 GB, 60 files).

Left behind for a hand purge: `/db/rehearsal` (the rehearsal store plus four increment directories, 77 GB, all regenerable; its container `tc-neo4j-rehearsal` was removed), and the `*.superseded.*` copies of the Nadal LMDB that repeated staging left in the build tree (`/scratch/projects/torchcell/database/data/torchcell/nadal_ribelles_perturbseq2025`, uid 7474).

## 2026.09.12 - Batch Admission: Several Datasets in One Increment

Admitting one dataset per run costs a full pipeline (stage, generate, constraint, filter, stop, import, start, verify, record) per dataset, and the store is stopped once per dataset. A batch runs the same pipeline once for several datasets: one CSV set, one constraint step, one `neo4j-admin database import incremental`, one manifest event.

### The batch verdict rule

Every member is checked by the unchanged `check_admission` against the SERVED manifest, and the batch is admissible only when every member is. Nothing is relaxed and no second rule is added, because a member is checked in isolation: none of the members is in the store, so nothing a member ADDS (a schema symbol, a graph node class, an adapter method) can appear as drift for another member. Two members introducing the same new class is therefore additive, and the report only names it.

The batch report (`BatchAdmissionReport`, pydantic, wrapping each member's `AdmissionReport`) carries two derived views:

- `co_introduced_symbols`: schema symbol to the members that introduce it, for symbols more than one member brings. Informational.
- `changed_symbol_importers`: for a symbol that changed relative to the served manifest, the SERVED datasets that import it. This is the inverse of the members' `stale_served` and is not a second computation, so a block reads as "served datasets X, Y import class Z", which is the sentence that decides whether the honest answer is a full rebuild. `format_batch_report` prints it together with the batch members whose own closure carries the symbol.

### CLI

`--dataset` repeats, and each value may be a comma-separated list:

```bash
python -m torchcell.knowledge_graphs.kg_manifest --manifest $BUILD_ROOT/database/kg_manifest.json \
    admit --dataset ADataset --dataset BDataset,CDataset --data-root $DEV_DATA_ROOT --report admission.json
```

One dataset keeps the old behavior exactly: the same `format_report` output, the same `AdmissionReport` JSON, the same exit codes (0 admissible, 1 blocked). More than one prints the batch report (every member's report indented, then the two views) and writes a `BatchAdmissionReport` JSON; the exit code is 1 when any member blocks.

`record` reads either shape (`load_report` recognizes a batch by its `members`) and takes the live experiment counts as `--n-experiments`: a bare count for a single dataset (`--n-experiments 6188`, the unchanged call), or `NAME=COUNT` repeated, one per member, for a batch. The names must be exactly the batch's members. A batch is recorded under ONE `incremental_admission` event listing every member, all sharing the `biocypher_out` directory they were imported from.

### Runner

`database/slurm/scripts/gilahyper_increment_kg-slurm_docker.slurm` takes `DATASET_CLASSES`; `DATASET_CLASS` still works and is treated as a one-element batch.

```bash
sbatch --export=ALL,DATASET_CLASSES=ADataset,BDataset,CDataset \
    database/slurm/scripts/gilahyper_increment_kg-slurm_docker.slurm
```

Per-dataset stages are looped (slug resolution, LMDB staging, verification); the shared stages run once (`datasets=[A,B,C]` in the one ephemeral generate container, the constraint, the existing-edge filter, the import, the record). Output file names keep the dataset name for a single admission and become `batch<N>` for a batch.

The verification expectation is now per member: the ExperimentMemberOf rows whose `:END_ID` is that dataset's `Dataset` node id (the id IS the class name, which is what the live verification query has always matched on), written to `${JOB}_expected_<tag>.txt`. Their sum must equal the increment's Experiment node rows, since an experiment belongs to exactly one dataset; a mismatch aborts before the import. Each member's live `ExperimentMemberOf` count is then compared against its own expectation and passed to `record` as `NAME=COUNT`.

The CSV generator needed no change: `create_scerevisiae_kg_small` already builds every dataset named in `datasets`, and `kg_increment.yaml` already documented it as one or a few.

Measured: `pytest tests/torchcell/knowledge_graphs -x -q` 26 passed; mypy clean on `kg_manifest.py`. The CLI was exercised against a COPY of the production manifest in a scratch directory (the production file itself read-only): single admit of `Bloom2019Dataset` ADMISSIBLE with output identical in shape to before, batch admit of `Bloom2019Dataset,SmfKuzmin2020Dataset` BLOCKED with exit 1 (the second is already served and its dev LMDB is stale), and both the batch and the single `record` paths writing the copy. No batch has been imported into a store yet.

## 2026.09.12 - The value surface: what the gate could not see

The admission check fingerprints the schema closure of every served dataset, the BioCypher
graph schema and the `CellAdapter` methods. All three are CODE and SCHEMA. The shared VALUES
that media and compound node ids are content-addressed from were unwatched, and they are the
two files the environment work edits most: `torchcell/datamodels/media.py` (the recipes every
dataset's `Media` resolves to) and `torchcell/datamodels/compound_identity_table.json` (the
curated identity rows), plus `torchcell/datamodels/compound_identity.py`, the resolver that
turns a name into a `Compound`.

Adding a component to YPD, or filling one compound's InChIKey, changes the content the
adapter serializes for that node without changing one line of adapter code or one schema
fingerprint. The served node keeps the id it was written under, the dataset being admitted
writes a node with a new one, and the store ends up holding two YPDs that no query joins.
Incremental import cannot update the old node, so this is exactly the class of change the
gate exists to catch.

`KgBuildManifest.value_surface` is now `relpath -> sha256 of file content` for those three
files, recorded by `bootstrap_manifest` (at the build commit, via `git show`, skipping files
that did not exist then) and by `_adopt_current_surfaces` on every `record` / batch `record`.
`check_admission` compares it and reports:

- `VALUE SURFACE CHANGED: ['torchcell/datamodels/media.py']` as a BLOCK, cleared by
  `--ack-value-drift '<why served ids are unchanged>'`, which is stored in the admission
  event as `acknowledged_value_drift` exactly the way an adapter-drift acknowledgment is.
- A file that joined the surface AFTER the build is `value_surface_added`: additive and only
  reported, since nothing served was built from it.
- A recorded file that is now missing counts as CHANGED, not as silence.

## 2026.09.14 - The full rebuild the gate asked for, and what the sweep found first

The batch admission of the 14 unserved datasets BLOCKED on every served dataset
([[plan.serve-all-50.2026.09.12]]), so the next store is a full build. Before launching it,
a sweep over the 50 mapped dev stores found that the stores themselves were not ready: the
`build_manifest.json` freshness check (`torchcell.provenance.build_manifest.check_manifest`
against `load_default_surface()`) read 28 of them STALE and 8 unmanifested. The 28 were built
in July, before the typed media library, the compound identity table and the `DoseBasis`
change, and their closures drift on `Compound`, `DoseBasis`, `Environment`, `Media`,
`MediaComponent` and `Phenotype`; the 8 (Caudal, both Sameith, both synth-leth DB, Ozaydin,
Cachera, Yoshida) predate build manifests. A full build that read those LMDBs would have
serialized the OLD media and compound content under new node ids, which is the join problem
the campaign set out to fix. Only the 14 stores the campaign rebuilt on 2026.09.13 read fresh.

So the full build is a chain, all on slurm: rebuild the 36 stores from their raw files
(`torchcell.database.build_dataset_lmdb`, one job each, the move-aside recipe of the
campaign), then the freshness gate over all 50 plus the L0-L4 sweep, then the all-50 adapter
rehearsal at 1,000 records per dataset, and only then the store build. The build itself is
`database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm`
([[database.slurm.scripts.gilahyper_live_rebuild-slurm_docker]]): CSVs are generated from
the dev tree mounted read-only, imported into a fresh data root on `/db`, validated (Dataset
count and per-dataset `ExperimentMemberOf` rows against the CSVs), and only then swapped
under `tc-neo4j-readonly` by two directory renames. The old store and the old
`kg_manifest.json` stay beside the new ones; the new manifest is bootstrapped from the live
store, so the next admission is judged against what the store actually holds.

One more thing the sweep turned up: 174 files under the dev tree's `preprocess/` directories
were owned by uid 7474 (hardlinked with the build tree, link count 2), so the first ten
rebuild jobs died on `PermissionError` writing `data.csv` / `gene_set.json`. They were
chowned to the dev user through a root container and resubmitted.

`format_report` gains one line, `value surface: unchanged (3 files)` /
`value surface: CHANGED: <files> (acknowledged: ...)` /
`value surface: not recorded (...)`.

Old manifests load unchanged: the field defaults to empty, and an empty stored surface reports
"value surface not recorded (manifest predates the value surface; nothing to compare against,
so this does not block)" instead of blocking. The production manifest will pick the surface up
at its next `record`; until then this check is informational for that store.

The hash is of file CONTENT, not of a parse of it. A comment-only edit therefore reads as
drift and needs a one-line acknowledgment, which is recorded; the reverse error, a value edit
the gate misses, silently splits a node and cannot be repaired incrementally.

Measured: `pytest tests/torchcell/knowledge_graphs -x -q` 30 passed (4 new: content hashing +
changed/added/missing drift, the block and the acknowledgment round trip, the unrecorded
surface reporting rather than blocking, and an old manifest JSON without the field loading);
mypy and ruff clean on `kg_manifest.py`. The batch path threads `--ack-value-drift` the same
way it threads `--ack-adapter-drift`.

## 2026.09.19 - An admission is a minor release

Stage 7 of `gilahyper_increment_kg-slurm_docker.slurm` now computes the admitted
datasets' content hashes (`database/scripts/kg_content_hashes.sh` against the served
container), stamps the manifest as an incremental release (minor version bump, release
id `<today>-<this commit>`, untouched datasets keep their hashes because incremental
import never touches existing nodes), rewrites the store's `KgRelease` node through the
same `server.databases.writable` window the constraints use, and prints the release
table. A served manifest without a version stops the runner: stamp the served release
first. See [[torchcell.knowledge_graphs.releases]].
