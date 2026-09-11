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

Result of the passing run: 12 node labels / 12 edge types, 18,688 distinct node ids, 0 external endpoints, 0 relationships between existing nodes; `Entity.id` constraint created and online within the poll interval on 7.18 M nodes; incremental import 5 s; nodes 7,180,465 to 7,199,149 (+18,684: the S288C genome, the YPD liquid medium, the 30 C temperature and the control environment already existed and were matched, the S288C genome node now spans 18 datasets); 6188 of 6188 Nadal experiments attached to the new Dataset node; the NaCl `environment perturbation` node connects to one environment carrying 3097 experiments, the control environment carries 3091; a stored experiment and its reference round-trip through `EXPERIMENT_TYPE_MAP["pseudobulk_expression"]`; all 6190 phenotype nodes carry `dispersion` (mean 1.09) and `n_cells` (max 1894). The manifest recorded the admission and the CSVs were archived.
