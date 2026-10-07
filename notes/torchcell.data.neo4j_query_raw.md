---
id: dhxcnez3o7iphm3s8qi4gbn
title: Neo4j_query_raw
desc: ''
updated: 1723490056267
created: 1716490548011
---
## 2024.05.23 - Issue with Hyperparameter Sweep

If not set to `readonly=True` we get this error.

```bash
wandb: Starting wandb agent 🕵️
2024-05-23 13:04:19,665 - wandb.wandb_agent - INFO - Running runs: []
2024-05-23 13:04:20,123 - wandb.wandb_agent - INFO - Agent received command: run
2024-05-23 13:04:20,123 - wandb.wandb_agent - INFO - Agent starting run with config:
 cell_dataset: {'graphs': None, 'max_size': 1000, 'node_embeddings': ['nt_window_5979']}
 data_module: {'batch_size': 8, 'num_workers': 6, 'pin_memory': True}
 models: {'graph': {'activation': 'gelu', 'hidden_channels': 512, 'norm': 'layer', 'num_node_layers': 4, 'num_set_layers': 2, 'out_channels': 64, 'skip_node': True, 'skip_set': True}, 'pred_head': {'activation': None, 'dropout_prob': 0, 'hidden_channels': 0, 'norm': None, 'num_layers': 1, 'out_channels': 1, 'output_activation': None}}
 regression_task: {'alpha': 0, 'boxplot_every_n_epochs': 5, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 1, 'learning_rate': 1e-06, 'loss': 'mse', 'target': 'fitness', 'weight_decay': 0}
 trainer: {'accelerator': 'gpu', 'max_epochs': 50, 'strategy': 'auto'}
2024-05-23 13:04:20,128 - wandb.wandb_agent - INFO - About to run command: /usr/bin/env python experiments/smf-dmf-tmf-001/deep_set.py
2024-05-23 13:04:25,138 - wandb.wandb_agent - INFO - Running runs: ['49txm81b']
wandb: Currently logged in as: mjvolk3 (zhao-group). Use `wandb login --relogin` to force relogin
wandb: WARNING Ignored wandb.init() arg project when running a sweep.
wandb: wandb version 0.17.0 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.16.0
wandb: Run data is saved locally in /scratch/bbub/mjvolk3/torchcell/wandb-experiments/3693862/wandb/run-20240523_130534-49txm81b
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run rose-sweep-6
wandb: ⭐️ View project at https://wandb.ai/zhao-group/torchcell_smf-dmf-tmf-001_deep_set_1e04_00
wandb: 🧹 View sweep at https://wandb.ai/zhao-group/torchcell_smf-dmf-tmf-001_deep_set_1e04_00/sweeps/pk6ek5mc
wandb: 🚀 View run at https://wandb.ai/zhao-group/torchcell_smf-dmf-tmf-001_deep_set_1e04_00/runs/49txm81b
Processing...
Done!
Starting Deep Set 🌋
wandb_cfg {'hydra_logging': {'loggers': {'logging_example': {'level': 'INFO'}}}, 'wandb': {'mode': 'offline', 'project': 'torchcell_test', 'tags': []}, 'cell_dataset': {'graphs': None, 'node_embeddings': ['codon_frequency'], 'max_size': 1000.0}, 'data_module': {'batch_size': 16, 'num_workers': 6, 'pin_memory': True}, 'trainer': {'max_epochs': 10, 'strategy': 'auto', 'accelerator': 'gpu'}, 'models': {'graph': {'in_channels': None, 'hidden_channels': 128, 'out_channels': 32, 'num_node_layers': 0, 'num_set_layers': 3, 'norm': 'batch', 'activation': 'gelu', 'skip_node': True, 'skip_set': True}, 'pred_head': {'hidden_channels': 0, 'out_channels': 1, 'num_layers': 1, 'dropout_prob': 0.0, 'norm': None, 'activation': None, 'output_activation': None}}, 'regression_task': {'target': 'fitness', 'boxplot_every_n_epochs': 1, 'learning_rate': 0.01, 'weight_decay': 1e-05, 'loss': 'mse', 'alpha': 0.01, 'clip_grad_norm': True, 'clip_grad_norm_max_norm': 10}}
data/go/go.obo: fmt(1.2) rel(2023-07-27) 46,356 Terms
/scratch/bbub/mjvolk3/torchcell/data/scerevisiae/nucleotide_transformer_embedding/processed/nt_window_5979.pt
=============
node.embeddings
{'nt_window_5979_max': NucleotideTransformerDataset(6607)}
=============
-------------------------
dataset_root:/scratch/bbub/mjvolk3/torchcell/data/torchcell/experiments/smf-dmf-tmf_1e03
-------------------------
================
raw root_dir: /scratch/bbub/mjvolk3/torchcell/data/torchcell/experiments/smf-dmf-tmf_1e03
================
Error executing job with overrides: []
Traceback (most recent call last):
  File "/scratch/bbub/mjvolk3/torchcell/experiments/smf-dmf-tmf-001/deep_set.py", line 252, in main
    cell_dataset = Neo4jCellDataset(
                   ^^^^^^^^^^^^^^^^^
  File "/projects/bbub/mjvolk3/torchcell/torchcell/data/neo4j_cell.py", line 264, in __init__
    self.raw_db = self.load_raw(uri, username, password, root, query, self.genome)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/projects/bbub/mjvolk3/torchcell/torchcell/data/neo4j_cell.py", line 326, in load_raw
    raw_db = Neo4jQueryRaw(
             ^^^^^^^^^^^^^^
  File "<attrs generated init torchcell.data.neo4j_query_raw.Neo4jQueryRaw>", line 19, in __init__
    self.__attrs_post_init__()
  File "/projects/bbub/mjvolk3/torchcell/torchcell/data/neo4j_query_raw.py", line 147, in __attrs_post_init__
    self.env = lmdb.open(self.lmdb_dir, map_size=int(1e12))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
lmdb.InvalidParameterError: mdb_txn_begin: Invalid argument
```

This indicates something wrong with accessing `lmdb`. I believe it to be readonly. This should allow multiple processes to access.

```python
self.env = lmdb.open(self.lmdb_dir, map_size=int(1e12), readonly=True)
```

## 2024.08.12 - Neo4jQueryRaw is Immutable

```python
neo4j_db[0] = None
Traceback (most recent call last):
  File "<string>", line 1, in <module>
TypeError: 'Neo4jQueryRaw' object does not support item assignment
```

## 2026.09.19 - Which knowledge-graph version a query reads

`Neo4jQueryRaw` no longer opens the session on a hardcoded `torchcell` database. It
carries a `version` attribute (None means `Neo4jConnectionSettings.version`, i.e.
`TORCHCELL_KG_VERSION`, default `latest`) and resolves it at fetch time with
`torchcell.knowledge_graphs.releases.resolve_database`: `latest` and `pinned` are
aliases on the served DBMS, a release id such as `2026.09.17-7715ee35` or a version
such as `1.0` is looked up in the release nodes, and an unknown name raises. The
resolved name is logged beside the version. Callers that pass uri/user/password are
unchanged; pinning a query to a release is `version="2026.09.17-7715ee35"` or one line
in `.env`.

## 2026.09.27 - Three fixes from the Phase 7 exact tests

Phase 7 of [[plan.test-suite-buildout.2026.09.25]] pinned three defects in [[tests.torchcell.data.test_neo4j_query_raw]] and the audit confirmed them against the source; this entry records the fixes (user go-ahead "fix the raw query bugs").

- `compute_phenotype_label_index` read `record["experiment"].phenotype.label`; the schema field is `label_name`, so `phenotype_label_index` raised `AttributeError` whenever it had to compute (only a JSON file from an earlier build was ever read). It now reads `label_name`; three fitness records index to `{"fitness": [0, 1, 2]}` and the file is written.
- The parallel reference-index path (`parallel_hash_computation` and this module's `compute_experiment_reference_index_parallel`) read the key `"reference"`, while raw-query records carry `"experiment_reference"` (the key the sequential path reads), so `compute_experiment_reference_index(records, num_workers > 0)` raised `KeyError`. Both now read `experiment_reference`, and the sequential, `num_workers=1` and helper results are equal. The helper of the same name that `torchcell.data` exports comes from `experiment_dataset.py`, whose items use `reference`, and is unchanged.
- `__len__` returned inside its transaction, so the `close_lmdb()` after it never ran and the environment stayed open. It now reads the count and closes. `_get_records_by_slice` calls `len` and then reads through the environment on a thread pool, so it now reopens with `_init_lmdb()` after taking the length.

## 2026.10.01 - Driver closed in finally; empty query writes no store

`fetch_data` now closes the driver in a `finally`, so a consumer that closes the generator early still closes it. `process()` reads the first record before opening the LMDB store for writing; zero records raise `EmptyQueryResultError` with no `data.mdb` on disk, so a retry on the same root runs the query again (before, the empty store was left behind, the gene-set setter failed, and the next construction reused the empty store). A failure in the middle of a non-empty query still leaves a partial store; that case is not addressed. Issue #541; tests `test_a_consumer_that_stops_early_still_closes_the_driver`, `test_a_query_with_no_records_writes_no_store_and_a_retry_reruns_the_query`.

## 2026.10.01 - Review fix: staged build

A query that failed partway (for example after one record) left a partial `data.mdb` and an open write handle; the next construction ran no query and served the truncated store. `process()` now writes into `raw/lmdb.partial` and moves it onto `raw/lmdb` with `os.replace` only after every record is written and the environment is closed. On any failure while writing it closes the environment, removes the staging directory it created, and re-raises. A staging directory found at the start (a build killed outright) raises `StaleStagingStoreError` before the query runs and is left untouched for inspection, the no-fallback reading: it may hold any prefix of the query, so it is neither reused nor silently replaced. Tests: `test_a_query_that_fails_midway_leaves_no_store_and_a_retry_rebuilds_it`, `test_a_leftover_staging_store_is_refused_before_the_query_runs`.

## 2026.10.06 - The pairing gate at connect

`_connect` now reads the resolved database's `KgRelease` node and runs `releases.require_paired` against the installed schema surface (`schema_deps.load_default_surface`) before any driver is opened. A release whose closures the installed `torchcell` does not reproduce for every served dataset, a release that recorded no closure for a served dataset, or a store with no release node raises `IncompatibleReleaseError` naming the paired package and the remedy; nothing is queried and no store is written. Each fetch worker connects on its own, so the check runs once per worker (one node read plus a 0.3 s parse of `schema.py` and `pydant.py`). The tests fake `releases.read_release` with a node whose closure is the installed surface's own fingerprints, and pin the three refusals.

## 2026.10.07 - The artifact resolvability gate (artifact tier phase 4)

Decision D5 of the artifact-tier plan. `process` now refuses to build a store whose records point at off-graph bytes that no artifact source holds, the same rule as a missing interned constant.

- `_render_batch` collects each record's distinct `ArtifactRef`s (`torchcell.artifacts.distinct_refs` over the validated experiment and reference) and carries them as a fifth element of the row. A record whose JSON lacks `ARTIFACT_REF_MARKER` (`"tier":`, a key every `ArtifactRef` dump writes) is not walked: the walk measured 0.256 ms per record on a 9.8 kB fitness record against 0.0014 ms for the substring test, beside a raw stage of about 0.3 ms per record (timeit over `test_neo4j_query_raw_single_pass._experiment(3)` and `_reference(3)`, 2000 repeats, 2026.10.07).
- `_commit_rows` runs the gate before it opens the write transaction, in the writing process on both paths (single session and partitioned, where the refs travel back from the forked workers in the rows). Each distinct `(tier, key, path, sha256)` is resolved once per `process` run; the set is `_checked_artifacts`, reset at the start of `process`.
- The resolver is a seam, `resolver: ArtifactResolver`, defaulting to `require_resolvable` = `torchcell.artifacts.resolve(ref, materialize=False)` (manifests only, nothing downloaded). A miss (`ArtifactUnresolvableError`) becomes `UnresolvableArtifactError(RuntimeError)` naming the record key (`data_<i>`), the `tc://` string and sha256, and the resolver's numbered source list; an `ArtifactIntegrityError` propagates unchanged. Either way the staging store is removed and nothing is left at `raw/lmdb`.
- New fields: `artifact_check: bool = True` (False skips the gate), `artifact_data_root: str | None = None` (None is `DATA_ROOT` after `load_dotenv()`), `artifact_client: RemoteSource | None = None` (None is tc-data from `TC_DATA_URL` when the local tier misses), `resolver`.
- `materialize(ref) -> Path`: `torchcell.artifacts.materialize` with the same `artifact_data_root` and `artifact_client`.

Records with no refs are untouched: byte-identical LMDB values, no resolver call. Tests: `tests/torchcell/data/test_neo4j_query_raw_artifacts.py` (12; real `FitnessExperiment` records whose `SequenceVariantPerturbation`s carry `sequence_ref`; no schema class is subclassed in tests, because the ontology tree tests walk the live hierarchy in the same process; no reference-side schema class holds a ref after phase 3, so the reference half of the walk is not pinned on a stored record). The attrs state pin in `test_close_lmdb_is_idempotent_and_a_closed_view_pickles_and_reopens` gained the five new fields.
