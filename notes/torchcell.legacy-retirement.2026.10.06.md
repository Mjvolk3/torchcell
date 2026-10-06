---
id: 1i6k0rhml571hqu5j6row1s
title: '06'
desc: ''
updated: 1791317187660
created: 1791317187660
---

## 2026.10.06 - Legacy retirement

### Decision

The owner approved, on 2026-10-06, retiring every module that `scripts/legacy_partition.py` classifies as unreached from every root (tests, scripts, database, experiments 016 and later, the Makefile, pre-commit, the workflows and the `[project.scripts]` entry points), after a two-reviewer pass over the cluster. The modules went to the graveyard through `scripts/deprecate.sh`, not into a `torchcell/legacy/` package: the plan's Decision 22 relocation is replaced by retirement behind a tag.

- Tag: the annotated tag `legacy-pre-move-2026.10` on `03b9d6bc8` holds the whole tree as it was. Frozen experiments 001 to 015 and the `DEPRECATED_*` folders rerun from that tag and were not edited.
- Graveyard: `/scratch/projects/torchcell-deprecated/2026-10-06_<HHMMSS>__<basename>/`, one directory per retired path, each with a `DEPRECATION.txt` recording the original path, host, user, git head and reason.
- Counts: 110 modules (the 109 of the reviewed list plus `torchcell/metrics/__init__.py`, an empty package once its two members left), 46,207 lines; 3 test files; 29 notes (the 27 paired module notes and the 2 notes of the NaN-tolerant metric tests); 2 generated API pages (`docs/source/modules/dataset_readers.rst`, `metrics.rst`).
- Two modules were live only because a test imported them: `torchcell/datasets/cell.py` (by `tests/torchcell/datasets/test_esm2.py`, now rewritten onto the live `torchcell.data.neo4j_cell.create_embedding_graph`) and `torchcell/trainers/fit_int_gat_diffpool_inception_regression.py` (by a test in `test_nan_tolerant_metrics.py` that pinned its broken import).
- The two NaN-tolerant metric modules (`torchcell/metrics/nan_tolerant_metrics.py`, `nan_tolerant_classification_metrics.py`) had only retiring 003 trainers as importers. The owner ruled them the wrong design: NaN masking belongs in the trainer, before a plain torchmetrics call (issue #617).

### Kept

| Module | Reason |
|---|---|
| `torchcell/datasets/scerevisiae/spell.py` | imported by `experiments/015-spell/scripts/run_phase1_spell_analysis.py`; needs a test |
| `torchcell/knowledge_graphs/build_time_projection.py` | imported by `experiments/database/scripts/project_build_time.py`, current KG tooling; needs a test |
| `torchcell/paper/signal.py` | live CLI over `paper.tables` (`python -m torchcell.paper.signal`, named in [[experiments.database.signal-gzip-procedure]] and [[paper.supported-datasets-and-databases]]); needs a test |
| `torchcell/adapters/conf/__init__.py` | the package holds live YAML package data |
| `torchcell/knowledge_graphs/conf/__init__.py` | the package holds live YAML package data |
| `torchcell/nn/stoichiometric_hypergraph_conv.py` | live and tested; kept although it loses every user (below) |

`scripts/legacy_partition.py --check` still exits 1 after the retirement, on exactly these five kept modules (spell, build_time_projection, signal and the two `conf/__init__.py`) and on `torchcell/models/constants.py`, which is init-only (only `torchcell/models/__init__.py` imports it) and was not on the reviewed list. Each needs a root (a test, or a root-side import) or an owner decision before `--check` can gate CI.

### Orphan: stoichiometric_hypergraph_conv

`torchcell/nn/stoichiometric_hypergraph_conv.py` loses all nine model users: `hetero_cell`, `hetero_cell_pma`, `hetero_cell_flex`, `hetero_cell_isab_split`, `isomorphic_cell`, `isomorphic_cell_attentional` and the three `cell_latent_perturbation*` models. The gene, reaction and metabolite hypergraph model (`MetabolismProcessor` with `StoichHypergraphConv`) existed only in those models; live metabolism work uses `FluxLayer`. The module stays live through its own test.

### DATA_ROOT directory that stays

`$DATA_ROOT/data/scerevisiae/sgd_gene_graph_hot` stays. `torchcell/datasets/sgd_gene_graph_hot.py` retired, but the live `GraphEmbeddingDataset` still writes to that directory.

### Retired modules

Lines are `wc -l` before the move. The newest experiment era is the newest experiment folder that imports the module (from the importer scan of the review pass; `none` means no experiment imports it). Successors are as the two reviewers named them (the second reviewer's table, 2026.10.06, filled the half that first read `not recorded`); a PyG class is an external-library successor.

| Module | wc lines | Newest experiment era | Live successor | Verdict reason |
|---|---:|---|---|---|
| `torchcell/cell.py` | 1 | none | none | shadowed by the cell/ package, never importable |
| `torchcell/config.py` | 30 | none | none | unreached from every root |
| `torchcell/data/sgd_expression.py` | 80 | none | none | unreached from every root |
| `torchcell/data_download_yeastmine.py` | 65 | none | `torchcell/sequence/genome/scerevisiae/s288c.py` (genomes registry) | broken import: intermine needs collections.MutableMapping |
| `torchcell/database/biocypher_out_combine.py` | 329 | none | the live rebuild and incremental import | unreached from every root |
| `torchcell/dataloading_lmdb.py` | 64 | none | `torchcell/data/neo4j_cell.py` | broken import: torchcell.datasets.CellDataset is gone |
| `torchcell/datamodules/dcell_DEPRECATED.py` | 83 | none | `models/dcell`, `models/dcell_opt` | unreached from every root |
| `torchcell/dataset_preprocess/__init__.py` | 1 | none | none | empty package once its members left |
| `torchcell/dataset_readers/__init__.py` | 5 | none | `torchcell/data/neo4j_cell.py` | empty package once its members left |
| `torchcell/dataset_readers/reader.py` | 115 | none | `torchcell/data/neo4j_cell.py` | unreached from every root |
| `torchcell/datasets/add_datasets_check.py` | 62 | none | none | unreached from every root |
| `torchcell/datasets/base_cell.py` | 350 | none | `data/neo4j_cell` | broken import: torchcell.data.Dataset is gone |
| `torchcell/datasets/cell.py` | 593 | none (reached only by test_esm2) | `data/neo4j_cell` (`Neo4jCellDataset`, `create_embedding_graph`) | reached only by test_esm2 (rewritten onto the live create_embedding_graph) |
| `torchcell/datasets/cell_scratch.py` | 122 | none | `torchcell/data/neo4j_cell.py` | broken import: torchcell.data_prior is gone |
| `torchcell/datasets/dcell_DEPRECATED.py` | 515 | none | `models/dcell`, `models/dcell_opt` | broken import: torchcell.models.DCellLinear is gone |
| `torchcell/datasets/dummy.py` | 6 | none | none | unreached from every root |
| `torchcell/datasets/experiment.py` | 358 | none | `data/neo4j_cell` | broken import: torchcell.data.Dataset is gone |
| `torchcell/datasets/genome.py` | 32 | none | none | unreached from every root |
| `torchcell/datasets/go.py` | 181 | none | `torchcell/graph/graph.py` | unreached from every root |
| `torchcell/datasets/ontology.py` | 256 | none | the `datamodels` schema | unreached from every root |
| `torchcell/datasets/pronto_ontology.py` | 91 | none | none | unreached from every root |
| `torchcell/datasets/scerevisiae/costanzo2016_deprecated.py` | 871 | none | `datasets/scerevisiae/costanzo2016` | broken import: torchcell.data.Dataset is gone |
| `torchcell/datasets/scerevisiae/mechanisitc_aware.py` | 32 | none | none | broken import: rpy2 not installed |
| `torchcell/datasets/scerevisiae/tutorial_joining_nucleotide_embeddings.py` | 31 | none | none | broken import: fungal_utr_transformer is gone |
| `torchcell/datasets/sgd_gene_graph_hot.py` | 262 | none | `datasets/sgd_gene_graph` | unreached from every root |
| `torchcell/delete_subset.py` | 53 | none | none | unreached from every root |
| `torchcell/go/__init__.py` | 1 | none | `torchcell/graph/graph.py` | empty package once its members left |
| `torchcell/go/check_deprecated.py` | 21 | none | `torchcell/graph/graph.py` | unreached; cwd-relative GODag load at import |
| `torchcell/graph/graph_analysis.py` | 658 | none | none | unreached from every root |
| `torchcell/graph/metabolism.py` | 54 | none | `torchcell/metabolism/yeast_GEM.py` | unreached; builds a metabolism graph at import |
| `torchcell/graph/uniprot_api_ec.py` | 36 | none | `torchcell/metabolism/enzyme_kinetics.py` / `parameters.py` | unreached; web API query at import |
| `torchcell/graph/validation/raw_structure.py` | 201 | none | none | unreached from every root |
| `torchcell/knowledge_graphs/create_pypy_scerevisiae_kg.py` | 63 | none | the live KG build | unreached; SSL env mutation at import; PyPy path abandoned |
| `torchcell/knowledge_graphs/gene_interactions_scerevisae_kg_small.py` | 206 | none | `torchcell/knowledge_graphs/create_kg.py` | unreached; SSL env mutation at import |
| `torchcell/knowledge_graphs/smf_kg.py` | 191 | none | `torchcell/knowledge_graphs/create_kg.py` | unreached; SSL env mutation at import |
| `torchcell/knowledge_graphs/smf_tmi_combine_kg.py` | 190 | none | `torchcell/knowledge_graphs/create_kg.py` | unreached; SSL env mutation at import |
| `torchcell/losses/SupCr.py` | 104 | none | `torchcell/losses/multi_dim_nan_tolerant.py` | broken import: pytorch_metric_learning not installed |
| `torchcell/losses/dcell_DEPRECATED.py` | 35 | none | `losses/dcell` | unreached from every root |
| `torchcell/metrics/__init__.py` | 1 | none | none | empty package once its two members left |
| `torchcell/metrics/nan_tolerant_classification_metrics.py` | 323 | 003-fit-int (through its trainers) | NaN mask in the trainer, then plain torchmetrics (#617) | only importers were retiring 003 trainers; NaN-tolerant metrics ruled the wrong design |
| `torchcell/metrics/nan_tolerant_metrics.py` | 572 | 003-fit-int (through its trainers) | NaN mask in the trainer, then plain torchmetrics (#617) | only importers were retiring 003 trainers; NaN-tolerant metrics ruled the wrong design |
| `torchcell/models/cell_diffpool_dense.py` | 831 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/models/cell_diffpool_sparse.py` | 813 | none | none | unreached from every root |
| `torchcell/models/cell_gin_diffpool_dense.py` | 795 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/models/cell_latent_perturbation.py` | 1670 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/models/cell_latent_perturbation_tform.py` | 1585 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/models/cell_latent_perturbation_unified.py` | 1774 | none | none | unreached from every root |
| `torchcell/models/cell_sagpool.py` | 574 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/models/cell_sagpool_inception.py` | 805 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/models/dcell_DEPRECATED.py` | 359 | none | `models/dcell`, `models/dcell_opt` | unreached from every root |
| `torchcell/models/dense_gat_conv.py` | 136 | none | PyG `torch_geometric.nn.DenseGATConv` (external library) | unreached from every root |
| `torchcell/models/early_cell_diffpool_dense.py` | 583 | none | none | unreached from every root |
| `torchcell/models/gat_diffpool.py` | 655 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/models/gat_diffpool_alt.py` | 635 | none | none | unreached from every root |
| `torchcell/models/gat_diffpool_inception.py` | 752 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/models/hetero_cell.py` | 726 | 003-fit-int | `models/hetero_cell_bipartite_dango*` | unreached; frozen-era code |
| `torchcell/models/hetero_cell_bipartite.py` | 646 | 003-fit-int | `models/hetero_cell_bipartite_dango*` | unreached; frozen-era code |
| `torchcell/models/hetero_cell_flex.py` | 928 | none | `models/hetero_cell_bipartite_dango*` | unreached from every root |
| `torchcell/models/hetero_cell_isab_split.py` | 1209 | none | `models/hetero_cell_bipartite_dango*` | unreached from every root |
| `torchcell/models/hetero_cell_nsa.py` | 716 | 003-fit-int | `models/hetero_cell_nsa_retry`, `nn/hetero_nsa` | unreached; frozen-era code |
| `torchcell/models/hetero_cell_pma.py` | 1229 | 003-fit-int | `models/hetero_cell_bipartite_dango*` | unreached; frozen-era code |
| `torchcell/models/hetero_gnn_pool.py` | 826 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/models/isomorphic_cell.py` | 1673 | none | none | unreached from every root |
| `torchcell/models/isomorphic_cell_attentional.py` | 1750 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/models/nsa_hetero_cell.py` | 1094 | none | `models/hetero_cell_nsa_retry`, `nn/hetero_nsa` | unreached from every root |
| `torchcell/models/self_attention_sag.py` | 184 | none | none | unreached from every root |
| `torchcell/models/species_aware_lm.py` | 243 | none | `torchcell/models/fungal_up_down_transformer.py` | unreached; HuggingFace download at import |
| `torchcell/ncbi.py` | 118 | none | `torchcell/sequence/genome/scerevisiae/s288c.py` (genomes registry) | shadowed by the ncbi/ package, never importable |
| `torchcell/ncbi/__init__.py` | 2 | none | none | empty package once its members left |
| `torchcell/ncbi/ncbi.py` | 37 | none | `torchcell/sequence/genome/scerevisiae/s288c.py` (genomes registry) | unreached; web API query at import |
| `torchcell/ncbi/sequence.py` | 172 | none | `torchcell/sequence/genome/scerevisiae/s288c.py` | unreached from every root |
| `torchcell/ncbi/sequence_scratch.py` | 70 | none | none | unreached; cwd-relative read_csv at import |
| `torchcell/neo4j_example.py` | 193 | none | `torchcell/adapters/` + `torchcell/knowledge_graphs/create_kg.py` | unreached from every root |
| `torchcell/neo4j_fitness_lmdb.py` | 40 | none | `torchcell/data/neo4j_cell.py` | unreached from every root |
| `torchcell/nn/aggr/__init__.py` | 1 | none | PyG `torch_geometric.nn.aggr` (external library) | empty package once its members left |
| `torchcell/nn/aggr/set_transformer.py` | 168 | none | PyG `torch_geometric.nn.aggr.SetTransformerAggregation` (external library) | unreached from every root |
| `torchcell/nn/flex_attention_graph.py` | 70 | none | none | unreached; dataset build and plt.show() at import |
| `torchcell/nn/flex_attention_graph_adj.py` | 116 | none | none | unreached; dataset build and plt.show() at import |
| `torchcell/nn/flex_attention_graph_nsa.py` | 402 | none | `nn/nsa_encoder` | unreached from every root |
| `torchcell/nn/sort_adj_block_model.py` | 122 | none | none | unreached; dataset build and plt.show() at import |
| `torchcell/prof.py` | 77 | DEPRECATED_dna_llm_viz | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/profilers/__init__.py` | 1 | none | `lightning.pytorch.profilers.PyTorchProfiler` | empty package once its members left |
| `torchcell/profilers/pytorch.py` | 660 | 003-fit-int | `lightning.pytorch.profilers.PyTorchProfiler` | unreached; frozen-era code |
| `torchcell/pypy_adapters/__init__.py` | 16 | none | `adapters/` | empty package once its members left |
| `torchcell/pypy_adapters/costanzo2016_pypy_adapter.py` | 1355 | none | `adapters/costanzo2016_adapter` | unreached from every root |
| `torchcell/pypy_adapters/kuzmin2018_pypy_adapter.py` | 2000 | none | `adapters/kuzmin2018_adapter` | unreached from every root |
| `torchcell/sc_graph.py` | 296 | none | the costanzo2016 and kuzmin2018 loaders | unreached from every root |
| `torchcell/sequence/data_scratch.py` | 13 | none | none | unreached from every root |
| `torchcell/sequence/sequence_plot.py` | 122 | none | none | broken import: torchcell.sgd is gone |
| `torchcell/sequence/sequence_scratch.py` | 28 | none | none | unreached from every root |
| `torchcell/trainers/cell.py` | 63 | none | none | unreached from every root |
| `torchcell/trainers/fit_int_cell_diffpool_dense_regression.py` | 514 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/trainers/fit_int_cell_gin_diffpool_dense_binary.py` | 546 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/trainers/fit_int_cell_sagpool_regression.py` | 486 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/trainers/fit_int_deep_set_regression.py` | 449 | 003-fit-int | `trainers/neo_regression` | unreached; frozen-era code |
| `torchcell/trainers/fit_int_gat_diffpool_inception_regression.py` | 589 | 003-fit-int | none (the GAT-DiffPool family is an abandoned 003 architecture; the module was on neither reviewer table) | broken import (NaNTolerantPearsonCorrCoef); reached only by a test pinning that failure |
| `torchcell/trainers/fit_int_gat_diffpool_regression.py` | 676 | 003-fit-int | none | unreached; frozen-era code |
| `torchcell/trainers/fit_int_hetero_cell.py` | 597 | 005-kuzmin2018-tmi | `trainers/int_hetero_cell.RegressionTask` | unreached; frozen-era code |
| `torchcell/trainers/fit_int_hetero_cell_multiset_decomposition.py` | 815 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/trainers/fit_int_hetero_gnn_pool_binary_classification.py` | 564 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/trainers/fit_int_hetero_gnn_pool_reg_categorical_entropy.py` | 551 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/trainers/fit_int_hetero_gnn_pool_regression.py` | 367 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/trainers/fit_int_isomorphic_cell_attentional.py` | 380 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |
| `torchcell/trainers/graph_convolution_regression.py` | 480 | none | `trainers/neo_regression` | broken import: WeightedMSELoss moved |
| `torchcell/trainers/int_hetero_cell_nsa.py` | 1370 | none | `trainers/int_hetero_cell` | unreached from every root |
| `torchcell/trainers/regression.py` | 459 | none | `trainers/neo_regression` | broken import: WeightedMSELoss moved |
| `torchcell/trainers/regression_deep_set_transformer.py` | 353 | none | `trainers/neo_regression` | broken import: WeightedMSELoss moved |
| `torchcell/trainers/utils.py` | 63 | none | none | broken import: pydantic-v1 ConstrainedStr |
| `torchcell/transforms/hetero_to_dense.py` | 176 | 003-fit-int (the 004-011 hits are commented-out imports) | `transforms/hetero_to_dense_mask.HeteroToDenseMask` | unreached; frozen-era code |
| `torchcell/viz/datamodules.py` | 195 | 003-fit-int | none | unreached; frozen-era code, abandoned architecture |

### Tests

- `tests/torchcell/datasets/test_esm2.py`: the one test that built `CellDataset.create_embedding_graph` over an `Esm2Dataset` now calls the live `torchcell.data.neo4j_cell.create_embedding_graph`, which min-max normalizes each feature before building nodes, so the pinned node vectors are the normalized rows ([1, 1, 0.5, 0.5], [0, 0, 0, 0], [1, 1, 1, 1]). It runs on a fresh construction over the same store: on a dataset whose items were already read, PyG serves its cached items and the nodes keep the raw rows (observed in the first run of the rewrite).
- `tests/torchcell/metrics/test_nan_tolerant_metrics.py` and `test_nan_tolerant_classification_metrics.py` retired with their modules. The one live behavior they touched, `EqualWidthStrategy.compute_onehot_labels` writing an all-NaN row for a NaN value, is already pinned by `TestBinningBranches::test_onehot_is_left_closed_clamped_and_nan_propagating` in `tests/torchcell/transforms/test_regression_to_classification.py`, so no test moved.
- `tests/torchcell/datasets/test_dummy.py` (a placeholder for `datasets/dummy.py`) retired. `tests/torchcell/models/test_hetero_cell_nsa.py` stays: despite its name it tests the live `nn.masked_attention_block` and `nn.self_attention_block`, not the retired model.
- `tests/torchcell/test_import_all.py`: `KNOWN_BROKEN` is empty (all 16 entries retired), `NEVER_IMPORT` keeps only the scratch and experiments carve-out, and the hard-coded-path allowlist drops `data/sgd_expression.py` and `scerevisiae/mechanisitc_aware.py`.

### Edits outside the package

- `pyproject.toml`: the ruff `extend-exclude` of `trainers/graph_convolution_regression.py`, the mypy `exclude` alternatives (`_DEPRECATED.py`, `costanzo2016_deprecated.py`, `_scratch.py`, `delete_subset.py`, `pypy_adapters/`), the `ignore_errors` overrides for `torchcell.pypy_adapters.*`, `torchcell.trainers.utils` and `torchcell.trainers.graph_convolution_regression`, and the seven matching `[tool.torchcell.test_exceptions]` paths are gone; the scratch, experiments and `hetero_cell_nsa_retry` carve-outs stay.
- `docs/gen_api_pages.py`: the `dataset_readers` and `metrics` descriptions and the retired names in `SKIP_MODULE` and in the `nn` / `transforms` descriptions are gone; `docs/source/modules/*.rst` regenerated (the run also picked up drift that predates this change in `data`, `datamodels`, `knowledge_graphs` and `metabolism`), `dataset_readers.rst` and `metrics.rst` retired, and `docs/source/index.rst` drops both from the toctree and the retired packages from "Not documented".
- `tests/torchcell/datasets/scerevisiae/test_raw_pins.py`: `costanzo2016_deprecated` left the pin-debt list.
- `.claude/`: two command notes mark their retired files; the ruff and update-src-notes skill examples name `torchcell/timestamp.py` instead of the retired `torchcell/cell.py`.
- Notes: 55 dendron links to retired notes, in 19 notes, now read as the plain path followed by "(retired 2026-10-06 to the graveyard)". Prose mentions and the coverage tables of [[test-campaign.2026.09.25]] are left as history.

### Coverage

`scripts/coverage_gaps.py --before coverage-p21.json --after coverage-retired.json`, TOTAL rows (before is PR-21 with the cluster present, after is without it):

| | | Statements | Before | After | Import-only | Delta |
|---|---|---:|---:|---:|---:|---:|
| TOTAL (line+branch) |  | 51648 | 68.1% | 91.2% | 16.9% | +23.1 |
| TOTAL (line only) | | 51648 | 68.2% | 91.5% | 21.3% | |

The rise is the denominator: the retired modules held 18,453 statements of which 543 (2.9%) were executed in the PR-21 run (`coverage-p21.json`), so removing them lifts the line+branch TOTAL from 68.1% to 91.2% with no new test.

Runs: retirement worktree on `03b9d6bc8` plus the retirement edits: full hermetic suite including `test_import_all.py`, 6998 passed, 125 skipped, 9 xfailed, 7408 warnings in 305.08s (0:05:05); behavioral coverage run (import-all deselected) 6675 passed, 1 failed, 125 skipped, 322 deselected, 9 xfailed in 430.21s, the one failure being the `costanzo2016_deprecated` pin-debt entry fixed after that run (the fixed file and `test_import_all.py` then 372 passed); `legacy_partition.py --check` exit 1 on the five kept modules and `models/constants.py` only; `test_quality_check.py` 348 files clean; `check_paired_tests.py --base origin/main` 0 added modules; ruff and mypy (CI form) clean on every changed .py; `gen_api_pages.py --check` exit 0; `sphinx-build` is not installed in the torchcell env, so the docs were not built.

## 2026.10.06 - Follow-up: the orphan constant and the closed gate

After the retirement `scripts/legacy_partition.py --check` still exited 1 on six modules: the five kept on purpose and `torchcell/models/constants.py`, whose one constant `DNA_LLM_MAX_TOKEN_SIZE` (5979) was re-exported by `torchcell/models/__init__.py` and read by nothing. Phase 22 (PR-22) closed the gate: behavioral tests make `spell.py`, `build_time_projection.py` and `paper/signal.py` live; a `package-data` category in the partition script covers the two `conf/__init__.py` files whose directories ship YAML listed under `[tool.setuptools.package-data]`; and `constants.py` went to the graveyard with its re-export removed (`/scratch/projects/torchcell-deprecated/2026-10-06_161754__constants.py`). The check now reports `live 317, init-only 0, legacy 0, carve-out 19, package-data 2` and runs as a blocking CI step.
