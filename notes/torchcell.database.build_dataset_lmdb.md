---
id: sp2cvo47dklh1bxyalm4wy5
title: Build_dataset_lmdb
desc: ''
updated: 1791801780043
created: 1791801780043
---

## 2026.10.12 - `--list-stale --organisms`

`mapped_store_status(data_root, include_private, organisms)` walks `build_adapter_map(include_private, organisms)`, the map the build itself selects, and `--list-stale` takes `--organisms <comma-separated set>` (default every organism). A yeast-only rebuild therefore lists only the yeast stores it will read, and the live rebuild's preflight, which calls the same function with `KG_ORGANISMS`, does not block on a stale bacterial store ([[database.slurm.scripts.gilahyper_live_rebuild-slurm_docker]]).

The deviation from the request: the organism option is on `build_dataset_lmdb --list-stale`, not on `python -m torchcell.provenance.build_manifest`. That CLI has no `--list-stale`; it scans every built directory under `DATA_ROOT/data/torchcell/` whether or not it is mapped, and it sits below `torchcell.data` in the import graph (`experiment_dataset` imports it), so it cannot read a loader's organism without an import cycle. The map-walking list is this one, and it is the one the preflight runs.

Measured on the shared dev tree on 2026-10-12, with `--include-private`:

| `--organisms` | stores listed |
|---|---:|
| (absent) | 55 |
| `scerevisiae` | 3 (`Boocock2025Dataset`, `TfDeletionPseudobulkJackson2020Dataset`, `InhibitorBioscreenVolk2021Dataset`) |
| `ecoli` | 24 |
| `pputida` | 28 |

3 + 24 + 28 = 55, and the union of the three lists equals the unfiltered list name for name. Without `--include-private`, `--organisms scerevisiae` lists 2 (the private Bioscreen store drops out). The same unfiltered command run from the primary checkout on `main` (`eb07bd213`) lists the same 55 stores, so declaring `organism` on 154 loader classes made no store stale: the build manifest's closure is computed from `torchcell.datamodels` imports alone, and `Organism` lives in `torchcell.data`.

Re-measured after the rebase onto `8bee8bb04` (Yoshikawa 2009 on main), 07:13 to 07:16 CDT, while another session was rebuilding dev stores (28 build manifests written in the preceding 15 minutes). The yeast list was the same 6 in two runs three minutes apart: `GeneEssentialitySgdDataset`, `SynthLethalityYeastSynthLethDbDataset`, `SynthRescueYeastSynthLethDbDataset`, `Boocock2025Dataset`, `TfDeletionPseudobulkJackson2020Dataset` and the private `InhibitorBioscreenVolk2021Dataset`. E. coli listed 1 (`MorphologyCampos2018Dataset`). P. putida listed 0, although the unfiltered walk a minute earlier (12 stores) had named the five Yunus 2026 stores: they were rebuilt between the two walks. Back-to-back walks of a tree being rebuilt need not partition; the partition itself is pinned against the real map, with no store built, by `test_an_organism_walk_lists_that_organisms_stores_and_no_other`.

## 2026.10.15 - Typed verify outcome and the lane walk (ontology unification guards)

Two changes from stage 1 of Path U ([[scratch.2026.10.15.ontology-unification-decision]]).

**`VerifyOutcome`.** `verify_after_build` returned `True` with "NO PER-DATASET VERIFIER" when `resolve_verifier` raised (lines 325 to 329 before this change), so a build log read a skipped verification as a pass ([[scratch.2026.10.15.ontology-review-3-serving]] finding 7). It now returns a `StrEnum` with three members and prints `VERIFY OUTCOME: <value>` in every case: `passed` (a per-dataset verifier ran and every report passed), `no_per_dataset_verifier` (no entry point; the manifest is kept because the family runner covers the store, and the log says so), `failed` (raised, no report, or a report did not pass; the manifest is retired, exit 1). `main` exits 1 only on `failed`, the same exit behavior the array sweep script relied on.

**`mapped_store_lanes(data_root, include_private, organisms)`.** The preflight's second walk beside `mapped_store_status`: the same `build_adapter_map` of the organism set, reading `perturbation_leaves` off each store's build manifest ([[torchcell.provenance.build_manifest]]) and checking every leaf against the node lanes the dataset's adapter conf enables through `torchcell.knowledge_graphs.perturbation_lanes.unserved_leaves` ([[torchcell.knowledge_graphs.perturbation_lanes]]). `StoreLaneStatus.blocks` is true for a manifest without the field (or no manifest) and for any leaf an enabled edge walks that no enabled node method emits; `describe()` is the refusal line. The live rebuild's preflight heredoc calls it right after the freshness walk, on the same three arguments ([[database.slurm.scripts.gilahyper_live_rebuild-slurm_docker]]).

Tests: `test_verify_after_build_returns_the_typed_outcome_for_each_case` (three toys built directly, one outcome each, the `failed` manifest retired), `test_mapped_store_lanes_checks_the_manifests_leaves_against_the_confs_lanes` (the toy's KanMX deletions served under the yeast lane, a gap under the bacterial-only lane with the edge, nothing without the edge, blocked without the field and without a manifest), and the existing no-verifier test now asserts the typed line.

## 2026.10.15 - The content-id collision gate in the rebuild preflight (lane V)

**`mapped_store_collisions(data_root, include_private, organisms)`.** The preflight's
third walk beside `mapped_store_status` and `mapped_store_lanes`: the same
`build_adapter_map` of the organism set, reading `content_id_collisions` off each store's
build manifest ([[torchcell.provenance.build_manifest]]) and comparing
`n_records_lost` with the dataset's entry in `EXPECTED_CONTENT_ID_COLLISIONS`.
`StoreCollisionStatus.blocks` is true for a manifest without the field (or no manifest)
and for any measured count that is not the declared one; `describe()` is the refusal
line and prints the example groups' LMDB keys so the rows can be read. The live
rebuild's preflight heredoc calls it right after the lane walk, on the same three
arguments ([[database.slurm.scripts.gilahyper_live_rebuild-slurm_docker]]).

It REFUSES rather than reports, and the argument is that reporting was the status quo.
The sink keeps one node per `Experiment` content id, so a record that shares one with
another record of its store is absent from the graph with no error; the number was
computable from the served store the whole time and nobody computed it, so 18 SGD
essentiality annotations, 31 Ozaydin measurements and 2 Mota rows went into three KG
releases unnoticed ([[experiments.database.scripts.content_id_collision_scan]]). A gate
that prints a line into an 8-hour build log would have been read the same number of
times. The build manifest still records the count whatever it is; only the rebuild is
gated.

The escape is one declaration, and it is written. `EXPECTED_CONTENT_ID_COLLISIONS` maps a
dataset class to the number of records its SOURCE released more than once,
byte-identically, with the released evidence in the loader module; it holds exactly one
entry, `EnvChemgenMota2024Dataset: 2`, for the two RNR4 rows Mota 2024 printed twice
([[torchcell.datasets.scerevisiae.mota2024]]). A declaration admits one number, not "at
least this many", so a dataset that starts losing a third record blocks even though it is
in the table.

Tests: `test_mapped_store_collisions_refuses_a_store_whose_collisions_are_not_declared`
(the toy's three distinct records pass at 0, a manifest reporting 1 blocks undeclared,
passes when declared 1, blocks again when declared 2, and blocks without the field and
without a manifest) and `test_the_only_declared_source_duplication_is_motas_two_rnr4_rows`,
which pins the table itself because an exception list is part of the gate.
