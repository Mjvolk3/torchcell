---
id: ak71trysgn3cwwc7w9zcivu
title: '25'
desc: ''
updated: 1790392161358
created: 1790392161358
---

## Context

The 030 build (`/db/experiments/030-solid-growth-multi-001-multi-build`, 13,525,088 records,
symlinked as `$DATA_ROOT/data/torchcell/experiments/030-solid-growth-multi/001-multi-build`)
keeps every source entry per genotype instead of merging to one value. Fifteen source datasets
contribute (`dataset_name_index.json`, names carry the `Dataset` suffix:
`DmfCostanzo2016Dataset` 13,087,332 ... `SmfKuzmin2020Dataset` 235). On the S3 closure
(1,121,662 records) the entries number 4,860,168 (`closure/entries.parquet`, written by
`closure_recompute_030.py`), and 1,396,361 of 2,237,611 (record, label) pairs carry more than
one source. The 025 trainer's `_coo_label` (W025 trainer L222-269) masks any batch row with two
values of one label, so on 030 it would drop about 94% of closure doubles and every single with
more than one entry; the per-entry path replaces it.

Deliverables, in order: (1) S3 subset, the 010 random-split (R) val/test pin, and a gene-level
essentiality holdout, regenerated ON 030 with counts checked against 025 (decisions 2, 11);
(2) per-entry training rows with a readout dataset token, the single-value path unchanged when
off, with tests (decisions 3-9); (3) first arm `cgt_030_s3_r_tok_embfit_001` (decision 13)
after a GilaHyper smoke test that shows the encoding is learned (decision 10), the 955 GB build
copy, and ONE seed-0 submission; (4) plan only: Kuzmin 2020 array SMF + S1/S3 table on every
2020 record, a full rebuild (Deferred).

Branch facts verified 2026-09-25 (they correct two scout claims): `main` has none of the 025
trainer work; `#346` (`feat/025-fitness-joint-head`, 29af19ebf) and `#430`
(`feat/025-label-named-metrics`, dcd9b04ee, contains 29af19ebf) are open; `#435`
(`feat/025-s3-closure-030`, f5190a01c, notes-tex + W&B view scripts, based on 29af19ebf, NOT on
`#430`) is a fourth stacked PR the scouts omitted; `#414` (`feat/030-solid-growth-multi`) holds
the 030 build scripts. `torchcell/data/label_policy.py` + `label_table.py` landed on `main` as
ae67e6738 (five additive files: two modules, one test, two notes); ae67e6738 is NOT an ancestor
of dcd9b04ee, so `import torchcell.data.label_policy` fails there until it is cherry-picked. The
fd-exhaustion fix f1bfa95be lives only on `multimodal-phenotype-retrospective`.

Scoring rule, unchanged from 025: mean validation trigenic Pearson over epochs 10-29 on the
pinned 010 triples, never a max; 025 references S0-R composite 0.433 (sd 0.001, n 3) and S3-R
table 0.479 (sd 0.001). The S3 gain is not attributed to the identity (memory
`025-s3-closure-recompute-findings`).

## Relevant Files

W025 = `/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/025-fitness-joint-head`;
W030 = `.../feat/030-solid-growth-multi`; WK = `.../feat/kinetics-equilibrator-datasets`; M =
`/home/michaelvolk/Documents/projects/torchcell` (main); E030 =
`experiments/030-solid-growth-multi`. NEW paths are repo-relative on the new branch.

| Path | Action | Purpose |
|---|---|---|
| W025 `torchcell/trainers/int_transformer_cell.py` | MODIFY | `_coo_label` L222-269 becomes per-entry rows; `_fitness_step` L331; `_perturbation_order` L366; `_update_order_metrics` L372; `_get_batch_size` L440; `_shared_step` L936; `validation_step` L1601 gains `dataloader_idx`; `_compute_metrics_safely` L1616 |
| W025 `torchcell/models/equivariant_cell_graph_transformer.py` | MODIFY | `PerturbationHead` mlp `Linear(2d, d)`, forward L1303; `GlobalHead` L1358; ctor L1885 (`perturbation_head` L2146, `perturb_cls` L2151, `global_head` L2233-2262); forward L2694, perturbed CLS L2868, readouts L2930-2952, returns `h_CLS_pert` L3023 |
| W025 `torchcell/data/graph_processor.py` | MODIFY | `Perturbation._add_phenotype_data` L2061; entry loop L2092; add parallel `phenotype_dataset_indices`; always-emit rule L2159 |
| W025 `torchcell/datamodules/cell.py` | MODIFY | `follow_batch` L231, `pinned_split_indices` L236, `index_subset` L237, `unpinned_to_train` L238, cache check L404, `val_dataloader` L717; add `extra_val_indices` |
| W025 `torchcell/transforms/coo_regression_to_classification.py` | MODIFY | `COOLabelNormalizationTransform` L21 fits on `dataset.label_df` (L66, last entry wins); add `fit_table`; clone path L151-164 |
| M `torchcell/data/label_policy.py` (cherry-pick ae67e6738) | REFERENCE | `source_key` L85, precedence kuzmin2018 > kuzmin2020 L195-208, `replicate_combination` L228, `policy_id` L262, `select` L278, `_combine` L298 |
| M `torchcell/data/label_table.py` (same commit) | REFERENCE | `entries_of_record` L119, `build_label_table` L199 |
| W025 `experiments/025-solid-growth/scripts/{subset_definitions,transfer_010_tmi_splits,make_010build_index_artifacts}.py` | REFERENCE | Copy patterns; roots hardcoded to 025 (`subset_definitions` L49-64, `transfer_010` L70-73, L119) |
| W025 `experiments/025-solid-growth/scripts/equivariant_cell_graph_transformer.py` | REFERENCE | Copy source; `load_index_artifact` L90, `follow_batch` L480-502, `CellDataModule` L505-522, split assert L528-541 |
| W025 `experiments/025-solid-growth/scripts/igb_mmli_cgt.slurm` | REFERENCE | Launcher template; `PROJECT_ROOT` L62, `BUILD_REL` L64, `--mem=250g` L44, `--time=4-00:00:00` L50, `torchrun` L135 |
| W025 `experiments/029-solid-growth-ko/scripts/{sync_igb_029_build.sh,gh_sync_igb_029.slurm}` | REFERENCE | rsync template: `processed/` + empty `raw/lmdb` stub; 2-day 8 GB wrapper |
| W025 `experiments/025-solid-growth/conf/cgt_s3_r_kl_embfit_034.yaml` (+ chain to `default.yaml`) | REFERENCE | The arm being ported; nine-file defaults chain |
| W030 `E030/scripts/closure_recompute_030.py` | REFERENCE | Per-entry LMDB scan (`_entries` L111-168, `scan` L240); writes `closure/{entries,triple_roles}.parquet` |
| WK `experiments/028-gene-essentiality/scripts/{build_essentiality_splits,train_essentiality}.py` | REFERENCE | `SplitReport` L79; `FCL_ESSENTIALITY_SPLIT` L1875 (`data/merzbacher2025_fcl/.../yeast_essentiality_test_split.csv`, `essential` column inverted, L1877-1881); `read_fcl_essentiality_split` L1951 |
| `E030/scripts/subset_definitions_030.py` | NEW | S3 on 030; writes `results/subset_S3_indices.json.gz` + summary |
| `E030/scripts/transfer_010_tmi_splits_030.py` | NEW | 010 seed-42 pin onto 030 by gene set; writes `results/pinned_splits_from_010_seed_42.json.gz` |
| `E030/scripts/build_essentiality_holdout_030.py` | NEW | Gene-level holdout (a) + (b); pydantic `EssentialityHoldoutReport`; writes `results/essentiality_holdout_030.json.gz` |
| `E030/scripts/make_pinned_eval_table_030.py` | NEW | LabelPolicy choice per pinned (record, label); `results/pinned_eval_table_030.parquet` |
| `E030/scripts/equivariant_cell_graph_transformer.py` | NEW | 025 script with 030 roots, `dataset_token` block, `val_ess` wiring |
| `E030/scripts/smoke_dataset_token.py` + `gh_smoke_dataset_token.slurm` | NEW | GilaHyper smoke job and its PASS/FAIL report |
| `E030/scripts/{sync_igb_030_build.sh,gh_sync_igb_030.slurm,igb_mmli_cgt_030.slurm}` | NEW | Build copy and mmli launcher |
| `E030/conf/{default,cgt_030_smoke_tok_000,cgt_030_s3_r_tok_embfit_001}.yaml` | NEW | Flattened config over a copied `default.yaml` |
| `tests/torchcell/trainers/test_int_transformer_cell_per_entry.py`, `tests/torchcell/models/test_dataset_token_readout.py`, `tests/torchcell/data/test_graph_processor_dataset_indices.py` | NEW | Per-entry loss/metrics, readout token, tensor emission |
| W025 `tests/torchcell/trainers/test_int_transformer_cell_coo_label.py` | MODIFY | Retarget from the conflict mask to the per-entry rows |

## Key Design Decisions

1. **Branch `feat/030-per-entry-dataset-token` from dcd9b04ee (`#430` head), PR against
   `feat/025-label-named-metrics`; first commit cherry-picks ae67e6738 (LabelPolicy), second
   f1bfa95be (fd fix).** The trainer work exists only on the 025 stack and `#430` already edits
   the metric names this plan extends; stacking avoids re-resolving the 025-vs-main conflicts
   (`gh_cgt.slurm`, `delta_preflight_025.sh`, `neo4j_query_raw.py`, two weekly notes, a FigS
   drawio) inside a feature branch. ae67e6738 adds only files absent from the stack, so the
   cherry-pick is conflict-free. Landing order `#346` -> `#430` -> new by rebase + ff; `#435`
   shares no code with this branch and lands independently (it and `#430` both touch the
   trainer relative to 29af19ebf, `#435` merely lacking `#430`'s +32/-3, so the merge queue
   serializes them). Nothing from `#414` is imported as code; `entries.parquet` is regenerated
   by its committed script if absent. Rejected: branching from `main` (re-implements
   `fitness_lambda`, `unpinned_to_train`, per-order metrics).
2. **Regenerate S3 and the R pin on 030 by sorted-gene-name-set join, never by index.** 030 has
   +17 records relative to 025 (1,121,662 vs 1,121,645 in S3), so 025 indices are off by
   construction. The join key is `tuple(sorted(systematic_gene_name))` (as
   `transfer_010_tmi_splits.py` L50); the script asserts each 010 triple matches exactly one 030
   record and hard-fails otherwise. Acceptance: singles 5,694, closure doubles 739,236, triples
   376,732, pinned val and test 37,673 each (10% of 376,732, equal to 025's report).
3. **Per-entry rows are built AFTER the encoder forward, not by expanding the dataset.** One
   LMDB read and one forward per genotype; `h_CLS_pert` `[B, d]` is gathered by
   `phenotype_values_batch` to `[E, d]` entry rows. Input-side expansion would multiply forwards
   by 4.3 (4,860,168 entries over 1,121,662 records), pushing a 36-minute 025 epoch to about
   2.6 h and 50 epochs past the 4-day mmli clock; materializing entries as dataset items would
   multiply reads on a 955 GB LMDB by the same factor.
4. **Source identity travels as `phenotype_dataset_indices`, a long tensor parallel to
   `phenotype_values`, emitted for every sample, listed in `follow_batch`, excluded from the
   transform clone path.** The vocabulary is the sorted key list of `dataset_name_index.json`
   (15 names), fixed at dataset construction and written into the run config; a name outside it
   raises. PyG 2.8.0 `Batch.from_data_list` raises `KeyError` when samples disagree on keys
   (<https://github.com/pyg-team/pytorch_geometric/releases>, 2.8.0), so the empty-phenotype
   placeholder branch (graph_processor L2154) emits the tensor too. The transform's `clone()`
   (L151, L210) copies only `phenotype_values`, so the new tensor is untouched by construction;
   a test asserts it. Rejected: a bit code (the user's rule: one-hot sized to the pool) and a
   dataset x temperature vocabulary (17; deferred).
5. **The token is `F.one_hot(idx, 15) @ W` with `W` an `nn.Linear(15, 8, bias=False)`,
   mathematically an embedding table.** Keeping the one-hot literal matches the decision and
   avoids `nn.Embedding`'s backward, which became non-deterministic in torch 2.11 (fused
   `compute_grad_weight`; PyTorch forum thread 2026-06-16,
   <https://discuss.pytorch.org/t/embedding-became-non-deterministic-in-pytorch-2-11/225105>;
   PyTorch 2.11 release 2026-03-30, <https://pytorch.org/blog/pytorch-2-11-release-blog/>);
   installed torch is 2.11.0+cu128. The trainer sets no deterministic algorithms either way;
   seed-to-seed variance is reported, not suppressed.
6. **Token enters at the READOUT (both heads); input-token kept as a later ablation.** (i) With
   a dataset-only vocabulary the sources differ by measurement pipeline acting on the readout
   scale (a Costanzo double at 26 C vs 30 C, an SGD converted 0 vs a measured smf); an input
   token would spend encoder capacity memorizing which genes each screen covered. (ii) An input
   token forces one forward per entry (decision 3). (iii) With the encoder identical to 025's,
   any change in the trigenic score is attributable to readout conditioning alone. (iv) The
   learnability test (decision 10) is exact at the readout: a pure offset must be recoverable by
   a linear head. Mechanics: `PerturbationHead` gathers `combined = [h_CLS || z_S]` to entry
   rows, concatenates the 8-dim token, and its first `Linear` widens from `2d` to `2d + 8`;
   `GlobalHead(linear=True)` widens from `d` to `d + 8`. With `dataset_token.enabled: false`
   widths and forward are byte-identical to today (`test_single_head_config_matches_prechange`
   must still pass).
7. **Loss = masked MSE over entry rows for both labels, `fitness_lambda` 1.0, orders read per
   row from the row's genotype.** `#410`'s Costanzo SMF 26/30 triplication is inside the
   expansion factor; it is accepted and counted (exact-duplicate `(genotype, label, value,
   dataset)` tuples in the pool summary), not deduplicated, because the query decides what the
   dataset is.
8. **Val/test primary metric: exactly one (prediction, target) per pinned triple per label, the
   entry chosen by LabelPolicy, scored under that entry's own token.** `select` (L278) drops
   converted zeros when a measurement exists, walks precedence (kuzmin2018 before kuzmin2020),
   and `_combine` (L298) returns a single entry as is; several same-source entries are combined
   by inverse variance when every SE is positive, otherwise by plain mean, with Stouffer p.
   In-batch the trainer takes the rows whose dataset equals the chosen source and averages their
   values (plain mean; predictions are identical because genotype and token are identical);
   `make_pinned_eval_table_030.py` records how many pinned pairs have >1 same-source entry and
   the max |plain mean - policy value|, so the deviation from inverse-variance weighting is a
   logged number, not an assumption. For the 363,818 single-entry triples the target equals
   025's, so the 37,673-triple populations hold. Token for interaction = the `Tmi*` dataset of
   the entry, for fitness = the `Tmf*` dataset. Secondary, logged every epoch: per-entry Pearson
   each under its own token, per-screen Pearson, and the cross-token score (2018 entries scored
   under the 2020 token) as the running ablation.
9. **Normalization: one set of per-label constants fitted on the train-row ENTRY table (pinned
   train intersected with S3 minus the essentiality exclusion), shared by every token.**
   `label_df` is last-entry-wins per record (neo4j_cell L718-722), so fitting on it would weight
   a 160-entry double as one row and a 1-entry double as one row; per-token constants would
   pre-absorb the very offset the smoke test must show learned. Pearson is invariant to this
   choice; the loss scale is not, so it is named in the config (`transforms.fit_on:
   train_entries`).
10. **Smoke-test pass criteria, fixed before the run.** Pool: all 5,694 singles + the first
    20,000 S3 closure doubles carrying a fitness entry, real values. Probe: vocabulary extended
    by one fictitious token B (16 in the smoke only); every entry cloned under B with value +
    delta, delta = 0.3 in normalized units, so the offset is the only learnable difference. PASS
    iff (i) mean over genotypes of pred(B) - pred(A) within 10% of delta and per-genotype SD
    below delta/5; (ii) swapping tokens at evaluation raises MSE by delta^2 within 20%; (iii)
    the `enabled: false` control on the same doubled pool sits at least delta^2/4 above the
    enabled run's loss (the floor of predicting the midpoint). Real-data probe, reported not
    gated: per-token mean residual before and after training, the 15 x 15 pairwise distance
    matrix of the learned vectors, 2018 triples scored under the 2020 token.
11. **Essentiality holdout by GENE at the single-record level, served through a second
    validation dataloader `val_ess`, not a fourth split.** `pinned_split_indices` admits only
    train/val/test (cell.py L305); a fourth split would change every consumer of `index`, a
    second loader changes one datamodule method and one trainer hook. Universe: 5,694 single
    records keyed by gene (verify one record per gene); positives = genes with a
    `GeneEssentialitySgdDataset` entry (1,140), negatives = measured-only singles (4,554).
    Holdout (a) = every Merzbacher 2025 released TEST gene that resolves as a 030 single (028
    found 195 of 223; the 030 count is reported, not assumed); (b) = 250 essential genes drawn
    from the 907 that also carry a measured single, plus 250 non-essential, nearest-neighbor
    matched on (S3 doubles containing the gene, triples containing the gene), disjoint from (a);
    acceptance: coverage count as a score gives AUROC in 0.5 +/- 0.03 on (b). The 028
    `ess_genome_heldout` confound (label origin == class; FBA 0.5, PPI degree 0.74) is why (b)
    is matched and (a) uses the 907 rule. The whole single record of every held-out gene
    (converted-0 AND measured entries) leaves the pool via `index_subset`; its doubles and
    triples stay, and no pinned val/test index is excluded. Hypothesis (untested): higher-order
    records containing a held-out gene leak little single-deletion information because their
    fitness is dominated by the partner genes; the deferred doubles-inclusive variant tests it.
    Score = predicted fitness under the `SmfCostanzo2016Dataset` token; AUROC (Mann-Whitney)
    per epoch to W&B on (a) and (b) separately, the SGD token as a diagnostic; the REPORTED
    number is at the epoch chosen by trigenic val Pearson (the holdout stays a test), with a
    2,000-draw gene bootstrap CI, the across-seed range, and PPI-degree AUROC (STRING, already
    loaded by the model) on the same genes as the confound control. Comparators on the 195: FCL
    0.742, FBA 0.640, transformer 0.893, GO-GCN 0.917 (028).
12. **DDP logging invariants.** Every per-token and per-order key is logged on every rank in a
    fixed order, never conditional on the rank having seen the token; a token absent from a
    rank's epoch (`SmfKuzmin2020Dataset` has 235 records, 472 entries over both labels, so a
    rank can see none) logs NaN through `_compute_metrics_safely`, never skips the key.
    Lightning 2.5.5 is installed (2.6.0 released 2025-11-28,
    <https://github.com/Lightning-AI/pytorch-lightning/releases/tag/2.6.0>); issue
    <https://github.com/Lightning-AI/pytorch-lightning/issues/20946> documents `sync_dist`
    reducing the wrong values when ranks log keys in different orders. torchmetrics 1.8.2
    per-token collections multiply state 15-fold; keep them to two labels x 15 tokens x Pearson
    only.
13. **First arm = 025's `_034` with three changes: the 030 build, the dataset token on, the
    essentiality exclusion.** Composite `[fudt_upstream, calm, prot_T5_all, fudt_downstream]` at
    width 339, learnable table off, `phenotype_labels [fitness, gene_interaction]`,
    perturbed-CLS linear global head, KL graph prior, constant AdamW 2.5e-4, batch 256,
    `val_batch_size` 32, 50-epoch cap, R pin. Read against 025 S3-R composite (the 034 seeds)
    for the trigenic score; the essentiality AUROC has no 025 counterpart.

## Approach

**Branch.** `/setup-worktree feat/030-per-entry-dataset-token` with base dcd9b04ee, the two
cherry-picks of decision 1, PR opened immediately. W025 (dirty, 145 behind main) stays
untouched; every command runs with `PYTHONPATH=<worktree>`.

**Splits and holdout (GilaHyper CPU; gzipped JSON + summary JSON each).**
`subset_definitions_030.py` reads 030's `perturbation_count_index.json` and
`closure/triple_roles.parquet` (regenerated through `closure_recompute_030.py` if absent)
instead of the 025 recapitulation CSV, derives closure doubles by pair membership, and asserts
the counts of decision 2. `transfer_010_tmi_splits_030.py` reads 010's
`data_module_cache/index_seed_42.json` and joins on 030 triples by gene set with a one-to-one
assertion; output `{"report": ..., "pinned": {train, val, test}}` in the 025 shape so
`load_index_artifact(name, "pinned")` keeps working. `build_essentiality_holdout_030.py`
resolves the released split CSV, builds (a) and (b) per decision 11, runs the coverage-AUROC
acceptance, and writes gene lists, record indices, and an `EssentialityHoldoutReport`
(pydantic: counts per class and source, matching distances, acceptance AUROC, released genes
absent from 030). `make_pinned_eval_table_030.py` applies `LabelPolicy()` to
`entries_of_record` for every pinned val/test record and writes the table of decision 8 with
`policy_id`.

**Per-entry data path.** `Perturbation._add_phenotype_data` (graph_processor L2092) appends
`dataset_name_to_index[item["experiment"].dataset_name]` per emitted value and stores it as
`phenotype_dataset_indices` beside `phenotype_values` in both branches (L2142, L2154); the
vocabulary comes from `Neo4jCellDataset`, which exposes it as `dataset.dataset_vocabulary`. The
030 script adds `"phenotype_dataset_indices"` to `follow_batch` when the token is enabled
(beside `phenotype_values`, L502) and passes `extra_val_indices={"val_ess":
holdout_record_indices}` to `CellDataModule`, whose `val_dataloader` returns `[val, val_ess]`
when the mapping is present. `COOLabelNormalizationTransform` gains `fit_table: pd.DataFrame |
None` (record index, label, value per entry) and fits on it when given; the script builds it
from `entries.parquet` restricted to the realized train indices (decision 9).

**Model readout token.** `CellGraphTransformer.__init__` takes `dataset_token: dict | None`
(`enabled`, `vocab_size`, `dim` 8, `mode: readout`); when enabled it owns `self.token_proj =
nn.Linear(vocab_size, dim, bias=False)` and constructs both heads with `token_dim=dim`.
`forward` accepts optional `entry_batch` (`phenotype_values_batch`) and `entry_dataset`
(`phenotype_dataset_indices`); when present, after the encoder (L2926) it projects the token,
gathers per decision 6, and both heads return `[E, 1]`. Disabled is the current code path.
`mode: input` is accepted by the config schema and raises `NotImplementedError` until the
ablation is built.

**Trainer.** `RegressionTask` gets `per_entry: bool`. When true, `_shared_step` selects a
label's entry rows by `phenotype_type_indices`, their genotype by `phenotype_values_batch`, and
their order by indexing `_perturbation_order(batch, batch_size)` with that genotype, replacing
`_coo_label`'s `[B, 1]` contract; both losses run on those rows, `_update_order_metrics` takes
the per-row order and logs `n_entries` beside `n_records` under the `#430` names, and the
val/test primary metric follows decision 8 using the batch's token ids. `validation_step(batch,
batch_idx, dataloader_idx=0)` routes `dataloader_idx == 1` to `_essentiality_step`, which
forwards the held-out singles under the Costanzo and SGD tokens and buffers `(gene, pred)`;
`on_validation_epoch_end` `all_gather`s the buffers on every rank and logs
`val_ess/auroc_released`, `val_ess/auroc_matched`, and the SGD-token variants (rank-zero
compute after the gather; keys exist on every rank). `_get_batch_size` is unchanged because
`perturbation_indices_batch` stays first in `follow_batch`.

**Config.** `E030/conf/default.yaml` copies 025's; `cgt_030_s3_r_tok_embfit_001.yaml` flattens
the nine-file `_034` chain into one file over it and adds `dataset.root_rel:
data/torchcell/experiments/030-solid-growth-multi/001-multi-build`, `subset.{indices,
split_file, split_key: pinned, unpinned_to_train: true, exclude:
essentiality_holdout_030.json.gz}`, `dataset_token.{enabled: true, mode: readout, dim: 8}`,
`transforms.fit_on: train_entries`, `regression_task.{fitness_lambda: 1.0, per_order_metrics:
true, per_entry: true}`, `trainer.max_epochs: 50`, W&B tags `[s3, split_R,
dataset_token_readout, per_entry, ess_holdout, ...]`. `cgt_030_smoke_tok_000.yaml` overrides
the pool, epochs 5, the synthetic token, and the control flag. Hydra `config_path` points at
this directory and `load_index_artifact` at `030-solid-growth-multi/results`.

**GilaHyper smoke (job names `030-r1w0-smoke-{cpu,tok,ctl}`, `--mem` about 2x a measured
peak).** `smoke-cpu` constructs the dataset with the FINAL S3 subset + R pin + essentiality
exclusion and calls `data_module.setup()` once, writing
`data_module_cache/index_seed_42<pin><sub>-utt.json` for 030, and asserts realized split sizes
against the artifacts (the 025 L528-541 rule). `smoke-tok` (1 GPU) trains the probe of decision
10 and prints pool sizes, entry-row counts per token, the three criteria with PASS/FAIL, and
the real-data probe tables; `smoke-ctl` trains the disabled control and prints the loss-floor
comparison; the report script writes `results/smoke_dataset_token_<jobid>.json`. All three pass
before any IGB step.

**IGB copy and launcher.** `sync_igb_030_build.sh` is the 029 script with
`REL=data/torchcell/experiments/030-solid-growth-multi/001-multi-build` (rsync follows the
symlink through `$SRC/processed/`), sending `processed/` (955 GB), the empty `raw/lmdb` stub,
and `data_module_cache/`; `gh_sync_igb_030.slurm` wraps it at 2 days, 8 GB, resumable; verify
by `find | wc -l` and `du -s` on both ends. `igb_mmli_cgt_030.slurm` is the 025 launcher with
`BRANCH`, `PROJECT_ROOT` default, `BUILD_REL`, and the conf path pointed at 030, and its
preflight refuses to start without `data_module_cache/`. The user creates one detached worktree
at the frozen commit (`git worktree add --detach <path> <sha>`; `spawn` workers with
`NUM_WORKERS` 14 and `persistent_workers` re-import live code, so it advances only between
jobs), then `sbatch
--export=ALL,PROJECT_ROOT=<worktree> -J 030-r1w1-s0 igb_mmli_cgt_030.slurm
cgt_030_s3_r_tok_embfit_001 +seed=0`. After the first cell: `sacct -j <id>
--format=MaxRSS,Elapsed`, epoch-0 and epoch-1 wall times to the weekly note, then seeds 1 and 2
(`--dependency=afterany`) with `--mem` and `max_epochs` re-sized (gotchas 3 and 4). Hypothesis
(untested): cold epoch 0 takes 5 to 8 h, warm epochs 45 to 60 min. W&B project
`torchcell_025-solid-growth_equivariant_cell_graph_transformer`, group = arm name.

**Deferred.** (4) Kuzmin 2020 loader: `M torchcell/datasets/scerevisiae/kuzmin2020.py` gains
array-strain SMF records per array strain and per table (SmfKuzmin2020 grows from 235) and a
`source_table: Literal["S1", "S3"]` field on every 2020 Dmf/Dmi/Tmf/Tmi record, values sourced
from the mirrored SI first. Adding a field to a served dataset's schema closure BLOCKS
incremental admission, so this is the full-rebuild path (`gilahyper_live_rebuild-slurm_docker.slurm`
under slurm) followed by a 030 build 002; nothing here is run. Also deferred: `dataset_token.mode:
input` ablation; the 17-token dataset x temperature vocabulary; the doubles-inclusive
essentiality holdout; the per-token normalization ablation.

## Gotchas

1. **Index cache race.** cell.py L404 is a bare `osp.exists`, no lock; a missing 030 index under
   4 DDP ranks is four concurrent scans of 13.5M records. `smoke-cpu` writes the cache, the sync
   copies it, the launcher preflight requires it. The filename embeds the subset hash
   (L380-383), so the subset must be FINAL (essentiality exclusion included) before warming.
2. **`_coo_label` had a documented conflict rule** (three 025 singles with Costanzo ~1.0 beside
   SynthLethDB 0.0); per-entry rows make both values training targets under different tokens.
   That is the intended semantics; the pool summary counts such records, and the val/test rule
   (decision 8) never averages across sources.
3. **Epoch budget vs the clock.** LMDB reads stay 1x but the readout runs 4.3x rows (decision 3)
   and the metric collections grow; the 50-epoch cap is sized from measured epoch-1 wall time,
   re-sized if epoch 1 exceeds 90 min. mmli `--time` is 4 days; a run that cannot finish 30
   epochs (the scoring window) is stopped early and its epoch count reported as partial.
4. **Memory.** `--mem=250g` was sized on the 554 GB 025 LMDB; per-rank RSS grows with the map.
   Read `MaxRSS` after the first cell and resize before seeds 1 and 2; never launch three seeds
   blind.
5. **Hardcoded 025 roots.** Every copied script and the launcher carries `025-solid-growth` in
   five places (Relevant Files); grep `E030/` for it before the smoke job and expect zero hits.
6. **`/db` at 90%.** The rsync reads only; do not stage a copy on `/db`. The 434 GB of
   deletable intermediates are the user's call (open question 2).

## Verification

- Unit tests from the worktree root, `PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python -m
  pytest <paths> -xvs`:
  - `tests/torchcell/data/test_label_policy.py` + `test_graph_processor_dataset_indices.py`:
    cherry-pick intact; tensor emitted on every sample incl. the empty branch; vocabulary rejects
    an unknown name.
  - `tests/torchcell/models/test_equivariant_cell_graph_transformer.py` +
    `test_dataset_token_readout.py`: `test_single_head_config_matches_prechange` and
    `test_perturb_cls_moves_only_the_cls` unchanged; enabled token changes outputs across tokens
    and only there; disabled token bit-identical to the pre-change forward; `mode: input` raises.
  - `tests/torchcell/trainers/`: per-entry loss equals the mean of per-row MSEs on a hand-built
    batch; per-order counts report `n_entries` and `n_records`; the val rule picks the
    LabelPolicy source and averages same-source rows; keys logged in a fixed order with NaN for
    an unseen token; `validation_step` with `dataloader_idx=1` logs the four `val_ess` keys.
  - `tests/torchcell/datamodules/test_cell.py`: 22 existing + `extra_val_indices` returns a
    second loader that never intersects train/val/test.
- mypy on the five W025 MODIFY modules (Relevant Files) and `ruff check` on them plus `E030/`.
- Artifact counts (printed and asserted by each script): S3 = 5,694 + 739,236 + 376,732 =
  1,121,662; pinned val = test = 37,673, train pin = the remainder of 376,732, zero unmatched
  010 triples; holdout (a) count reported with the 223 released test genes minus those absent
  from 030 singles; (b) exactly 250 + 250, coverage AUROC in [0.47, 0.53]; excluded record
  indices all order 1 and disjoint from pinned val/test; pinned eval table rows = 2 x (37,673 +
  37,673) minus pairs the policy leaves empty (count printed); exact-duplicate tuple count
  printed.
- Smoke (decision 10): the three criteria printed as PASS; the 030 index cache present with
  realized sizes equal to the artifacts; `grep -c 025-solid-growth E030/scripts/* E030/conf/*`
  = 0.
- IGB: the copy and first-cell checks of Approach recorded; W&B run shows
  `val/n_records/gene_interaction/order3` = 37,673, the 15-token metric keys, and
  `val_ess/auroc_*` every epoch.

## Open Questions

1. Land `#346` and `#430` first, or stack this branch and land all three in order (`#435`
   independently)? The plan assumes stacking, order `#346` -> `#430` -> new.
2. Storage: delete the 029 build from IGB scratch (not needed, 346 TB free, no quota shown; the
   plan leaves it), and reclaim the 434 GB of deletable intermediates on `/db` (90% full)?
3. Size of the matched genome-wide essentiality set: 250 + 250 proposed; larger sets tighten the
   CI but eat into the 907 measured essentials available for training.
4. Temperature in the token vocabulary for arm 1: no, by the user's rule (dataset one-hot only);
   the 17-token variant is a deferred ablation.
