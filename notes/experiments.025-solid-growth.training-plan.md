---
id: valhzat7nky3ct86cwh30e8
title: Training Plan
desc: ''
updated: 1788502935383
created: 1788502935383
---

Training campaign on the 025 all-solid-growth build ([[experiments.025-solid-growth]]),
revised from the 2026.09.04 discussion. Destination: the paper's R2 claim (state-of-the-art
trigenic interaction prediction; DANGO 0.367, 010 CGT 0.454 on the random split) and the
guilt-by-association GO annotation supplementary figure. Thesis mechanism under test:
interaction scores are deterministic functions of fitness values, so bringing fitness and
lower-order interactions to bear should improve tmi prediction if the network can exploit
the identity

    tau_abc = f_abc - f_ab*f_c - f_ac*f_b - f_bc*f_a + 2*f_a*f_b*f_c.

## 2026.09.04 - Revised Plan

### 0. What the evidence base says going in

- Additive null (B1 per-gene ridge) reaches test Pearson 0.400 on the random-over-records
  010 split vs the CGT's 0.443-0.455, and COLLAPSES to 0.127 +/- 0.033 on the
  query-pair-disjoint split (5-fold CV); the embedding MLP (B5), the only baseline that can
  represent a three-way term, inverts BELOW the additive null there (0.058). The
  transformer has never been refit on the disjoint split
  ([[experiments.010-kuzmin-tmi.additive-baseline-analysis]], the 010-positive-panel
  report Sec 2.5 and 10.2).
- Every 010 triple carries exactly one recurring query pair (420 pairs, count
  distribution bimodal with nothing between 2 and 200), so the disjoint split is exact
  and cheap to build (`experiments/010-kuzmin-tmi/scripts/query_pair_disjoint_split.py`).
- 025 holds the 376,732 triples with 010's split pinned exactly
  (`results/pinned_splits_from_010_seed_42.json`), plus 5,694 singles and 13,142,648
  doubles, each group carrying both `fitness` and `gene_interaction` labels where they
  exist (fitness 13,525,071 / gene_interaction 13,515,659).

### 1. Gate first: can 025's fitness recapitulate its own tmi

`experiments/025-solid-growth/scripts/recapitulate_tmi_from_fitness.py`
([[experiments.025-solid-growth.scripts.recapitulate_tmi_from_fitness]]) recomputes tau
per triple from the stored smf/dmf/tmf and compares against the stored tmi, two ways:

- **Variant A "aggregate"**: fitness = mean over ALL of a genotype group's fitness
  entries (Costanzo + Kuzmin + essentiality-derived), which is exactly the target a
  per-entry MSE trains toward -- the fitness the MODEL will see.
- **Variant B "kuzmin"**: fitness restricted to Kuzmin-sourced entries, the screen family
  that produced the tau labels -- isolates whether cross-source aggregation is what
  breaks the identity.

Plus the digenic identity (eps_ab = f_ab - f_a*f_b vs stored dmi) on the closure pairs,
closure-coverage counts, and positive-call confusion at tau > +0.08/+0.16/+0.20.

Decision rule:

- **A high (r >= ~0.9)**: aggregated fitness carries the tau signal; ladder proceeds as
  designed.
- **A poor, B high**: aggregation across sources breaks the identity; revise the fitness
  channel before training (options: per-entry COO supervision with source identity kept,
  or source-faithful fitness for closure records).
- **Both poor**: the stored tmi is not consistent with any stored fitness; the
  fitness-helps mechanism is unsupported in this data as built -- verify against the raw
  Kuzmin release before any training, and consider ingesting Kuzmin's own
  single/double-fitness columns as the supervision source.

Caveats stated up front: the recapitulation r is a REFERENCE, not a strict model ceiling
(the model also uses gene identity and graph structure, and stored tmi is itself a noisy
mean); and the wet-lab normalization finding (corr(eps, f_a*f_b) = -0.926 in the
010-positive-panel report) concerns OUR assay, not Kuzmin's published values -- this
script quantifies the dataset-side analog.

### 2. The subset ladder

Design rules that make the ladder interpretable:

- **Evaluation is fixed across all arms.** Metrics are computed on tmi triples only, on
  identical val/test records within a split regime. Arms differ ONLY in training data.
- **Two split regimes, both run**: **R** = random-over-records, pinned to 010
  (continuity + SOTA comparison); **Q** = query-pair-disjoint (the honest generalization
  regime; the additive bar there is 0.127). Q is built by porting the 010 disjoint-split
  logic onto 025's triples via gene-set identity, non-triple records all assigned to
  train.
- **Q leakage rule**: in Q arms, doubles whose gene pair IS a val/test query pair are
  EXCLUDED from training (a per-pair fitness signal could stand in for the screen batch
  offset that B4 exploited). One diagnostic arm includes them, to measure the effect.

Arms:

| arm | training data | question it answers |
|---|---|---|
| S0 | triples, gene_interaction only | anchor: new architecture (hard mask, sum pooling) vs 010's 0.443-0.455 |
| S1 | triples, gi + fitness (tmf) | does the triple's OWN fitness help tau |
| S2 | S1 + all 5,694 singles (smf) | do the equation's f_a, f_b, f_c terms help |
| S3 | S2 + closure doubles (dmf + dmi on pairs inside triples) | does the full tau closure help -- the thesis arm |
| S4 | S2 + size-matched random non-closure doubles | ablation: "the equation's doubles" vs "any doubles" |
| S6 | dmi (+ fitness, all orders), gi head never sees a triple | zero-shot order transfer: the cleanest "learned the general equation" test -- if tmi beats the additive null with no trigenic gi supervision, the head is computing the equation, not memorizing order-3 labels |
| S5 | full 025 (13.5M) | endpoint, later; compute-heavy |

On size-matching (the saturation concern): closure doubles are ~600-740k records
(recapitulation reports the exact count), 1.6-2.0x the triple count -- near size-matched
already, and maximally relevant since they are exactly the lower-order terms of the test
triples' equation. Recommendation: keep closure COMPLETE and handle imbalance with
per-phenotype/per-order LOSS WEIGHTS (recorded in config) rather than dropping equation
terms; if strict matching is wanted, rank closure pairs by the number of training triples
they support and take the top 376,732. S4 exists to show the choice matters.

Statistical discipline: >= 3 seeds per arm, val-selected checkpoint (never max over
epochs), mean +/- sd. With n_test = 37,673 the Pearson null width is ~0.005; at 3 seeds
treat arm gaps < ~0.015 as unresolved rather than ranked.

### 3. Mechanics

- **Splits/subsets as committed artifacts**: a `subset_definitions.py` script in
  `experiments/025-solid-growth/scripts/` reads `perturbation_count_index.json` +
  the recapitulation closure table and writes one JSON index list per arm to `results/`;
  the Q split ports `query_pair_disjoint_split.py` (same output schema as
  `index_seed_42.json`).
- **`CellDataModule` needs one small feature**: an `index_subset` parameter restricting
  the record pool before splitting (hashed into the split-cache tag, as
  `pinned_split_indices` already is). Without it a subset cannot be expressed -- pinning
  assigns but never excludes.
- **Baselines on every arm/split**: B0/B1/B2/B5 from
  `experiments/010-kuzmin-tmi/scripts/additive_baseline_gene_interaction.py` pointed at
  025's `label_df.parquet` -- cheap, and every CGT number needs its additive null beside
  it.

### 4. Model

- `CellGraphTransformer` with **hard masking**: `model.attention_mask.enabled: true`,
  `graph_regularization.graph_reg_lambda: 0.0`, following
  `experiments/019-simb-multimodal/conf/cgt_expr_010.yaml` (1.5-1.7x faster/epoch;
  mask-vs-KL ACCURACY is unmeasured -- S0 is the bridge to 010's KL numbers).
- **Free heads**: assign the 9 graphs via `attention_mask.head_graphs` and leave extra
  heads unassigned (e.g. 12 heads = 9 masked + 3 free), so the model keeps unconstrained
  attention capacity alongside the graph-structured heads.
- **Decoders (Type II virtual instruments)**: ONE `PerturbationHead` for
  `gene_interaction` serving orders 2 AND 3 through the same parameters -- the head is a
  set function on the perturbed-gene tokens, `pooling: sum` (cardinality-aware; `mean`
  measured cardinality-blind), which is what makes "one head approximates the general
  equation" a testable claim rather than a metaphor. A SECOND head instance for `fitness`
  across orders 1-3 (second `PerturbationHead` preferred over `GlobalHead s1_pool`;
  fitness is also a set function, and the shared trunk carries the cross-task transfer).
- Per-order metrics reported from the single gi head (order-2 and order-3 separately),
  plus retrieval metrics (P@100 and average precision at tau > +0.08 / +0.16 / +0.20 and
  the negative side) since the panel work consumes retrieval, not regression.

### 5. Guilt-by-association GO annotation (after model selection)

Best predictor -> genome-scale predicted digenic interaction profiles (trigenic where
affordable) -> profile similarity -> annotation transfer, evaluated on held-out GO terms
against the measured-profile baseline (Costanzo-style). Feeds the `profile-enrichment`
and `inference-scale` supplementary figures. Calibration carries over from the panel
report (regression of actual on predicted slope 0.877).

### 6. Open items and risks

- **Label parity 010 vs 025**: aggregation may have shifted the shared triples' label
  values; run a quick paired check (reuse the transfer script's gene-set maps) before
  reading S0 vs 010 as an architecture comparison.
- The smoke-flagged genotype that aggregated 23 `SmfKuzmin2020` experiments -- inspect
  before training (dedup identity may be coarser than intended).
- 185 groups mix essentiality-derived fitness with smf measurements; their means pull
  toward 0. Only matters if such genes enter closure.
- Compute: S3 trains on ~1.1M records, roughly 3x 010's per-epoch cost; solo-GPU for
  long arms per the 019 packing measurements. Cluster allocation decided at launch time.

### 7. Ordered execution

1. Recapitulation run -> report + decision (running now).
2. Label-parity check 010 vs 025.
3. `subset_definitions.py` + Q split port -> committed JSON artifacts.
4. `CellDataModule.index_subset` + tests.
5. Configs + trainer for S0/S1 (base: the maintained multitask trainer
   `experiments/019-simb-multimodal/scripts/train_cgt_multitask.py`), R and Q regimes.
6. Baselines on the same subsets/splits.
7. Launch S0/S1, 3 seeds; then S2/S3/S4/S6 as results arrive; S5 last.

## 2026.09.04 - Gate Measured, Subsets Materialized

**The gate fired on the "both poor" branch** (full numbers and figures in
[[experiments.025-solid-growth.scripts.recapitulate_tmi_from_fitness]]): recomputing tau
from the build's own fitness reaches r = 0.230 (aggregate) / 0.354 (Kuzmin-sourced) /
0.213-0.272 (stored-dmi variant) against the stored tmi, with 99.99% closure coverage,
digenic slope 0.99, and essentiality contamination ruled out as the driver (<= 0.06
effect). The identity holds in expectation and drowns in propagated error: recomputed
rmse 0.11-0.17 against a label sd of 0.0633. The additive null fit to labels (0.400)
beats the physical equation evaluated on stored values.

Plan consequences, applied:

- The thesis mechanism is REVISED from "the network computes the equation from fitness
  inputs" (capped near 0.35 here) to "fitness/dmi supervision improves the learned
  representation" -- which is exactly what the S-ladder measures, so the ladder stands
  unchanged. Expected effect sizes are tempered; the MDE discipline (gaps < 0.015 at 3
  seeds unresolved) matters more, not less.
- S6 (zero-shot order transfer) is PROMOTED in importance: beating the disjoint additive
  null (0.127) with no trigenic gi supervision would show transferable interaction
  structure that the noisy identity cannot supply.
- Per-entry COO supervision with source identity (rather than group means) is the
  data-side lever if aggregate fitness proves too noisy a channel -- the entries are
  already stored; a config choice, not a rebuild.
- Label parity 010 vs 025 is CLOSED: all 376,732 genotypes matched, gi labels identical
  to float precision, and 010 carried no fitness labels at all -- S0 is a clean
  architecture anchor and every fitness value in S1+ is new signal.

**Subset artifacts landed** (`subset_definitions.py`, summary in
`results/subset_definitions_summary.json`): S0 = 376,732; S2 = 382,426; S3 = S4 =
1,121,645 (closure doubles found 739,219). Query-pair-disjoint split over 025: **420
recurring pairs, exactly 010's count**, train 301,236 / val 37,705 / test 37,791 records
over 331/43/46 pairs (`results/query_pair_disjoint_splits_025.json.gz`); the strict leakage
rule removes 85 training doubles whose pair is a held-out query pair
(`results/subset_Q_excluded_doubles.json.gz`).

Next implementation steps (unchanged order): `CellDataModule.index_subset`, S0/S1
configs + trainer port, baselines on the same subsets, launch at 3 seeds.

## 2026.09.04 - Trainer, Configs, and Where the Free Attention Goes

Steps 4 and 5 of the ordered execution are implemented: `CellDataModule.index_subset`
plus the three pieces the 010 trainer needed to run on a multi-order, multi-label build,
and three arm configs. The first arm launches tonight.

### The three arms, and why in this order

| config | subset | split | graph prior | question |
|---|---|---|---|---|
| `cgt_s0_r_kl_000` | S0 | R (010's) | KL, 010 verbatim | does the 025 data path reproduce 010's 0.4520 / 0.4472 / 0.4619 |
| `cgt_s0_r_mask_001` | S0 | R | hard mask | what does swapping KL for masking cost or gain |
| `cgt_s0_q_mask_002` | S0 | Q (disjoint) | hard mask | the honest generalization number, against the additive 0.127 |

The replication runs first and alone because it is the only arm whose answer is already
known. If 025 + the new subset machinery gives 0.45 on the R split, every later number
rests on a data path that has been checked against an independent build. If it does not,
nothing further is interpretable and the gap is the thing to chase.

Running the mask swap on R before Q is what keeps a Q result readable: Q changes the
split AND the graph mechanism relative to 010, so a drop there has two candidate causes
unless the mask has been priced on a split where a reference exists.

### Four ways the 025 build silently breaks a ported 010 trainer

Each of these was measured on this build, not assumed, and each produces a plausible run
rather than an error:

1. **Two labels per record.** 025's `experiment_types.json` holds `gene interaction` and
   `fitness`, so `phenotype_values` is the concatenation over both and a batch of B
   records carries 2B targets. `RegressionTask` reads that tensor as its label vector.
   Fixed by `Neo4jCellDataset(phenotype_labels=["gene_interaction"])`, which also fixes
   the ORDER: `_load_phenotype_info` returned `list(set_of_classes)`, whose iteration
   order over type objects is not guaranteed to agree between DDP ranks, and
   `phenotype_type_indices` is positional.
2. **Pinning cannot express a subset.** `pinned_split_indices` assigns records but never
   excludes them, so pinning the 376,732 triples still leaves 13,142,648 doubles and
   5,694 singles seed-assigned into train. `index_subset` restricts the pool before any
   splitting, and is hashed into the split-cache filename separately from the pin tag
   because the ladder varies pool and split independently.
3. **Normalization fitted on the wrong population.** `gene_interaction` has sd 0.0444
   over all 13,525,071 records and **0.0633 over the 376,732 triples**, which is the value
   010 normalized by (its min/max, -1.0816 / +1.1280, match the triples exactly). Fitting
   on the whole column divides every trigenic target by a constant 1.43x too small and
   raises the effective learning rate with it. `COOLabelNormalizationTransform` now takes
   `fit_indices`, resolved through `label_df`'s `index` COLUMN since its rows are not
   positional.
4. **Relation names.** `cell_graph` names the nine gene-gene relations
   `physical_interaction`, `regulatory_interaction`, `tflink`, and six `string12_0_*`.
   010's config regularizes `physical` and `regulatory`, and
   `compute_graph_regularization_loss` skips an unmatched name with a bare `continue`, so
   **two of nine heads carried no graph prior in every 010 run**. The replication config
   keeps the mismatch on purpose: 0.4520 / 0.4472 / 0.4619 were produced by a model
   regularized on seven graphs, and correcting it would make a disagreement
   uninterpretable. The mask arms use the true names and raise on a mismatch.

Measured together end to end (`probe_025_datamodule.py`, kept in scratch): dataset opens
in 10.5 s at 5.5 GB RSS, splits realize as 301,386 / 37,673 / 37,673 exactly matching the
pinned artifact, every split record is a triple, and a batch of 8 carries 8 labels and 24
perturbation indices.

### Free heads: separate rows, not parallel heads in one layer

The open design question was whether unconstrained attention should sit BESIDE the
graph-carrying heads inside the same layer, or in layers of its own. The concern with the
parallel form is that both paths write into the same residual stream at the same depth
and the unconstrained one is strictly easier to fit, so the gradient could route around
the graph entirely.

`_build_head_mask` allows either: heads with no assigned graph stay fully free, and
`attention_mask.layers` chooses which layers are masked at all. The configs take the
vertical form, **SSMMMMSS**: layers 0-1 and 6-7 fully unmasked, layers 2-5 fully masked
with all nine heads carrying a graph.

The evidence is Buterez et al. 2024, whose entire architectural claim is that masked and
unmasked attention combine vertically rather than horizontally. Their Table 8 ablation
over layer orderings reports that "the top configurations tend to include self-attention
layers at the front, with masked attention layers in the middle and self-attention layers
at the end, surrounding the PMA readout", and that "naive configurations such as
all-masked layers or simply alternating masked and self-attention layers do not tend to
be optimal". The horizontal form is SAN's, mixing masked and unmasked scores inside one
layer by a weight gamma, and ESA outperforms it across their 70 tasks.

Two limits on that transfer, both worth stating because neither is settled by their work:

- The gradient-bypass mechanism is a **hypothesis (untested)**. Buterez et al. measure
  which orderings score best; they do not measure why a horizontal mix underperforms, and
  neither does any arm here.
- Their tokens are edge sets on molecular, vision, and social graphs. Ours are 6,607 gene
  tokens plus a CLS, and the mask is a gene-gene adjacency rather than an edge-adjacency.
  The ordering is being carried across domains as a prior.

Note also that 010's KL arrangement was already vertical in effect: all nine heads of
layer 1 were assigned, leaving seven of eight layers unconstrained. The mask configs
widen the constrained band from one layer to four while keeping half the depth free.

### Mechanics

- Launcher: `experiments/025-solid-growth/scripts/gh_cgt.slurm <config-name>`, 4 GPUs,
  12 h, 250 GB. Twelve hours rather than 010's seven days because these arms are read at
  a checkpoint; the val-Pearson callback keeps the best epoch, so a truncated run still
  answers its question.
- `pinned_splits_from_010_seed_42.json` is now written gzipped (3.7 MB to 0.85 MB) so it
  is a committed artifact beside the subset and Q-split indices rather than a large
  untracked file the configs depend on.
- The masked arms set `graph_reg_lambda: 0.0`, which stops the model requesting attention
  weights and lets the fused SDPA kernel run, so the `[batch, 9, 6608, 6608]` score
  matrix is never formed. 019 measured 1.5-1.7x faster per epoch from this; the ACCURACY
  of mask against KL is unmeasured, which is what `cgt_s0_r_mask_001` is for.

## 2026.09.05 - First Launch OOM-Killed at Worker Spawn

Job 1597 (`cgt_s0_r_kl_000`, 4 GPUs, `--mem=250g`) was OOM-killed after 4 m 56 s. All
four ranks had built the dataset and realized the split correctly
(`train=301386 val=37673 test=37673` on every rank) and the sanity check passed; it died
starting the training dataloader, surfacing as `_pickle.UnpicklingError: pickle data was
truncated` inside `spawn_main` when the killer took a process mid-transfer.

**Cause, measured.** DataLoader workers are spawned, so each receives the whole dataset
object as a pickle. On this build that pickle is 0.65 GB, of which 0.62 GB is six
lazily-loaded caches: `_label_df` 0.30, `_phenotype_label_index` 0.13,
`_dataset_name_index` 0.13, `_perturbation_count_index` 0.06, and the two `_is_any_*`
indices unset. Four ranks at 14 workers is 56 copies, and the unpickled objects are
larger than the wire format. 010 never hit this because its build was 376,732 records
rather than 13,525,071.

**Fix.** `Neo4jCellDataset.__getstate__` now drops those six alongside the LMDB env,
taking the worker pickle to **0.04 GB**. Nothing in `get()` reads them: an item needs the
LMDB record, `cell_graph`, and `phenotype_info`. They are consumed once, in the parent,
by `CellDataModule._compute_and_save_index` at datamodule construction. The index
properties also re-read their JSON on every access rather than returning the cached
attribute, so these fields were not serving as caches on the read path to begin with.

The parent keeps them: `__getstate__` copies `__dict__` before clearing, which is
asserted in `tests/torchcell/data/test_neo4j_cell.py` because clearing in place would
work once and then strip the process that actually uses them.

`--mem` stays at 250 g. Measured need after the fix is roughly 50 GB (4 ranks at 5.6 GB
plus 56 worker interpreters), so the allocation is not the thing that was marginal.
