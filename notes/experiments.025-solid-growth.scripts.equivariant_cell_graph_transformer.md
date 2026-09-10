---
id: t3zbu45qsge27ox14lppfos
title: Equivariant_cell_graph_transformer
desc: ''
updated: 1788575502881
created: 1788575502881
---

## 2026.09.04 - Port of the 010 Trainer onto the 025 Build

Trains the equivariant `CellGraphTransformer` on one subset/split arm of the 025
all-solid-growth build. Ported from
`experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer.py`. Design,
arm table, and the masking-layer decision live in
[[experiments.025-solid-growth.training-plan]].

### What the arm config has to say, and why each is load bearing

010 trained on 376,732 records holding exactly the trigenic triples and exactly one
label. 025 holds 13,525,071 records over three perturbation orders and two labels, so
four things that were implicit there have to be stated here. Each produces a plausible
run rather than an error when omitted:

| config key | without it |
|---|---|
| `cell_dataset.phenotype_labels` | a triple carries fitness AND gene_interaction, so a batch of B records supplies 2B targets |
| `subset.indices` | pinning assigns but never excludes, so the 13,142,648 doubles join training |
| `subset.split_file` + `split_key` | no pinned split; R and Q store their lists under different field names |
| `transforms.fit_on_subset` | normalized by the whole column's sd 0.0444 instead of the triples' 0.0633 |

The script asserts the realized split equals the pinned artifact intersected with the
subset, per split, before training starts. That assertion is the one place a subset and a
pin can silently disagree: an index named by the pin but absent from the subset is
dropped rather than placed, which is correct behavior and also exactly how an arm could
train on fewer records than its name claims.

### Verified against the build

- Dataset opens in 10.5 s at 5.5 GB RSS; `len` 13,525,071; perturbation orders
  1: 5,694 / 2: 13,142,648 / 3: 376,732.
- The nine gene-gene relations are `physical_interaction` (144,211 edges),
  `regulatory_interaction` (44,310), `tflink` (207,250), `string12_0_coexpression`
  (1,002,806), `string12_0_experimental` (828,701), `string12_0_neighborhood` (153,320),
  `string12_0_database` (79,224), `string12_0_fusion` (18,394),
  `string12_0_cooccurence` (17,692); 6,607 gene nodes.
- Normalization on S0: mean -0.008024, sd 0.063264, min -1.0816, max 1.128043, matching
  the 010 report's label statistics.
- R split realizes 301,386 / 37,673 / 37,673; Q split realizes 301,236 / 37,705 / 37,791.
  Both equal their artifacts exactly, and every split record is a triple.
- Model instantiates at 4,774,861 parameters under both configs, the same count as 010.
- One-epoch smoke at batch 8 on a single GPU: 21 batches in 17 s under KL, 8 s under
  masking. The direction agrees with 019's 1.5-1.7x, but batch 8 is not the production
  regime and this is not a throughput measurement.

### Reference the replication arm is read against

010's three checkpoints reached validation Pearson 0.4520 (M01, `lzs9pcj3`), 0.4472
(M02, `yv4r30bi`) and 0.4619 (M03, `c7671wgj`). Their best-Pearson checkpoints sit at
epochs 24 and 25, and the cosine schedule's first cycle is 30 epochs, so the 12 h wall
clock is being spent in the range where 010 peaked rather than truncating a long climb.

## 2026.09.09 - The Normalizer's Fit Population Is Named, Not Assumed

`transforms.fit_on_subset: true` became `transforms.fit_on: subset | train`. `subset`
is the old behavior, the mean and sd of every record in the arm, which is what 010 did
and what the replication arm must keep. `train` fits on the pinned training split
intersected with the subset, so no validation or test label reaches the two constants.
Any other value raises. The run logs `arm/norm_fit_on`, `arm/norm_fit_records` and the
resulting mean and sd to W&B so the population is on the record.

Every 025 config carries the key: the seven that had run or been queued before this
change say `subset`, so they still describe the runs they produced; `cgt_s0_q_kl_004`
(job 1609, the query-pair-disjoint KL arm) says `train`. Smoke-tested on CPU with one
batch of two records under the 004 config: the normalizer reports 301,236 records,
mean -0.007844, sd 0.063776, matching
[[experiments.025-solid-growth.scripts.label_normalization_constants]].

## 2026.09.08 - Joint Fitness Head (opt-in)

### Where the arms stand on GilaHyper

Read from W&B on 2026.09.08 (project `torchcell_025-solid-growth_equivariant_cell_graph_transformer`):

| job | config | run | val Pearson | note |
|---|---|---|---|---|
| 1598 | `cgt_s0_r_kl_000` (KL, 010 verbatim) | `0yw7moue` | best 0.4463 at epoch 14; 0.4425 at epoch 35 when the 12 h clock ended it | train Pearson 0.5555 at the end; the working reference |
| 1606 | `cgt_s0_r_mask_003` (mask layer 1) | `7f1yrsq9` | 0.3066 at epoch 1, then 0.00 +/- 0.01 through epoch 49 | collapsed; train Pearson -0.002 |
| 1607 | `cgt_s0_r_mask_001` (mask layers 2-5) | `4qmgkcgn` | 0.00 +/- 0.01 for 17 epochs, 0.2537 at epoch 17 (in flight) | partial; train Pearson 0.0004 |

"Best" is a max over epochs, an upward-biased order statistic. The replication is the
only arm that trains, so the joint-fitness arms build on it.

### What the head is

The interaction head (`PerturbationHead`) is a two-layer MLP, 360 to 180 to 1 with ReLU
and dropout 0.1, on `[h_CLS || z_S]`: `h_CLS` is the class token of the encoder run on
the wild-type graph, identical for every strain in the batch, and `z_S` is the sum of the
perturbed-gene embeddings of the strain's deleted genes after the equivariant
perturbation transform. All strain dependence enters through `z_S`.

The fitness head is `GlobalHead` (`model.heads.global`), the existing whole-cell readout:
the same shape of MLP on `[h_CLS || mean_i H_genes_pert[b, i]]`, a mean over all 6,607
perturbed gene embeddings. Predicting fitness from the class token alone is not possible
in this architecture, since that vector carries no strain; `use_gene_pool: true` is what
makes the head strain-dependent.

### What changed

- `RegressionTask(fitness_lambda=...)` in `torchcell/trainers/int_transformer_cell.py`.
  `None` is the single-label path, unchanged. A float decodes both labels from the COO
  fields by type (`_coo_label`, rows from `phenotype_values_batch`), adds
  `fitness_lambda * MSE(global_head, fitness)` on the standardized scale to the 010
  objective, and logs `{stage}/fitness/{MSE,RMSE,Pearson}`,
  `{stage}/transformed/fitness/...` and `{stage}/fitness_loss` beside the existing
  gene_interaction metrics. Checkpoints still follow `val/gene_interaction/*`.
- The script reads `regression_task.fitness_lambda` and `model.heads`, passes
  `heads_config` to the model, appends `phenotype_values` to `follow_batch` on the joint
  path, and refuses a joint config whose labels or heads are incomplete.
- Configs `cgt_s0_r_kl_fit_008` (weight 1.0) and `cgt_s0_r_kl_fit_009` (weight 0.1)
  compose on top of `cgt_s0_r_kl_000`, so the diff to the replication is the label list,
  the fitness normalizer, the head, and the weight.
- Launcher [[experiments.025-solid-growth.scripts.igb_mmli_cgt]].

Hypothesis (untested): the trigenic interaction score is a residual of the triple's
fitness against its subsets, so the fitness target should shape a representation the
interaction head can use. The three IGB runs (weight 1.0, control, weight 0.1, all seed
42) are what measures it.

## 2026.09.09 - Which Fit Population the Fitness Arms Use, and What Was Cancelled

The leak is the one the section above fixes, and the fix landed on main from the
additive-baselines side while this branch was open: `transforms.fit_on: subset | train`,
with the replication arm deliberately keeping `subset` so it still reproduces 010's two
constants. This branch had briefly forced train-only on every arm, which would have
broken exactly that replication; main's version is the one that survived the rebase.

The fitness arms take `fit_on: train` (set in `cgt_s0_r_kl_fit_008`, inherited by `_009`
through `_011`), and the experiment's control is `cgt_s0_r_kl_ctrl_012`, which is
`cgt_s0_r_kl_000` with the same key changed and nothing else. A named control rather
than a launch-time override, because a control that differs from its treatment arms by
the normalizer as well as the head answers nothing, and an override typed at launch is
invisible in the config on a rerun. All four arms therefore standardize by the same
301,386 training records; against job 1598 and the 010 checkpoints they are comparable in
raw units only.

Cancelled for carrying the all-record constants: GilaHyper 1609 (soft KL, disjoint, had
not started; resubmitted as 1640 from main, which has both the fix and the config), the
Delta canary 21895901 at 19 h, and the 26 pending sweep jobs 21917113 to 21917138. The
Delta fitness replicates 21919310 to 21919313 are still queued and still run the `_008`
pool-readout design.

`gh_cgt.slurm` now runs from the submitting checkout (`SLURM_SUBMIT_DIR`) and accepts
Hydra overrides after the config name, like `delta_cgt.slurm`, so a branch can be
launched on GilaHyper before it lands.

## 2026.09.09 - Fitness from the Perturbed CLS

The `_008` arm read fitness from `[h_CLS || mean pool over genes]`, and the CLS half of
that is the same vector for every strain: the encoder runs once on the wild-type graph and
the token is sliced off before the perturbation operator (across-strain sd 0.0 against
0.973 for z_S, measured in the 019 expression strand). The 019 review of perturbation
operators (see the note on `perturb_cls` in
[[torchcell.models.equivariant_cell_graph_transformer]]) found nothing pre-encoder had
ever been run and that the cheapest of the three designs in
`notes-tex/019-simb-multimodal-expression/sections/3-next.tex` is to run the existing
operator on the CLS too.

That is `model.perturb_cls: true`: the CLS is query row 0 of the same cross-attention,
over the same deleted-gene keys, so 010's operator is unchanged, the gene rows are
bit-identical (test `test_perturb_cls_moves_only_the_cls`), and `h_CLS_pert` is
strain-specific. The one-key degeneracy the expression strand fought does not arise here:
every 025 record is a triple, so the softmax has three keys.

Arms, both composing on `_008` and read against the same control:

| config | fitness head | weight |
|---|---|---|
| `cgt_s0_r_kl_fit_010` | linear probe of `h_CLS_pert`, no gene pool | 1.0 |
| `cgt_s0_r_kl_fit_011` | same | 0.1 |

The interaction head keeps the wild-type CLS (`perturbation_head_cls: wildtype`), so the
fitness gradient reaches the interaction prediction only through the shared trunk. The
`perturbed` setting is the follow-up arm. Logged: `{stage}/cls_pert_strain_sd`.

Queued on IGB mmli in place of the `_008`/`_009` pair (cancelled before starting):
`cgt_s0_r_kl_fit_010`, `cgt_s0_r_kl_ctrl_012`, `cgt_s0_r_kl_fit_011`, seed 42. The Delta
replicates 21919310 to 21919313 still run `_008` and `cgt_s0_r_kl_000`, so they answer
the same question with the pool readout and the replication's normalizer.
