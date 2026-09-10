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

Later the same evening the IGB set was torn down, the same three arms were queued on
GilaHyper as jobs 1656 to 1658 behind the disjoint arm 1640, and then replaced by the
constant-rate versions below before any of them started.

## 2026.09.09 - Constant-Rate Protocol (`ctrl_013`, `fit_014`, `fit_015`)

Under CosineAnnealingWarmupRestarts (peak 5e-4, floor 1e-7, 30-epoch cycle, 0.7 per
restart) the learning rate is a function of the epoch, so a best-validation epoch is
partly a statement about where the rate was, and arms stopped at different epochs are
read at different rates. The cosine runs all peak inside the first cycle: job 1598 at
epoch 14, the 010 checkpoints at 24 and 25, the Delta canary at 8. Consistent with the
schedule shaping the curve; not proof of it.

`cgt_s0_r_kl_ctrl_013` removes the schedule: `regression_task.lr_scheduler: null`, so
`RegressionTask.configure_optimizers` returns the bare AdamW at `lr: 2.5e-4`, and
`trainer.max_epochs: 30` inside the 12 h clock. The rate is a choice, not a measurement:
the mean of the cosine over its first cycle, so the 30-epoch rate integral matches the
run it replaces. No 025 or 010 model has trained at a constant rate before; the first
three runs are the measurement, and a collapse or a crawl points at the rate first.
Every arm under the protocol is read twice, at epoch 30 (fixed) and at its best
validation epoch (an upward-biased max), and the two are reported together.

`ctrl_013` is written to be the graph-regularization sweep's lambda 1e-3 point on the
random split: no penalty, the lambda ladder, the hard mask and the random-graph control
are this config with one key changed each. The 010 checkpoints and job 1598 stop being
the figure's anchors, since their schedule and normalizer differ. The figure's design
strip (`notes/assets/drawio/FigS-graph-regularization-sweep.gen.py`) still describes the
cosine, 24 h, Delta design and needs regenerating for this protocol; on Delta A40s at 79
min/epoch, 30 epochs is 40 h.

| GilaHyper job | config | arm |
|---|---|---|
| 1640 (running, cosine) | `cgt_s0_q_kl_004` | soft KL, disjoint split, the additive-baselines report's run |
| 1659 | `cgt_s0_r_kl_fit_014` | fitness from the perturbed CLS, weight 1.0, constant rate |
| 1660 | `cgt_s0_r_kl_ctrl_013` | control, constant rate |
| 1661 | `cgt_s0_r_kl_fit_015` | fitness from the perturbed CLS, weight 0.1, constant rate |

Job 1640 keeps the cosine on purpose: the report reads it against job 1598 and the 010
band, which share that schedule. On Delta the four `_008` / `kl_000` replicates
(21919310 to 21919313, cosine, old commit) were cancelled and replaced by 21934081
(`fit_014`, seed 1), 21934082 (`ctrl_013`, seed 1), 21934083 (`fit_014`, seed 2) and
21934084 (`ctrl_013`, seed 2), 48 h each so 30 epochs fit at the canary's 79 min/epoch.
With GilaHyper's seed 42 that is three seeds of the weight-1.0 arm and its control.

## 2026.09.10 - Delta fitness arms lost to a file-lock timeout, resubmitted staggered

Measured. 21934081 and 21934083 (`fit_014`, seeds 1 and 2) both exited FAILED at
06:12:41 after 30 and 28 minutes, one rank of each raising
`filelock._error.Timeout` on `processed/phenotype_label_index.json.lock` under the
Taiga-backed 025 build (`FileLockHelper.default_timeout` is 60 s; the index file is
248,745,547 B, read under the lock by every rank). The two control jobs 21934082 and
21934084 (`ctrl_013`, same code path through the index) survived and stood at epoch 7
after 6 h 26 m and 6 h 06 m, about 50 min/epoch, faster than the canary's 79. GilaHyper
1659 (`fit_014`, local disk) passed the same step and was at epoch 2 after 57 min.

Hypothesis (untested): four jobs, sixteen ranks, reached the index read within the same
few minutes, and a rank whose 60 s wait outlasted the holders' NFS reads of the 248 MB
JSON timed out; the fitness arms lost because of where their ranks landed in that queue,
not because of anything in the fitness code. Resubmitted from the same commit (33802c10)
with the second job held until 30 minutes after the first starts:

| Delta job | config | seed | note |
|---|---|---|---|
| 21947151 | `cgt_s0_r_kl_fit_014` | 1 | replaces 21934081 |
| 21947152 | `cgt_s0_r_kl_fit_014` | 2 | replaces 21934083; `--dependency=after:21947151+30` |
| 21934082 | `cgt_s0_r_kl_ctrl_013` | 1 | running |
| 21934084 | `cgt_s0_r_kl_ctrl_013` | 2 | running |

The GilaHyper set ran one at a time on the local build: 1640 (`cgt_s0_q_kl_004`,
cosine) timed out at 12 h having reached epoch 36; 1659 (`fit_014`) started at 11:09 as
it ended, with 1660 (`ctrl_013`) and 1661 (`fit_015`) queued behind.

### 12:52 - GilaHyper cleared for development; the whole protocol moves to Delta

1659 was cancelled at 1 h 42 m (a partial run, no result) and 1660 / 1661 before
starting, to free the GilaHyper GPU. The protocol now runs entirely on Delta at three
seeds per arm, nine jobs, each 48 h, submitted as one chain in which every job is held
until 30 minutes after the previous one starts, so no two are in dataset init on the
Taiga build at once (the lock timeout above):

| Delta job | config | seed | dependency |
|---|---|---|---|
| 21934082 | `cgt_s0_r_kl_ctrl_013` | 1 | running |
| 21934084 | `cgt_s0_r_kl_ctrl_013` | 2 | running |
| 21947151 | `cgt_s0_r_kl_fit_014` | 1 | none |
| 21947152 | `cgt_s0_r_kl_fit_014` | 2 | after 21947151 + 30 min |
| 21947958 | `cgt_s0_r_kl_fit_015` | 1 | after 21947152 + 30 min |
| 21947959 | `cgt_s0_r_kl_fit_015` | 2 | after 21947958 + 30 min |
| 21947960 | `cgt_s0_r_kl_fit_014` | 3 | after 21947959 + 30 min |
| 21947961 | `cgt_s0_r_kl_ctrl_013` | 3 | after 21947960 + 30 min |
| 21947962 | `cgt_s0_r_kl_fit_015` | 3 | after 21947961 + 30 min |

Seed 42 is no longer part of the design; the GilaHyper 1659 partial is discarded. The
readout is unchanged: `val/gene_interaction/Pearson` per arm at epoch 30 and at the best
epoch, three seeds each, `val/fitness/Pearson` and `val/cls_pert_strain_sd` alongside.

## 2026.09.10 - Sequence embeddings against the learnable table on the disjoint split (IGB mmli)

Job 1640 put the soft-KL model on the query-pair-disjoint split inside the additive band
(0.199 max over epochs, 0.131 at epoch 35; ridge 0.185, MLP 0.141). The model's only
per-gene information is a free 180-vector shaped by the triples the gene appeared in,
plus the nine graphs. The next round replaces that vector with a fixed sequence
embedding of the whole gene and asks whether the disjoint number moves.

**Protocol.** The constant-rate protocol of `cgt_s0_r_kl_ctrl_013` on the Q split: AdamW
at 2.5e-4, no schedule, 30 epochs, normalizer fit on train, `perturb_cls: true` (inert
for the interaction prediction without a fitness head; logged), interaction head only,
one seed. Everything after the gene embedding is identical across arms.

**Parameter matching.** The learnable table is 6,607 x 180 = 1,189,260 parameters and
feeds the transformer directly (no preprocessor at width 180). A pre-computed embedding
enters through the model's 2-layer preprocessor MLP, whose hidden width is now a config
key (`model.learnable_embedding.preprocessor.hidden_dim`; the default stays the midpoint
between input and model width). The width is chosen so the embedding side matches the
table: for a 3,328-dim input, `3511 h + 540 = 1,189,260` gives h = 339 (1,190,769); for
the 1,000-dim random control, h = 1,005 (1,189,455). The trunk, perturbation operator and
heads are unchanged, so the arms differ only in what the gene vector holds.

**The composite** (`cell_dataset.node_embeddings: [fudt_upstream, calm, prot_T5_all,
fudt_downstream]`) reads each functional region once: promoter (1,003 bp 5', species
fungal UTR transformer, 768), ORF as codons (CaLM, 768), ORF as protein (ProtT5-XL, 1024),
terminator (300 bp 3', 768). The Nucleotide Transformer windows are left out (2,560 dims
each, at zero in the 019 kNN probe) and ESM2 duplicates ProtT5's region. Measured
elsewhere, not on interactions: ProtT5 minus a random 1024-vector was +0.068 on
expression (019 v10 grid); the two flanking models sat near the probe's noise floor. They
stay in so the same gene vector can later carry promoter and terminator choices.

**Whether the table is cold on Q.** Counted by
[[experiments.025-solid-growth.scripts.query_pair_disjoint_gene_coverage]]: 99.8 percent
of val genes and 99.7 percent of test genes appear in a training triple, so unlike the
019 expression setting the table is trained for nearly every held-out gene. The
learnable-versus-sequence contrast on Q is about what the vector can contain, not
whether it was trained. 18.7 percent of val records do carry one of the 8 genes that
never appear in training (they are held-out query-pair members).

| config | gene vector | learnable table | embedding-side params | model total | role |
|---|---|---|---:|---:|---|
| `cgt_s0_q_kl_ctrl_016` | none | on (180) | 1,189,260 | 4,774,861 | Q baseline under the protocol |
| `cgt_s0_q_kl_emb_017` | composite 3,328 | off | 1,190,769 (h 339) | 4,776,370 | the question, parameter-matched |
| `cgt_s0_q_kl_rand_018` | `random_1000` | off | 1,189,455 (h 1,005) | 4,775,056 | content control, parameter-matched |
| `cgt_s0_q_kl_embtab_019` | composite 3,328 | on (180) | 2,441,049 | 6,026,650 | table on top of the composite; not matched |

Counts are the model's own `Parameter counts` line from the CPU smokes of 2026-09-10; the
three matched arms sit within 0.03 percent of each other on the total.

Readings: `_017` against `_016` is the sequence-information question; `_017` against
`_018` separates content from a fixed identity; `_019` against `_017` says whether a free
row still adds anything once the gene has a sequence vector. All four are read at epoch
30 and at the best validation epoch. Run one at a time on the IGB mmli node (4 x A100,
58 min/epoch measured for the KL arm on the 025 build), about 29 h each. The launcher's
preflight now checks the four embedding builds on IGB scratch.
