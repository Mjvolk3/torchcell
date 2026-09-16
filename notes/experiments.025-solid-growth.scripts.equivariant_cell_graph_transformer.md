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

Submitted on IGB mmli (4 x A100, one job at a time, each held until the previous one
ends, 4-day clock; W&B offline there, sync with `igb_login_wandb_sync.sh`). The first
chain, 2390540 to 2390543 from 6b69399c, died at preflight because the compute nodes
have no git ([[experiments.025-solid-growth.scripts.igb_mmli_cgt]]); resubmitted from
a5fd743e with the worktree prepared on the login node:

| IGB job | config | dependency |
|---|---|---|
| 2390616 | `cgt_s0_q_kl_ctrl_016` | first |
| 2390617 | `cgt_s0_q_kl_emb_017` | afterany 2390616 |
| 2390618 | `cgt_s0_q_kl_rand_018` | afterany 2390617 |
| 2390619 | `cgt_s0_q_kl_embtab_019` | afterany 2390618 |

Readings: `_017` against `_016` is the sequence-information question; `_017` against
`_018` separates content from a fixed identity; `_019` against `_017` says whether a free
row still adds anything once the gene has a sequence vector. All arms are read at epoch
30 and at the best validation epoch. Run one at a time on the IGB mmli node (4 x A100,
58 min/epoch measured for the KL arm on the 025 build), about 29 h each. The launcher's
preflight now checks the four embedding builds on IGB scratch.

### Revised chain (13:45): composite first, single regions next, control last

The random-vector control (`_018`) and the composite-plus-table arm (`_019`) are held
until an embedding arm moves the disjoint number. Two single-region arms take their
place, parameter-matched the same way:

| config | gene vector | preprocessor hidden | embedding-side params |
|---|---|---:|---:|
| `cgt_s0_q_kl_calm_020` | CaLM alone, 768 | 1,250 | 1,189,290 |
| `cgt_s0_q_kl_prot_021` | ProtT5 alone, 1,024 | 985 | 1,189,435 |

The chain runs in reverse of the first design, so the full-gene vector is the first
result: composite (`_017`), CaLM (`_020`), ProtT5 (`_021`), then the learnable control
(`_016`). Chain 2390616 to 2390619 was cancelled (the control had run 10 minutes). The
run script now also logs `model/params_embedding_preprocessor`, so the parameter match is
visible in W&B beside `model/params_gene_embedding`. Submitted from cf5fee7a:

| IGB job | config | dependency |
|---|---|---|
| 2391132 | `cgt_s0_q_kl_emb_017` (composite) | first |
| 2391133 | `cgt_s0_q_kl_calm_020` | afterany 2391132 |
| 2391134 | `cgt_s0_q_kl_prot_021` | afterany 2391133 |
| 2391135 | `cgt_s0_q_kl_ctrl_016` | afterany 2391134 |

Synced from the login node (`wandb sync --include-offline` on each of the four per-rank
offline dirs; rank 0 is `s1vx2zgw` and carries the validation metrics). The composite arm
runs at about 17 min/epoch on compute-5-7 (8 h 41 m to epoch 29), a third of the 58
measured for the KL replication there in September, so the four-arm chain is about 36 h,
not 5 days.

Composite arm, one seed, read at 22:40 with epoch 29 still running (partial by one
epoch). `val/gene_interaction/Pearson` by epoch, with `train/gene_interaction/Pearson`:

| epoch | 0 | 1 | 2 | 3 | 4 | 6 | 10 | 11 | 15 | 17 | 20 | 23 | 26 | 28 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val | 0.151 | 0.209 | 0.263 | 0.241 | 0.196 | 0.226 | 0.251 | 0.238 | 0.238 | 0.235 | 0.175 | 0.227 | 0.202 | 0.173 |
| train | 0.030 | 0.218 | 0.309 | 0.345 | 0.359 | 0.374 | 0.393 | 0.399 | 0.422 | 0.434 | 0.451 | 0.466 | 0.482 | 0.493 |

Max over epochs 0.263 at epoch 2 (an upward-biased order statistic over 29 epochs);
epochs 10 to 19 sit at 0.22 to 0.25; epoch 28 is 0.173. Train Pearson climbs
monotonically to 0.49 while val drifts down after epoch 19, so the arm overfits the
training pairs under the constant rate. For scale, not as a matched comparison: job 1640
(learnable table, cosine, 36 epochs) reached 0.199 max and 0.131 at epoch 35 on this
split; additive ridge 0.185, MLP 0.141. The matched control `_016` is last in the chain.
`val/cls_pert_strain_sd` runs 0.31 to 0.43 from epoch 2 on (wild-type CLS is 0.0).

## 2026.09.11 - Composite and CaLM complete, ProtT5 at epoch 18; both Delta controls lost to an NCCL timeout

Synced 13:55. Rank-0 runs: composite `s1vx2zgw` (2391132, COMPLETED 9 h 03 m), CaLM
`8aa08xx0` (2391133, COMPLETED 8 h 40 m), ProtT5 `pmkzwwzw` (2391134, running, epoch
18 of 30). One seed each; the matched learnable control (`_016`, 2391135) has not run,
so no arm has its matched comparison yet.

| arm | epochs | max over epochs (epoch) | epoch 29 | train Pearson at last epoch |
|---|--:|---|---|---|
| composite | 30 | 0.263 (2) | 0.222 | 0.502 |
| CaLM alone | 30 | 0.252 (7) | 0.208 | 0.532 |
| ProtT5 alone | 19 (partial) | 0.270 (3) | 0.123 at epoch 18 | 0.454 |

Composite by epoch: 0.151 0.209 0.262 0.241 0.196 0.185 0.226 0.225 0.208 0.190 0.251
0.238 0.236 0.223 0.218 0.238 0.220 0.235 0.215 0.220 0.175 0.181 0.223 0.227 0.193
0.193 0.202 0.218 0.173 0.222. CaLM: 0.089 0.176 0.249 0.210 0.225 0.217 0.162 0.252
0.218 0.213 0.191 0.226 0.201 0.236 0.212 0.230 0.229 0.212 0.194 0.189 0.192 0.222
0.198 0.209 0.162 0.173 0.210 0.212 0.165 0.208. ProtT5 through 18: 0.168 0.205 0.223
0.270 0.237 0.236 0.238 0.238 0.247 0.204 0.240 0.181 0.179 0.241 0.147 0.177 0.179
0.190 0.123.

Every arm peaks in its first ten epochs and swings 0.05 to 0.10 epoch to epoch while
train climbs monotonically past 0.45, so the epoch-30 reading sits 0.04 to 0.06 under
the max in every case; the two numbers are reported together as the protocol says.
Against the scale marks on this split (1640: 0.199 max, 0.131 at epoch 35; ridge 0.185)
all three embedding arms' maxima are higher and their late-epoch values are near or
above the ridge, but the schedule and the seed differ, so this is not a comparison until
`_016` runs. ProtT5 alone at epoch 18 (0.123) is the lowest late value of the three;
partial.

**Delta.** Both constant-rate controls died: 21934082 (seed 1) FAILED at 16 h 42 m in
epoch 16, 21934084 (seed 2) FAILED at 25 h 49 m in epoch 18, each with rank 0 exiting
-6 after the NCCL watchdog caught an `ALLREDUCE` collective timeout (1,800 s, the
30-minute process-group default) on another rank, in the middle of a training epoch
(epoch 16 at 65 percent; epoch 18 at 3 percent). Plausible mechanism, unverified: one
rank stalled on an LMDB page read through the Taiga NFS mount for over 30 minutes. The fitness arm 21947151 has
waited 26 h and the scheduler estimates a 2026-09-12 14:08 start; seven more jobs chain
behind it. Delta at ~60 min/epoch over NFS against IGB at 17 min/epoch on local disk
puts the whole Delta design in question; decision pending.

### 14:30 - ProtT5 alone complete; the 100-epoch round queued on mmli and cabbi

ProtT5 alone (2391134, `pmkzwwzw`, COMPLETED 8 h 42 m): max 0.270 at epoch 3, then
0.12 to 0.16 from epoch 18 on, 0.135 at epoch 29; its validation point loss rose from
0.90 at epoch 3 to 1.05 at epoch 29. So of the three embedding arms it is the one that
follows the learnable model's pattern (1640: 0.96 to 1.22), while composite and CaLM hold
flat near 0.94. One seed each; no mechanism claimed.

Validation point loss (z-scored MSE on gene interaction) at matched epochs, the reading
that separates the arms more cleanly than the noisy Pearson:

| epoch | 3 | 9 | 15 | 21 | 27 |
|---|---|---|---|---|---|
| 1640, learnable, cosine | 0.956 | 0.991 | 1.054 | 1.175 | 1.215 |
| composite, constant | 0.918 | 0.979 | 0.925 | 0.967 | 0.942 |
| CaLM, constant | 0.928 | 0.936 | 0.927 | 0.947 | 0.946 |
| ProtT5, constant | 0.898 | 0.967 | 1.006 | 1.023 | 1.031 |

Whether the input helps is not yet answered: the matched 30-epoch control (`_016`,
2391135) started at 13:50 and reads out about 23:00. The graph penalty is still falling
at epoch 30 in every embedding arm (composite 3.36, CaLM 2.98, ProtT5 similar, against
0.33 for the learnable model by epoch 9), so the 30-epoch table is of unconverged
penalties. The 100-epoch round (`_022` to `_025`, `trainer.max_epochs: 100`, one key
each on the 30-epoch configs) runs in two lanes from a second login-node worktree
(`~/projects/torchcell.worktrees/025-fitness-joint-head-b` at a7f701e4, passed to the
launcher as `PROJECT_ROOT`, so the running `_016` job's worktree is untouched). The
cabbi lane uses 4 of compute-3-3's 8 RTX 6000s, the hardware the 010 M01 to M03 runs
trained on with this batch size and precision; it is slower than the A100s, and cross-lane
comparisons cross hardware, which changes speed and floating-point nondeterminism, not
the computation. Runs are watched and any arm can be stopped early; the val-Pearson
checkpoint keeps the best epoch.

| IGB job | partition | config | dependency |
|---|---|---|---|
| 2394963 | mmli | `cgt_s0_q_kl_emb_022` (composite, 100 ep) | afterany 2391135 |
| 2394964 | mmli | `cgt_s0_q_kl_calm_023` (CaLM, 100 ep) | afterany 2394963 |
| 2394965 | cabbi | `cgt_s0_q_kl_prot_024` (ProtT5, 100 ep) | running |
| 2394966 | cabbi | `cgt_s0_q_kl_ctrl_025` (learnable control, 100 ep) | cancelled 14:50 before starting |

The 100-epoch learnable control was cancelled: the learnable model's held-out optimum
falls early (epoch 14 in the 010 replication, 2 to 12 in job 1640) and its validation
loss only climbs after, so the 30-epoch control (`_016`, 2391135) already contains its
best epoch, and a 100-epoch run would read a memorized state. The 100-epoch embedding
arms are read at their best epoch and at epoch 30 against `_016`, and at epoch 100 with
no matched control. The cabbi slot after 2394965 goes to the flanks-only arm
`cgt_s0_q_kl_fudt_026` (fudt_upstream + fudt_downstream, 1,536 dims, hidden width 692,
1,190,088 embedding-side parameters, model total 4,775,689 from its CPU smoke), job
2395008, which completes the single-region set: what the regulatory DNA carries with no
ORF view. It runs from a third login-node worktree (`025-fitness-joint-head-c` at
baba07c8) so neither running job's checkout moves. compute-3-3's GPUs are RTX 6000 Ada,
48 GB, the same card as GilaHyper, so the two lanes differ less than first stated.

## 2026.09.12 - Delta reads the 025 build from local scratch, not Taiga

The build Delta trains on has been the Taiga copy, through a symlink from
`/scratch/bbub/mjvolk3/torchcell/data/torchcell/experiments/025-solid-growth/001-full-build`
to `/taiga/illinois/eng/chbe/zhao5/mjvolk3/projects/torchcell/...`. Both constant-rate
controls that read it died mid-epoch on the 30-minute NCCL watchdog, and the LMDB read
through that mount is the only remote I/O in the training loop; unverified as the cause.
Delta's bbub scratch quota is 9.8 TB with 2.8 TB used, so the 554 GB build is being
copied to local Lustre by `delta_copy_025_build_local.slurm` (job 21984419, CPU
partition, two rsync passes, size check against 554,075,586,560 B). After it prints
CLEAN the symlink is swapped for the local directory, before the pending fitness chain
(21947151 first, estimated start 2026-09-12 14:08) starts reading. Done at 00:35: the
copy took 32 minutes (about 290 MB/s), the second pass moved nothing, both `data.mdb`
sizes read 554,075,586,560 B, and `001-full-build` is now a real directory on Lustre
with the old link kept beside it as `001-full-build.taiga-link`. The pending jobs need
no resubmission: they open the build through the symlink path. The two lost controls are
appended to the chain with the same 30-minute stagger, restoring three seeds per arm:

| Delta job | config | seed | dependency |
|---|---|---|---|
| 21947151 | `cgt_s0_r_kl_fit_014` | 1 | pending, priority |
| 21947152 | `cgt_s0_r_kl_fit_014` | 2 | after 21947151 + 30 min |
| 21947958 | `cgt_s0_r_kl_fit_015` | 1 | after 21947152 + 30 min |
| 21947959 | `cgt_s0_r_kl_fit_015` | 2 | after 21947958 + 30 min |
| 21947960 | `cgt_s0_r_kl_fit_014` | 3 | after 21947959 + 30 min |
| 21947961 | `cgt_s0_r_kl_ctrl_013` | 3 | after 21947960 + 30 min |
| 21947962 | `cgt_s0_r_kl_fit_015` | 3 | after 21947961 + 30 min |
| 21984678 | `cgt_s0_r_kl_ctrl_013` | 1 | after 21947962 + 30 min, replaces 21934082 |
| 21984680 | `cgt_s0_r_kl_ctrl_013` | 2 | after 21984678 + 30 min, replaces 21934084 |

### 22:30 - The local copy was slower still; the chain resubmitted with node-local staging

21947151 started at 01:09 (34 minutes after the swap) and did 5 epochs in 21 hours,
3.3 to 3.7 h each, against 63 min through Taiga: learning normally (val interaction
Pearson 0.397 to 0.433 over epochs 0 to 4, val fitness Pearson 0.935) but unable to
reach epoch 30 inside 48 h. The node was ours alone with CPU load 1.15 on 16 cores, so
the ranks were waiting on LMDB page reads from Lustre (the file sits on 4 of 114
stripes). Three things checked before changing anything:

- **Gradient recording is not the cause.** The trainer has no `wandb.watch`, no gradient
  norm logging, and no step-level logging; every `self.log` is per epoch, sample plots
  run once per epoch, transformer diagnostics and edge recovery every 10. The only
  gradient operation is clipping. The panel-c gradient probe for the KL-versus-mask
  question is not implemented yet, so nothing of that kind runs. The same code does 17
  min per epoch on IGB and 19 on GilaHyper.
- **The Delta docs prescribe node-local disk for this pattern.** Data Management: "The
  high performance ssd storage (740GB CPU, 1.5TB GPU) is available in /tmp" and "Codes
  that need to perform i/o to many small files should target /tmp on each node of the
  job". `/projects` is 500 GB, `/work/hdd` 1 TB on 12 OSTs, and `/work/nvme` (96 OSTs,
  about 800 GB/s aggregate) is "available upon request", which would remove the per-job
  copy if granted.
- **IGB's `gpu` partition cannot take these jobs as designed.** It is three nodes of two
  A40s each; the protocol is single-node 4-GPU DDP at batch 256 per rank, and a 2-GPU
  world size changes the effective batch and step count under the constant rate.

All nine jobs (5 epochs of 21947151 discarded) were cancelled and resubmitted from
b0f279f5 with the staging launcher ([[experiments.025-solid-growth.scripts.delta_cgt]]),
controls interleaved so a complete seed lands first:

| Delta job | config | seed | dependency |
|---|---|---|---|
| 22020651 | `cgt_s0_r_kl_fit_014` | 1 | pending, priority |
| 22020652 | `cgt_s0_r_kl_ctrl_013` | 1 | after 22020651 + 30 min |
| 22020653 | `cgt_s0_r_kl_fit_015` | 1 | chained |
| 22020654 | `cgt_s0_r_kl_fit_014` | 2 | chained |
| 22020655 | `cgt_s0_r_kl_ctrl_013` | 2 | chained |
| 22020656 | `cgt_s0_r_kl_fit_015` | 2 | chained |
| 22020657 | `cgt_s0_r_kl_fit_014` | 3 | chained |
| 22020658 | `cgt_s0_r_kl_ctrl_013` | 3 | chained |
| 22020659 | `cgt_s0_r_kl_fit_015` | 3 | chained |

The first job's log reports the copy time and the first epochs' pace; that is the
measurement of whether staging works. On IGB the flanks-only job 2395008 was briefly
released from its dependency, then re-held behind ProtT5 at 23:15 because the user wants
cabbi's other four GPUs for different work.

### 2026.09.13 07:45 - Stage-in works; the mirror's symlinked lock files did not

22020651 started at 06:53 on gpub021, seventeen hours before the scheduler's estimate,
and copied the LMDB to the node's NVMe in 940 s (about 590 MB/s). It then died in
dataset init: `OSError: [Errno 40] Too many levels of symbolic links` on
`processed/gene_set.json.lock`. Delta's filelock (newer than GilaHyper's 3.20.0) opens
lock files with `O_NOFOLLOW`, so the mirror's symlinked `*.lock` files raise ELOOP. The
mirror now leaves `*.lock` unlinked and `FileLockHelper` creates them locally
(aa44be08). The control 22020652 was held before it could start on the old launcher,
the worktree advanced, and the arm resubmitted as 22030924 with 22020652 re-chained
30 minutes after it; the rest of the chain is unchanged.

**Measured (10:00):** 22030924 started at 08:54, staged the LMDB in 1,007 s, and reached
epoch 0 at 09:26 and epoch 1 at 09:53: **27 min per epoch** on A40s from node-local NVMe,
against 63 through Taiga and 200 to 220 from Lustre scratch; the validation curve
(0.397, 0.410) matches the earlier controls at the same epochs. Thirty epochs is about
13.5 h plus the 17-minute stage-in, so the arm finishes around 23:00 today, before the
Monday scheduler maintenance. The read-path table for the 025 build is now complete:

| where the LMDB lives | min per epoch |
|---|---|
| IGB local disk (A100) | 17 |
| GilaHyper NVMe (RTX 6000 Ada) | 19 |
| Delta node NVMe via stage-in (A40) | 27 |
| Delta Taiga NFS (A40) | 63, with 30-minute stalls |
| Delta Lustre scratch (A40) | 200 to 220 |

The control 22020652 then failed on the same lock-file error, 47 minutes in, because
`sbatch` stores a copy of the batch script at submission: the seven jobs submitted at
22:29 all carried the old launcher regardless of the worktree advance, and only the
resubmitted 22030924 had the fix. All seven were cancelled and resubmitted from aa44be08,
chained 30 minutes apart behind 22030924, in the same order:

| Delta job | config | seed |
|---|---|---|
| 22030924 | `cgt_s0_r_kl_fit_014` | 1 (running, epoch 6 at 11:53) |
| 22034665 | `cgt_s0_r_kl_ctrl_013` | 1 |
| 22034666 | `cgt_s0_r_kl_fit_015` | 1 |
| 22034667 | `cgt_s0_r_kl_fit_014` | 2 |
| 22034668 | `cgt_s0_r_kl_ctrl_013` | 2 |
| 22034669 | `cgt_s0_r_kl_fit_015` | 2 |
| 22034670 | `cgt_s0_r_kl_fit_014` | 3 |
| 22034671 | `cgt_s0_r_kl_ctrl_013` | 3 |
| 22034673 | `cgt_s0_r_kl_fit_015` | 3 |

## 2026.09.13 - Disjoint split: sequence embeddings against the matched learnable control

All 30-epoch arms and two of the 100-epoch arms are complete; the table is produced by
[[experiments.025-solid-growth.scripts.disjoint_embedding_readout]] from W&B by config
tag and written to `results/disjoint_embedding_readout.csv`. One seed each, split Q, the
constant-rate protocol, embedding side parameter-matched to the table. "max" is the max
over the epochs run, an upward-biased order statistic; "mean 10 to 29" is a 20-epoch
window average, not a max; "epoch 29" is the protocol's fixed reading.

| config | arm | budget | max (epoch) | epoch 29 | mean 10 to 29 | mean 60 to 99 | val point loss at 29 | train P last | graph penalty last |
|---|---|--:|---|---|---|---|---|---|---|
| `ctrl_016` | learnable table | 30 | 0.195 (2) | 0.130 | 0.140 | | 1.217 | 0.644 | 0.34 |
| `emb_017` | composite | 30 | 0.263 (2) | 0.222 | 0.215 | | 0.950 | 0.502 | 3.36 |
| `calm_020` | CaLM alone | 30 | 0.252 (7) | 0.208 | 0.204 | | 0.950 | 0.532 | 2.98 |
| `prot_021` | ProtT5 alone | 30 | 0.270 (3) | 0.135 | 0.162 | | 1.046 | 0.524 | 3.07 |
| `emb_022` | composite | 100 | 0.263 (2) | 0.222 | 0.215 | 0.164 | 0.950 | 0.754 | 2.48 |
| `prot_024` | ProtT5 alone | 100 | 0.248 (3) | 0.175 | 0.175 | 0.197 | 0.977 | 0.764 | 2.33 |
| `calm_023` | CaLM alone | 100, at 32 | 0.252 (7) | 0.208 | 0.204 | partial | 0.950 | 0.544 | 2.90 |
| `fudt_026` | promoter + terminator | 100, at 9 | 0.153 (6) | partial | partial | partial | partial | 0.378 | 6.81 |

**Findings, one seed each.**

- **The learnable control reproduces job 1640 under the constant rate**: 0.195 max at
  epoch 2 against 1640's 0.199, 0.130 at epoch 29 against 0.131 at epoch 35. The
  schedule was not what held the disjoint number down.
- **Sequence input moves the disjoint number, and the move is not one lucky epoch.**
  Over epochs 10 to 29 the composite averages 0.215 and CaLM 0.204 against the control's
  0.140, with the control never above 0.163 in that window and the composite never
  below 0.173. The maxima order the same way (0.263, 0.252 against 0.195). ProtT5 alone
  has the highest max (0.270 at epoch 3) but averages 0.162 over the window: it peaks and
  then follows the control down.
- **The mechanism visible in the loss is memorization, and the composite resists it.**
  The control's validation point loss climbs from 0.935 at epoch 2 to 1.217 at epoch 29,
  past the label variance, while train Pearson reaches 0.64: the free rows fit the
  training pairs. Composite and CaLM hold 0.95 through epoch 29. ProtT5 alone climbs to
  1.05. Hypothesis, untested: a fixed gene vector cannot be moved to fit a training pair,
  and a 3,328-dim vector through a width-339 projection is a tighter bottleneck than
  ProtT5's 1,024 through width 985.
- **The graph penalty tells the opposite story from the score.** The learnable model
  drives the layer-1 attention onto the graphs (penalty 0.34 by epoch 29) and generalizes
  worst; the embedding arms sit at 2.3 to 3.4 and generalize best. Matching the graphs is
  easy for free rows and does not carry to unseen pairs.
- **100 epochs adds nothing and costs something.** The composite's first 30 epochs are
  bit-for-bit its 30-epoch run (same seed, same node type), then it decays to a mean of
  0.164 over epochs 60 to 99 with validation loss at 1.05 and train Pearson at 0.75.
  ProtT5 at 100 epochs holds 0.197 late. The best epoch is inside the first ten for every
  arm, so the 30-epoch budget was sufficient on this split and the question is now
  replicates, not epochs.
- **The flanks alone are at epoch 9 with 0.153 max; partial**, but already below every
  ORF-carrying arm at the same epochs (composite 0.208 to 0.263 over epochs 1 to 9).
- Scale marks on this partition: additive ridge 0.185, MLP 0.141. The composite's window
  mean sits above the ridge; the control's sits at the MLP.

What this does not yet establish: the size of the composite-minus-control gap with error
bars. Within a run the epoch-to-epoch swing is about 0.05, and no arm has a second seed.
Two more seeds of `_016` and `_017` on the 30-epoch budget (about 9 h each on IGB) would
put a spread on the 0.07 window gap.

Sync note: `wandb sync` of an offline run rewrites the run's tags from its record, so
tags added through the API are lost on the next sync; retag after the final sync, or
rely on the config-name tags the run script now attaches at init.

### 23:40 - CaLM at 100 decays like the composite; the flanks sit between control and ORF arms

Synced at 23:39. CaLM 100-epoch (`_023`, epoch 80): validation loss 1.044 and train
Pearson 0.73 at 80, the composite's decay. Flanks alone (`_026`, epoch 42, partial):
max 0.208 at epoch 20, mean 0.164 over epochs 10 to 29, 0.146 at 29, validation loss
1.013 at 29: above the learnable control (0.140) and below every ORF-carrying arm
(0.204 to 0.215). Hypothesis, untested: part of the gain is regularization from any
frozen per-gene vector, part is ORF content; the random-vector control separates them.

Next round on mmli, queued behind the CaLM job so the node does not idle:
`cgt_s0_q_kl_rand_018` (random 1,000-vector, width 1,005, parameter-matched, disjoint
split, 30 epochs) as IGB job 2397827 from worktree `-c` at 8f3d2b11. Queued behind it
at 23:50, one after another, about 9 h each:

| IGB job | config | seed | purpose |
|---|---|---|---|
| 2397827 | `cgt_s0_q_kl_rand_018` | 42 | frozen random vector: content versus frozen identity |
| 2397845 | `cgt_s0_q_kl_ctrl_016` | 1 | control replicate |
| 2397846 | `cgt_s0_q_kl_emb_017` | 1 | composite replicate |
| 2397847 | `cgt_s0_q_kl_ctrl_016` | 2 | control replicate |
| 2397848 | `cgt_s0_q_kl_emb_017` | 2 | composite replicate |
| 2397876 | `cgt_s0_q_kl_embfit_027` | 42 | composite + fitness head (weight 1.0) on the disjoint split |

`_027` composes on `_017` and adds `fit_014`'s fitness side: both COO labels, a
standard normalizer on fitness fit on train, the 181-parameter linear probe of the
perturbed CLS, `fitness_lambda: 1.0`. Its control is `_017`. Model total 4,776,551. Its
16-record CPU smoke died after epoch 0 in the W&B checkpoint logger with "Out of range
float values are not JSON compliant: -inf": Lightning substitutes -inf for a NaN
monitored metric, and a 16-record subset of the disjoint split gave a NaN validation
Pearson; the 64-record smoke passed. Not a full-scale risk (every disjoint run logs the
metric at every epoch on 37,705 validation records). Expected end of the chain: Wednesday
about 10:30, from the measured 8 h 40 m to 9 h 03 m per 30-epoch run on this node.

## 2026.09.14 - The graph-regularization sweep runs from the 025 build; a whole-build arm for cabbi

### Sweep on Delta, composed on ctrl_013

The sweep for `FigS-graph-regularization-sweep` no longer needs the 010 build. Every arm
is `cgt_s0_r_kl_ctrl_013` with one key changed, and the three ctrl_013 seeds already in the
Delta fitness chain (22034665, 22034668, 22034671) are the ladder's lambda 1e-3 point:

| arm | override on `ctrl_013` | seeds | jobs |
|---|---|---|---|
| no penalty | `model.graph_regularization.graph_reg_lambda=0` | 1, 2, 3 | 3 |
| ladder 1e-5, 1e-4, 1e-2, 1e-1, 1 | same key | 1, 2, 3 | 15 |
| hard mask, layer 1, nine heads | config `cgt_s0_r_mask_028` (mask_003's block on ctrl_013, lambda 0, loss lambda 0, edge-recovery plots off) | 1, 2, 3 | 3 |
| random graphs, degree-matched | not yet: needs the rewiring option | | |

`delta_submit_sweep.sh sweep` submits the 21 as one chain (`after:<prev>+30`, 24 h
clocks, seed-major with 0 and mask first inside each seed). Both configs were composed
with Hydra to check every key: the mask arm has `attention_mask.enabled=true, layers=[1]`,
nine `head_graphs`, per-head KL lambdas 0, loss lambda 0; the ladder arms carry the
override into all nine `regularized_heads` through the interpolation, and the seed lands.
Submission is scheduled for 09:01 CDT today, after the 08:00 to 09:00 scheduler
maintenance; a one-shot reminder in this session fires it.

A correction to the 2026.09.04 entry of [[experiments.025-solid-growth.training-plan]]:
the KL arms regularize all nine graphs, not seven. `_normalize_adjacency_matrices`
registers each `*_interaction` relation under its unsuffixed name too, so `physical` and
`regulatory` resolve, and the loss raises rather than skips on a name it cannot find. The
mask arm therefore masks the same nine heads the ladder regularizes.

Two things the runs will not log and the panels must read from checkpoints: at lambda 0
the model returns before computing the divergence, and under the mask the KL heads are
declared at lambda 0, so `train/graph_reg_loss` exists only for the ladder. Panel a's
lambda-0 and mask points come from the saved best checkpoints, which do survive the
NVMe mirror: `models/checkpoints` is a symlink back to `/scratch`, and job 22030924
(fit_014 seed 1, COMPLETED 00:08 today, 16 h 01 m, 30 epochs) left
`b3n4ax4a-best-pearson-epoch=18-val/gene_interaction/Pearson=0.4523.ckpt` there. Its 27
min/epoch is the rate the 24 h clock is sized on.

### The whole-build arm for cabbi

Asked for: train on the random-split triples with the fitness head, and add every
double (dmf + dmi), every single, essential genes and synthetic lethality. Counted from
the build's `dataset_name_index` x `perturbation_count_index` (scratch script, 819
composite keys):

| records | count | labels |
|---|---|---|
| singles | 5,694 | smf (fitness only) |
| doubles | 13,142,648 | dmf + dmi; 13,993 of them carry a SynthLethDB record converted to fitness (3,720 SynthLethDB-only, 3,899 shared with Costanzo dmf) |
| triples | 376,732 | tmf + tmi |
| total | 13,525,071 | fitness on all, gene_interaction on 13,515,659 |

Gene essentiality is absent: `dataset_name_index` has no key containing
`SgdEssential`, and the served graph has no `GeneEssentialitySgdDataset` node (probe
2026-09-14 01:00, count 0), so the query block returned nothing at build time. Essential
genes appear only where a screen measured them. Adding them is a KG increment plus a
requery, not a config.

So the set the question describes is the whole build, S5 of the ladder, and it now has
a config: `cgt_s5_r_kl_fit_029` (fit_014's joint objective, `subset.indices: null`,
`unpinned_to_train: true`). The datamodule change that makes it an evaluation on the
triples: `CellDataModule(unpinned_to_train=True)` moves every pool record the pinned
split does not name into train, so val and test stay the 37,673 + 37,673 pinned
trigenic records; without it the unpinned remainder is seed-split 80/10/10 and the
validation Pearson would mix dmi with tmi. The normalizer under `fit_on: train` then
fits on pinned train plus the unpinned records. Two tests cover it, and the trainer's
post-setup assertion checks the realized splits against that rule.

Cost, hypothesis by scaling and not a measurement: 35.9x the S0 records per epoch,
against 17 min/epoch on mmli (2391135) and about 24 on cabbi (2395008 at partial epoch
42 after 16 h 47 m), is 10 to 14 h per epoch, two weeks for 30 epochs on one node. The
closure arm `cgt_s3_r_kl_fit_030` (S3: 1,121,645 records, the 739,219 doubles inside
some triple plus all singles) is the same design at 2.98x, about 36 h on cabbi. A
per-epoch cap on the doubles for S5 would need a sampler and is not written.

cabbi at 00:50: our `025-q-fu` (2395008) running on compute-3-3 with three `dyna_seq`
tasks of another user and a fourth pending on Resources, so no free lane there tonight.

### The last two pieces, so the figure can be left to run

**Random-graph control (panel f).** `torchcell.graph.rewire.degree_preserving_rewire`
rewires an edge list by seeded double-edge swaps, (a, b) + (c, d) to (a, d) + (c, b),
rejecting self-loops and duplicates, so every gene keeps its out-degree and in-degree
(the configuration-model null of Maslov and Sneppen 2002). A graph stored with both
directions of every edge would be swapped on its undirected edge set and re-symmetrized;
in the 025 `cell_graph` none of the nine passes that test (each has at least one
one-directional edge), so all nine are rewired as directed graphs. `rewire_cell_graph`
copies the HeteroData with every gene-gene `edge_index` replaced and leaves the
dataset's graph untouched. The 025 trainer applies it under `model.random_graph.enabled`
(seed = the run seed unless `random_graph.seed` is set) and hands the rewired graph to
the model and the task, so the KL prior, the mask and the edge-recovery diagnostics all
see the same rewired graphs. Config `cgt_s0_r_kl_rand_031` = ctrl_013 + rewiring, at
lambda 1e-3 (the mock-up's "best lambda from the sweep" is not known at submission; it is
one override away). Four tests in `tests/torchcell/graph/test_rewire.py`.

Measured on the real graphs in the CPU smoke (seed 1, five attempts per edge, under a
minute for all nine): the fraction of original edges surviving is 0.13 (STRING
coexpression, 1.0M edges), 0.14 (database), 0.18 (experimental), 0.19 (physical), 0.20
(neighborhood), 0.32 (regulatory), 0.37 (fusion), 0.44 (co-occurrence), 0.46 (TFLink).
The survivors are what the degree constraint forces: an edge between two hubs has few
places to go. The overlap is logged per graph as `random_graph/<rel>/edge_overlap`.

**Gradient probe (panel c).** `RegressionTask(gradient_probe_epochs=[...])`: on the first
training batch of each listed epoch, the gradient of each weighted loss term (point,
distribution, graph penalty, fitness when present) is taken separately against every
parameter with `torch.autograd.grad(retain_graph=True)` and its global L2 norm logged as
`probe/grad_norm/<term>`, with the summed loss as `probe/grad_norm/total` and the ratio
`probe/grad_ratio/graph_reg_to_point`; a term with no graph (lambda 0, the mask) logs 0.
`PointDistGraphReg` keeps its weighted term tensors as `last_terms` for this. The numbers
are also printed to the SLURM log. `cgt_s0_r_kl_000` now carries
`gradient_probe_epochs: [0, 1, 2, 5, 10, 20]`, so every 025 arm composed on it logs the
probe, and every Delta job still pending picks it up at start (Delta jobs read the
training script and configs from the worktree when they begin; only the sbatch launcher
is copied at submission). The IGB chain runs from its own frozen worktrees and does not.

## 2026.09.15 - Per-order metrics for the mixed-order arms

`RegressionTask(per_order_metrics=True)` (config `regression_task.per_order_metrics`, on
in `cgt_s5_r_kl_fit_029` and so in the closure arm `_030`) keeps a second set of the
MSE / RMSE / Pearson collections per perturbation order, fed with the rows of each batch
whose record perturbs 1, 2 or 3 genes (`bincount` of `perturbation_indices_batch`), and
logs them at epoch end as `<stage>/gene_interaction/order<k>/<metric>` and, on the joint
path, `<stage>/fitness/order<k>/<metric>`, plus `<stage>/n_records/order<k>`; an order
that received no row in the epoch is skipped. The pooled metrics are unchanged. On the
closure and whole-build arms the pooled training Pearson mixes dmi and tmi; the order-3
line is the trigenic fit alone, the order-2 line the digenic one. Validation and test on
the pinned trigenic splits are order 3 only, so there the order-3 line equals the pooled
one, which is the check that the split is what it claims. Three tests in
`tests/torchcell/trainers/test_int_transformer_cell_per_order_metrics.py`.

## 2026.09.15 - Disjoint split at two and three seeds; the random-vector control

Synced from IGB (20 offline runs of jobs 2397827, 2397845, 2397846, 2397847 and the
100-epoch flanks run 2395008) and read with `disjoint_embedding_readout.py`, now one row
per seed. Validation Pearson on the 420 held-out query pairs; max is a max over 30
epochs and biased upward, the window mean over epochs 10 to 29 is the steadier reading:

| arm | seed | max (epoch) | mean 10 to 29 | val point loss at 29 |
|---|---|---|---|---|
| learnable table, control | 42 | 0.195 (2) | 0.140 | 1.217 |
| learnable table, control | 1 | 0.226 (3) | 0.144 | 1.193 |
| learnable table, control | 2 | 0.163 (7) | 0.125 | 1.172 |
| random 1,000-vector, matched | 42 | 0.187 (13) | 0.151 | 1.017 |
| composite sequence embedding | 42 | 0.263 (2) | 0.215 | 0.950 |
| composite sequence embedding | 1 | 0.275 (15) | 0.235 | 1.018 |
| CaLM alone | 42 | 0.252 (7) | 0.204 | 0.950 |
| ProtT5 alone | 42 | 0.270 (3) | 0.162 | 1.046 |
| promoter + terminator alone, 100 ep | 42 | 0.208 (20) | 0.164; 0.157 over 60 to 99 | 1.013 |

Three seeds of the control give window means 0.140, 0.144, 0.125 (spread 0.019); two
seeds of the composite give 0.215 and 0.235. The gap between the arms, about 0.08 on the
window mean, is four times the control's seed spread, so with two composite seeds the
ordering is established even if a third composite seed is still owed for a proper
interval. The random 1,000-vector control, parameter-matched like the composite, sits
at 0.151, above the learnable table and well below the composite: a fixed input vector
by itself removes some of the memorization (its validation loss holds near 1.02 where
the table's climbs to 1.2), and the sequence content adds the rest. Flanks alone after
100 epochs stay at 0.16, in the band of the random vector. The composite-plus-fitness
arm (2397876) has not run yet; the composite seed-2 replicate (2397848) is at epoch 11.

The report's run set (`split_Q` tag) is the group view of all of these:
<https://wandb.ai/zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer/reports/025-disjoint-split:-sequence-embeddings-against-the-learnable-table--VmlldzoxNzk0MTc0Nw==> (version of 2026-09-15; the first version is VmlldzoxNzkyNTI2OA==)
