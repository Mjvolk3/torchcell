---
id: 2yecgmklqjrpiuyog11ns71
title: 'Metabolism _003 -- optimizer x distributional sweep'
desc: 'Design + measured justification for the three-arm metabolism sweep on GilaHyper, ported from expression round _007'
updated: 1785209036038
created: 1785209036038
---

## 2026.07.27 - Round _003 design

The metabolism port of expression round `_007` ([[experiments.019-simb-multimodal.fig3-expression-experiments]]).
Three arms, one GPU each on GilaHyper, running concurrently:

| arm | experiment | target | F | ranked on | ceiling |
|---|---|---|---:|---|---:|
| betaxanthin | `020-cachera-betaxanthin` | Cachera 2023 corrected fluorescence | 1 | Pearson | 0.914 |
| beta_carotene | `021-ozaydin-beta-carotene` | Ozaydin 2013 colour score (ordinal) | 1 | **Spearman** | ~0.54 |
| mulleder19 | `022-mulleder-metabolome` | Mulleder 2016 19 amino acids | 19 | Pearson | unmeasured |

Every number below regenerates from
[[experiments.020-cachera-betaxanthin.scripts.analyze_002_noise_and_lambda]]
(`experiments/020-cachera-betaxanthin/scripts/analyze_002_noise_and_lambda.py` ->
`results/analysis_002_noise_and_lambda.json`).

### The measurement that shaped the round

`_002` used a TPE sampler, which re-proposes points -- so several configurations ran more than
once with **identical hyperparameters and an identical seed** (42 throughout; that sweep never
overrode `cfg.seed`). Those accidental repeats are the only direct measurement of the noise
floor every config comparison is made against:

| arm | repeated trials | values | pooled sigma | dof |
|---|---|---|---:|---:|
| betaxanthin | 3 / 10 / 11 | 0.4216 / 0.4095 / 0.3584 | **0.0302** | 3 |
| betaxanthin | 19 / 31 | 0.4211 / 0.3896 | | |
| beta_carotene | 18 / 22 | 0.1993 / 0.1494 | **0.0249** | 3 |
| mulleder19 | 7 / 10 | 0.1798 / 0.1780 | 0.0012 | 1 (unusable alone) |

The mechanism is structural, not incidental. `pearson_per_feature` is a **mean over features**,
so its noise falls as `1/sqrt(F_eff)`. Expression has F = 6,127 and is denoised by its own
metric; the two scalar arms have F = 1, where the objective is a single correlation over ~473
validation strains. Same metric name, same code path, two different statistical regimes --
which is why `_007`'s single-stage design is right for expression and insufficient here.

Consequences for betaxanthin:

- top five `_002` configs span **0.021 = 0.68 sigma** -> statistically **indistinguishable**
- selection inflation for a best-of-36 is `sigma * E[max_36]` = **0.064**, so the reported
  **0.4301 sits on a bias-corrected floor near 0.366**
- resolving a 0.02 gap needs `sigma_eff <= 0.010`, i.e. **R = 10 replicates**

mulleder19's dof-1 sigma estimate is not usable on its own (a chi-square interval on 1 dof
spans roughly 0.4x-30x the point estimate); plan on ~0.007 from the `1/sqrt(19)` argument.

### What that buys: a two-stage sweep

| stage | what runs | answers |
|---|---|---|
| `screen` | ~60 QMC trials, one seed each | which axes matter |
| `confirm` | top-K screen configs x 5 seeds, ranked on the **mean** | which config is actually best |

`cfg.seed` drives **both** model init and the `CellDataModule` split (`random_seed=seed`), so
the confirm stage is repeated-random-subsampling validation rather than an init re-roll -- the
stronger estimator for "best in expectation over splits", which is the quantity worth selecting
on. R = 5 takes betaxanthin's sigma to 0.013.

The launcher resolves `screen | confirm | done` from the study's own completed-trial counts at
each job start, so the deadline-guarded requeue chain carries an arm through both stages
unattended and stops taking a GPU once confirm is full.

### Inherited from _007 unchanged

- `lr` / `weight_decay` continuous log-uniform, `dropout` categorical -- the `hp_profile`
  bundle **decomposed**. The bundle was as unattributable here as in expression, and TPE
  starved it worse: `baseline` got n = 5 / 4 / **2** across the three arms while posting the
  best mean on two of them.
- Halton QMC (low discrepancy on the float axes; optuna delegates categoricals to a seeded
  random sampler, which is what balances the level counts `_002` lacked).
- Ranking on the peak `_max` of the phenotype-namespaced metric via `DistHead.point()`.
- Calibration recorded `_at_peak`, never ranked.

### What had to change, and the measurement forcing each

- **`energy` / `energy_rank` dropped on the scalar arms.** At F = 1, `Sigma = diag(sigma^2) +
  V V^T` has `V` of shape `[1, k]`, so `V V^T` is a scalar and the energy score degenerates
  into a Monte-Carlo CRPS -- noisier than the closed form already swept. A phantom dimension.
- **`energy_rank` for mulleder19 is a grid {0, 8}, not `_007`'s 32.** A 19x19 covariance has
  rank <= 19, so 32 cannot be realized, and `DistHead` has no guard -- it would allocate
  `V [19, 32]` and waste the excess. `_007`'s 32 was measured (participation-ratio rank 32.8
  of the 6,127-gene expression residual); the equivalent has **not** been measured for 19
  amino acids. Follow-up: re-run `residual_covariance_diagnostic.py` on this target.
- **Per-arm lambda grids.** The same nominal lambda buys very different prior strengths:

  | arm | measured ratio at 6.5e-5 | parity lambda | `_003` grid |
  |---|---:|---:|---|
  | betaxanthin | 0.41 | 1.6e-4 | {0, 1.6e-6, 1.6e-5, 1.6e-4, 1.6e-3} |
  | beta_carotene | 0.17 | 3.4e-4 | {0, 3.4e-6, 3.4e-5, 3.4e-4, 3.4e-3} |
  | mulleder19 | 0.29 | 2.2e-4 | {0, 2.2e-6, 2.2e-5, 2.2e-4, 2.2e-3} |

  A shared lambda is **not** a shared prior. `_002` ran all three at 6.5e-5.

  A 5-epoch smoke run checked the extrapolation directly. mulleder19 landed at
  `ratio_to_data = 0.992` at lambda 2.2e-4 -- parity, exactly as predicted. **beta_carotene
  landed at 8.6**, nine times the prediction, so its parity may sit nearer 4e-5 than 3.4e-4.
  The two disagreeing estimates come from different reductions (`_002` = ratio at last epoch
  of a full run; smoke = `_at_peak` of a 5-epoch run), and the four-decade grid brackets both,
  so the sweep settles it. The arm's *declared* lambda was lowered to 3.4e-5 accordingly --
  that value governs only a manual non-sweep run, where a prior at 8.6x the data loss would
  be training the prior rather than the data.
- **`num_transformer_layers` stays swept {2, 4}.** `_007` could freeze it because `_006` swept
  {2,4,6,8} on expression and found it inert; `_002` measured a large, monotone, **arm-dependent**
  effect (betaxanthin 2 -> 0.288, 4 -> 0.197, 6 -> 0.124, while beta_carotene preferred 4).
- **`graph_reg_depth` not swept** -- with L swept down to 2 (layers 0..1), `_007`'s middle `[2]`
  is out of range half the time and the axis would mean different things in different trials.
- **`node_embeddings` cut to {prot_T5_all, random_1000}**, keeping the width-matched control
  that makes "content, not identity" a measurement.

### Three defects carried over from _002, fixed here

1. **Sub-module dropouts were never bound.** `base.yaml` sets a literal `0.1` on
   `perturbation_head.dropout` and `learnable_embedding.preprocessor.dropout`, and the model
   only falls back to `model.dropout` when the key is **absent**. The `_002` driver overrode
   `model.dropout` alone -- so a trial sampling `dropout = 0.0` still ran those sub-modules at
   0.1. **`_002`'s dropout attribution is partly void.** Same defect class `_007` fixed for
   expression.
2. **`perturbation_head.num_heads` was a literal 6** while expression used 9. Both divide
   hidden = 90, so nothing crashed -- expression and metabolism were quietly running different
   perturbation operators. Now bound to `${model.num_attention_heads}`. Note this **changes
   the model relative to `_002`**.
3. **`min_delta: 0.0`** let any improvement reset early-stopping patience, including a noise
   blip, so a noisy metric kept runs alive on nothing. Now 1e-4, with `max_epochs` 250 -> 200.

### The run-length confound

Ranking on `_max` takes a maximum over however many validation epochs a run survived, and early
stopping makes that length depend on the hyperparameters. Correlation between trial duration and
the ranked objective:

| arm | r(duration, objective) | within hidden=90 |
|---|---:|---:|
| mulleder19 | **+0.754** | **+0.799** (n=23) |
| betaxanthin | +0.248 | +0.121 (n=23) |
| beta_carotene | +0.208 | +0.239 (n=32) |

On mulleder19 that is r^2 = 0.57 of the between-trial variance. Two readings cannot be separated
from this data -- better runs keep improving so early stopping lets them run longer (benign), or
longer runs take more draws at the max (mechanical) -- and either way any axis that changes run
length, **`lr` above all**, gets credit partly through draw count. Three additions make it
auditable rather than silent, all recorded and none ranked:

- `{metric}_smooth3_max` -- max of a 3-epoch rolling mean. A config whose `_max` far exceeds its
  `_smooth3_max` peaked on one lucky epoch rather than on a plateau.
- `val/n_val_epochs`, `val/peak_epoch` -- the covariates. A peak near the end means the run was
  still improving; peaks scattered early with a long tail is the noise-max signature.
- `{phenotype}/pred_sd_ratio` -- `sd(pred)/sd(target)`, the mean-collapse diagnostic **for a
  scalar head**. `pearson_per_instance` correlates across features within a strain and is
  undefined at F = 1, so it came back `None` on two of three arms in `_002`, leaving them with
  no collapse signal at all.

### Launch

```bash
cd <worktree-root>
ARM=betaxanthin   sbatch --job-name=020-bx-003  experiments/020-cachera-betaxanthin/scripts/gh_optuna_metabolism_003.slurm
ARM=beta_carotene sbatch --job-name=021-bc-003  experiments/021-ozaydin-beta-carotene/scripts/gh_optuna_metabolism_003.slurm
ARM=mulleder19    sbatch --job-name=022-m19-003 experiments/022-mulleder-metabolome/scripts/gh_optuna_metabolism_003.slurm
```

`DEADLINE` defaults to `2026-07-30T22:00:00` (a **fixed** date -- a relative one would be
recomputed by every requeue and the chain would never end). Override at submit time if needed.
Trial economics from `_002`: median 35 / 25 / 46 min, so ~60 screen + 8x5 confirm fits the
window on each arm.

### Open, and deliberately not in this round

- **The Merzbacher nested split is built but NOT wired into training.** No config references
  `results/merzbacher_nested_split.json`; `CellDataModule` does an 80/10/10 random split keyed on
  `random_seed`, so roughly 511 of their 639 test genes currently sit in our **train** set. The
  comparison 020 exists for cannot be made from these runs. See
  [[experiments.020-cachera-betaxanthin.merzbacher-comparison]].
- The Cachera build is stale w.r.t. the shared name resolver (issue #195), costing 10 of the 639.
- The flux layer stays deferred -- published k_cat covers 79 of Yeast9's 1,161 genes (6.8%).
  [[plan.cgt-metabolism-flux-layer.2026.07.26]]

## 2026.07.28 - Round _004: full_007 embedding axis + the pinned Cachera split

`_003` was cancelled at 26 / 28 / 18 of 60 screen trials. Three changes, two of which force a
fresh study (`suggest_categorical` rejects a changed value list, and 020's data changed):

1. **`node_embeddings` restored to _007's full EIGHT.** `_003` cut it to
   {prot_T5_all, random_1000} to buy resolution on the optimizer axes. That traded away the
   cross-round comparison, which is the reason to mirror_007 at all.
2. **The Cachera/Merzbacher test split is WIRED IN** on the betaxanthin arm
   (`data_module.pinned_test_split_file`).
3. **`ranked_smooth3`** replaces `_003`'s `pearson_smooth3`, derived from `OBJECTIVE_METRIC`
   rather than hardcoded -- beta-carotene ranks on Spearman, so its spikiness check had been
   comparing a Spearman maximum against a Pearson rolling mean. The tell was arithmetically
   impossible values (`smooth3 > max`, which cannot occur within one metric).

### What `_003` measured before it was stopped

Its headline question is already answered, consistently across all three arms: **the
`hp_profile` bundle was `lr`, not `weight_decay`.**

| | betaxanthin | beta_carotene | mulleder19 |
|---|---|---|---|
| lr, low / mid / high tercile | 0.132 / **0.177** / 0.087 | 0.120 / **0.132** / 0.097 | 0.059 / **0.092** / 0.031 |
| top-quartile lr median | **1.4e-4** | **1.4e-4** | **1.6e-4** |
| weight_decay terciles | 0.130 / 0.134 / 0.123 | 0.121 / 0.116 / 0.110 | 0.046 / 0.059 / 0.077 |

A clean inverted-U on lr, three independent arms landing on ~1.4e-4 -- between 010's 1e-4 and
`baseline`'s 3e-4, and well below `aggressive`'s 1e-3, which is why `_002` saw "aggressive
loses" without being able to say why. Weight decay is nearly flat (rho = -0.21 / -0.19 /
+0.09), vindicating the suspicion that `_002`'s naive bundle reading contradicted 010's
checkpointed 1e-8.

The new diagnostics also paid for themselves immediately:

- **`pred_sd_ratio`**: 21/26, 24/28 and 11/18 trials were COLLAPSED (< 0.05) -- predicting a
  near-constant while Pearson still read 0.14-0.18, because Pearson is scale-free. The top
  betaxanthin configs were healthy (0.618 / 0.288 / 0.566), so the ranking was selecting real
  models; **beta-carotene's leader was not** (0.001).
- **run length**: r(n_val_epochs, objective) = **+0.687** betaxanthin, +0.415 mulleder19,
  +0.125 beta-carotene. Real, but NOT distorting the leaders -- top-5 by `_max` and by
  `_smooth3_max` are identical in order on both Pearson arms, with top-3 gaps of
  0.020 / 0.011 / 0.009 (plateaus) against 0.059 / 0.061 for mid-ranked trials (spikes).

### The pinned split, and the bug it exposed

Wiring it surfaced a defect that would have invalidated the comparison in the opposite
direction from the one being fixed. The first implementation mapped gene names through
`is_any_perturbed_gene_index`, which resolved **639 genes to 4,885 of 4,930 records**, leaving
**train = 28**. It trained anyway.

The cause is real biology, not a coding slip. Both pigment cassettes contain NATIVE ORFs that
also exist in the deletion collection, and they are emitted as `gene_addition` / `allele`
perturbations on EVERY strain:

| cassette member | systematic | records | in Merzbacher's test list |
|---|---|---:|---|
| `ARO4` (K229L) | YBR249C | 4,669 | **yes** |
| `ARO7` (G141S) | YPR060C | 4,669 | no |
| `BTS1` | YPL069C | 4,406 | **yes** |
| `CYP76AD1`, `DOD`, `crtI`, `crtYB` | -- | 4,406-4,669 | no (heterologous) |

So `ARO4` as a *deletion target* is one record, but as a *cassette member* it is the whole
screen -- and two such genes are in the pinned list. This is exactly the distinction
[[torchcell.data.genotype_aggregate]]'s `DeletionKeyedGenotypeAggregator` exists to make
("the cassette belongs to the reference cell, not to the perturbation"); the split wiring
simply used the wrong index.

**Fix:** a new `Neo4jCellDataset.is_any_deletion_gene_index` (deletion-only counterpart,
disk-cached like its sibling), plus a hard assertion that a pinned test set is under half the
dataset. Verified live: **639 genes -> 639 records, 13.0% of 4,930**, splits
train 3,703 / val 294 / test 933, with zero pinned records in train or val.

Nine tests in `tests/torchcell/datamodules/test_cell.py` (the first tests this datamodule has
had) cover the properties that are invisible at runtime: pinned records land in test and
nowhere else, the pin changes the cache key so a pinned run cannot reuse an unpinned cached
index, the same pin reuses its own cache across a requeue, absent genes are reported not
fatal, and the pin survives a seed sweep so the confirm stage's 5 seeds re-roll train/val
without ever moving the comparison genes.

### Reading `_004` numbers

**Val remains the ranking metric and stays comparable to `_002`/`_003`.** The betaxanthin
TEST split is now larger (933 vs ~490) and contains Merzbacher's genes by construction, so it
is NOT comparable to earlier test numbers -- it is the comparison set, to be read against
their Fig 4b via [[experiments.020-cachera-betaxanthin.merzbacher-comparison]].

## 2026.07.28 - Delta: four jobs, and the controlled auxiliary-task experiment

Four 2-day / 4-GPU jobs on Delta (NCSA), alongside the GilaHyper `_004` sweep. Three mirror
the GilaHyper arms at 4x the GPU count; the fourth is a new experiment.

| job | ARM | base config | what it is |
|---|---|---|---|
| 1 | `betaxanthin` | `delta_betaxanthin_000` | mirror of the GH arm |
| 2 | `beta_carotene` | `delta_beta_carotene_000` | mirror of the GH arm |
| 3 | `mulleder19` | `delta_mulleder19_000` | mirror of the GH arm |
| 4 | `bx_pair` | `delta_bx_m19_000` | **the controlled pair**, 2 GPUs per side |

### The question job 4 answers

**Does the rest of the metabolism signal improve betaxanthin prediction?** Two arms, one base
config, differing in EXACTLY one thing -- whether the 19-AA metabolome head is attached:

```text
ARM=bx_ctrl   active_heads=[betaxanthin]                 control
ARM=bx_m19    active_heads=[betaxanthin, mulleder19]     joint
```

Both rank on `val/betaxanthin/pearson_per_feature`. The metabolome is an **auxiliary task**,
not a second objective: a Pareto front over both would answer "which configs are good at
both" and would make the two arms incomparable. `bx_m19 - bx_ctrl` is then the auxiliary-task
effect, read on one metric, at the confirm stage (top-K x 5 seeds) -- betaxanthin's replicate
sigma is 0.030, so a single-seed difference under ~0.06 is not readable.

Both run as ONE slurm job (2 GPUs each) so they cannot drift apart in node, queue position or
software state. A paired comparison split across two jobs days apart is a worse comparison
for no benefit.

### The control needed a new mechanism

`require_modalities` intersects on phenotype LABEL, which separates expression from calmorph
but **cannot** separate betaxanthin from the metabolome: both are `metabolite_level`, and
what distinguishes them is which KEYS their value dict carries (`{betaxanthin: ...}` vs the
19 amino acids). `require_modalities: [metabolite_level]` is a no-op here -- it would have
silently left the two arms on different instance sets, which is precisely the confound the
control exists to remove.

New `cell_dataset.require_head_targets` resolves each named head through
`multitask.head_phenotype_keys` and keeps genotypes carrying at least one key for every
listed head. Measured on this build:

| | genotypes |
|---|---:|
| betaxanthin | 4,669 |
| mulleder19 | 4,678 |
| beta_carotene | 4,406 |
| **betaxanthin AND mulleder19** | **4,432** |
| all three | 4,023 |

So the restriction costs only **237 rows (5%)** -- a far better position than the 019
expression/morphology control, whose both-modality intersection was 1,440. Verified live:
4,930 -> 4,432 (train 3,820 / val 311 / test 301), and both heads train.

The joint arm also **drops the pinned Merzbacher split** (`pinned_test_split_file: null`).
That pin is right for the head-to-head against their Fig 4b, but this is an INTERNAL contrast
between two of our own arms, and both sides should see the same ordinary seeded split rather
than whatever remains after a 639-gene block is removed.

Recorded alongside the headline: `aux_mulleder19_pearson`, the metabolome head's OWN score in
the joint arm. "The metabolome helped betaxanthin" and "the metabolome head learned anything"
are different claims -- an auxiliary head at r ~ 0 that still moves the primary metric would
mean the gain came from regularization, not from shared metabolic signal.

### max_epochs 400

Up from 200, on all four Delta configs. On `_003`/`_004` the 200 cap NEVER bound (0/24 runs;
longest run 175 epochs, latest peak at 134) -- but the BEST run was also the latest-peaking
one, leaving only 66 epochs of headroom. Raising it is nearly free: a converged run stops on
`patience` regardless, so the extra ceiling is only ever spent by a run genuinely still
improving. See [[experiments.022-mulleder-metabolome.scripts.analyze_training_length]].

**GilaHyper `_004` keeps 200** deliberately -- it is mid-flight, and changing the cap under a
running study would mix provenance within one study for no gain.

### Delta specifics

- Repo `/projects/bbub/mjvolk3/torchcell` + `rockylinux_9.sif`; **DATA_ROOT
  `/work/hdd/bbub/mjvolk3/torchcell`** (the large space; `/projects` has a tight quota and
  `run_training` resolves the dataset from DATA_ROOT).
- Account **`bbtp-delta-gpu`**, partition **`gpuA40x4`**.
- Env **`/work/hdd/bbub/miniconda3` / `torchcell313`** -- Delta's stock torchcell envs are
  Python 3.11 and the repo uses PEP 695 generics, a SyntaxError there.
- **W&B ONLINE** -- Delta compute nodes have internet, so no offline/sync dance.
- Four workers share ONE study per arm: they sit on one node and one filesystem, so
  concurrent SQLite is safe. The Delta and GilaHyper studies are deliberately NOT pooled --
  different filesystems, and cross-cluster SQLite has no coherent locking. They are pooled by
  READING, not by sharing a file.

### Transfer

`experiments/020-cachera-betaxanthin/scripts/sync_delta_fig6.sh` moves ~13 GB: the
`fig6_pigment_transfer` dataset (293 MB), all seven embedding trees the eight-way axis can
draw (1.1 GB), and the genome/graph trees (`sgd/genome` 11 GB, `string` 326 MB, `tflink`
80 MB, `go` 33 MB). Directory names were verified against `NodeEmbeddingBuilder`'s
`root_path` entries rather than guessed -- a missing embedding does not fail at submit time,
it fails ~20 minutes into whichever trial first draws it, so the script refuses to transfer a
partial set.

**Duo:** Delta requires 2FA per SSH connection. The script keeps it to **ONE push** via
`ControlMaster` multiplexing plus `--rsync-path="mkdir -p ... && rsync"` (the destination is
created inside the rsync's own session rather than by a separate `ssh mkdir`). `DRY_RUN=1`
does a purely LOCAL inventory and opens no connection at all -- `rsync --dry-run` would still
authenticate, which defeats the point.

rsync is incremental, so if the July-2026 fig3_core transfer already placed the shared trees
on Delta, only `fig6_pigment_transfer` is genuinely new.

### Launch

```bash
# from GilaHyper -- one Duo push
DRY_RUN=1 bash experiments/020-cachera-betaxanthin/scripts/sync_delta_fig6.sh   # inventory
bash experiments/020-cachera-betaxanthin/scripts/sync_delta_fig6.sh            # transfer

# on Delta
cd /projects/bbub/mjvolk3/torchcell
for A in betaxanthin beta_carotene mulleder19 bx_pair; do
  ARM=$A sbatch --account=bbtp-delta-gpu --job-name=020-$A-delta \
    experiments/020-cachera-betaxanthin/scripts/delta_metabolism_000.slurm
done
```

**Open before launch:** confirm `torchcell313` still exists on Delta and that `optuna` is
installed into it -- the July note flagged both as blockers, and neither is verifiable from
GilaHyper.
