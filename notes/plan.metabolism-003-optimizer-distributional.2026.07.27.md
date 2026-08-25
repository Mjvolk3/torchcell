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
- Env **`/work/hdd/bbub/miniconda3/envs/torchcell`** -- the name is `torchcell`, NOT the
  `torchcell313` the July notes used. Validated interactively 2026-07-28: torch 2.8.0+cu128,
  PyG 2.7.0, torch_scatter 2.1.2+pt28cu128, torchcell 1.2.0, full `torchcell.datasets` import
  chain clean. Delta's stock envs are Python 3.11 and hit a SyntaxError on the repo's PEP 695
  generics, which is why a dedicated env exists at all.
- **NO SINGULARITY.** An earlier draft of the launcher wrapped every call in
  `singularity exec --nv rockylinux_9.sif`, inherited from the 019 launchers. The env that
  was actually validated runs NATIVELY -- its torch is built against the host CUDA, and the
  smoke test that cleared the whole import chain called the env python directly. Running a
  host-built CUDA wheel inside a container with its own driver stack is an untested second
  variable with nothing to gain, so the launcher calls `$DELTA_PY` directly.
- **Install with the env's EXPLICIT pip**
  (`/work/hdd/bbub/miniconda3/envs/torchcell/bin/pip`). A bare `pip` on the Delta login node
  is system python3.9 and silently drops packages into `~/.local`, where this env never sees
  them -- which is how a "successful" install still leaves a `ModuleNotFoundError` at run
  time. This cost most of one evening.
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

**Env status (2026-07-28): CLEARED.** The `torchcell` env imports the full chain
(`torch_scatter` ABI, `torch_geometric.utils._subgraph`, `torchcell` 1.2.0, and the ten
previously-missing deps including GEOparse). What remains unproven is a real trial on a
COMPUTE node -- GPU binding and a cold LMDB build. Prove it with a one-trial interactive run
before committing four 2-day jobs:

```bash
srun --account=bbtp-delta-gpu --partition=gpuA40x4-interactive \
  --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64g --time=01:00:00 \
  bash -c '
cd /projects/bbub/mjvolk3/torchcell
export PYTHONPATH=/projects/bbub/mjvolk3/torchcell:$PYTHONPATH PYTHONUNBUFFERED=1
export DATA_ROOT=/work/hdd/bbub/mjvolk3/torchcell
export ARM=betaxanthin STAGE=screen OPTUNA_N_TRIALS=1 WANDB_MODE=offline \
       OPTUNA_STORAGE=sqlite:////tmp/tc_metab_smoke.db METAB_BASE_CONFIG=delta_betaxanthin_000
/work/hdd/bbub/miniconda3/envs/torchcell/bin/python -u \
  experiments/020-cachera-betaxanthin/scripts/optuna_metabolism_sweep.py
'
```

Reaching a `val/betaxanthin/...` line means green for all four sbatch jobs. A cold LMDB build
may exceed the 1 h interactive limit -- if so, rerun on the non-interactive `gpuA40x4` with a
2 h limit rather than assuming failure.

## 2026.07.31 - Four Delta jobs: max out betaxanthin, metabolome, the joint pair, beta-carotene

**Both branches are landed on `main`** -- `8b918333` (scaffolding + the two defect fixes) and
`e9351938` (the long-run redesign). The Delta launch mechanics that ran alongside this design are
not carried over; what follows is the design and what it rests on.

### The finding that changed the design -- read this first

You asked for 8 runs × 2 days per experiment and said "we have to see saturation". My first
cut treated 2 days as a *replication* budget (8 settings × many seeds, ~30 min each). Then I
read [[experiments.019-simb-multimodal.wave6-design]], and your instinct is backed by a measurement I
had not weighted:

> Smoothed val Pearson peaks ≈0.14 at epoch 85-136, falls to 0.08-0.11 by epoch 200-300, then
> rises to a project-best **0.1980 at epoch 1367**. *Every arm ever scored at 300-400 epochs
> was scored in the dip.* Nothing had converged at 1,500 epochs. Val **loss** and val
> **Pearson** move in opposite directions.

Now put the metabolism history next to it. Every `_002`/`_003`/`_004` run early-stopped on a
patience of 40-50 and finished at a **median of ~71-135 epochs** -- i.e. at or before the first
bump. Same model, same trainer, same loss family.

**So the 0.4301 we keep quoting may not be this model's ceiling; it may be its first local
maximum.** That is a hypothesis, not a fact -- nobody has run metabolism past ~200 epochs. It
is also the cheapest possible thing to test, and it is exactly what a 2-day A40 node buys.

Hence: **8 settings, each trained for the whole 48 h.** Not 8 settings × 12 seeds.

| | old cut (wrong) | what ships |
|---|---|---|
| per-run length | ~30-90 min, early-stopped at patience 50 | the full window, ES **off** |
| epoch cap | 400 | 10,000 (a ceiling; the clock decides) |
| what stops a run | `EarlyStopping` | `trainer.max_time_s` → Lightning `Timer` |
| runs in flight | 4 (1/GPU) | 8 (2/GPU) -- all eight settings share the window |
| what round 0 answers | "which setting wins" | **"where does this target saturate"** |

Round 0 is 8 long runs at **one seed**, so it does *not* resolve a 0.02 gap between settings
(σ = 0.030 is still the floor). Read round 0 as **curves**, not a leaderboard. Rounds 1-2 are
replicate seeds and only run if a setting converges early enough to free a worker.

The `Timer` is what makes this runnable: it stops training *gracefully*, so `fit` returns and
the metric snapshot, the test pass and the prediction dump all still happen. A slurm kill
mid-`fit` loses all three and leaves the Optuna trial `RUNNING` instead of `COMPLETE`.

### Two defects that would have made the jobs worthless

1. **`num_workers=0` was still unconstructible.** `2cca6d83` guarded `prefetch_factor` but not
   `timeout`, so torch asserts `_SingleProcessDataLoaderIter requires timeout == 0` from
   `iter(dataloader)` -- *after* the dataset and embeddings load. That is how Delta job
   20556837 lost 119 trials while exiting `COMPLETED 0:0`. `NUM_WORKERS=0` is the Delta
   default (spawn re-imports the stack off the parallel filesystem: a 38-minute sanity check),
   so this gated **every** Delta run. Fixed in `torchcell/datamodules/cell.py`.
2. **The pinned Merzbacher split was never scored.** `run_training` called `fit` and stopped,
   so `pinned_test_split_file` held their 639 genes out of training and then nothing looked at
   them. Added `trainer.run_test` (on the **best** checkpoint -- these runs mean-collapse, so
   the last model reports ~0 for a run that worked) + `trainer.dump_test_predictions`.

### The four jobs

| # | question | experiment | arms | settings |
|---|---|---|---|---|
| 1 | max out betaxanthin, beat Merzbacher **on their split** | `020-cachera-betaxanthin` | `betaxanthin` | 8 |
| 2 | max out the 19-AA metabolome | `022-mulleder-metabolome` | `mulleder19` | 8 |
| 3 | does the metabolome help betaxanthin? | `023-metabolome-betaxanthin-joint` *(new)* | `bx_ctrl` + `bx_m19` | 4 × 2 arms |
| 4 | max out beta-carotene | `021-ozaydin-beta-carotene` | `beta_carotene` | 8 |

**The 8 settings are a 2×2×2 factorial**, fully crossed, defined in
`experiments/019-simb-multimodal/scripts/metabolism_grid_runner.py`:

- **dropout {0.1, 0.3}** -- and it goes *up*. 019 paired: dropout 0 → train 0.72 / val 0.14-0.16;
  dropout 0.1 → train 0.64 / val 0.178-0.198. At 10,000 epochs over-fitting is the binding
  constraint, and metabolism has never swept dropout above 0.2.
- **L {2, 6}** -- a direct disagreement. 019 wave-6 fixes L=6; metabolism `_002` measured a
  monotone preference for L=2 (2→0.288, 4→0.197, 6→0.124). Both can't be right, and the
  metabolism reading came from ~100-epoch runs, exactly where a deeper model is still behind.
- **graph_reg λ, per arm** -- decades bracketing each arm's own measured parity (ratio = 1 at
  1.6e-4 / 3.4e-4 / 2.2e-4). A shared λ is not a shared prior strength.

**Frozen, deliberately:** `dist` (019 `_007` at n≈60/mode: the distributional axis is *not*
the lever -- each arm takes its own measured best), `target_norm` = zscore, `prot_T5_all`,
`learnable_embedding=false`, `perturbation_head.num_heads=6`.

Exceptions: **mulleder19** swaps depth for `dist {quantile, energy}` -- it is the only arm with
a joint distribution to model (F=19; at F=1 the energy score degenerates to a noisy CRPS), and
019's finding was about *marginals*. **Experiment 3** runs 4 settings × 2 arms instead of 8 ×
1, because the pairing *is* the experiment: both arms share every (setting, seed) cell, hence
the same split and the same init, so `bx_m19 − bx_ctrl` is a **paired** difference.

**One deliberate revert on experiment 1:** `perturbation_head.num_heads` back to 6. `_004`
rebound it to 9 *in the same round* it introduced the pinned split, and that round's best fell
to 0.2469 from `_003`'s 0.4050. Two changes, one drop, no attribution. The pinned split is
required by the question; the operator change is not.

### Read-out

W&B is **online** (Delta compute nodes have internet), projects
`torchcell_{020_betaxanthin,021_beta_carotene,022_mulleder19,023_bx_m19}_delta_grid`.

**The primary artifact is the val-Pearson-vs-epoch curve, not the final number.** The question
is whether metabolism shows the 019 dip-then-climb. Log `val/<pheno>/pearson_per_feature`
against epoch for all 8 settings; if any is still climbing at 48 h, the follow-up is more
time, not more settings.

Per-arm summary from the study (worker 0 prints it, or read it yourself):

```bash
$PY -c "
import optuna
s = optuna.load_study(study_name='betaxanthin_grid_000',
      storage='sqlite:///experiments/020-cachera-betaxanthin/optuna/optuna_020-cachera-betaxanthin_betaxanthin_grid.db')
for t in sorted((t for t in s.trials if t.values), key=lambda t: -t.values[0]):
    print(round(t.values[0],4), t.params['setting'], 'peak_ep', t.user_attrs.get('peak_epoch'),
          'n_ep', t.user_attrs.get('n_val_epochs'))"
```

**Experiment 1 → the Merzbacher head-to-head** (back on GilaHyper, after rsyncing the dumps
from `$DATA_ROOT/test-predictions/`):

```bash
python experiments/020-cachera-betaxanthin/scripts/evaluate_merzbacher_head_to_head.py
```

Their released labels as truth, bin scale fitted on the train pool only, MCC + top-k
high-producer enrichment next to accuracy, aggregated over runs. Their bar
(`RandomForestClassifier_Resampled`, from their own shipped predictions): accuracy 0.700 vs a
majority rate of 0.673, MCC 0.205, high-producer recall 0.18, **94.8 % of genes called
medium**. If ours is also ~95 % medium we reproduced their failure mode and must say so.

**Experiment 3 → the paired difference:** per (setting, seed) cell, `bx_m19 − bx_ctrl`, then
the mean paired difference with its SE. Report `aux_mulleder19_pearson` beside it -- an
auxiliary head at r≈0 that still moves the primary metric means the gain was regularization,
not shared metabolic signal.

### Open, flagged not resolved

- **Throughput is unmeasured on A40.** 019 §8 wants ≤17.3 s/epoch for 5,000 epochs in 24 h and
  measures 32 s/epoch on mmli. If Delta lands near 32 s/epoch we get ~5,000 epochs in 46 h,
  which clears the 1,367-epoch peak with room -- but nobody has timed this model on an A40.
  Check the first log for epoch wall time.
- **019 §8's throughput levers are not implemented here** (host-RAM record cache, `batch_size`
  8→32, `train_eval_every`). Metabolism already runs at batch 128, so the largest lever is
  spent; the RAM cache is the remaining one and is out of scope for tonight.
- **Experiment 4's second half is unspecified** -- you wrote "same for beta carotene - for this
  we plan to see" and it trails off. I built it as max-out only. If you meant "does the
  metabolome/betaxanthin help beta-carotene", that is a fifth job (or a 023-style pair), say
  the word.
- **The Cachera build is stale w.r.t. the name resolver** (issue #195). The split is built from
  the raw screen so it is unaffected, but **training data still inherits it**.

## 2026.07.31b - REVISED after the 019 expression work landed (`3bf3dbe4`)

Three things changed. The first two are bugs that would have killed all four jobs.

### Metabolism was BROKEN on main

The masked-label objective (v9) added `observed_values` / `observed_mask` to
`CellGraphTransformer.forward` and the trainer passes them unconditionally, but
`CellGraphTransformerMetabolism.forward` still named the old arguments -- so every metabolism
run died with `TypeError: forward() got an unexpected keyword argument 'observed_values'`, at
the **first training batch**, ~20 min in, after the dataset and embeddings had loaded. The 7
existing tests all call `model(cell_graph, batch)` positionally while the trainer calls by
keyword, which is exactly why none caught it. Now a `*args/**kwargs` pass-through, with a
regression test that derives its argument set from the *parent's* signature.

Second: two `ModelCheckpoint`s on the same monitor share a `state_key` and Lightning refuses
to build the Trainer. `metric_monitor` is now the arm's *other* correlation -- also what the
Merzbacher recipe reports.

### λ was the wrong knob (and expensive)

`graph_reg_lambda` is the **KL penalty to the adjacency matrices**. `cgt_expr_010` already
replaced it with **hard graph masking** (`attention_mask.enabled: true`, nine relations, one
per head) and set λ to 0. Measured under identical packing:

| | s/epoch |
|---|--:|
| `D2_mask` (mask on, λ=0) | **28.0** |
| the seven KL runs | 42.3 - 48.7 |

The KL needs attention *weights*, so it must materialize a `[1, 9, 6608, 6608]` matmul and
cannot use the fused kernel. **1.5-1.7×** -- in a round whose question is where the curve
saturates, that is a third of the epochs. λ is not frozen, it is *gone*.

### The grid now matches the current best

| | before | now |
|---|---|---|
| graph prior | KL, λ swept | **hard mask**, λ = 0 |
| L | {2, 6} swept | **6** (`_012`) |
| dropout | {0.1, 0.3} | **0.1** frozen |
| seed | init **and** split | **init only** (`split_seed: 0`) |
| factor 1 | dropout | **`hadamard {off, replace}`** |
| factor 2 | λ | **mask depth `{[1],[1,3]}`** (mulleder19: `dist {quantile, energy}`) |
| runs | 8 settings × 1 seed | **4 settings × 2 init seeds** (023: 2 × 2 arms × 2 seeds) |
| -- | -- | `train_eval_every: 25`, best-by-metric checkpoint |

**Why `hadamard` is the right spend:** 019 proved the additive operator has *no* pair-(p,i)
term at |S_b| = 1 -- re-drawing `W_Q`, `W_K` at std 10 changes the output by **exactly 0.0**,
leaving 16,200 of 32,760 attention parameters dead. Every metabolism strain is a single
deletion, so this holds on all four arms. `replace` swaps in `h_i ⊙ (1 + γ(c_b))`, a genuine
rank-90 interaction, identity at init.

**Why replication is affordable now:** pinning the split makes `seed` vary init only. 019
measured between-seed sd 0.0444 against across-arm 0.0058 when one knob drove both -- the
nuisance axis was 7.7× the signal axis. Cost, stated plainly: the absolute level now belongs
to one validation draw; rankings transfer, the number does not.

### Still open

- **Beta-carotene inference on the CIT2 double-KO panel.** `CIT2` appears only as prose in
  the 021 configs -- **there is no dataset and no loader in the repo**. Where is that panel?
  Until it exists, experiment 4 is max-out only and the inference step cannot be written.
- **Throughput on A40 is still unmeasured.** Indicative only (contended GPU, GilaHyper, L=6,
  masking on): the smoke ran ~30 s/epoch, which over 46 h would be ~5,500 epochs -- clearing
  019's 1367-epoch peak with room. Confirm from the first Delta log rather than trusting it.
- **`GRID_WORKERS_PER_GPU=2` at L=6 is untested for memory.** Masking *reduces* memory
  (~35 vs ~40 GB per GPU pair, per `_010`), but if the log shows CUDA OOM, resubmit with
  `--export=ALL,GRID_WORKERS_PER_GPU=1` -- the queue is unchanged, runs just serialize.

## 2026.08.01 - FINAL: the 24-run grid, ready to launch (`f0a22be7`)

Superseded everything above after reviewing 019 wave 6 + the full 020 betaxanthin W&B history.

### What the review killed

| axis | verdict | evidence |
|---|---|---|
| pair term (`hadamard`) | **null** -- frozen `off` | wave 6 ran ranks 0/9/16/32/64/90 across six mechanisms at ~2,425 epochs. **Rank-0 baseline placed 2nd of 12** (0.2107). `replace` = 0.1965, *below* baseline. |
| dropout up | **negative** -- frozen 0.1 | 0.2 → 0.1881; **0.3 → collapsed to exactly 0** (peak 0.1262 @ ep 225, then nmse 1.002 / pred_sd_ratio 0) |
| mask depth | **null** -- frozen `[1]` | wave 5, 4 seeds each: `[3]` 0.1609 vs `[1,3]` 0.1597 |
| `graph_reg_lambda` (KL) | **gone** | `_010` replaced it with hard masking; KL blocks the fused kernel (28.0 vs 42-49 s/epoch) |

### The grid: `L{2,6,4} × mask{on,off} × lr{1e-4,1e-3} × target_norm{zs,yj}` = 24

Three waves of eight, depth-major, ordered **2 → 6 → 4**. Each wave is a complete 2×2×2 at one
depth; all cells cost the same, so a slow node squeezes the *last* wave -- losing the
interpolation point, keeping both contrast blocks. Main effects on 12-vs-12, SE ≈ 0.012.

Per-arm: **022** swaps `target_norm` → `dist {quantile, energy}`; **023** swaps it → `{bx_ctrl,
bx_m19}` (12 per arm), giving a paired control-vs-joint at every (depth, mask, lr).

Frozen: `prot_T5_all` · hidden 90 · 9 heads · hadamard off · dropout 0.1 · wd 1e-8 · batch 128 ·
seed 42 · `split_seed=0` · **`max_epochs: 1000`** · no early stopping · **ckpt on best val
pearson** · Merzbacher's 639 genes pinned to test on 020.

W&B: `torchcell_020_betaxanthin_v4`, `torchcell_021_beta_carotene_v4`,
`torchcell_022_mulleder19_v4`, **`torchcell_023_bx_m19_v1`** (new).

### Budget

```
measured 18.5 s/epoch (GilaHyper, CONTENDED; L=2 and L=6 alike -- depth is nearly free)
x1.4 A40 derate -> ~25 s/epoch -> 1000 ep + startup = ~7.2 GPU-h/run
24 runs x 7.2 = 174 GPU-h / 4 GPUs = 43.4 h = 1.81 d   (48 h wall)
```

Caveat: the derate is unmeasured on Delta. Protection is that an over-running run is
**Timer-stopped gracefully** -- checkpoint, metrics and the 020 test dump all still land.
