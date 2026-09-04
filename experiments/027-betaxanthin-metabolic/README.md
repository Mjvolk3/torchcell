# 027 -- Does the metabolic flux module help on betaxanthin?

Decides whether the enzyme-constrained thermodynamic flux module, specialized to the
betaxanthin route, predicts the Cachera 2023 genome-wide betaxanthin screen better than
(a) the same CGT with the module removed and (b) Flux Cone Learning (Merzbacher 2025,
`RandomForest_Resampled`), scored on Merzbacher's own 639-gene test split.

**NOTHING HERE HAS BEEN SUBMITTED.** Every file is prepared and locally validated; the launch
commands at the bottom are for the author to run after reading the prerequisites.

## Why this experiment is not just "026 with more epochs"

026 ran 98 cells and could not distinguish any arm from any other, or from nothing at all.
Re-reading its own results files gives the three reasons, and each one has a fix here.

| 026 defect | measured | 027 fix |
| --- | --- | --- |
| the headline number was a **maximum over epochs of validation Pearson** -- the same curve used to pick the epoch | with n_val = 319-362 the null width of that statistic is 1/sqrt(n-3) = 0.054, and the arms were separated by 0.004-0.08 | the epoch is selected on validation; the number reported is Spearman on a **disjoint pinned test set** |
| the **split was re-rolled per seed**, so seed moved the nuisance and the signal together | paired arm gap sd 0.0303 (real) and 0.0722 (permuted); 019 measured between-seed sd 0.0444 against within-seed across-arm sd 0.0058 | `split_seed` pinned at 0; seed varies weight initialization only |
| the arm gap was **never compared to its own permuted-label gap** | real gap +0.0735, PERMUTED gap +0.0302 -- more than 40% of "flux helps" survives destroying the labels | permuted-label arms are first-class, and the decision rule is the *difference of the two gaps* |

The last row is the finding that invalidated 026's headline. Against zero the real gap is
7.7 sigma. Against its own null it is +0.0433 at 1.9 sigma. Both nulls are in
`experiments/026-metabolism-flux/results/sweep_arms_{a,b}.json` and `sweep_null.json`.

## Arms

Six, fully crossed with seeds. Each arm file in `conf/` changes exactly two fields
(`base_arm`, `permute_train_targets`) against `conf/base.yaml`, so a difference between two
arms cannot come from a hyperparameter that moved with the module.

| arm | metabolic module | thermodynamics + kinetics | labels | what it isolates |
| --- | --- | --- | --- | --- |
| `pooled` | off | -- | real | the no-module control; the 020_v4 architecture |
| `flux_off` | on | none (mass balance + GPR only) | real | does routing through the stoichiometric network help at all? |
| `flux_free` | on | learned potential (loop-freedom) | real | is structural loop-freedom enough? |
| `flux_anchored` | on | tabulated delta_f G', kcat capacity, protein budget, dissipation | real | **the treatment** |
| `null_pooled` | off | -- | **permuted** | null for `pooled` |
| `null_anchored` | on | full | **permuted** | null for `flux_anchored` |

The flux settings themselves are **not redefined here**: `base_arm` is a key into 026's
`ARMS` registry, and `train_bx.py` asserts it resolves. 026 stays the single source of truth
for what the module does.

**Dropped: `flux_nullspace`.** Measured over 10 seeds it peaked at 0.0381 +- 0.0359 with a
median peak epoch of 0.5 and a last-five mean of **exactly 0.0000 in all ten runs** -- it
collapses onto a constant. Including it would spend a sixth of the allocation on an arm whose
outcome is already known.

**Why no hyperparameter search.** 026's `reg` grid crossed lr x weight decay x hidden width at
n=2 per cell; every cell sat inside one null width of every other. At this sample size a
search cannot resolve hyperparameters, only replication can resolve arms. The incumbent
(lr 1e-3, wd 1e-4, hidden 32, L 2, batch 128) is fixed in `conf/base.yaml` and is identical
across all six arms.

## Seeds: 24, and 3 is provably not enough

Three replicates is the usual floor. The measured spread says three cannot decide this
question, so the arithmetic is given rather than the convention.

Two independent estimates of the across-seed sd of the paired arm contrast agree closely:

- **026, val-peak statistic:** paired (`flux_anchored` - `pooled`) sd 0.0303 with real labels
  (n=10) and 0.0722 with permuted labels (n=12); **pooled 0.0554**.
- **020_v4, the actual 027 endpoint** (test Spearman on these same 639 genes, 6 lr=1e-4
  Delta cells): **sd 0.0541**.

The quantity the decision rests on is the *excess of the real gap over the permuted gap*, so
its standard error is `sd * sqrt(2/n)`:

| seeds per arm | SE(excess) | 2-sigma MDE | z on 026's observed excess (+0.0433) |
| --: | --: | --: | --: |
| 3 | 0.0452 | 0.090 | 0.96 |
| 6 | 0.0320 | 0.064 | 1.35 |
| 12 | 0.0226 | 0.045 | 1.92 |
| **24** | **0.0160** | **0.032** | **2.71** |

At n=3 the experiment cannot see an effect twice the size of the one it is looking for. 24 is
the smallest count that puts the 026-observed excess clearly outside the noise, and it is
what 9 nodes buys in one wall-clock window. 0.0554 is also a **conservative** input: it was
measured with the split re-rolling per seed, which 027 pins, so the realized MDE should be
better than the table says. It will be measured directly from the completed grid rather than
assumed.

The grid is enqueued **round-major and sharded by whole seed**, so a wall-clock kill leaves
every arm at the same seed count -- 12 balanced seeds rather than 24 unbalanced ones.

## Scoring rule, fixed before launch

Written in `conf/base.yaml` under `scoring:` and implemented in `train_bx.py`. Do not edit
either after the first submission.

- **Primary endpoint: Spearman** between predicted and measured betaxanthin over the pinned
  Merzbacher test genes. Spearman rather than accuracy because on a 67%-majority problem
  accuracy actively selects against the capability the task is about -- FCL clears the
  majority rate by 0.027 by calling 94.8% of genes medium.
- **Epoch selection: argmax of a centered 5-epoch moving average of validation Pearson.**
  Smoothing is not cosmetic: the raw argmax of a 200-point noisy curve is itself a max order
  statistic. Implemented with a 5-deep ring of trainable-parameter snapshots, and the online
  selection is cross-checked against an offline recompute from the saved curve; a
  disagreement raises.
- **Selection and reporting use disjoint data.** Validation is the seeded 10% of the
  non-pinned pool (294 records); the reported number is on the 639 pinned genes, which no arm
  and no epoch ever saw.
- **NEVER a maximum over seeds, arms or cells.** An arm's result is the mean over seeds with
  its across-seed sd and n. 020's Fig 6 is the cautionary case: CGT cells spanned 0.013-0.158
  Spearman and 0.406-0.695 AUC, so any single-number claim is dominated by which cell is
  quoted.
- **200 epochs, no early stopping.** 026 never ran past 30 and peaked at a median epoch of
  3.5-17, so "the curve is finished by 20" is an extrapolation its budget never tested. 019
  measured the opposite shape on expression -- peak ~0.14 at epoch 85-136, a fall to 0.08-0.11
  by 200-300, then a project best 0.198 at epoch 1367. 200 is the smallest budget that
  separates those two shapes. The extra epochs cannot inflate the reported number, because
  the number is a test score at a validation-selected epoch.
- **Secondary, all prespecified:** test Pearson; top-10/25/50 enrichment for high producers;
  the same metrics on the whole test split; and the same metrics stratified by yeast-GEM
  membership, since the flux module has features for only 19.4% of the screen.

### The split, as it actually resolves

Measured by the local smoke runs, not assumed. All 639 Merzbacher test ORFs resolve in the
current build (the split file's `availability_in_current_build` warns of 10 missing as of
2026-07-28; the build has since been refreshed, and the trainer reports the count either way).

| split | records | with a betaxanthin measurement | null width of a correlation |
| --- | --: | --: | --: |
| train | 3,703 | -- | -- |
| val (selection only) | 294 | 270 | 0.061 |
| test | 933 = 639 pinned + 294 other | 905 | -- |
| **test, pinned only (reported)** | **639** | **629** | **0.040** |

The reported endpoint sits on more than twice the observations that 026's selection statistic
did, and it is not the statistic used for selection. Both facts shrink the noise floor the
arms have to clear.

### Null calibration and the decision

`analyze_bxfx.py` implements this and nothing else:

- **D1** -- `mean_s[anchored - pooled] - mean_s[null_anchored - null_pooled] > 0`, with a
  seed-wise percentile-bootstrap 95% interval excluding 0. Resampling is joint across arms so
  the pairing is preserved. A one-sided bootstrap of the null gap alone is reported alongside,
  because 026's null gap was skewed (sd 0.0722 against the real gap's 0.0303) and a symmetric
  interval understates how often a null can look like a result.
- **D2** -- `mean_s[test_spearman(flux_anchored)]` exceeds FCL RF's **+0.0391** by more than
  one across-seed standard error.

Both must hold. A D1 failure with a positive raw gap is the 026 result reproduced with enough
seeds to see it: the gap is architecture, not biology.

**FCL baselines, read from the file that wrote the figures**
(`experiments/020-cachera-betaxanthin/results/merzbacher_comparison_figures.json`), not from
prose: Spearman +0.0391, high-producer AUC 0.5696, precision@10/25/50 = 8/19/20.

**What 027 does not re-derive.** The 3-class head-to-head against Merzbacher's released
labels stays in 020's `plot_merzbacher_comparison.py` / `evaluate_merzbacher_head_to_head.py`,
which own that label provenance. 027 writes per-gene test predictions in a shape those scripts
read (`$DATA_ROOT/test-predictions-027/<arm>_s<seed>.json`).

**Two limits to state in any writeup, because a reviewer will raise both.** First, the
Merzbacher labels agree with our copy of the screen on only 81.2% of genes, so ~19% of any
model's charged error is label disagreement -- that is the ceiling every number sits under.
Second, yeast-GEM covers 915 of the screen's 4,721 deletions and misses 73% of its high
producers, so a flux module can only reach 19% of the panel; the stratified endpoint is what
makes that visible rather than averaged away.

## Node-hour estimate

**Measured inputs, three of them, none guessed:**

| source | s/epoch | conditions |
| --- | --: | --- |
| 026, median over 98 completed cells | 51.4 | GilaHyper RTX 6000 Ada, `num_workers=3`, two heads; min 50.9, max 51.8, flat across arms and across hidden 32 vs 64 |
| 027 smoke, `pooled` | **50.6** | this machine, one betaxanthin head, 6 epochs |
| 027 smoke, `flux_anchored` | **70.0** | same, sharing the host with two other runs |
| 027 grid-runner smoke, `flux_anchored` | **70.3** | same cell through `grid_runner.py`, 8 epochs, host otherwise idle |

The 027 rows are `wall_time_s / len(history)` in `results/smoke_pooled.json`,
`results/smoke_flux_anchored.json` and `results/smoke_grid_runner_w0.json`.

026 reported its per-epoch cost as flat across arms (51.08 s `pooled`, 51.38 s
`flux_anchored`), but it trained two heads and used 3 dataloader workers. On the 027
single-head path a flux arm costs **1.39x** a pooled one, and the third row settles that this
is the layer rather than host contention: it reproduces 70.0 s on an otherwise idle machine.
The budget below therefore charges the four flux arms at 70 s and the two pooled arms at 51.

| quantity | value |
| --- | --- |
| cells | 6 arms x 24 seeds = **144** |
| epochs per cell | 200 |
| solo GPU-hours, 4 flux arms | 96 x 200 x 70.0 s = **373** |
| solo GPU-hours, 2 pooled arms | 48 x 200 x 50.6 s = **135** |
| total solo | **508 GPU-hours** |
| packing (2 runs/GPU, 019-measured 1.39x aggregate) | **366 packed-GPU-hours** |
| GPUs | 9 nodes x 4 = **36** |
| **binding constraint: a flux worker's 2 cells** | 2 x 200 x 70.0 s x 1.44 = **11.2 h** |
| Delta band (A40 + `num_workers=0`, 1.0-1.5x) | **11.2-16.8 h** |
| requested | `--time=20:00:00` x 9 nodes = **180 node-hours** |
| expected consumption | **101-151 node-hours** |

The 1.0-1.5x Delta band is the only extrapolation left, and `delta_bxfx_canary.slurm` exists
to replace it with a measurement before the grid is submitted. It prints steady-state s/epoch
and the arithmetic that turns it into `--time`.

**If the canary comes back slower than 1.5x, halve the seeds rather than the epochs.**
`GRID_ROUNDS=12` keeps the shard math valid (12 groups, 1 cell per worker), finishes in
~5.6 h, and still yields a balanced 12 seeds at a 2-sigma MDE of 0.045. Cutting the epoch
budget instead would change the experiment, because the selection search widens with it.
The deadline guard already delivers this outcome automatically if the wall clock runs out
mid-grid.

Account `bfjt-delta-gpu` holds 12,797 hours (checked 2026-09-03). **`bbtp-delta-gpu`, which
every older Delta launcher in this repo names, no longer exists on this allocation** -- it is
absent from `accounts` output. Both slurm files carry `--account=bfjt-delta-gpu` in the
header rather than leaving it to the submit line.

`gpuA40x4` has 98 nodes and a 48 h limit, so the 20 h request is legal and 9 nodes is a
modest fraction of the partition. **The expression sweep `019-expr-v10` (array 21796128) is
PENDING on the same partition with a 48 h limit**, which is why 027's job name, slurm output
paths, W&B project and optuna directory are all distinct from it.

## Sharding: 72 workers, no shared SQLite

Optuna's WAITING-trial pop races on SQLite across processes. Measured on IGB job 2332400: six
workers claimed three distinct cells. The same race fired on the Delta grids --
`s03_L2_maskon_lr0.001_energy` ran four times, `bx_ctrl s01` twice -- while every log looked
healthy.

`experiments/019-simb-multimodal/scripts/delta_grid_common.sh` **still has this bug**: it sets
`OPTUNA_WORKER_ID` but points every worker at one storage file and never exports
`GRID_SHARD_COUNT`. Only the IGB launcher was fixed. That is the main reason 027 carries its
own `delta_bxfx_common.sh` rather than sourcing the 019 include; the second reason is that
the 019 include drives `metabolism_grid_runner.py` -> `train_cgt_multitask.py`, which never
passes `flux_layer=` and so cannot build the module at all.

- `GRID_SHARD_COUNT=72` = 9 nodes x 4 GPUs x 2 workers/GPU; each worker gets its own
  `optuna/optuna_bxfx_w<id>.db`.
- Shard **by whole seed**, copied from `metabolism_grid_runner._owns_cell`: with A arms and W
  workers there are G = W // A groups; group `WORKER_ID // A` takes every G-th round and runs
  arm `WORKER_ID % A`. Here A = 6, W = 72, G = 12, so 12 complete seeds are in flight and
  finish together. `GRID_SHARD_COUNT` must be a multiple of the arm count and exits at import
  if it is not.
- A flat `cell_index % W` stride would also cover every cell once but would give each worker
  one arm across scattered seeds, so a kill would leave controls at seeds the treatments never
  reached -- paired differences with no partner.
- `study.optimize` is called with **no `catch=`**. `catch=(Exception,)` turns a crash into a
  FAILED trial and the worker marches on, exiting `COMPLETED 0:0` with nothing to show. Each
  worker owns its own study, so a loud death costs 2 cells, not the grid.

## Prerequisites -- all four, in order

Delta charges service units, so a job that dies on a missing prerequisite costs real
allocation. Verified read-only on Delta 2026-09-03.

1. **CODE.** `/projects/bbub/mjvolk3/torchcell` is on branch
   `multimodal-phenotype-retrospective` at `c69601ed`, and its `torchcell/metabolism/` holds
   only `enzyme_kinetics.py` and `yeast_GEM.py`. **`flux_layer.py` is not on Delta.** Push the
   branch carrying 027 (which must itself be rebased onto main -- see below) and check it out
   there. `bxfx_preflight` refuses to start without `flux_layer.py`.
2. **THE OPEN ENZYME DATABASE MIRROR, and this one is silent if you skip it.**
   `$DATA_ROOT/data/enzyme_kinetics/open_enzyme_database/scerevisiae` (512 KB, 2 files) is
   **not on Delta**. `resolve_kcat_table` catches `FileNotFoundError` and returns an empty
   record list, so every kcat becomes the organism default: the enzyme-constrained arm runs
   with no enzyme constraints, completes normally, and reports a plausible number. Both the
   shell preflight and a `train_bx.py` assertion refuse a flux arm whose experimental kcat
   fraction falls below 0.02 (GilaHyper resolves 0.0397).

   ```bash
   rsync -av /scratch/projects/torchcell-scratch/data/enzyme_kinetics/open_enzyme_database/scerevisiae/ \
     mjvolk3@login.delta.ncsa.illinois.edu:/scratch/bbub/mjvolk3/torchcell/data/enzyme_kinetics/open_enzyme_database/scerevisiae/
   ```

3. **DATA.** `$DATA_ROOT/data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer`
   and `data/torchcell/yeast-GEM` are both present on Delta. Nothing is rebuilt: the dataset
   is opened, never processed, and a compute node has no Neo4j.
4. **DIRECTORIES.** `mkdir -p experiments/027-betaxanthin-metabolic/{slurm/output,optuna,results}`
   on Delta. `--output` points into `slurm/output/`, which is gitignored; slurm will not
   create it and the job dies at t=0 with no log anywhere.

### Branch hazard

027 was authored in the worktree `feat/kinetics-equilibrator-datasets`, whose branch
**predates the 026 landing**: it has neither `torchcell/metabolism/flux_layer.py` nor
`experiments/026-metabolism-flux/scripts/train_flux.py`. **This branch cannot run the
experiment until it is rebased onto main.** For local smoke testing, `TC_FLUX_SCRIPTS_DIR`
points `train_bx.py` at the primary checkout's 026 scripts; on Delta and after the rebase it
is unset and the default path is correct.

## Files

| file | what it is |
| --- | --- |
| `conf/base.yaml` | everything the arms hold constant: scoring rule, FCL baselines, split pinning, hyperparameters, the kcat coverage floor |
| `conf/arm_*.yaml` | one per arm; `base_arm` + `permute_train_targets` and nothing else |
| `scripts/train_bx.py` | one cell: betaxanthin-only head, pinned test split, pinned partition, ring-buffer epoch selection, test pass + per-gene dump |
| `scripts/grid_runner.py` | sharded optuna work ledger; owns `_owns_cell` and the deadline guard |
| `scripts/delta_bxfx_common.sh` | Delta environment + node launch body, with the sharding fix the 019 include lacks |
| `scripts/delta_bxfx_canary.slurm` | 1 GPU, 1 h -- **run first**; measures s/epoch on Delta and exercises every path |
| `scripts/delta_bxfx_grid.slurm` | the decision run: array 0-8, 9 nodes, 72 workers, 144 cells |
| `scripts/analyze_bxfx.py` | the prespecified analysis and the D1/D2 decision |

## Launch (NOT RUN -- for the author, after the prerequisites)

```bash
# 0. FROM GILAHYPER -- the 512 KB kcat mirror Delta does not have. Skipping this does not
#    fail the job, it silently makes every flux arm unconstrained (the preflight catches it,
#    but only after the job has already started).
rsync -av /scratch/projects/torchcell-scratch/data/enzyme_kinetics/open_enzyme_database/scerevisiae/ \
  mjvolk3@login.delta.ncsa.illinois.edu:/scratch/bbub/mjvolk3/torchcell/data/enzyme_kinetics/open_enzyme_database/scerevisiae/

# 1. ON DELTA -- code and directories.
ssh mjvolk3@login.delta.ncsa.illinois.edu
cd /projects/bbub/mjvolk3/torchcell
git fetch origin && git checkout <branch-with-027-rebased-on-main>
mkdir -p experiments/027-betaxanthin-metabolic/{slurm/output,optuna,results}

# 2. CANARY FIRST -- 1 GPU, under 1 h. Read the s/epoch it prints.
sbatch experiments/027-betaxanthin-metabolic/scripts/delta_bxfx_canary.slurm

# 3. THE DECISION RUN -- 9 nodes, array 0-8. Adjust --time from the canary first.
sbatch experiments/027-betaxanthin-metabolic/scripts/delta_bxfx_grid.slurm
```

Then, once the array finishes:

```bash
/work/hdd/bbub/miniconda3/envs/torchcell/bin/python \
  experiments/027-betaxanthin-metabolic/scripts/analyze_bxfx.py
```

## 2026.09.04 -- Design review before launch: the backbone is wrong

**Do not launch the six arms above as they stand.** Reviewing every prior betaxanthin run
turned up a defect that no amount of replication fixes, and it is upstream of everything
026 and 027 were built to correct.

### The finding

Betaxanthin runs in this repo fall into two architecture families, and the family explains
almost all of the score variance.

| family | configuration | measured val Pearson |
| --- | --- | --- |
| weak | 2 graphs, learnable embeddings, MSE, hidden 32, 4 heads | 0.04 to 0.16 |
| strong | 9 graphs, `prot_T5_all`, no learnable embedding, CRPS, hidden 90, 9 heads | 0.32 to 0.43 |

The best betaxanthin result ever measured is **0.43399** (GilaHyper job 1344, config
`gh_metabolism_002`, seed 42, peak at epoch 39 of 80, `val/betaxanthin/pearson_per_feature`
as a max over epochs, n_val = 340). Independently, Optuna study `betaxanthin_002` trial 15
reached 0.4301 with the same family.

`experiments/026-metabolism-flux/scripts/train_flux.py` builds the **weak** family, and
`027/conf/base.yaml` inherits its hyperparameters by design ("the 026 incumbent"), while
`027/scripts/train_bx.py` sets `learnable_embedding_config={"enabled": True}`,
`num_attention_heads=4` and `nn.MSELoss`. So 026 compared a flux module against a CGT
running roughly 5x below its own demonstrated ceiling, and 027 would repeat that comparison
with better statistics.

That reframes 026's negative result. Its `flux_off` arm at 0.0837 is the correct control
*for that sweep*, and it is not a CGT baseline. Nothing in 026 tested whether the module
helps a CGT that works.

### Consequences for the design

1. **The backbone becomes `gh_metabolism_002`, not the 026 incumbent.** 9 graphs,
   `prot_T5_all`, `learnable_embedding: false`, CRPS, hidden 90, 9 heads, 2 transformer
   layers, `target_norm: yeo_johnson`, `graph_reg_lambda` 6.5e-5. Held identical across
   every arm.
2. **`min_kcat_experimental_fraction` is now the wrong gate.** It asserts 2 percent
   *experimental* kcat coverage against the Open Enzyme Database. The predicted tables cover
   95.3 percent of catalytic units, so the gate should assert on the resolved table the arm
   actually loads, and record which predictor produced it.
3. **Predictor choice is a hyperparameter with a measured effect size.** The five kcat
   predictors move a single gene's value by a median of 1.23 decades, which is larger than
   the 10th-to-90th spread of kcat across genes (0.94 decades). Fixing one predictor
   silently is a choice, so it is swept.

### Revised arms

Seven, on one backbone. Each changes exactly one thing against `cgt`.

| arm | module | parameters supplied | labels | isolates |
| --- | --- | --- | --- | --- |
| `cgt` | off | -- | real | the real baseline, target ~0.43 |
| `flux_struct` | on | stoichiometry + GPR only | real | does routing through S help at all? |
| `flux_kcat` | on | + kcat capacity bound | real | does enzyme capacity help? |
| `flux_full` | on | + delta_r G' second law, dissipation | real | **the treatment** |
| `flux_shuffled` | on | kcat **permuted across reactions** | real | **is it chemistry or capacity?** |
| `null_cgt` | off | -- | permuted | null for `cgt` |
| `null_flux_full` | on | full | permuted | null for `flux_full` |

`flux_shuffled` is the control 026 lacked. It holds parameter count, architecture and
optimization fixed and destroys only the correspondence between a reaction and its
turnover number. If `flux_full` and `flux_shuffled` land together, the module is acting as
a capacity regularizer and the kinetic tables are decoration.

### Two stages

**Stage A, screen (18 runs).** `flux_full` only, kcat table crossed over
{consensus median, DLKcat, UniKP, EITLEM, TurNuP, DeepEnzyme} x 3 seeds. Either one table
wins, or predictor choice does not move the endpoint, and that null is itself worth
reporting given the 1.23-decade spread.

**Stage B, confirm (70 runs).** 7 arms x 10 seeds with the Stage A table. Ten seeds is set
by the measured noise: across-seed sd is 0.030 to 0.036 on this endpoint, so sigma_eff
reaches 0.010 only near R = 10.

**Decision rule, fixed before launch.** Primary endpoint is Spearman on the pinned
Merzbacher 639-gene test split at the validation-selected epoch, reported as a mean over
seeds with its sd and n. The claim "the module helps" requires
`(flux_full - cgt) - (null_flux_full - null_cgt)` to exceed zero by more than 2 standard
errors of the paired difference, **and** `flux_full` to exceed `flux_shuffled` on the same
rule. Either one alone is insufficient.

### How the three parameter tables actually enter, and where they do not

- **kcat** enters as the capacity bound `v_j <= kcat_j * E_j`. Covers 95.3 percent of
  yeast-GEM catalytic units across five independent tables.
- **delta_f G'** enters as the second-law penalty and the dissipation bound. Covers 42.5
  percent of reactions with a real standard error, median 2.30 kJ/mol. The uniform-pH build
  is the one to use, which sets transport thermodynamics to zero by construction.
- **K_M** is built for three predictors and **is not wired into the flux layer**. The
  saturation term is drawn as inactive in the module diagram and it remains inactive. K_M
  contributes nothing to this experiment and must not be described as if it did.

**The pathway is the least parameterized part of the model.** `CYP76AD1` and `DOD` are
heterologous, have no reaction in yeast-GEM, and therefore have no predicted kcat from any
of the five tables. Three of the five betaxanthin intermediates cannot be assigned a
formation energy without a ChemAxon license. So the module cannot currently constrain the
heterologous steps at all.

What it can constrain is **native precursor supply**: the shikimate and aromatic-amino-acid
route to tyrosine, whose genes (`ARO1`, `ARO2`, `ARO3`, `ARO4`, `ARO7`, `TYR1`, `ARO8`,
`ARO9`, `PHA2`) are all parameterized. That is the mechanism under test, and stating it
narrowly is what makes the experiment falsifiable: the hypothesis is that constraining
native precursor flux improves prediction of a heterologous product's titer, not that the
model simulates betaxanthin synthesis.

## 2026.09.04 -- Launch record

Submitted to Delta on `bbub-delta-gpu` (3,851 h available; `bbtp` no longer exists on this
allocation and `bfjt` is wanted elsewhere).

### What changed before launch

The backbone, per the design review above. `conf/base.yaml` now carries the 019
`gh_metabolism_000` lineage rather than the 026 incumbent, and three couplings had to move
with it, each found by a local smoke run rather than by reading:

1. **The perturbation head defaults to 8 heads and 90 is not divisible by 8.** Its
   cross-attention refuses to build. 019 uses 6, which divides 90 into 15.
2. **The model reads `cell_graph["gene"].x` only when it is told it has precomputed
   embeddings.** With `learnable_embedding` off and `node_embeddings` unset it finds neither
   and raises. It also indexes the embedding dataset to infer its width, so it needs the
   objects themselves, not their names.
3. **Dropping 026's `build_dataset` dropped its module globals with it.** `DATASET_ROOT` and
   `QUERY_PATH` are defined here now.

Local smoke, `pooled` seed 0, two epochs on one GPU: loss 1.0586 to 1.0088, val Pearson
-0.0214 to +0.1055, pinned test Spearman +0.0261 on n = 629. Two epochs is not a result;
what it establishes is that the nine-graph prot_T5 backbone builds, trains, and scores
against the pinned Merzbacher split end to end.

### Environment facts confirmed on Delta, not assumed

| fact | value |
| --- | --- |
| repo | `/projects/bbub/mjvolk3/torchcell`, branch `feat/kinetics-equilibrator-datasets` |
| `DATA_ROOT` | `/scratch/bbub/mjvolk3/torchcell` (the Delta `.env` still says `/work/hdd`, which is wrong) |
| interpreter | `/work/hdd/bbub/miniconda3/envs/torchcell/bin/python` |
| GPU | NVIDIA A40, 46,068 MiB, 4 per node |
| `NUM_WORKERS` | 0, forced (spawn re-imports off Lustre and stalls in sanity check) |
| W&B | online, authenticated via `~/.netrc`, project `torchcell_027_bxfx` |

The Open Enzyme Database mirror was **absent on Delta** and was rsynced (516 KB). Without it
`resolve_kcat_table` catches the missing file, returns an empty record list, and every
measured kcat is silently replaced by the organism default, so the enzyme-constrained arm
would have run with no enzyme constraints and reported a plausible number. The
`min_kcat_experimental_fraction` gate exists to abort that, and it can only fire if the
mirror is there to be measured.

### Shape of the grid

Nine nodes, 4 GPUs each, **2 runs per GPU** = 72 workers. Six arms divide 72 into 12
seed-groups, so every arm gets 12 seeds. The shard rule assigns by whole seed and every
worker owns its own Optuna database, which is what avoids the SQLite race that had multiple
workers claiming one trial.

### Known limitation carried into this run

The predicted kcat and K_M tables are built but **not wired into the flux layer**, which
still resolves kcat from the Open Enzyme Database (4.0 percent experimental, the rest
default). K_M is not consumed at all. So this run tests the flux module's structure and
thermodynamics, not the predicted kinetics. Wiring the predictor registry in, and adding the
`flux_shuffled` control, are the next round.
