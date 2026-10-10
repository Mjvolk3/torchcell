---
id: o87q6yna943u6gwo7qf7xe9
title: Train_mixture
desc: ''
updated: 1791620016894
created: 1791620016894
---

## 2026.10.10 - The dose- and mixture-aware trainer: model, data, losses, evaluation, round 1

Step 3 of the 040 plan, the model the in-house inhibitor data will be scored against.
Nothing has been fitted yet: every number in this section is from the CPU smoke, which
runs two steps and is noise by construction. The bars to beat are the model-free ones of
claim 1 (Loewe growth AUROC 0.958 served, Bliss-from-ex23-singles fitness Spearman 0.573)
and, on the gene-level side, nested ridge on FCFP4 counts (compound-cold median centered
Spearman 0.359 on the corrected store, 038 round 1, slurm 3374).

### Files

- `experiments/040-inhibitor-synergy-wetlab/scripts/mixture_data.py` -- data assembly.
- `experiments/040-inhibitor-synergy-wetlab/scripts/train_mixture.py` -- model, training,
  evaluation, sweep CLI.
- `experiments/040-inhibitor-synergy-wetlab/conf/mixture/cgt_model.yaml` -- the cell graph
  transformer and the nine gene graphs, copied from 038's `conf/default.yaml` with the
  OmegaConf interpolations written out. Not a sweep file.
- `experiments/040-inhibitor-synergy-wetlab/conf/mixture/smoke.yaml` and ten
  `conf/mixture/r1_mix_*.yaml` round-1 arms.
- `experiments/040-inhibitor-synergy-wetlab/scripts/gh_train_mixture.slurm`,
  `scripts/delta_train_mixture.slurm`.
- `experiments/040-inhibitor-synergy-wetlab/results/vanacloig_doses.csv`.

Experiment 038 is a read-only reference. `TokenLayer`, `StrainBatch`, `strain_batch`,
`build_encoder`, the OneCycle / autocast loop, the sweep CLI, and on the data side
`VanacloigCells`, `Fold`, `load_cells`, `make_folds`, `subsample_pool`, `ceiling` and
`score_compounds` are restated from its `scripts/train_factorized.py` and
`scripts/vanacloig_data.py` (commit 9427c0d4e) with an attribution comment, rather than
imported across worktrees, so the 040 tree rsyncs to Delta on its own.

### The model

The 038 round-10 environment encoder generalized in four switchable pieces. The encoder
runs once on the wildtype cell graph, giving gene tokens `H` [N, d] with N = 6,607 and
d = 180; the strain's post-deletion rows `h_del` come from the same forward's deletion
operator. A CONDITION is a medium: a set of compounds with their doses.

| switch | on | off |
|---|---|---|
| `compound_set` | `tokens`: one token per inhibitor, all placed in front of the gene tokens | `mean`: the tokens are averaged into one, which is 038's single-token model |
| `dose` | `film`: an MLP of the standardized log10 molar dose produces a scale and a shift on each token; its last layer starts at zero, so the model starts as the dose-free token | `off`: 038's dose-free token |
| `host_head` | always built; `lambda_host` weights its loss | `lambda_host` 0 |
| `sources` | `hoepfner_hop` and `hillenmeyer_het` behind the Vanacloig task | `vanacloig` alone |

The condition's tokens are concatenated in front of `H` and run through `env_layers`
`TokenLayer`s, giving the cell in that medium `H^c`. The GENE readout is 038's, with one
block added: the rows of `H^c` at the strain's deleted genes summed, `h_del`, the mean of
`H^c` over genes, the mean of the compound tokens after they read the genome, and a
learned source token, through a LayerNorm and an MLP, beside a per-source gene bias and a
compound offset. The HOST readout is MLP([mean of `H^c` over genes ; mean of the compound
tokens]) through a sigmoid: it reads no genotype, which is what the bAID host is, so it
needs no empty-genotype path through the encoder.

Conditions are grouped by their number of compounds and one group runs per pass, so no
attention mask is needed and the SDPA flash kernel stays available on a GPU. A pass
reduces `H^c` to its three summaries inside the chunk loop, as 038 does, so the
[conditions, N, d] state is never held for the whole condition set.

### Data

Counts measured on build 002 of the 033 cell table, the served growth call, and the
rebuilt Vanacloig dev store (`results/mixture/<sweep>/<name>_counts.csv` records them per
run):

| source | cells | genes | compounds | conditions | with molar dose |
|---|---|---|---|---|---|
| vanacloig | 97,772 | 3,587 | 28 | 28 | 28 |
| hoepfner_hop | 1,234,523 | 4,471 | 145 | 275 | 275 |
| hillenmeyer_het | 2,345,856 | 5,813 | 287 | 401 | 382 |
| host anchors | -- | -- | 102 | 103 | 103 |
| host ex21 | 180 wells | -- | 6 | 55 | 55 |
| host ex23 | 197 wells | -- | 6 | 64 | 64 |
| host isoboles | 600 wells | -- | 4 | 300 | 300 |

Table 1. Gene-level sources and host records as `mixture_data.assemble` loads them.
Vanacloig is 20,890 cells short of the served 118,662 because DMSO and MBO are dropped
(`EXCLUDED_CONDITIONS`, as 038 drops them) and the four percent-dose compounds leave the
panel under `require_molar_dose`. Hillenmeyer loses 228,279 cells to its two-compound
environments and its rows with no InChIKey. The host conditions count the compound-free
medium of each run, so ex21's 55 are 54 single-agent cells plus one control and ex23's 64
are the 63 combinations plus one.

ORIENTATION IS PER SOURCE AND HAPPENS BEFORE ANYTHING IS POOLED (the 031 rule). Vanacloig
(log2 inhibitor over control) and Hoepfner (adjusted MADL sensitivity, "negative =
hypersensitive, positive = resistant" in `hoepfner2014.py`) are already sick-negative and
are kept as served; Hillenmeyer HET is `log2(mean control intensity / treatment
intensity)`, a fitness defect, so it is NEGATED. Each source is then standardized on its
own fitted conditions, and the compound features and the dose scale are standardized on
the fitted conditions of every source plus the host training records.

DOSES. Vanacloig's served `conc_values` are empty, so the dose is read from the interned
environments of the rebuilt dev store (`results/vanacloig_doses.csv`, one record index
and record count per condition). Of the 32 published compounds 28 convert to molar: mM and
uM directly, and 2,2'-bipyridine's 18 ug/mL through the RDKit molecular weight of its
curated-identity SMILES (0.115 mM). Four do not, because a percent states no basis or
density: ethanol 4 percent, gamma-valerolactone 1.5 percent, isobutanol 0.75 percent and
methyl methanesulfonate 0.01 percent. Those four leave the scored panel whenever
`require_molar_dose` is true, which every round-1 arm sets, so the dose ablation compares
like for like; the consequence is that the panel is 28 and the ridge bar of 0.359 was a
median over 32, so that comparison is approximate and every per-compound score is written
to `<name>_vanacloig_compounds.csv` beside it.

FINGERPRINTS are FCFP4 counts from the 031 table, keyed by InChIKey because Hoepfner
serves two names ("Cleisthantin derivative", "Valinomycin Derivative") against two
structures each. All 433 distinct gene-level InChIKeys have a row
there. Only TWO of the six wet-lab inhibitors need featurizing from SMILES, not three:
formic acid (BDAGIHXWWSANSR) and lactic acid (JVTAAEKCZFNVCJ) have no row, and acetic acid
DOES (QTBSBXVTEAMEQO-UHFFFAOYSA-N) and is taken from the table. The two are featurized
with the recipe `inhibitor_profiles.py` verified reproduces every npz row exactly.

ANCHORS AND LEAKAGE. An anchor names a compound and its IC30 dose, so an anchor for a
held-out compound would show the compound encoder a test compound's structure. Every host
record carrying one of the fold's test compounds is therefore dropped from the host
training set, and every auxiliary condition whose compound is in the Vanacloig panel is
dropped from the gene task (as 038's `train_multisource` removes them). Hoepfner compounds
served at several concentrations have no unambiguous IC30 among them, so only its 74
single-dose compounds are anchored.

### Losses and the run order

A step takes one gene batch from every active source against that source's fitted
conditions (all 28 for Vanacloig, a random `cond_batch` of 24 for an auxiliary source) and
one random `host_batch` of 16 host records. The loss is the Vanacloig masked MSE on
standardized values, plus `aux_weight` times each auxiliary source's, plus `lambda_host`
times the host MSE, plus `cgt_lambda` times the graph penalty (0 in every round-1 arm,
following 038 round 10). The host pass reuses the Vanacloig pass's wildtype `H`, which
does not depend on the strain, so the host task costs no extra encoder forward. An epoch
is one pass over the Vanacloig genes; the auxiliary sources are sampled, not exhausted.

`host_train` is the wet-lab question, and it has three values:

- `anchors`: public anchors only, so ex21 is held out and its score is a prediction.
- `anchors+ex21`: the anchors and the 54 ex21 cells jointly, throughout.
- `finetune_ex21`: stage 1 is `anchors`; stage 2 then trains `finetune_epochs` (5) epochs
  on the ex21 cells alone at `finetune_lr` (1e-4). `finetune_scope` `host_head` unfreezes
  the host readout only; `host_head+env_encoder` also unfreezes the compound MLP, the FiLM
  MLP, the compound offset and the medium transformer layers. The gene encoder is frozen in
  both, so the wildtype gene tokens are computed once in eval mode and reused.

ex23 and the isobole grids are never trained on under any value.

### Evaluation

`results/mixture/<sweep>/<name>_scores.csv` is long form (task, subset, metric, value, n,
reference_rule, reference_value, beats_reference, fold, member, name, fold_seed, call), one
block per fold and seed plus a seed-ensemble block, and the run prints the fold mean of the
ensemble beside the reference. Predictions go to
`$DATA_ROOT/experiments/040-inhibitor-synergy-wetlab/predictions/<sweep>/` as the gene
matrix and the host vector per fold.

| task | subset | metric | reference |
|---|---|---|---|
| vanacloig | test_compounds | centered and raw Spearman median | nested ridge FCFP4, 0.359 |
| ex21 | all_wells, and one row per compound | fitness Spearman over wells, named `_train_fit` when ex21 was trained on | -- |
| ex23 | combinations | growth AUROC over the 63 | `loewe_ex21` 0.958 |
| ex23 | combinations_that_grew | fitness Spearman over the 21 | `bliss_ex23` 0.573 |
| isobole_ex26/27/28 | interior_cells | mean of observed minus predicted over 81 cells | `bliss_ex21` -0.082, -0.446, -0.017 |

Table 2. The evaluation rows and the model-free bar each is printed against. Observed host
fitness reads a no-growth well as zero and a cell as the mean over its replicates, which is
`mixture_rules.py`'s rule, so the model and the reference rules are scored on the same
numbers; `fitness_grown` (the mean over the replicates that grew) is what the ex23 fitness
Spearman uses, again as `mixture_rules.py` does. A per-compound ex21 Spearman is NaN
whenever the prediction is constant across that compound's nine doses, which is the
structural outcome for a `dose: off` arm and is reported rather than papered over (038's
`CONSTANT_SD` rule).

### The round-1 sweep

Ten arms, one file each, five configs per file (the five compound-cold folds of fold seed
0), three seeds per config, 50 epochs keeping the last, fitted on the non-test pool,
`gene_batch` 128, `env_chunk` 16. Round 1 scores fold seed 0 only; the replication on fold
seeds 1 and 2 (the 123 evaluations 038 round 10 reports) is round 2 on the arms that
survive. `mix_van_ex21_film_L1_lh1` is the best-guess joint arm, and the fine-tune arms and
the capacity arms are that arm with one field changed, so the three `host_train` values are
a clean comparison on one backbone.

| file | sources | host_train | dose | env_layers | lambda_host | estimated wall |
|---|---|---|---|---|---|---|
| `r1_mix_van_anchors_off_L1_lh1.yaml` | van | anchors | off | 1 | 1 | 5 h |
| `r1_mix_van_anchors_film_L1_lh1.yaml` | van | anchors | film | 1 | 1 | 5 h |
| `r1_mix_van_ex21_film_L1_lh1.yaml` | van | anchors+ex21 | film | 1 | 1 | 5 h |
| `r1_mix_van_ex21_off_L1_lh1.yaml` | van | anchors+ex21 | off | 1 | 1 | 5 h |
| `r1_mix_van_ft_film_L1_lh1.yaml` | van | finetune_ex21, host_head | film | 1 | 1 | 5 h |
| `r1_mix_van_ftenv_film_L1_lh1.yaml` | van | finetune_ex21, host_head+env_encoder | film | 1 | 1 | 5 h |
| `r1_mix_van_ex21_film_L2_lh1.yaml` | van | anchors+ex21 | film | 2 | 1 | 5 h |
| `r1_mix_van_ex21_film_L1_lh0p1.yaml` | van | anchors+ex21 | film | 1 | 0.1 | 5 h |
| `r1_mix_vanhop_ex21_film_L1_lh1.yaml` | van + hop | anchors+ex21 | film | 1 | 1 | 7.5 h |
| `r1_mix_vanhophet_ex21_film_L1_lh1.yaml` | van + hop + het | anchors+ex21 | film | 1 | 1 | 10 h |

Table 3. The ten round-1 arms, each a 12 h job at PARALLEL=2. The wall times are
ESTIMATES, not measurements: 038's 15-config round of this recipe took 5 h 50 min at
PARALLEL=2 on a GilaHyper RTX 6000 Ada (slurm 3080), about 47 min of single-GPU time per
three-seed config; a step here touches about 1.5 times as many conditions with one gene
source, 2.3 with two and 3.1 with three, three configs land on each of two workers, and an
A40 is taken as 1.4 times slower than the Ada. The grid the plan named (three source sets
crossed with two host_train values and two dose values) is 12 arms before the capacity and
fine-tune arms; it is pruned to these ten by crossing dose with the Vanacloig backbone only
and running the source axis at the best-guess host and dose setting, because the dose switch
is a property of the readout and not of which screens are pooled (a choice, not a measured
result).

### The smoke

```bash
cd /home/michaelvolk/Documents/projects/torchcell.worktrees/exp/040-inhibitor-synergy-wetlab
DATA_ROOT=/scratch/projects/torchcell-scratch PYTHONPATH=$PWD WANDB_MODE=offline \
  timeout 1200 ~/miniconda3/envs/torchcell/bin/python \
  experiments/040-inhibitor-synergy-wetlab/scripts/train_mixture.py \
  --sweep experiments/040-inhibitor-synergy-wetlab/conf/mixture/smoke.yaml
```

Three configs, CPU, 7 to 9 s each, exit 0, writing a complete scores table for every arm of
the ladder: `smoke_tokens_film_anchors` (tokens, FiLM, anchors), `smoke_mean_doseoff_joint`
(single averaged token, dose off, joint ex21, the full 32-compound panel) and
`smoke_aux_finetune` (Hoepfner behind Vanacloig, two-stage fine-tune with the env encoder
unfrozen). The numbers are noise at two steps; what the smoke checks is that every
evaluation column is populated and that no switch crashes. Outputs under
`results/mixture/smoke/`.

### What is not done

- No GPU run: the GilaHyper cards were taken until about 07:00, so nothing has been fitted
  and no wall time is measured. The estimates in Table 3 are arithmetic on 038's slurm 3080.
- A step runs one encoder forward per gene source. Batching strains of different deletion
  orders into one forward is possible (the encoder takes a flat perturbation index with a
  genotype assignment, so the order may vary per genotype) and would make the three-source
  arm roughly as cheap as the one-source arm. Not done; the cost is carried instead.
- Hillenmeyer's media are pooled (YPD, YP, SD) behind one source token, and its
  heterozygotes are modeled as deletions by the encoder's deletion operator. Both are
  stated simplifications, not measured choices.

## 2026.10.10 - Both growth calls on the wet-lab metrics, and the worst-scenario ranking

Supersedes the evaluation schema of the section above (its Table 2 scored the wet-lab
metrics under the served call only). The training path and the model are unchanged.

**A prediction is call-independent**, since the model is handed a medium and never a
growth call, so the same prediction vector is scored once per call against that call's own
grown set, tau and fitness scale, and every row of
`results/mixture/<sweep>/<name>_scores.csv` now carries the `call` it belongs to.
`mixture_data.MixtureData` gained `host_by_call` (the same conditions keyed by record key,
one view per call) and `runs_by_call`; a scorer looks a record up in the view and indexes
the prediction by its position in the training-call list. The config's `call` field is now
documented as the call the HOST HEAD TRAINS on and no longer decides what is reported; the
scores table records it as `training_call`.

**Finding that constrains the change: the software call covers ex23 and nothing else.**
Read off `results/wetlab_wells.csv` by `mixture_data.runs_per_call`, which asserts a call
is present for all of a run's wells or none of them:

| run | wells | served `grew` | software `grew_software` |
|---|---|---|---|
| ex21 | 180 | 180 | 0 |
| ex23 | 197 | 197 | 197 |
| ex26 | 200 | 200 | 0 |
| ex27 | 200 | 200 | 0 |
| ex28 | 200 | 200 | 0 |

Table 4. Growth-call coverage per run. The Bioscreen software generation times were
produced for the ex23 plate set alone, so `fitness_software` and `grew_software` are empty
for every ex21 and isobole well, and `results/isobole_summary.csv` holds one set of rows,
computed on the served call. The isobole metrics are therefore emitted for the served call
only, and the ex21 and Vanacloig rows carry `call` = `both` rather than being duplicated.
This is not a choice between calls: scoring the grids under the software call would mean
reading an empty growth flag through `astype(bool)`, which is True, and silently calling
all 243 interior cells grown. `runs_per_call` is what stops that, and
`load_host_records` now drops the runs a call does not cover instead of reading them.

**New metric.** Beside the AUROC, `worst_scenario_spearman` is the Spearman of predicted
fitness against observed fitness over ALL 63 combinations with a combination that did not
grow scored at zero, which is `observed_zero_mean` of
`results/mixture_combinations.csv`. It reads the model as a ranking of which combination
scenarios are worst, for prioritizing detoxification targets, rather than as a growth
classifier plus a separate fitness regression on the survivors. Its reference is Loewe
from the ex21 Hill fits against the same observed values, computed here from that
committed artifact because `mixture_scores.csv` reports its Spearman over the grown subset
only: **0.772 served, 0.800 software** (Bliss from the ex23 singles reaches 0.643 and
0.684, so Loewe is the bar on this statistic as it is on growth).

**The per-call records reproduce the model-free artifact exactly.** Over the 63 ex23
combinations of each call, `observed_zero_mean`, `observed_grown_mean` and `grew` match
`results/mixture_combinations.csv` with zero mismatches: 21 grew served, 25 software. The
four combinations whose call differs are 5-HMF + lactic acid, 5-HMF + levulinic acid,
furfural + acetic acid + 5-HMF and furfural + 5-HMF + formic acid, three replicates each,
which are the 12 disputed wells, all of them 5-HMF combinations. So the model and the
reference rules are scored on the same numbers under both calls, and an arm whose apparent
success hinges on the call will show it as a gap between its two ex23 rows.

### The smoke, re-run

Same command as above, exit 0, 8 s per config on CPU. The ex23 and isobole block of
`smoke_tokens_film_anchors` (two steps, so the values are noise; what matters is that every
new row is populated and carries its call and its call's reference):

```
  task          subset                    metric                        call         value      n  reference
  ex23          combinations              growth_auroc                  served       0.454     63  loewe_ex21 0.958
  ex23          combinations              growth_auroc                  software     0.397     63  loewe_ex21 0.952
  ex23          combinations              worst_scenario_spearman       served      -0.049     63  loewe_ex21 (all 63, no growth 0) 0.772
  ex23          combinations              worst_scenario_spearman       software    -0.147     63  loewe_ex21 (all 63, no growth 0) 0.800
  ex23          combinations_that_grew    fitness_spearman              served       0.328     21  bliss_ex23 0.573
  ex23          combinations_that_grew    fitness_spearman              software     0.332     25  bliss_ex23 0.865
  isobole_ex26  interior_cells            mean_observed_minus_predicted served      -0.626     81  bliss_ex21 -0.082
  isobole_ex27  interior_cells            mean_observed_minus_predicted served      -0.451     81  bliss_ex21 -0.446
  isobole_ex28  interior_cells            mean_observed_minus_predicted served      -0.619     81  bliss_ex21 -0.017
```

The ex21 and Vanacloig rows are unchanged and read `call` = `both`. The printed summary and
the W&B summary keys now group by call as well, so a key is
`ex23/combinations/growth_auroc/served`.
