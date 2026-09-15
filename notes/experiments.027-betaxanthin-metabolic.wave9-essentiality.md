---
id: y0ujf0ebicfe59r1gt2qb0w
title: Wave9 Essentiality
desc: ''
updated: 1789443243887
created: 1789443243887
---

## 2026.09.14 - Gene essentiality as an endpoint: head built, dataset BLOCKED

Plan rank 11, moved forward because the amino-acid target cannot decide anything: its
ceiling is about 0.17 Pearson (022's Optuna sweep) and every arm measured to date sits at
0.05 to 0.07. Essentiality is where the published methods are far enough apart to separate.

### Correction: the 93.5 and 95 percent figures are E. coli, not yeast

The brief cited FBA at 93.5 percent and FCL at 95 percent. Those are the **E. coli**
accuracies, from the mirrored `paper.md` line 54: "FBA delivers a maximal accuracy of
$9 3 . 5 \%$ correctly predicted genes for E. coli growing aerobically in glucose with
biomass synthesis as optimization objective" and "achieving an average $9 5 \%$ accuracy
for all test genes across training repeats".

**Yeast is a much harder target and the gap is much narrower.** From the mirrored
`si/si1.md`, Supplementary Table S5 (table at line 33, caption at line 35: "Classification
performance metrics for Flux Cone Learning (FCL) and Flux Balance Analysis (FBA) in yeast
(S. cerevisiae) and Chinese Hamster Ovary (CHO) cells (Figure 3)"):

| model | accuracy | precision | recall | F1 | PR AUC | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| FBA | 0.81 | 0.91 | 0.97 | 0.89 | 0.92 | **0.69** |
| FCL (5-fold CV) | 0.90 +/- 0.008 | 0.91 +/- 0.008 | 0.98 +/- 0.010 | 0.94 +/- 0.007 | 0.94 +/- 0.021 | **0.78 +/- 0.065** |
| FCL (test) | 0.89 | 0.89 | 0.99 | 0.94 | 0.93 | **0.73** |

**Accuracy is the wrong headline here and AUROC is the right one.** The released test split
is 192 nonessential against 31 essential, a majority-class rate of **0.861**, and FBA's
published accuracy of 0.81 is BELOW it. On ROC AUC the separation is real but modest: 0.69
for FBA against 0.73 for FCL on the test split. These are recorded as data in
`FclEssentialityComparators` so a result file carries the comparison rather than a note
claiming it.

The protocol, `paper.md` line 122: "For essentiality prediction, a classstratified $20 \%$
of deletions (192 nonessential; 31 essential) was held out as a test set. The remaining
$80 \%$ of deletions (772 nonessential; 126 essential) were split into 5-fold
cross-validation sets ... The minimum deletion-level accuracy was $8 7 . 5 \%$, the maximum
was $9 0 . 3 \%$. ... The FBA baseline was computed using the single_gene_deletion function
in Cobrapy for all genes with glucose as the carbon source and the standard biomass
reaction." So the FBA baseline is **glucose minimal, aerobic, standard biomass objective**.

### The released split carries an INVERTED label, and nothing in it says so

Counted on `$DATA_ROOT/data/merzbacher2025_fcl/deletionprediction-main/data/yeast_essentiality_test_split.csv`:
1,159 rows, of which `test=0` holds 772 at `essential=1.0` and 126 at `0.0`, and `test=1`
holds 192 at `1.0` and 31 at `0.0`. The paper's Methods describe exactly those splits with
the labels the other way round, and Supplementary Table S4 agrees at the total (N=964
nonessential, N=157 essential). **The column named `essential` is a viability flag: 1.0
marks a NONESSENTIAL gene.** Reading it at face value inverts every label, which a
classifier fits happily and reports as a plausible AUROC. 38 rows carry an empty label and
are dropped.

`read_fcl_essentiality_split` applies the inversion once and asserts the four class counts
against the paper's own sentence, so a re-released file with different contents is refused
rather than quietly changing the comparison.

### Dataset route: BLOCKED, and a query is needed

| build | holds `GeneEssentialitySgdDataset` | `phenotype_label_index` keys |
|---|---|---|
| `019-simb-multimodal/fig6_build` | no | `visual_score`, `metabolite_level` |
| `003-fit-int/001-small-build` | yes, 1,140 records | `fitness`, `gene_interaction` |
| `025-solid-growth/001-full-build` | yes, 1,140 records | `fitness`, `gene_interaction` |

So the label is reachable in two builds but **not as `is_essential`**: both ran a converter
that turned the boolean into a fitness value, and an essentiality-only record in the 003
build reads `label_name: fitness` with value 0.0. No built CellDataset on disk carries the
essentiality label itself, so per the brief this stops at the head plus a unit test.

**The query that is needed**, patterned on `fig6_pigment_transfer.cql`, is a single block
with no converter:

```cypher
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'GeneEssentialitySgdDataset'
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (g)<-[:PerturbationMemberOf]-(p:Perturbation)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
MATCH (e)<-[:PhenotypeMemberOf]-(phen:PhenotypicFeature)
MATCH (e)<-[:EnvironmentMemberOf]-(env:Environment)
WHERE phen.graph_level = 'node'
 AND ALL(dp IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
 WHERE (NOT dp.perturbation_type ENDS WITH 'deletion')
 OR dp.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert)
 WHERE pert.perturbation_type ENDS WITH 'deletion' | pert]) = 1
WITH DISTINCT e, ref ORDER BY e.id RETURN e, ref
```

Three requirements on the build, each of which the current 003/025 builds violate:

1. **`converter=None`.** The converter is what replaced `is_essential` with `fitness`.
2. **Single-deletion genotypes only** (`SIZE(...) = 1`), because the FCL split is keyed on
   one knockout per row.
3. **No co-grouped fitness dataset**, or the deduplicator merges the boolean into a fitness
   record again. A dedicated root, not a union build.

Until that dataset exists the four arm cells are not writable, so none were written: a cell
naming a root that does not exist cannot be verified and would fail after a queue wait.

### What was built, and how it is scored

`--head essentiality`, label `is_essential`, width 1, and the parts that differ from every
continuous head:

* **No standardization.** A 0/1 label is handed mean 0 and sd 1 and the train-split pass is
  skipped. Subtracting a mean from a class label and dividing by its sd leaves the log-loss
  scoring a rescaled variable that is not a probability's target.
* **Loss is `binary_cross_entropy_with_logits`**, and `--crps` is REFUSED on this head:
  CRPS is a proper scoring rule for a continuous target, and a Gaussian over a 0/1 label is
  not the right object. The head emits one logit, `param_dim = 1`.
* **Selection is validation AUROC under the SAME centered 5-epoch moving average**, which is
  the natural analog of the Pearson rule and is the statistic the comparator table reports.
  A validation split carrying one class returns `nan`, which `_select_epoch` already treats
  as unselectable.
* **Readouts** are deletion-level: AUROC, accuracy, and precision and recall PER CLASS at
  the 0.5 probability threshold, reported beside the majority-class rate on the same genes,
  because that is the number FBA's 0.81 accuracy fails to beat.
* **The flux arm reads the biomass column.** `FluxBiomassReadout` takes `v[:, biomass]` and
  maps it to a logit with a scale and an intercept. Verified on yeast-GEM 9.0.2 today: the
  objective is `r_2111` ("growth"), reaction index 1499, and `build_gem_tensors` resolves
  `gem.biomass_index` to exactly it; `r_4041` is the biomass PSEUDOREACTION and is not the
  objective. This is the physically meaningful readout, and it is the same quantity FBA
  thresholds in its own baseline.

**The medium, and it needs a decision before the cells run.** FBA's published baseline is
glucose minimal aerobic. The SGD essentiality phenotype is scored on rich medium (YPD) in
the deletion-collection literature, not on glucose minimal. Those are different experiments,
and an arm run on the shipped yeast-GEM bounds is running the FBA baseline's medium rather
than the phenotype's. The `--apply-media` flag from wave 9 can set either. Whichever is
chosen must be stated in the cell and in the result, and the honest default for a comparison
against the published FBA number is the FBA medium, with the mismatch recorded.

### The linear baseline needs edits to a script I do not own

`linear_baselines_bx.py` cannot take a binary column as it stands. Its `TARGETS` registry
has no essentiality entry and no dataset to point one at, and more fundamentally it is a
REGRESSION pipeline throughout: it scores with per-feature Pearson, mean Spearman and NMSE,
and its B0 sanity check asserts per-feature Pearson 0 and `0.9 < nmse < 1.15`, neither of
which means anything for a ridge fit to a 0/1 label. Spec for the owner:

1. A `TARGETS["essentiality"]` entry: label `is_essential`, `columns = ["is_essential"]`,
   the new dataset root, `dump_head = "essentiality"`.
2. A classification scoring path beside the regression one: logistic regression rather than
   ridge for B1/B2, and AUROC plus per-class precision and recall rather than Pearson/NMSE.
3. The B0 sanity check gated on target kind: for a binary target the constant baseline's
   AUROC is 0.5, not a Pearson of 0.
4. The same label-inversion guard when the FCL split is read, or the baseline scores the
   opposite class from the arms.

### Tests

`tests/torchcell/metabolism/test_essentiality_head.py`, 8 tests on a synthetic batch: the
width-1 0/1 decode with no transform, the head's membership in the scalar and binary
families, AUROC as a rank statistic (perfect, inverted, tied at chance, single-class nan),
the per-class block against a hand-checked confusion matrix, the released split's inversion
and counts, the comparators being the yeast numbers, and `FluxBiomassReadout` reading
exactly the objective column and holding exactly two parameters.

## 2026.09.15 - Build BLOCKED on an empty graph; cells preflighted; FBA row specified

### The blocker, checked twice

The served Neo4j is **reachable and EMPTY**. Measured at 2026-09-14 22:5x and again on
2026-09-15:

| check | result |
|---|---|
| `bolt://localhost:7687` reachable | yes, container listening on 7687 and 7474 |
| `MATCH (n) RETURN count(n)` | **0** |
| `MATCH (d:Dataset) RETURN d.id` | **0 datasets** |
| `GeneEssentialitySgdDataset` experiments | **0** |
| `$DATA_ROOT/database/kg_manifest.json` | still lists **36** served datasets, including `GeneEssentialitySgdDataset` (file dated Sep 11) |
| KG rebuild jobs in the queue | none (1896 `kg_rehearsal_50` and 1897 `live_rebuild_kg` are gone from `squeue`) |

So the manifest and the live store disagree: the store was cleared and not repopulated, and
nothing is currently repopulating it. **The query has NOT been run and the dataset has NOT
been built.** There are no job ids to report for the build, and no class counts or split
coverage, because both come from the build.

`build_essentiality_dataset.py` refuses an empty graph before it opens the dataset, rather
than letting `Neo4jCellDataset` write an empty LMDB that reports success. That is why
`gh_cpu_build_essentiality.slurm` was not submitted: it would fail in its first seconds.

**What the build will report once the store is back**, all of it asserted rather than
printed: the four released-split class counts against the paper's Methods sentence (772 / 126
train, 192 / 31 test, with the label inversion applied), the build's own
`phenotype_label_index` containing `is_essential`, and the FCL split's coverage as
found/absent per class. `found_essential > 0` is a hard assert, because a build covering no
essential genes can only score the majority class.

### The shuffled-kcat control did not exist, and now does

`gh_wave9_ess_shuffled_*` referenced `--flux-shuffle-kcat`, which I had written into the
cells on an assumption. It did not exist anywhere: no flag, no arm, no mechanism. Implemented
now on the construction-time pattern, permuting the resolved kcat vector ACROSS catalytic
units with a recorded seed, in `build_flux_layer` right after `resolve_kcat_table`. The
architecture, the parameter count, the tier counts and the value DISTRIBUTION are all
unchanged; only the correspondence between a turnover number and its reaction is destroyed.
`flux_meta.kcat_shuffle` records the seed, the unit count, how many units moved, and the
value sum before and after as the multiset witness.

### The four cells, preflighted

All ten essentiality cells pass both checks with no dataset present: every flag resolves
against the real argparse parser, and every configuration constructs through `build_cell`
(41 call-site keywords accepted).

| cell | arm resolved | media | shuffle seed |
|---|---|---|---|
| `gh_wave9_ess_pool_e80_s270{0,1,2}` | `pooled` | none (no layer) | none |
| `gh_wave9_ess_flux_e80_s270{0,1,2}` | `flux_projection_anchored` | dataset (YPD) | none |
| `gh_wave9_ess_shuffled_e80_s270{0,1,2}` | `flux_projection_anchored` | dataset (YPD) | 2700 |
| `gh_wave9_ess_genoref_e80_s2700` | `flux_nullspace_kcat_sigma2` + channel `warmstart_k100_boxforce` | dataset (YPD) | none |

The genoref arm's `base_arm` resolves through its arm file to `flux_nullspace_kcat_sigma2`,
and the file's `flux_parameterization: warmstart_k100_boxforce` is what selects the warm
start, so the three warm-start flags are live on it.

None submitted. They cannot run until the dataset exists.

### FBA essentiality on both media: SPEC for the baselines owner

`cbm_baseline_bx.py` cannot do this without edits. Its `--target` choices are
`betaxanthin`, `amino_acids`, `isobutanol`; it has a `CONDITIONS` registry keyed on those,
it builds a bench model per medium through `build_bench_model(media_key, cassette=...)`, and
it never calls `single_gene_deletion`. What it already has that makes the addition small:
`FBA_MEDIA` and `media_to_bounds` are imported, and `build_bench_model` already refuses a
medium with unresolved components.

Spec, so the row is comparable to both the published FBA number and our own arms:

1. Add `--target essentiality` with `CONDITIONS["essentiality"]` carrying TWO conditions
   rather than one: `glucose_minimal` (`media_key` `SM`) and `rich` (`media_key` `YPD`).
   The first reproduces the published baseline's medium, the second matches the medium the
   SGD phenotype was scored on and the medium our flux arms run.
2. For each condition: `cobra.flux_analysis.single_gene_deletion(model)` over the genes of
   the released split, then threshold the predicted growth. Merzbacher used a fraction of
   wild-type growth (its E. coli cutoff was 0.41 against a wild-type 0.81, i.e. 50 percent);
   apply the same RULE, 50 percent of the wild-type growth on that medium, and record both
   the wild-type value and the absolute cutoff per condition, since they differ between the
   two media (measured: shipped bounds 0.0858, SM 0.3138, YPD-approx 0.5352 1/h).
3. Score with the same block the trainer uses, `train_bx.binary_block`, so the FBA row and
   the arm rows are computed by one function: AUROC (on the negated growth rate, since lower
   growth means more essential), accuracy, and per-class precision and recall, beside the
   majority-class rate.
4. Read labels through `train_bx.read_fcl_essentiality_split`, which applies the column
   inversion once and asserts the counts. A baseline that reads the CSV directly scores the
   opposite class from the arms.
5. Report against the published yeast row (FBA accuracy 0.81, ROC AUC 0.69), and expect our
   glucose-minimal number to differ from it: different GEM version, different gene list, and
   a threshold rule reapplied rather than copied.

It is a CPU job, minutes, and it needs no GPU.

## 2026.09.15 - CORRECTION: the store was never empty, my probe read the wrong database

The 2026-09-15 blocker section above is **wrong and is retracted**. The served store was not
empty; my probe queried the wrong Neo4j database.

The server at `bolt://localhost:7687` hosts three databases: `neo4j` (the default, EMPTY),
`system`, and **`torchcell`**, which holds the build. A `driver.session()` opened without
`database=` goes to `neo4j` and returns zero rows for every query, which is
indistinguishable from an empty or mid-rebuild store, and that is exactly what I reported.
Re-measured with `database="torchcell"`:

| check | wrong database (`neo4j`) | right database (`torchcell`) |
|---|---:|---:|
| `MATCH (n) RETURN count(n)` | 0 | **83,066,931** |
| `MATCH (d:Dataset)` | 0 | **36** |
| `GeneEssentialitySgdDataset` experiments | 0 | **1,140** |
| phenotype rows | none | 1,140 at `graph_level` node, label `is_essential` |
| deletions per genotype | none | 1,140 genotypes, all with exactly **1** |

`neo4j_query_raw.py:174` has hardcoded `database="torchcell"` at its session all along,
which is why every dataset build worked while my ad-hoc probe saw nothing. The dataset build
was never blocked; only my reading of the store was.

**The fix, so this cannot recur.** `Neo4jConnectionSettings` gains a `database` field
resolved from `NEO4J_DATABASE` with the default `torchcell`, alongside the URI, user and
password it already resolves. Nothing is hardcoded in a committed file: the credentials come
from the environment exactly as before. `build_essentiality_dataset.py` opens its preflight
session with `database=settings.database` and records the database name in the build
manifest beside the URI.

Not copied from `query_fig6.py`: that script hardcodes the NCSA URI
`neo4j+s://torchcell-database.ncsa.illinois.edu:7687` together with a readonly credential
pair. The NCSA host has been faulting on I/O since 2026-08-08, and a password in a committed
file is not a pattern to spread.

**The lesson worth keeping.** A query against the wrong Neo4j database does not error, it
returns an empty result. "0 nodes" and "the store is being rebuilt" look identical from the
client, and I reported the second when the first was true. Any future probe that reads zero
should check `SHOW DATABASES` before concluding anything about the store.

**Build submitted: job 1988**, `gh_cpu_build_essentiality.slurm`, nice 1000, 8 cores, 24 GB,
4 h limit.

### Build ran (job 1988, 14 s) and the dataset has NO negative class

`gh_cpu_build_essentiality.slurm`, job **1988**, COMPLETED in 14 seconds. 1,140 genotypes,
0 deduplicated, `phenotype_label_index` is `{"is_essential": 1140}`, so the converter defect
is gone and the label survives as itself. Manifest at
`$DATA_ROOT/data/torchcell/experiments/027-betaxanthin-metabolic/essentiality_build/build_manifest.json`.

Then the numbers that stop the endpoint as designed:

| quantity | value |
|---|---:|
| records in the build | 1,140 |
| **distinct `is_essential` values** | **`True` only, 1,140 of 1,140** |
| FCL split genes found | 166 / 1,121 |
| of those, essential | 155 |
| of those, nonessential | 11 |
| FCL split genes absent | 955 |

**`GeneEssentialitySgdDataset` is the ESSENTIAL-gene list, not an essentiality label over
genes.** It records essential genes and almost nothing else, so the build is single-class: a
binary classifier has no negative class, the cross entropy has nothing to separate, and
AUROC is undefined. The `found_essential > 0` assert passed and did not catch this, because
it guards the opposite failure. **The four arms cannot train on this build.**

The 11 genes the split calls nonessential and the build calls essential are a genuine
disagreement between SGD and the labels Merzbacher took "from the literature", and they are
worth a look on their own.

### Where the negative class is, measured

Single-deletion genotype coverage of the split's two classes, per dataset in the served
graph:

| dataset | nonessential covered | essential covered |
|---|---:|---:|
| `GeneEssentialitySgdDataset` | 11 / 964 | 155 / 157 |
| `SmfCostanzo2016Dataset` | 820 / 964 | 1 / 157 |
| `SmfKuzmin2018Dataset` | 192 / 964 | 0 / 157 |
| **union of the three** | **826 / 964 (85.7 %)** | **156 / 157 (99.4 %)** |

The union reaches **982 of the 1,121** split genes. So the negative class lives in the
single-mutant fitness datasets, which is what a viable deletion collection IS, and the
positive class lives in the SGD essential list.

### The fork this creates, and I am not choosing it silently

A union build reintroduces exactly the defect the dedicated query was written to avoid: with
a fitness dataset co-grouped, `MeanExperimentDeduplicator` merges `is_essential` into a
`fitness` record, which is what produced `fitness`-only labels in the 003 and 025 builds.

Two routes, and they differ in what the LABEL is:

1. **Label from the released split, genotypes from the union build.** The binary label is
   joined per gene from
   `yeast_essentiality_test_split.csv`, and the dataset supplies only the genotype and the
   graph. This is what Flux Cone Learning itself did: it trained on flux cones keyed by
   knockout with labels taken from the literature, not from a phenotype record in a graph.
   It gets 982 of 1,121 genes and both classes, and it sidesteps the deduplicator entirely
   because no phenotype is being decoded. It requires changing `--head essentiality` to read
   its target from the split file rather than from `is_essential` in the batch, which is a
   real change to what I built today.
2. **Label from the graph, union build with the merge suppressed.** Keeps the head as built,
   but needs the union query to avoid the fitness merge, which means either a converter-free
   aggregator for this build or a post-hoc reconstruction of the boolean. More machinery,
   and the label still has to be reconciled against the split for the comparison to mean
   anything.

**Route 1 is the recommendation**, because the comparator numbers were computed on those
exact labels and any label we derive ourselves is a different experiment from the one the
table reports. It is not implemented: the head currently decodes `is_essential` from the
batch, and no cell was submitted.

## 2026.09.15 - Union build, label from the split, two cells submitted

### The build (job 1989, 54 s)

`essentiality_build` rebuilt as the UNION of `GeneEssentialitySgdDataset`,
`SmfCostanzo2016Dataset` and `SmfKuzmin2018Dataset`, single deletions, `converter=None`,
`DeletionKeyedGenotypeAggregator`. The SGD-only root was moved aside rather than deleted and
is named in the manifest as `superseded_root`.

| quantity | value |
|---|---:|
| genotypes | **5,656** |
| phenotype labels present, ALL ignored by this head | `fitness` 4,540, `is_essential` 1,140 |
| FCL split genotype coverage | **982 / 1,121** |
| found essential / nonessential | **156 / 826** |
| absent essential / nonessential | 1 / 138 |

Three coverages are distinct and all three matter:

* **cell-graph gene coverage 1,121 / 1,121** (157 essential, 964 nonessential): every split
  gene is a node, so no label is lost to the gene set;
* **genotype coverage 982 / 1,121**: only those have a single-deletion genotype to attach a
  prediction to;
* **test-gene coverage 195 / 223**, giving 195 records pinned to test, 3.4 percent of the
  build, of which **31 are essential**, matching the paper's own held-out essential count.

### The label comes from the split, not from the graph

`--head essentiality` no longer decodes a phenotype. `essentiality_gene_labels` builds one
value and one `known` flag per gene node from `read_fcl_essentiality_split` (inversion
applied once, counts asserted), and `gene_keyed_target` joins it per genotype through
`deletion_indices`, the deletion-only field added earlier the same day. A genotype whose
gene the split does not name, or which does not delete exactly one gene, arrives masked
False and contributes to no loss, no selection metric and no test score. `extract_targets`
is not called for this head at all, which is exactly why the union's fitness merge is
harmless: nothing reads the merged column.

The held-out set is Merzbacher's own 223-gene list rather than the betaxanthin 639-gene pin,
routed through a new per-head `held_out_mask`.

### Smoke, and the bug it caught

One CPU epoch of the `ess_pool` cell's exact command. The first run died AFTER training, in
the headline print: `result['test']['pinned']['spearman']`, a key `binary_block` does not
emit and should not, since a rank correlation between a logit and a 0/1 label is not the
endpoint. Fixed in both the headline and the W&B summary, which now report AUROC, accuracy,
the majority rate and per-class recall. The rerun:

```
[pooled s2700] ep 0 loss 1.4478 data 0.4751 val_loss 0.3725 val_r +0.4327 sd_ratio 0.0002
  held-out AUROC +0.4386 (accuracy 0.8410 against a majority rate of 0.8410; n=195, 31 essential)
```

At one epoch the head predicts the majority class on every gene (recall_essential 0.000,
recall_nonessential 1.000, accuracy exactly the majority rate), so the 0.4386 is not a
result, it is one epoch. `val_r` is the validation AUROC under the centered 5-epoch rule.

### Submitted

| job | cell |
|---:|---|
| **1990** | `gh_wave9_ess_pool_e80_s2700.slurm` |
| **1991** | `gh_wave9_ess_flux_e80_s2700.slurm` |

Both through `freeze_and_submit.sh`, queued behind the running cells. The other eight
essentiality cells are preflighted against the new root and NOT submitted.

### The 11 genes SGD calls essential and Merzbacher's labels call nonessential

`YBR121C, YBR128C, YCL004W, YCR068W, YFR021W, YJL012C, YKL192C, YKL212W, YNL280C, YPL120W,
YPL268W`

These are in the build as SGD-essential and carry a nonessential label in the released split,
which Merzbacher took "from the literature". Since the trainer uses the SPLIT's label, these
11 train and score as nonessential. They are 11 of 1,121, so they cannot move a headline on
their own, but they are a real disagreement between two label sources and worth a look
before the comparison is written up: GRC1/YBR121C, ERG13/YBR128C is a candidate misread, and
YKL212W is SAC1, all of which have essentiality that is known to be condition-dependent.
Hypothesis (untested): the disagreement is medium-dependent essentiality rather than error
in either source.

### Deviation to flag

The coordinator asked for train and validation to be class-stratified over the non-test
genes. They are NOT: the data module takes its own seeded random draw over the remaining
records, and I did not replace that. With 86/14 imbalance over roughly 800 labeled non-test
records a random draw is close to stratified in expectation, but it is not guaranteed, and
the realized per-split class balance should be read out of the result before any contrast is
believed.

## 2026.09.15 - First reads: the backbone beats the published comparators and the module destroys it

Both seed-2700 cells landed after three crashes (a missing medium entry, a biomass index read
off the wrong object, and a target allocated on the host while indexed from the GPU; the last
is now covered by a GPU-gated test).

| arm | job | select ep | AUROC | accuracy | majority | recall essential | prediction spread ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| `pooled`, no flux layer | 2009 | 60 | **0.7927** | 0.8564 | 0.8410 | 0.323 | 3.12 |
| `flux_projection_anchored` | 2008 | 26 | 0.5398 | 0.8410 | 0.8410 | 0.000 | 0.0005 |

Published on this split, from the mirrored paper: flux cone learning 0.73, flux balance
analysis 0.69.

* **The no-module control beats both published numbers**, at n = 195 of the split's 223 test
  genes, 31 of them essential. This is the first endpoint in 026 or 027 where the backbone
  clears a published flux-model comparator rather than a null.
* **The flux arm is at chance and calls every gene nonessential.** Its accuracy equals the
  majority rate to four decimals and its recall on essential genes is 0.000, which is what a
  constant predictor scores. The paired difference at seed 2700 is -0.253 AUROC.
* **Caveats, all of which matter before this is quoted.** One seed each, so jobs 2021 to 2024
  replicate both arms at 2701 and 2702. Our test set is 195 of the paper's 223 genes, the
  subset the union build covers, so the comparison to 0.73 and 0.69 is close but not
  like-for-like. Train and validation are not class stratified. The label is the released
  split's, with its inversion fixed.
* **It fits the collapse pattern exactly.** The pooled arm's prediction spread ratio is 3.12
  and the flux arm's is 0.0005. Every readout through the flux layer collapses; no readout on
  the pooled context does.
