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
