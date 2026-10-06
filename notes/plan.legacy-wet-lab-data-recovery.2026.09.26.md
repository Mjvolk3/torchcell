---
id: 7xdoe8ofn2hj5yhitrw4pra
title: Legacy Wet-Lab Data Recovery (pre-2023)
desc: ''
updated: 1790476374242
created: 1790476374242
---

## 2026.09.26 - Inventory and recovery plan

Locating and packaging the wet-lab data collected before roughly 2023, so it can be used
in the environment-conditioned models, cited in the dissertation, and published alongside
whatever it supports. Everything below was located by search on the M1 Mac on 2026.09.26.

**The central risk: almost none of this is on local disk or in git.** The raw instrument
files live in Box, the analysis code lives in two local projects that are not part of
torchcell, and two of the three source projects have no GitHub remote. Nothing has a
provenance record. This note is the map.

### Where it lives

| Tier | Location | Under version control |
| --- | --- | --- |
| Raw instrument files | `~/Library/CloudStorage/Box-Box/AAA Lab Research/Wet_Lab/experiments/BioscreenC/` (69 files) | No, Box only |
| Analysis and processed results | `~/Documents/projects/multi_knockout/` (5,859 files) | Local git, no remote found |
| Design, costing, R library | `~/Library/CloudStorage/Box-Box/AAA PC_copy/Desktop/Mutation_Combinatorics/` | `github.com/Mjvolk3/Mutation_Combinatorics` |
| Isobole plotting | `~/Documents/projects/inhibitor_tolerance/` | Local only |
| Old-machine mirror | `~/Library/CloudStorage/GoogleDrive-.../Other computers/My MacBook Pro/projects/` | No |

The Google Drive mirror of the old MacBook holds: ART, backwardian_continuum, BIONIC,
chbe478_bioenergy_technology, CompGCN, Ferm_Opt, GNN_ScLipid_shared, GraphGym,
inhibitor_tolerance, Kbase, life_long_learning, noteTraits, trigenic_temp, vault,
Vicissitudes, Zotero, plus Deep-Generative-and-Dynamic-Models. A second machine mirror
sits at `Box-Box/AAA PC_copy/`, which is where Mutation_Combinatorics actually lives.

### Strain background (resolved)

**BY4741.** Measured, not inferred: the preprocessed tables carry an explicit `strain`
column whose value is `BY4741`, in `multi_knockout/experiments/bsc/ex24/MV_ex24_preprocessed.csv`
and set in `ex25/ex25_preprocessing.ipynb`. The single-mutant position lists
(`ex16/position_list.csv`) list `BY4741` as the wild-type control across replicates
`br1` to `br5`.

Unresolved: `MV_ex22_Magic_Calibration` and `MV_ex23_Magic_inhibitor_combinations` carry
"Magic" in the filename. No strain string appears in the ex23 processed outputs.
Hypothesis (untested): "Magic" names the MAGIC system of
`lianMultifunctionalGenomewideCRISPR2019`, not a different strain background. **Confirm
before publishing anything that states the genotype of the inhibitor assays.**

### What the data actually is (measured)

#### Single-mutant fitness (smf)

Four Bioscreen C runs, `ex16` to `ex19`, MATalpha deletion collection, July to August 2020.
Processed outputs `MV_ex{16,17,18,19}_fitness.csv` all share the schema
`experiment_id, name, fitness_mean, fitness_std`, so replicates are already aggregated.

| Run | Rows | Unique ORFs | Raw file date |
| --- | --- | --- | --- |
| ex16 | 51 | 51 | 2020-07-16 |
| ex17 | 51 | 51 | 2020-08-08 |
| ex18 | 63 | 63 | 2020-08-22 |
| ex19 | 63 | 63 | 2020-08-24 |

**Union across all four runs: 122 unique ORFs.** ex16 and ex17 repeat one gene set, ex18
and ex19 repeat a second, so there are two panels measured twice each, which gives a
between-run reproducibility check for free.

Earlier single-mutant work: `Mv_ex12_single_mutants.bsm` and `_.csv` (2020-03-10), with
`OD_calculation_single_mutants.xlsx` (2020-03-08).

#### Triple-mutant fitness (tmf)

`ex24` (2021-04-25) and `ex25` (2021-06-28). Six genotypes measured against the Costanzo
literature values, in `ex24/bar_graph_comparison.xlsx` and `ex24/1_rk1_literature.csv`.
This is the source of the "Fitness Trigenic Interactions" slide.

| Genotype | Measured fitness | Measured SD | Literature fitness | Literature SD |
| --- | --- | --- | --- | --- |
| WT | 1.017 | | | |
| YEL063C YOR128C YNL268W (A3) | 0.650 | 0.130 | not in Costanzo | |
| YGL263W YDR210W YOL031C | 0.956 | 0.080 | 1.0389 | 0.0200 |
| YIL149C YDR210W YOL031C | 1.115 | 0.420 | 1.0336 | 0.0459 |
| YJL016W YDR210W YOL031C | 0.812 | 0.190 | 1.0667 | 0.0424 |
| YLR309C YDR210W YOL031C | 0.951 | 0.032 | 1.0304 | 0.0357 |
| YMR219W YDR210W YOL031C | 0.793 | 0.143 | 1.0200 | 0.0427 |

Replicates exist (a standard deviation is reported per genotype), but the replicate
**count** per genotype has not been read out of the raw file yet. Do that before any
reuse: the `n` drives the uncertainty the loader would store.

`1_rk1_literature.csv` carries the full Costanzo record per triple: raw and adjusted
epsilon, p-value, query double-mutant fitness, array single-mutant fitness, combined
fitness and its SD. So the comparison is already joined to the published values.

Five of the six share the `YDR210W YOL031C` double background, so this is one query double
crossed against five array singles, plus one unrelated triple (A3). **That is a very thin
panel.** Treat any model-ranking use as a spot check, not an evaluation.

#### Inhibitor dose response

`MV_ex21_inhibitor_titration` (2021-04-08). Six inhibitors, three biological replicates
each, doubling time against concentration, with the literature concentration marked:
lactic acid, formic acid, acetic acid, levulinic acid, HMF, furfural. The reference
concentrations come from the lab-scale hot-water pretreatment table of
Cheng, Dien, Lee and Singh (Bioresource Technology), 190 C for 10 min.

Working concentrations chosen from it, per the combinations slide: furfural 1.5, acetic
acid 2, HMF 2.52, formic acid 1, levulinic acid 6, lactic acid 20, all g/L.

#### Inhibitor combinations at a single concentration

`MV_ex23_Magic_inhibitor_combinations` (2021-04-20), calibrated by `MV_ex22` (2021-04-13).
`ex23/MV_ex23_fitness.csv` has 63 rows of which **25 have a fitness value**; the remaining
38 are empty. The 25 measured conditions are the six singles plus pairs and a few triples:

`AA, FA, FF, HMF, LA, LVA, AA_FA, AA_FA_LVA, AA_HMF, AA_LA, AA_LVA, FA_LVA, FF_AA,
FF_AA_FA, FF_AA_HMF, FF_FA, FF_FA_LVA, FF_HMF, FF_HMF_FA, FF_LA, FF_LVA, HMF_FA, HMF_LA,
HMF_LVA, LVA_LA`

The 38 empty rows are worth explaining before publication: they are either combinations
that were planned and not run, or wells that failed QC. Unknown at present.

#### Isoboles

Three two-way isobole experiments, each a full concentration grid rather than a single
point:

| Experiment | Pair | Date |
| --- | --- | --- |
| `MV_ex26_inhibitor_isobole_FF_AA` | furfural + acetic acid | 2021-09-11 |
| `MV_ex27_inhibitor_isobole_FA_AA` | formic + acetic acid | 2022-06-16 |
| `MV_ex28_inhibitor_isobole_HMF_AA` | HMF + acetic acid | 2022-06-16 |

Parsed traits for ex26 are local at
`inhibitor_tolerance/MV_ex26_inhibitor_isobole_FF_AA_Traits.txt`, columns
`Container Name, Lag, GT, Yield, Problem found, Details`, per well, with a QC flag. `GT`
is the doubling time. Plotting in `inhibitor_tolerance/notebooks/n2--plotting_isoboles.ipynb`.

These are the highest-value inhibitor records: a grid over two concentrations gives an
interaction surface, not a single fitness value, so the antagonism or synergy question is
answerable from them.

### Processing code that would have to be published

Any dataset built from the above depends on this code, so it has to be recovered and
committed, not just the numbers.

| Purpose | Path |
| --- | --- |
| Per-run preprocessing | `multi_knockout/experiments/bsc/ex{16,17,18,19,23,24,25}/*_preprocess*.ipynb` |
| smf comparison and the confusion matrix | `multi_knockout/bsc_fitness_ex16_ex17_ex18_ex19_ex24.ipynb` |
| Inhibitor combinations | `multi_knockout/bsc_fitness_ex23.ipynb` |
| Fitness statistics | `multi_knockout/bsc_fitness_mean_std.ipynb`, `bsc_fitness_plots_prelim.ipynb` |
| Interaction error propagation | `multi_knockout/digenic_interaction_std_calculation_check.ipynb` |
| Bioscreen parsing | `multi_knockout/process_bsc.py` |
| Isobole plotting | `inhibitor_tolerance/notebooks/n2--plotting_isoboles.ipynb` |
| Worklist generation | `inhibitor_tolerance/notebooks/MV_BioScreen_Worklist.ipynb` |
| R fitness library | `Mutation_Combinatorics/lib/{mutant_fitness,plot_fitness,plot_compare_fitness,process_PRECOG,pipette_error}.R` |
| Randomized plating, iBioFAB worklists | `Mutation_Combinatorics/{BC_rand_plate,BC_ibiofab_worklist,process_BC}.R` |

The R side matters for a reason beyond provenance: `pipette_error.R` and `BC_rand_seed.R`
say the plate layout was randomized and pipetting error was modeled. That is the argument
that the measurements are not confounded by plate position, and it is worth stating.

Published reference already on hand:
`Mutation_Combinatorics/published_data/10.1038-nmeth.1534_S1_Single-mutant_fitness_standard.xls`.

### The validation result already computed

`multi_knockout/experiments/bsc/bsc_statistics.txt` holds an agreement analysis between
these measurements and literature values, across ex16 to ex19 and ex24:

| Quantity | Value |
| --- | --- |
| True positive | 125 |
| True negative | 52 |
| False positive | 34 |
| False negative | 20 |
| Recall (TPR) | 0.8621 |
| Specificity (TNR) | 0.6047 |
| Precision (PPV) | 0.7862 |
| Negative predictive value | 0.7222 |
| Accuracy | 0.7662 |
| F1 | 0.8224 |
| Matthews correlation coefficient | 0.5324 |

Figures: `fitness_assay_comparison_ex16_ex17_ex18_ex19_together_confusion_mtx_red_line.png`
and `fitness_assay_comparison_singles_triples_cofusion_mtx_red_line.png`, same directory.
The threshold that turns a continuous fitness into the binary call is set inside the
notebook and must be stated wherever this table is reproduced; a confusion matrix over a
thresholded continuous variable is meaningless without it.

### Candidate uses, honestly rated

**Agreement with published trends (strongest).** The table above already is that result.
122 single mutants measured in a liquid assay against colony-size-derived literature
fitness is a real methods comparison. Specificity 0.60 against recall 0.86 means the assay
calls more mutants "fit" than the literature does.

Hypothesis (untested, and the one the author raised): the liquid Bioscreen assay
systematically **overestimates** fitness relative to colony size. The data to test it is
in hand, since `MV_ex{16..19}_fitness.csv` joins to literature values through
`fit_compare.csv` and `fitness_compare.csv`. The test is a signed residual distribution,
not a confusion matrix. Run it before claiming the direction.

**Environment-conditioned model input (weak on its own, useful as held-out).** 25 measured
inhibitor conditions on one wild-type background is not a training set: there is no gene
perturbation dimension, so it cannot constrain a genotype-by-environment model. What it
can do is serve as a measured environment axis for the reference cell, or as a held-out
check on a medium encoding.

**Isobole resolution (promising, unquantified).** Three concentration grids. Whether a
two-dimensional dose-response surface can be resolved from them depends on grid density
per experiment, which has not been counted yet. Count the unique concentration pairs in
ex26 to ex28 before planning anything on them.

**Trigenic model ranking (do not attempt as an evaluation).** Six genotypes, five sharing
one double background. Ordering model predictions on this panel is a demonstration, not a
benchmark, and should be labeled as such in the dissertation.

### Open questions, in priority order

1. Replicate count per genotype in ex24 and ex25. Drives the stored uncertainty.
2. What the 38 empty rows in the ex23 fitness table are.
3. The fitness threshold used for the confusion matrix.
4. Whether "Magic" in ex22 and ex23 names a strain or the MAGIC system.
5. Grid density of the three isobole experiments.
6. Whether `multi_knockout` has a remote, and whether it should be published as-is or
   re-expressed as a torchcell dataset.
7. Whether the A3 triple (YEL063C YOR128C YNL268W) has any literature value anywhere.

### Recovery steps

1. **Copy the raw tier out of Box** into the raw-data mirror under a citation-key-style
   directory, with `source_url` recorded as the Box path, `retrieval_method`
   `manual_browser`, and a sha256 per file. Box is not a durable dependency.
2. **Give `multi_knockout` a remote**, or vendor the processing notebooks into a torchcell
   experiment directory. Today the only copy of the analysis is one unsynced local folder.
3. **Answer questions 1 through 5** by reading the raw files, not by recollection.
4. **Re-run one preprocessing notebook end to end** from the mirrored raw file to confirm
   the pipeline still reproduces a committed `MV_ex*_fitness.csv`. Until that passes, the
   data is not recoverable, only present.
5. **Then** decide the dataset shape: a single-mutant liquid-fitness dataset, an
   inhibitor-condition dataset, and an isobole dataset are three different schemas.

### Separate proposal: OCR the PhD slide decks

The `Box-Box/AAA Lab Research/Meetings/` tree holds the ML4BSD and subgroup decks from
2020 through 2025, and the research updates. They are the only continuous record of what
was tried and why, including work whose data was never written up. Several of the figures
traced in this note were found only because they appear in those decks.

Proposal: run the decks through the same MinerU OCR path the literature mirror uses, so
the deck text becomes searchable and quotable with a sha256 per file. That makes the
narrative of the dissertation recoverable from primary sources rather than memory. Scope
it from the start of competence onward rather than everything.

This note is the data half. The slide OCR would be the narrative half and should get its
own note if it goes ahead.

## 2026.10.03 - Strain Correction and Archive Location

Correction to the strain section above, sourced from the submitted 2021 prelim report
(`thesis_archive/reports/Prelim_Report_2021_Michael_Volk.pdf`, section 1.2.3 and 2.2.1):

- Single mutants ex16-ex19 were the BY4742 MATalpha deletion library borrowed from
  Dr. Jin, not BY4741. Report text: "BY4742 MATα single deletion strain library was
  borrowed from Dr. Jin and used for preliminary testing of the BioscreenC".
- The inhibitor runs used the MAGIC strain BY4742-iAID6. Report text: "MAGIC strain
  BY4742-iAID6 was grown over a gradient of inhibition concentrations". This resolves the
  "Magic" in the ex22/ex23 filenames: it names the strain, not only the system.
- The BY4741 `strain` column in the ex24/ex25 processed files covers only those
  triple-mutant runs. The report passages read so far do not name the triples' parent
  strain, so BY4741 for the triples is still the processed files' claim, unconfirmed.

Every source in the inventory table is now copied, sha256-verified, into Box
`AAA Lab Research/thesis_archive/data/` (29,438 files, `MANIFEST.tsv`), so the single-copy
risk for `multi_knockout` and `inhibitor_tolerance` is closed. Run index and
prelim-figure-to-data map: `thesis_archive/README.md`.
