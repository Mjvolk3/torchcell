---
id: fnn78zmlzdeqhiygg3i6myg
title: thompsonMassivelyParallelFitness2019_release_inventory
desc: ''
updated: 1791613421479
created: 1791613421479
---

## 2026.10.10 - Thompson 2019 lysine is subsumed by the Borchert 2024 compendium

Schedule row "Thompson 2019 lysine" (Thompson MG et al. 2019, mBio 10:e02577-18, DOI 10.1128/mBio.02577-18, key `thompsonMassivelyParallelFitness2019`). Decision: **subsumed, no loader**, on the Wetmore 2015 / Thompson 2020 provenance-record pattern. Generating script: `experiments/036-dataset-fixes-before-kg-build/scripts/thompsonMassivelyParallelFitness2019_release_inventory.py`; results `experiments/036-dataset-fixes-before-kg-build/results/thompsonMassivelyParallelFitness2019_release_inventory.json` and the per-cell table `..._table_s1_cells.csv`.

### Mirror route

The paper is not in the literature mirror (no manifest names the DOI), and nothing went to Zotero. Its open-access copy is in the PMC Article Datasets bucket under `PMC6509195.1`, so three objects were retrieved with `torchcell.literature.retrieve.pmc_cloud_object` (`pmc_cloud`, the Mohiuddin 2022 route) into `$DATA_ROOT/torchcell-raw/thompsonMassivelyParallelFitness2019/` with a `manifest.json`:

| file | sha256 | bytes | read for |
|---|---|---|---|
| `paper/PMC6509195.1.txt` | `aedb5fb0...` | 73,809 | every quote (Methods 'RB-TnSeq', Results) |
| `si/mBio.02577-18-st001.csv` (Table S1) | `b4718373...` | 4,059 | the 156 per-gene cells |
| `si/mBio.02577-18-st002.csv` (Table S2) | `907aed18...` | 50,030 | shape only |

All eight quotes re-audit verbatim against the pinned text (the PMC text uses thin spaces before units and no-break spaces after "Table"; the quotes carry those code points).

### Release, measured

- Table S1: 39 rows, 39 distinct loci, 4 value columns (D-Lysine, L-Lysine, 5-AVA, Glucose), 156 cells. Re-applying the Results rule (below -2 on 5AVA, D- or L-lysine, not below -0.5 on glucose) selects all 39, matching "Fitness profiling revealed 39 genes".
- Table S2: 540 rows of pairwise tests (Prot, Carbon1, Carbon2, Stat, pval, Sig, Bon_Corr) over 15 proteins and 6 carbon sources of WILD-TYPE targeted proteomics. Test statistics, no per-sample abundance and no perturbation, so no record of any existing class.
- Genome-wide fitness: released only on the Fitness Browser, which answered HTTP 403 on 2026-10-10.
- Deletion-strain growth curves (Figs S3, S5-S7): TIF images only.

### Duplication, by content

Each Table S1 column was compared against all 332 sample columns of the compendium release (`fModule_Metadata.xlsx`, sha256 `4d649385...`), scored by max absolute difference over the 39 loci:

| Table S1 column | matched sample | max abs diff | median abs diff | r | runner-up (max abs diff) |
|---|---|---|---|---|---|
| D-Lysine | set7IT062 | 0.0460 | 0.0033 | 0.999993 | set7IT056 (2.14) |
| L-Lysine | set7IT055 | 0.0194 | 0.0009 | 0.999997 | set12IT023 (3.22) |
| 5-AVA | set7IT044 | 0.0207 | 0.0024 | 0.999989 | set7IT057 (1.43) |
| Glucose | set6IT057 | 0.0082 | 0.0007 | 0.999945 | set8IT058 (0.58) |

The served `RbTnseqBorchert2024Dataset` (`$DATA_ROOT/data/torchcell/rbtnseq_borchert2024`, 1,372,280 records) holds each matched sample as 4,732 records whose values equal the compendium matrix exactly (max difference 0.0), i.e. 18,928 records, and all 156 Table S1 cells are among them (served fraction 1.0 over an independent release count).

Unlike rows 24, 28 and 29, the agreement is NOT within the 0.0005 rounding bound of the compendium's three decimals: only 28 of 156 cells are. The identification is still unambiguous (worst match 0.046 against a nearest wrong sample at 0.58). Hypothesis (untested): the compendium values come from a later Fitness Browser recomputation of the same assays, whose normalization differs slightly from the 2019 export.

### Findings for the compendium loader

- The loader's UNTESTED hypothesis for set7 ("Thompson 2019 lysine") is now measured for three samples (set7IT044, set7IT055, set7IT062), and the glucose control is set6IT057, which the loader's `HYPOTHESES` attributes to nobody in particular (set6 is listed for valerolactam). Attaching `thompsonMassivelyParallelFitness2019` as a `SourceStudy` would change the stored publication on 18,928 records and needs a rebuild of the 1.37 M-record store; left as an owner decision, not done here.
- The compendium metadata labels set6IT057 "48 well microplate; Tecan Infinite F200", while the Methods state "Cells were grown in 50-ml culture tubes". The served Environment follows the compendium.
- Same-condition samples Table S1 does not use: set7IT056 (D-lysine) and set7IT057 (5-aminovalerate), each from the same set and date. Not measured whether they are this paper's unreported replicates or another study's samples.
