---
id: 9nud26qgeqwlorz6hxe6ivi
title: Wetmore2015_release_inventory
desc: ''
updated: 1791695037274
created: 1791695037274
---

## 2026.10.10 - Wetmore 2015's own release, measured off the deposited bytes

Candidate row 2 (*E. coli*, "Transposon fitness"): Wetmore KM, Price MN, Waters RJ, et al. Rapid quantification of mutant fitness in diverse bacteria by sequencing randomly bar-coded transposons. mBio 2015;6:e00306-15, doi:10.1128/mBio.00306-15. Loader: [[torchcell.datasets.ecoli.wetmore2015]] (`RbTnseqWetmore2015Dataset`). Compendium it is checked against: [[torchcell.datasets.ecoli.price2018]].

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/wetmore2015_release_inventory.py`. Results: `experiments/036-dataset-fixes-before-kg-build/results/wetmore2015_release_inventory.json`. Run from the repo root:

```bash
python experiments/036-dataset-fixes-before-kg-build/scripts/wetmore2015_release_inventory.py
```

The script reads the sha256-pinned files with pandas, NOT through the loader's readers, and restates the paper's five quality rules itself, so its counts are an independent re-derivation of the loader's inventory. It refuses to write when any pin is broken.

### Files (all 12 intact against their pins)

Data Set S1 (`si/si1.xlsx`, literature mirror) and eleven raw-mirror files under `$DATA_ROOT/torchcell-raw/wetmoreRapidQuantificationMutant2015/`: the seven the 2026-10-07 provenance record deposited, plus `fit_t.tab`, `fit_standard_error_obs.tab`, `fit_standard_error_naive.tab` and `fit_logratios.tab`, retrieved 2026-10-10 by the recorded `direct_url` route (the old seven were re-fetched in the same run and are byte-identical upstream).

### Samples

| measured | value |
|---|---|
| `Expts_Keio` rows / time-zero rows / condition samples | 111 / 10 / 101 |
| successful by Data Set S1 `u` | 92 |
| the paper's five rules reproduce `u` | yes, for all 101 |
| failed, with the rule each trips | 9: `set1IT030` (gMed 49), `set1IT033`, `set1IT034` (gMed 0); `set2IT024`, `set2IT034`, `set2IT045`, `set2IT055`, `set2IT079`, `set2IT090` (cor12 < 0.1) |
| condition samples by medium | M9 noCarbon 64, M9 noNitrogen 26, LB 7, MOPS Rich Defined noCarbon 4 |
| successful by medium | M9 noCarbon 61, M9 noNitrogen 26, MOPS Rich Defined noCarbon 4, LB 1 |
| sucrose / D-mannitol samples (the 2021 withdrawal) | `set1IT007`, `set1IT008` (Sucrose), `set1IT043`, `set1IT044` (D-Mannitol); all four successful |

### Values

| measured | value |
|---|---|
| genes of the gene-fitness table (all `sysName` are b-numbers) | 3,646 |
| sample columns: successful-sample table / all-sample table | 92 / 106 (the 101 condition samples and 5 time-zero columns) |
| successful values equal between the two tables | max difference 0.0 |
| max \|t - fit / sqrt(0.1^2 + max(se_obs, se_naive)^2)\| over all 106 x 3,646 values | 1.99e-13 |
| successful values whose read-count SE exceeds the strain-consistency SE | 24.35% |

### Against the compendium release (Price 2018, statistics version 1.0.3)

On the 3,646 shared genes (the compendium has 3,789), every one of the 92 samples both releases carry, `set2IT045` included through the all-sample table: per-sample Pearson r min 0.98895, median 0.99713, none below the bar of 0.87 (Price 2018's median correlation between exact replicates of defined-media experiments). Largest single-gene difference 2.5725 log2 (`set1IT028`, `b2500`). This release's samples the compendium release lacks: `set1IT029` (fumarate, successful here, failed at 1.0.3) and the eight samples both releases fail.

### Against the compendium store (`data/torchcell/rbtnseq_price2018_ecoli`)

553,896 records, built from `52d376d6d` with `torchcell_dirty: true`. Of its 162 Table S5 samples 92 are this paper's; 84 of those are stored and 8 dropped by Price's own ledger (the four withdrawn samples, `authors_withdrew_sucrose_and_mannitol_2021`; the four MOPS Rich Defined samples, `medium_not_in_media_library`). This paper's successful samples absent from the store: 9 (the four withdrawn, `set1IT029`, the four MOPS Rich Defined). The store's phenotype carries no t statistic.
