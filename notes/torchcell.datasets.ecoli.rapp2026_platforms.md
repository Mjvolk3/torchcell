---
id: c5nailv535w8f5tlf09nzkl
title: Rapp2026_platforms
desc: ''
updated: 1791436537233
created: 1791436537233
---

## 2026.10.08 - Three further Rapp 2026 quantity families

Rows 4, 5 and 6 of the E. coli SI audit's ranked loadable-now table
([[plan.bacteria-si-phenotype-audit-ecoli]]). `torchcell/datasets/ecoli/rapp2026_platforms.py`
adds three dataset classes beside [[torchcell.datasets.ecoli.rapp2026]]'s FI-MS
fold-change store, one per released scale, so no two platforms share a record set.

| dataset | records | values | source | stored statistic |
|---|---|---|---|---|
| `GrowthAucRapp2026Dataset` | **1,514** | 1,514 | Table S2 (`si3.xlsx`) + Table S1 | trapezoid AUC of the 181-point OD600 curve, mean of 3 cultures, over the 48 control cultures' mean |
| `TargetedMetabolomeRapp2026Dataset` | **406** | **1,244** | Table S6 (`si7.xlsx`) + S1 + S3 + S5 + S9 | targeted LC-MS/MS EIC peak-height fold change vs the library median |
| `MetaboliteIntensityRapp2026Dataset` | **406** | **1,373** | Table S5 (`si6.xlsx`) + S1 + S3 + S9 | absolute FI-MS intensity of the annotated feature, mean of 2 plates |

### Where the audit's counts move, and why

The audit measured **411 records** for each accumulation family from
`df["Gene"].nunique()`. Measured here: **4 of those 411 tokens are CONTROL wells**
(`ctrl4`, `ctrl7`, `ctrl8`, `ctrl11`, carrying 1, 6, 1 and 2 rows), which cannot be
records because they have no knockdown, and `phnE` is dropped by the same
`b_number_remapped_by_the_annotation` rule the metabolome loader applies. So each
accumulation family is **406 records**, and the value counts are 1,256 - 10 - 2 = 1,244
and 1,385 - 10 - 2 = 1,373. The growth count is the audit's: 1,515 Table S2 genes minus
`phnE`. `argR`, the metabolome loader's other drop, is absent from Table S1, Table S2 and
both accumulation tables, so it never arises here.

### Non-duplication, reproduced

The audit's statistic for Table S6 against the stored FI-MS values reproduces exactly,
and the build refuses any other answer (`preprocess/platform_agreement.json`):

| statistic | audit | measured here |
|---|---|---|
| joined pairs (1:1 on gene, abbreviation, mode) | 1,256 | 1,256, 0 unmatched |
| Pearson r, linear fold change | 0.6722 | 0.6722046 |
| Pearson r, log2 | 0.6701 | 0.6701132 |
| median absolute log2 difference | 1.1649 | 1.1648917 |

Two further measurements on the same join: the rank correlation is 0.5533, and log2 of
the released targeted fold change exceeds 1 on 1,062 of 1,256 rows (0.8455), which is the
paper's "confirmed metabolite increase in 85% of the tested pairs". So the second platform
agrees on the CALL and not on the number, which is what makes it a measurement rather than
a copy.

### The reference of each family, and the one value that is not sourced

- **Growth.** `fitness` is the strain's mean AUC over the grand mean AUC of the 48
  released control cultures (16 `ctrlN` wells x 3), so the reference is `fitness = 1.0`
  with `n_samples = 48` `biological_replicate`. Measured control AUC **20.270572**
  (sd 3.205158). The uncertainty is the SAMPLE SD of the three per-replicate ratios, so
  `fitness_se` is that over sqrt(3).
- **Intensities.** The reference level is the **per-batch median intensity**, back-solved
  as `Mean_Int / Mean_FC`. That is a measurement, not a reading: `R1_Int / R1_FC` equals
  `R2_Int / R2_FC` on all 1,385 rows and is constant across the strains of a batch to
  **3.26e-16** relative spread over the 246 multi-strain (feature, batch) groups
  (`preprocess/batch_median_check.json`, 717 groups). `n_replicates` on the reference is
  that batch's sample count (560 / 564 / 574 / 546 / 560 / 222, summing to 3,026).
- **Targeted LC-MS/MS.** `fold-change` is a ratio to a library-wide median, so a strain at
  that median has 1.0, and the per-record reference is 1.0 on every key the record
  measures. **`n_replicates` on that reference is the one value here that is not sourced
  per record**: it is the MEASURED number of strains the released Table S6 carries for
  that m/z feature (1 to 67, median 2), a lower bound on the population the median was
  taken over, because the release names it only as "all other strains" (Table S6 legend)
  or "the whole library" (Figure 2B) and never enumerates it. Flagged for review.

### Sourcing

Every quote is verbatim in a sha256-pinned artifact and audited at verification time:
`paper.md` (`3c63b665c7e69d8e433f3b5a48a579a01956be919bc22669cbc789bf74d643e5`) and
`si/si1.md` (`809c383ac09cab361a09206522bb1d0ca928853f5984985bc41be3f2cff58b01`).

| key | value | where | quote (abridged) |
|---|---|---|---|
| `growth_assay` | trapezoid AUC (MATLAB `trapz.m`) | Methods, Growth analysis of arrayed library | "OD600 was measured every 10 min for 24 h at 37 C and 800 rpm. Data was analyzed with custom MATLAB scripts to obtain the area under the curve using the trapz.m function." |
| `growth_replicates` | 3 | Figure S1 caption | "Growth curves show means from n=3 cultures cultivated in minimal glucose medium in a plate reader." |
| `auc_cutoff` | 18.0 | Figure S1 caption | "The area under the curve (AUC) was used to classify CRISPRi strains with and without growth defect (AUC cutoff of 18)." |
| `back_dilution_hours` | 6.0 | Figure S1 caption | "The cultures were back diluted into fresh medium at t = 6 h." |
| `growth_defect_split` | 489 / 1,026 | Results | "Of the 1,515 CRISPRi strains, 489 CRISPRi strains showed growth defects in minimal glucose medium, whereas 1,026 CRISPRi strains showed no tangible growth defect" |
| `targeted_pairs` | 1,256 | Results | "We selected 1,256 strain-metabolite pairs with the strongest accumulation for targeted LC-MS/MS analysis at three collision energies (10, 20, and 40 eV)" |
| `targeted_statistic` | EIC peak height vs the library median | Figure 2B caption | "Fold changes of the peak height in the extracted ion chromatogram (EIC), compared with the median peak height of the whole library, were used to quantify the accumulation" |
| `targeted_max_per_strain` | 10 | Methods, Targeted LC-MS/MS measurements | "A maximum of 10 metabolites were measured per strain (those with highest mean fold change)." |
| `targeted_qc_passed` | 569 | Results | "Of the 1,256 measurements, 569 fulfilled these criteria" |
| `targeted_confirmation_share` | 0.85 | Results | "These data confirmed metabolite increase in 85% of the tested pairs" |

**Column legends are checked against the workbook bytes, not through the text audit.**
`audit_sourced_value` reads an artifact as text and these descriptions live in a sheet of
a zipped workbook, so `SourcedColumn` + `check_column_legends` re-read the `Legend` sheet
out of the sha256-pinned file at build time and refuse unless it still reads verbatim.
That is the stronger check: the quote is re-read from the same bytes the values come from.
Eight columns are pinned that way (Table S5 `Mean_Int`, `R1_Int+R2_Int`, `R1_FC + R2_FC`,
`LC-MS/MS`; Table S6 `fold-change`, `QC passed`, `Intensity PrecMz`, `Datafile Name`).

### The 30 hour axis, resolved

Table S2 releases 181 columns at 10 min spacing spanning 0 to 30 h, while the Methods name
a 24 h measurement. The Figure S1 caption settles it: aTc induction at t = 0 and "the
cultures were back diluted into fresh medium at t = 6 h", so the released axis is the 6 h
before the back dilution plus the 24 h after it. Evidence that the full axis is what the
paper's `trapz.m` consumed: the trapezoid over all 181 points puts **490** strains below
the AUC 18 cutoff and 1,025 at or above, against the reported 489 / 1,026, and the AUC
range is 0.3306 to 26.2601. The build pins the measured 490 and records the one-strain
difference (`preprocess/growth_defect_check.json`). Trapezoid integration is linear, so
the AUC of the mean curve equals the mean of the three AUCs identically; the ledger
records both, and the agreement is arithmetic rather than evidence.

### Deliberately not loaded

- **Table S6 `Intensity PrecMz`** (audit rank 7). An instrument-scale precursor intensity
  with no normalization, and MEASURED not to be the numerator of the released fold change:
  `Intensity PrecMz / fold-change` varies within a feature key by a median relative spread
  of 0.185 (maximum 3.34) over the 173 multi-strain keys, so it reconstructs neither the
  statistic nor its denominator. Agreed with the audit's exclusion, now with a measurement
  behind it.
- **Table S7's annotated features** (audit rank 8). Gated on the release disagreement the
  audit records: joining Table S5 to Table S7's annotated rows gives 689 pairs, 0 of which
  share a `Mean_FC`, and Table S7's own `Mean_Int` holds the fold change rather than an
  intensity on all 9,462 rows. Agreed with the audit's exclusion.

### Raw mirror

`rapp2026.RAW_FILES` now pins **seven** workbooks under one manifest for the citation key.
Two are new, pinned there and read here:

| mirror path | SI table | bytes | sha256 (first 12) |
|---|---|---|---|
| `data/si3.xlsx` | Table S2 (`mmc3`) | 7,390,438 | `bc7ff53a40a5` |
| `data/si7.xlsx` | Table S6 (`mmc7`) | 590,835 | `c4957a1d7966` |

`MetabolomeRapp2026Dataset.raw_file_names` is now its own five (`PLATFORM_ONLY_FILES`
names the two it does not read), so each family links and sha256-checks exactly what it
parses. `NOT_MIRRORED` no longer claims `mmc3` and `mmc7` are unconsumed.

### Build and verification

```
python -m torchcell.datasets.ecoli.rapp2026_platforms build  {growth|targeted|intensity}
python -m torchcell.datasets.ecoli.rapp2026_platforms verify {growth|targeted|intensity}
```

| family | records | store | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|---|
| growth | 1,514 | 5.9 MB | 1,514 validated | count 1,514; pair uniqueness 1,514 | 1,514 values + SE; no zero dispersion | reference fitness == 1.0; media + compound identity; 10 provenance audits | 1.000 of 1,514 are MG1655 genes |
| targeted | 406 | 20 MB | 406 validated | count 406; genotype uniqueness 406 | 1,244 values | reference finite + key-subset; one `measurement_type` | 406 of 406 MG1655 gene rows |
| intensity | 406 | 25 MB | 406 validated | count 406; genotype uniqueness 406 | 1,373 values + 1,373 SE | reference finite + key-subset; one `measurement_type` | 406 of 406 MG1655 gene rows |

All three PASS. Registered in `runners.FITNESS_DATASETS` (growth, the first bacterial
entry there) and `runners.METABOLITE_DATASETS` (the other two).

### Tests and adapters

`tests/torchcell/datasets/ecoli/test_rapp2026_platforms.py`: 43 hermetic + 21 data-gated,
64 passing with `--data`; 88% hermetic line+branch coverage of the loader and 90% diff
coverage over the whole change. The hermetic screen writes all six workbooks and builds
all three families against the real `EcoliK12MG1655Genome` over the synthetic MG1655
assembly, with `b0099` a `gene_synonym` of the unreleased `b0005` so the remap rule fires
exactly as `phnE` does on the real table.

Three adapters, one module and one conf each, registered in `dataset_adapter_map`
(84 entries to 87) and in `kg_bacteria.yaml` (21 datasets to 24):
`GrowthAucRapp2026Adapter` (`fitness phenotype`), `TargetedMetabolomeRapp2026Adapter` and
`MetaboliteIntensityRapp2026Adapter` (`metabolite phenotype`). All three enable the
`crispr construct` and `environment perturbation` pairs, like the metabolome adapter, and
serve the `bacterial perturbation` class rather than the yeast `perturbation` class. The
data-gated graph check passes on each dev store. No KG build was run.
