---
id: pfd06dhqkekjqzgio8icvtv
title: Schmidt2016_replicate_intensity
desc: ''
updated: 1795586645972
created: 1795586645972
---

## 2026.11.24 - Table S8 per-culture intensities, one record per culture (#1178)

Loader: `torchcell/datasets/ecoli/schmidt2016_replicate_intensity.py`, class `ProteomeReplicateIntensitySchmidt2016Dataset`, store `proteome_replicate_intensity_schmidt2016`. Part of #1174 and #1176, policy #1178.

### What a record is

One BW25113 culture of the data-set-2 LC-MS experiment: 22 conditions x 3 cultures = **66 records**, the three glucose cultures included. `protein_abundance` is the culture's released Table S8 `normInt_<file>.1` for each of the 2,058 Table S8 rows, verbatim; `n_replicates` 1 per key; no SE (typed gap `not_applicable_single_measurement`); `replicate_id` is the Table S25 file name (`A14-07036`). `measurement_type` is `label_free_ms1_intensity_safequant_normalized_dataset2`, a different quantity from the copies/cell of `ProteomeSchmidt2016Dataset`, which is why this is its own dataset (one `measurement_type` per dataset, the same reason `schmidt2016_srm` is two).

The reference is the released `medianNormInt_Glucose.1` (the authors' median of the three glucose cultures), `n_replicates` 3, `replicate_id` None.

### Sourcing (verbatim)

- `paper.md` (sha256 `67bedae8...a8710`), Online Methods, Sample preparation: "All samples of data set 2 were prepared in biological triplicates."
- `si2.xlsx` (sha256 `3280a13f...1912f`), Table S8 row 2 block label: "Normalized Abundance of Biological Replicates Determined by LFQ using SafeQuant (sample names see below)"; Table S25 header: "Growth Condition (in biological triplicates)".
- `si/si1.md` (sha256 `24660453...c334a`), SafeQuant Description: "This analysis included global data normalization by equalizing the total MS1 peak areas across all LC-MS runs, summation of MS1 peak areas per protein and LC-MS/MS run, followed by calculation of protein abundance ratios."

### Keys

Joined through the UniProt accession to the Table S6 row, so each key is the copies/cell store's key. Table S8 prints older or truncated symbols on 5 rows (`yacF` for `zapD`, `yhcM` for `zapE`, `yt` for `ytfL`, `Mett` for `MettuDRAFT_4149`, empty for `B8LFD5`). `P63284` (`clpB`) is on two rows in both tables, so its two Table S8 rows are keyed `clpB | Table S8 row 1040` and `clpB | Table S8 row 1973`. Released symbol and stored key per row: `preprocess/protein_identifiers.csv`.

### Not stored: `Glucose.2` (A14-07088..90)

`si/si1.md`, Supplementary Note 1: "Further, the 3 glucose growth condition samples were analyzed at the beginning and end of the quantitative LC-MS experiment to check if the data was impaired by any systematic performance changes of the LC-MS platform employed throughout the whole analysis time of the high number of $\mathtt { > 6 0 }$ samples." So the second glucose triplicate re-analyzes the same three samples. The release does not pair a repeat file with its culture (measured log10 Pearson r: each `Glucose.2` column correlates 0.954 to 0.963 with every `Glucose.1` column, with no k-th to k-th pairing), so it can be neither stored as three more cultures nor combined into the culture it repeats. Left to the owner (needs a technical-repeat field on `ProteinAbundancePhenotype`, or a decision to store the repeats as their own records). Listed in `preprocess/dropped_records.json` under `not_stored`.

Hypothesis (untested): the same SI's statistics section ("All LC-MS analysis runs are acquired from independent biological samples.") could be read as the opposite, which is why this is an owner call and not a loader call.

### Not read: Table S9

Table S9 is a separate SafeQuant run with its own absolute calibration over 18 files: BW25113 LB and glucose (A14-07032..38, the SAME files as Table S8) plus MG1655 and NCM3722 in LB and glucose (A14-07016..30, 12 cultures). Measured: its BW25113 `normInt` differs from Table S8's for the same files (median ratio 1.11 to 1.17) and its BW25113 copies/cell from Table S6's (median relative difference -22% LB, -18% glucose). Storing it means a second processing of six already-stored cultures; proposed shape for the owner in the PR.

### Build and verification (scratch)

`$DATA_ROOT/tmp/scratch/per-replicate-schmidt2016/after/proteome_replicate_intensity_schmidt2016`: 66 records x 2,058 keys. Verifier PASS: L0 66 validated; L1 count 66/66; L1 culture_uniqueness 66 distinct (environment, culture) keys, 22 environments x 3; L2 value_fidelity 135,828 values; L3 reference_finite 135,828; L3 single measurement_type; L3 replicate_id_names_one_replicate 66 of 66; L3 assembly_pin; L4 containment reported (keys are released symbols).

Median reproduction (`experiments/036-dataset-fixes-before-kg-build/scripts/schmidt2016_per_replicate_counts.py` -> `results/schmidt2016_per_replicate_counts.json`): the median over each condition's three records equals Table S8's released `medianNormInt` for all 45,232 (protein, condition) values compared (the two P63284 rows skipped), max abs diff 0.0; the reference equals `medianNormInt_Glucose.1`, max abs diff 0.0.
