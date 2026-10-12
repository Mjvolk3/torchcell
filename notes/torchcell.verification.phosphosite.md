---
id: hhl3igflv5tz2a9jw1tfmq4
title: Phosphosite
desc: ''
updated: 1791817508517
created: 1791817508517
---

## 2026.10.12 - The phosphosite-intensity family gate (#921)

`torchcell/verification/phosphosite.py`, `verify_phosphosite_dataset`: the L0-L3 gate every `PhosphositeIntensityPhenotype` store shares. The proteome family's shape rules ([[torchcell.verification.protein]]) restated for sites, plus the two only a site-level release has:

| level | rule | what it catches |
|---|---|---|
| L0 | `structural` | a record that does not validate against `ExperimentType` |
| L1 | `count` | a record count off the loader's pin |
| L1 | `one_record_per_sample` | two records naming one released sample |
| L1 | `site_list_is_shared` | a record or reference whose site key set differs from the others (the identification is the release's, not a sample's) |
| L2 | `intensities_finite_positive` | a non-finite or non-positive stored intensity |
| L2 | `site_evidence_is_per_site` | one site key carrying two different identification records |
| L3 | `measurement_type_consistent` | two measurement types in one store |

A loader's `verify_build` calls it and adds its release's pins (per-sample counts, a re-read of every cell, the sourced quotes) and its cross-table L4. There is no assembly-containment L4: the sites are stored under their released protein ids and never resolved to a locus, so the bacterial families' containment rule has nothing to contain. Registered in `torchcell/verification/runners.py` as `PHOSPHOSITE_INTENSITY_DATASETS` / `run_phosphosite_intensity`, called by `run_all` after the bacterial fold-change family. First store: [[torchcell.datasets.ecoli.potel2018]]. Tests: `tests/torchcell/verification/test_phosphosite.py` (each rule passes on a well-formed store and fails on its one defect).
