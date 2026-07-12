---
id: t5pg308ay1dtt7ejmdbcrws
title: Smith2006
desc: ''
updated: 1783834308152
created: 1783834308152
---

## 2026.07.12 - Smith 2006 fatty-acid clear-zone screen build

`FattyAcidSmith2006Dataset` (`torchcell/datasets/scerevisiae/smith2006.py`). Smith et al.
2006, *Mol Syst Biol* 2:2006.0009, doi:10.1038/msb4100051, PMID 16738555. Genome-wide YKO
clear-zone (fatty-acid utilization) screen → `EnvironmentResponsePhenotype`. **L0-L4
verified, 14,163 records** (4,721 unique-ORF strains × 3 conditions).

- **Source (open-access SI, mirror + hash-pin):** Supplementary Table 1
  `msb4100051-s1.xls` (sha256 `7048663f…b283a4`), fetched once from the Europe PMC
  supplementary bundle for PMC1681483 into the library mirror `…/data/`; loader copies +
  verifies sha256. Legacy .xls, header row 23, 4,770 strain rows.
- **Model:** KO × carbon-source condition. Each condition = Media (YPBO/YPBM/YPBA) + a
  `SmallMoleculePerturbation` for the carbon species (oleic acid 0.1% / myristic acid
  0.125% / acetate 2%, percent_w/v), 30 C, aerobic, solid. One record per (strain, condition).
- **Ordinal score** on `environment_response` (float) + semantic `category`
  (`measurement_type=categorical`; the enum has no ordinal member). Clear-zone 4/3/2/1
  (enhanced/wild_type/reduced/defective); acetate growth 3/2/1 + undocumented 2.5 kept
  ("intermediate"). Reference = BY4742 WT, `environment_response=None`/`category=wild_type`
  (the verifier's L3 `reference_zero` requires numeric reference==0, so None is the honest
  ordinal encoding). `n_samples=3` (triplicate replicate plates; quadruplicate pinning is
  within-plate technical, not counted). No SE released.
- **Gene resolution:** 4,721/4,770 → current R64 ORFs; 49 dropped (26 non-current/dubious;
  23 alias-collisions where the alias target is already a direct-ID row, e.g. YOR240W→
  YOR239W/ABP140 — dropped to avoid mislabeling a distinct dubious strain). 22 legit renames
  (e.g. YGR272c→YGR271C-A) kept.
- Skipped the sparse 'Glucose (YEPD)' NG/LG flag column (not an ordinal score). Verifier
  wired as `env_chemgen_smith2006` in `runners.py`.
