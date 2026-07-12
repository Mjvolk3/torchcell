---
id: gm71e202w4ajfj5zfs1ox97
title: Yoshida2012
desc: ''
updated: 1783835401860
created: 1783835401860
---

## 2026.07.12 - Yoshida 2012 organic-acid titer build

`OrganicAcidYoshida2012Dataset` (`torchcell/datasets/scerevisiae/yoshida2012.py`). Yoshida
& Yokoyama 2012, *J Biosci Bioeng* 113(5):556, doi:10.1016/j.jbiosc.2011.12.017, PMID
22277779. HPLC organic-acid titers of 17 selected KO overproducers → `MetabolitePhenotype`.
**L0-L4 verified, 17 records.**

- **Born-digital source (OCR was garbled):** the data is Table 3 in the paper (no SI file).
  MinerU OCR scrambled Table 3's numeric cells; recovered cleanly via `pdftotext -layout`
  on the sha256-pinned mirror `paper.pdf` (WT row + 17 genes × {OD, Ace, Cit, Mal, Pho, Pyr,
  Suc}, mM, mean±SD, n=3). Values transcribed into a vetted module-level `TABLE_3` literal
  (deterministic; PDF not re-parsed at build). Illustrates the born-digital-first rule.
- **17 records** (one per gene). metabolite_level = {acid: mM} for acetate/citrate/malate/
  pyruvate/succinate + phosphate. **OD dropped** (biomass, not a metabolite).
  measurement_type=`hplc_organic_acid_titer_mM`, n_replicates=3, se=SD/√3 (sample SD).
- **target_metabolite_ids → Yeast9 s_NNNN** (via `build_metabolite_s_id_map`): acetate s_0362,
  citrate s_0522, malate s_0066, pyruvate s_1399, succinate s_1458. Phosphate unmapped (inorganic).
- **Measured WT reference** = the Table 3 WT (BY4742) row, restricted per record to measured
  analytes. Static YPD liquid, 25 C, 72 h. All 17 genes resolved to R64 ORFs.
- **DEFERRED layers:** the 36-gene BCP-halo categorical screen (Table 2 ordinal per-acid
  fold-change) — no clean existing schema fit; Table 4/5 out of scope (overexpression / chem
  sensitivity). Table 3 titers are the clean quantitative layer.
