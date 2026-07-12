---
id: n6srg6orgqp9kedz8yhg3b4
title: Dasilveira2014
desc: ''
updated: 1783835203529
created: 1783835203529
---

## 2026.07.12 - da Silveira 2014 lipidomics build

`MetaboliteDaSilveira2014Dataset` (`torchcell/datasets/scerevisiae/dasilveira2014.py`).
da Silveira dos Santos et al. 2014, *Mol Biol Cell* 25(20):3234, doi:10.1091/mbc.E14-03-0851,
PMID 25143408. Kinase/phosphatase deletion lipidomics → `MetabolitePhenotype`. **L0-L4
verified, 127 mutant records × up to 147 lipid species.**

- **Source (open-access SI, mirror + hash-pin):** Table S4 `TableS4_complete_dataset_all_lipids.xlsx`
  (sha256 `91409229…bced3894`, Quant sheet = relative abundance a.u.) + Table S10 ChEBI ids,
  fetched once from the Europe PMC supplementary ZIP (PMC4196872).
- **WT reference discovered in-data:** the Quant sheet has **3 WT control rows** (Y7092,
  Y7220, BY4741) — NOT ORFs, so excluded from mutant records; their per-lipid MEAN is the
  measured WT reference (`reference_centered=False`, restricted per record to measured
  lipids). This resolves the paper's "127 mutants" (130 rows − 3 WT = 127) and is strictly
  more faithful than a mutant population mean.
- **Value:** relative abundance in arbitrary units (`measurement_type="lipidomics_ms_relative_abundance_au"`,
  NOT concentration). `n_replicates=2` (biological duplicates, sourced verbatim; up-to-6
  technical reps not independent → not counted); no per-strain SE released → `se=None`.
- **Target ids DEFERRED:** lipids are acyl-resolved species mapping to ChEBI (Table S10,
  147/147), not Yeast9 `s_NNNN` — so `target_metabolite_ids=None`; the lipid→ChEBI map is
  emitted to `preprocess/lipid_chebi.csv` for a follow-up.
- Background `BY4741` (evidence-based: BY4741 WT control row), YPD liquid 30 C. 0 ORFs dropped.
