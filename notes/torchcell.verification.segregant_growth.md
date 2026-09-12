---
id: tm4v5qd0a59w5gpnxng88qy
title: Segregant_growth
desc: ''
updated: 1789203212517
created: 1789203212517
---

## 2026.09.12 - Verifier for haplotype-mosaic genotypes

`torchcell/verification/segregant_growth.py`, registered in `torchcell/verification/runners.py` as `SEGREGANT_GROWTH_DATASETS` / `run_segregant_growth` and hooked into `run_all`. Written for [[torchcell.datasets.scerevisiae.bloom2019]]; the environment-response verifier does not apply because its strain signature reads `genotype.perturbations`, its "every record carries an environmental edit" rule would flag the control plates, and its single-measurement-type rule would flag the residual/absolute split.

One streaming pass over the LMDB records and one pass over the raw files:

- L0 structural against `ExperimentType`.
- L1 count, uniqueness on (segregant id, condition signature), id bijection between the phenotype table and the 16 genotype matrices, the provenance-gap census.
- L2 value fidelity: every stored value equals the released tsv cell parsed with `float_precision="round_trip"` (the default parser differs in the last digit, which is why both loader and verifier use round_trip); mosaic round trip: every segregant's blocks re-expand to its released marker row; block invariants per cross.
- L3 measurement partition (36 residual, 2 absolute); reference 0; every environment is one of the 38 documented conditions and exactly the two control plates have an environment equal to their control's; sourced-value audits (text anchors through `audit_sourced_value` against the raw mirror root, which is root-agnostic; xls anchors by re-reading the sheet and matching the quoted row, since a binary xls cannot be substring-searched).
- L4 the 15 Peter ids resolve in the 1011 assembly member index; xls parent pairs equal the README pairs and per-cross counts equal the xls; a marker sample's ref allele equals the S288C base at that position (settles the coordinate system empirically); the derived gene set is a subset of the SGD gene universe.

Tests: `tests/torchcell/verification/test_segregant_growth.py` builds a synthetic 17-segregant release in `tmp_path` and checks that a tampered value, a tampered block, and a duplicated record each fail the right level.
