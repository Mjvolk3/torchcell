---
id: tcdmdu6tuzqg1zk9ng12r3g
title: Segregant_expression_reference_panel
desc: ''
updated: 1791733641931
created: 1791733641931
---

## 2026.10.11 - A verifier for pooled cells matched to a reference panel

`torchcell/verification/segregant_expression_reference_panel.py`, written for [[torchcell.datasets.scerevisiae.nguessan2025]]: cells of many pre-genotyped segregants grown in one culture, each cell assigned to a strain of a reference panel, one record per strain (its cells summed, the panel's posterior as a haplotype mosaic). It is a sibling of [[torchcell.verification.segregant_expression]] rather than a branch of it, because that verifier is keyed on Boocock's panel specs and re-reads Boocock's RDS release.

What is new relative to the Boocock verifier is that the genotype and the expression arrive from two deposits joined by an index: the release's 0-based batch-1 index of the matched strain, mapped to the panel's `Number` through the batch-1 rows of `segregant_info.txt`. Two L4 rules test that join instead of assuming it:

- `mat_fixation`: the panel's reference pseudobulk is MATa by expression (a-specific to alpha-specific UMIs at least 10:1, `MIN_A_TO_ALPHA`), so the stored posterior at the SNP nearest MAT (chrIII:200,000) must be fixed: its mean within 0.05 of 0 or 1. The message names which parent the fixed allele is, which the paper does not state.
- `cis_concordance`: for every gene detected in at least half the strains and placed on S288C by the genome, the absolute Pearson r between the strain's log CPM and its P(RM) at the gene's nearest SNP; the null permutes which strain's genotype goes with which strain's expression (20 permutations, seed 0) and takes its 99th percentile of |r| as the threshold. The rule passes when at least 5% of tested genes exceed it, 5x the null rate. The 5x margin is a choice made before any build, not tuned on one; with a wrong index map the observed rate should sit at the null rate (shown on synthetic data in the tests).

The other rules mirror the Boocock verifier: L0 structural, L1 count and uniqueness plus the shared rules, L2 value fidelity against the summed cells (the stage partition must be a typed gap), the mosaic round trip within half the 0.05 grid (an exact 0.5 posterior is a tie), the reference pseudobulk, L3 Table 1 numbers and the quotes of both mirrors (`audit_sourced_value` against `torchcell-library` for paper quotes and `torchcell-raw` for release quotes), L4 parents, marker reference and gene containment.

On the synthetic release of `tests/torchcell/datasets/scerevisiae/_nguessan2025_release.py` every rule passes but `cis_concordance` (three strains cannot power it, and the test asserts that it reports rather than passes). No real store exists yet (#1014).
