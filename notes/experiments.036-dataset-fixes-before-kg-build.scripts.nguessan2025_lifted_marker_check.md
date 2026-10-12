---
id: s7y1iunk2w8b5cm211hndkh
title: Nguessan2025_lifted_marker_check
desc: ''
updated: 1791814086763
created: 1791814086763
---

## 2026.10.12 - The two L4 genome-reference rules on the lifted SNPs

`experiments/036-dataset-fixes-before-kg-build/scripts/nguessan2025_lifted_marker_check.py` writes `experiments/036-dataset-fixes-before-kg-build/results/nguessan2025_lifted_marker_check.json`. It applies the verifier's `marker_reference` and `mat_fixation` ([[torchcell.verification.segregant_expression_reference_panel]]) to the derived lifted table ([[experiments.036-dataset-fixes-before-kg-build.scripts.nguessan2025_snp_liftover]]) instead of the released positions, with every input pinned. Run under slurm at `1d395d4aa` (job 4321, 39 s, 2.5 GB peak RSS):

| rule | on the released positions (store, job 4320) | on the lifted SNPs |
|---|---|---|
| `marker_reference` | 0.6740 of 500 sampled SNPs | 41,578 of 41,587 lifted SNPs (0.999784), bar 0.95 |
| `mat_fixation` | mean P(RM) 0.1097 at the SNP nearest chrIII:200,000 | 0.0438 at the lifted SNP nearest it (chrIII:201,156, released `snp5020`), within 0.05 of 0 |

On S288C coordinates the genotype passes both rules, which confirms both the lift and the panel's MATa fixation at MAT. The stored records keep the released positions; these numbers describe the derived table only.
