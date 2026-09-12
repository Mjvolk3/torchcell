---
id: igrrtlb2p30b6rk653n61oz
title: Nadal_split_half_signal
desc: ''
updated: 1789186817619
created: 1789186817619
---

## 2026.09.11 - Do the labeled cells carry any reproducible signal?

Script: `experiments/028-knockout-expression/scripts/nadal_split_half_signal.R`; output
`recomputed/split_half_signal.tsv`; summary in
`results/nadal_readthrough_and_signature_tests.json`.

Independent of Kemmeren: each genotype's cells split in half; a 50-gene signature (largest
|pseudobulk log2 FC|, >= 20 summed UMI) derived from half A against half of the WT cells
(`wt_ref`); half B scored against the held-out WT half (`wt_test`). No cell on either
side of the AUROC was used to choose the genes. Null: WT cells of the median genotype size
playing a genotype, same held-out structure, 40 draws.

- 207 genotypes with >= 40 cells: split-half AUROC median 0.52 (IQR 0.49-0.57); null median
  0.49 (range 0.39-0.57); 18 above 0.65, 6 above 0.8.
- Spearman with the Kemmeren-signature AUROC on the same genotypes: 0.24; the genotypes that
  separate are largely the same (`YLR176C` rfx1 0.90, `YGR122W` 0.84, `YNL229C` ure2 0.83,
  `YLR025W` snf7 0.83, `YHL027W` rim101 0.82) plus a few Kemmeren does not flag (`YDL160C`
  dhh1 0.90, `YCR033W` snt1 0.78, `YMR070W` mot3 0.77).

A first version of this script derived the signature against ALL WT cells and scored
against the same cells; its "WT control" read 0.16, i.e. the AUROC was measuring the
overfit of gene selection, and the genotype AUROCs (median 0.56) were inflated the same
way. The held-out version above replaces it.

Reading: for about 90% of genotypes the labeled cells carry no reproducible
genotype-specific transcriptome at this depth, by their own signature or Kemmeren's; for
the ~10% with strong regulatory deletions, a signal is present in roughly half the cells.
