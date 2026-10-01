---
id: ecrh5qmk9sefr007o482avp
title: Common
desc: ''
updated: 1790872920010
created: 1790872920010
---

## 2026.10.01 - Empty gene set passes containment; American spelling

Previously, with `sgd_genes` given and no gene perturbations, `gene_containment_sgd` defined the overlap as 0.0 and failed ("0.000 of 0 measured genes ...") while `current_genome_genes` passed the same empty set. The overlap of an empty measured set is now 1.0 (vacuous containment), the result passes, and its message reads "no measured genes (the measured gene set is empty); containment holds vacuously". The `uncertainty_sanity` messages now spell "labeled". No code parses these messages; historical dataset notes that quote old reports were left verbatim. The same 0.0 convention still exists in `runners._l4_rnaseq_gene_containment` and `segregant_growth`, outside this fix. Issue #541; tests `test_gene_containment_passes_a_dataset_with_no_genes_vacuously`, `test_uncertainty_*` in [[tests.torchcell.verification.test_common]].
