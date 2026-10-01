---
id: ecrh5qmk9sefr007o482avp
title: Common
desc: ''
updated: 1790872920010
created: 1790872920010
---

## 2026.10.01 - Empty gene set passes containment; American spelling

Previously, with `sgd_genes` given and no gene perturbations, `gene_containment_sgd` defined the overlap as 0.0 and failed ("0.000 of 0 measured genes ...") while `current_genome_genes` passed the same empty set. The overlap of an empty measured set is now 1.0 (vacuous containment), the result passes, and its message reads "no measured genes (the measured gene set is empty); containment holds vacuously". The `uncertainty_sanity` messages now spell "labeled". No code parses these messages; historical dataset notes that quote old reports were left verbatim. The same 0.0 convention still exists in `runners._l4_rnaseq_gene_containment` and `segregant_growth`, outside this fix. Issue #541; tests `test_gene_containment_passes_a_dataset_with_no_genes_vacuously`, `test_uncertainty_*` in [[tests.torchcell.verification.test_common]].

## 2026.10.01 - Review fix: measured_genes_present

The vacuous pass above removed the only failing result for a dataset whose records carry no gene perturbation (review of PR #588: three wild-type records at three temperatures, or one wild-type record, passed with nothing failing). When `sgd_genes` is given and the measured set is empty, a FAILING L4 `measured_genes_present` result now precedes the two vacuous passes ("no measured genes: none of the N records carries a gene perturbation outside the background genes, ..."), and `current_genome_genes` also names the empty set. The row is emitted only for an empty set, so reports of datasets that measure genes keep their rows. Per the reviewer, none of the 43 stored reports carrying `gene_containment_sgd` has `n_measured == 0`, so no stored `passed` flag would flip. Tests: `test_gene_containment_with_no_genes_fails_on_measured_genes_present`, `test_background_only_genes_count_as_no_measured_genes`, `test_a_dataset_with_no_gene_perturbations_fails_for_no_measured_genes`.
