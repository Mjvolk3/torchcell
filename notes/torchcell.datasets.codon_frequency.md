---
id: ex37g9pb0ed00i8aytbyj06
title: Codon_frequency
desc: ''
updated: 1790818970045
created: 1790818970045
---

## 2026.09.30 - Logged skip of a refused CDS (issue #570)

`CodonFrequencyDataset.process` computes `compute_codon_frequency` on each gene's CDS. That function refuses by name an empty CDS (`Empty CDS string; a codon frequency needs at least one codon.`), a CDS whose length is not a multiple of three, and a CDS with a base other than A, T, G, C. Since PR #568 an empty CDS is such a named refusal, and the skip in `process` is what keeps the build going past it.

- Previous behavior: `except ValueError: continue`, with no log line, so the store silently lacked those genes.
- Now: the gene is still excluded, and each skip is logged at WARNING as `skipping <gene_id>: <message>`; after the loop the total is logged once as `skipped <n> gene(s) whose CDS has no codon frequency`. A build with no refused gene logs no warning.
- The module header now names this file and its test (it carried the `fungal_up_down_transformer` header).
- Evidence: `tests/torchcell/datasets/test_codon_frequency.py` (`test_empty_cds_gene_is_logged_counted_and_excluded`, `test_no_refused_gene_logs_no_warning`).
