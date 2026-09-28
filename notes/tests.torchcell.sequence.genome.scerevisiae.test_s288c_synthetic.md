---
id: hq3zv1lz31a0feeflqe1ayz
title: Test_s288c_synthetic
desc: ''
updated: 1790562740030
created: 1790562740030
---

## 2026.09.27 - SCerevisiaeGenome on a tiny fake genome

Forty-two tests on a two-chromosome FASTA and a small GFF under `tmp_path`, built with `overwrite=False` on a `data.db` seeded with the constructor's own `create_db` arguments (one test covers `overwrite=True`); the genomes-registry `resolve` is stubbed at its import site. Gene set, `alias_to_systematic`, `feature_index`, `resolve_gene_name` on current, renamed and retired names, strand-aware sequence windows as exact strings, `gene_attribute_table` rows and each error message. Module coverage 92%. Findings: `__getitem__` catches `KeyError` but gffutils raises `FeatureNotFoundError`, so an unknown id raises instead of returning None (s288c.py line 964); the plus-strand 5' window without the start codon includes the first CDS base while the minus strand excludes it (line 287); the 3' UTR error message lacks a space before its parenthesis (lines 362 and 378); the gene repr names `DnaSelectionResult` with a double space (line 398); `SCerevisiaeGene.alias_to_systematic` reads a `gene_set` a gene does not have (line 220); `get_seq` passes `self.id`, which a genome never defines (line 867). Unreachable: the `is_obsolete` branch (line 624), since `GODag` skips obsolete terms by default. Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
