---
id: hq3zv1lz31a0feeflqe1ayz
title: Test_s288c_synthetic
desc: ''
updated: 1790562740030
created: 1790562740030
---

## 2026.09.27 - SCerevisiaeGenome on a tiny fake genome

Forty-two tests on a two-chromosome FASTA and a small GFF under `tmp_path`, built with `overwrite=False` on a `data.db` seeded with the constructor's own `create_db` arguments (one test covers `overwrite=True`); the genomes-registry `resolve` is stubbed at its import site. Gene set, `alias_to_systematic`, `feature_index`, `resolve_gene_name` on current, renamed and retired names, strand-aware sequence windows as exact strings, `gene_attribute_table` rows and each error message. Module coverage 92%. Findings: `__getitem__` catches `KeyError` but gffutils raises `FeatureNotFoundError`, so an unknown id raises instead of returning None (s288c.py line 964); the plus-strand 5' window without the start codon includes the first CDS base while the minus strand excludes it (line 287); the 3' UTR error message lacks a space before its parenthesis (lines 362 and 378); the gene repr names `DnaSelectionResult` with a double space (line 398); `SCerevisiaeGene.alias_to_systematic` reads a `gene_set` a gene does not have (line 220); `get_seq` passes `self.id`, which a genome never defines (line 867). Unreachable: the `is_obsolete` branch (line 624), since `GODag` skips obsolete terms by default. Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Phase 16: CDS selection branches, stale caches, main

Thirty-four to forty-eight tests, 92 to 97 percent (the data-gated `test_s288c.py` skips whole, so the stub-file file holds these). The CDS-selection branches (a one-bp 5' CDS on each strand, a middle intron, the single Verified CDS); stale caches; `main`'s constructor arguments.

Findings: a 5' intron with no CDS, or with no Verified CDS, raises `UnboundLocalError` (lines 141-161); a CDS missing `orf_classification` reads as "gene not found" (151 with 964); a `.` strand gives `seq` None and every window call and `get_seq` then raises `UnboundLocalError` (183-188, 256, 283, 344, 862-865); `get_seq` accepts a FASTA-key chromosome then fails validation (868); `drop_chrmt` leaves `feature_index` and `go_genes` stale so Q0010 still resolves as CURRENT (906-922); `main` builds with `overwrite=True` (983).

## 2026.09.30 - Plus-strand 5' window (issue #543)

Retired the Finding that the `+` 5' window included the first CDS base. Now asserted: YAL002W `window_five_prime(5)` is `CHR_I[28:33] = TAGGA` at `(28, 33)`, the undersized window is `CHR_I[0:33]`, and window 40 is refused as `7bp outside`.

## 2026.09.30 - Findings retired (issue #538)

- Retired: the unbound `feature` for a 5' intron with no usable CDS, the "not found" read of a CDS without `orf_classification`, the `.`-strand gene with `seq` None, `get_seq` accepting a FASTA key, the stale `feature_index`/`go_genes` after `drop_chrmt`, and `main`'s `overwrite=True`.
- Now asserted: the exact `ValueError` messages (gene id, CDS id and span, strand), `get_seq` refusing a FASTA key, an absent chromosome number and a `.` strand by name, Q0010 resolving as RETIRED and GO:0000002 mapping to YAL002W only after the drop, and `overwrite=False` in `main`. The `self.id` and `FeatureNotFoundError` findings remain pinned.

## 2026.09.30 - drop_empty_go cache reset asserted (issue #570)

- Added `test_drop_empty_go_rebuilds_the_locus_index_and_go_genes`: with both caches warm, YBL001W resolves CURRENT before `drop_empty_go` and RETIRED after; `feature_index["genes"]` is the five surviving genes; `go_genes` is a new object with the exact three-term map. Fails on the previous source (YBL001W stayed CURRENT).
