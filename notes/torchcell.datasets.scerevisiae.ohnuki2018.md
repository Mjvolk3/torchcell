---
id: afpnvd6q0b1naaim7sibhqa
title: Ohnuki2018
desc: ''
updated: 1783884471903
created: 1783884471903
---

## 2026.07.12 - Ohnuki 2018 essential-gene het morphology build

`ScmdOhnuki2018Dataset` (`torchcell/datasets/scerevisiae/ohnuki2018.py`). Ohnuki & Ohya
2018, *PLoS Biol* 16(5):e2005130, doi:10.1371/journal.pbio.2005130, PMID 29768403.
CalMorph morphology of the essential-gene HETEROZYGOUS diploid deletion collection →
`CalMorphPhenotype`. **L0-L4 verified, 1,112 records.**

- **Source (SCMD2, mirror + hash-pin):** `ess1112data.tsv` (1,112 het-diploid mutant
  averages, sha256 `2d168bd1…`) + `wt114data.tsv` (114 BY4743 WT reps, sha256 `f48d42da…`),
  fetched from SCMD2 per the paper's Data Availability. 501 CalMorph traits (281 base +
  220 CV) — byte-identical vocabulary to Ohya 2005 (CALMORPH_LABELS/STATISTICS, zero mismatch).
- **Het deletion = 50% dosage, NOT a KO:** genotype uses
  `EngineeredCopyNumberPerturbation(copy_number=1, reference_copy_number=2, marker="KanMX")`
  (the schema names HIP het deletion as its motivating case; SO:0001019). ReferenceGenome
  BY4743 **diploid**. Env YPD **25 C** liquid (log-phase; sourced verbatim).
- Raw per-strain CalMorph averages (not z-scores); zero NaN; 0 ORFs dropped. Only the
  YPD/25 C arm built (SD/37 C poor-medium arm has no released matrix).

## 2026.07.15 - Migrated to the shared layered gene-name resolver

Reconciliation of the 2018 ORF names now goes through
`SCerevisiaeGenome.resolve_gene_name` (the layered resolver landed on main in
[[torchcell.sequence.genome.scerevisiae.s288c]], PR #98), replacing the old FASTA-based
R64 gene-set validation. Same pattern as the sibling
[[torchcell.datasets.scerevisiae.ohya2005]] migration: retain-all + collision-safe.

- Removed `_load_sgd_genes` + `_SGD_GENE_FASTAS`; the loader no longer drops any strain
  for a naming reason. Added a `genome` ctor param (lazy-built from `DATA_ROOT` if not
  injected) and `_reconcile_orf_names`.
- Rebuild result: **1112 strains, unchanged.** Reconciliation over 1112 unique names:
  1111 `current`, **1 `non_gene_feature`** (a valid non-`gene` locus, now retained with
  its correct systematic id rather than being at the mercy of the FASTA ORF-coding set);
  0 remapped, 0 retired, 0 merge-collisions.
- L0-L4 verifier PASS (`run_morphology_ohnuki`): 1112 records, L4 SGD gene containment
  1.000. `EngineeredCopyNumberPerturbation` genotype and round-trip unchanged.
- Unchanged (out of scope): this loader still imputes missing CalMorph values to 0.0 in
  `create_calmorph_experiment` rather than dropping incomplete rows like Ohya/Ohnuki 2022
  — flagged for a future consistency pass, not touched here.
