---
id: bfl3irwxk8p3wfk1k14mq4p
title: Protein
desc: ''
updated: 1783563814904
created: 1783563814904
---

## 2026.07.08 - Protein verifier: gate WS9 proteome records beyond what the schema encodes

This verifier exists because `ProteinAbundancePhenotype` datasets (WS9 Zelezniak SWATH-MS kinase knockouts) carry invariants the schema validator alone cannot express -- one record per knocked-out ORF, an absolute WT reference (not a centered 0), and a single measurement_type across the set. The schema already enforces non-empty abundances, matching abundance/replicate keys, and non-negative SE, so L0 subsumes those; this module adds the L1-L3 checks that make the built LMDB trustworthy for training.

- L3 `reference_finite` is the key distinction from the centered-score families: protein abundance is ABSOLUTE, so the WT reference must be finite and key-matched to the experiment, NOT identically zero.
- L1 `orf_uniqueness` guards against a knockout appearing in multiple records; L3 `measurement_type_consistent` refuses to silently mix assays.
- Exposes `protein_gene_set` (the knocked-out ORF union) as the L4 overlap key; the cross-source containment check against the deletion collection is asserted by [[torchcell.verification.runners]], not here.
- Built from the shared checks in [[torchcell.verification.levels]] and reported via [[torchcell.verification.report]].
