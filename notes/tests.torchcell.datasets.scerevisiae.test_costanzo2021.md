---
id: og3pkma0w9vii5roqolvkdj
title: Test_costanzo2021
desc: ''
updated: 1790762183461
created: 1790762183461
---

## 2026.09.30 - Phase 13: a real xlsx in the released layout

Nine to thirty tests, 75 to 95 percent (the rest is `main()`). The fixture is an xlsx with a decoy first sheet, the `Diff. Mutant fitness_Conditions` sheet and a `" benomyl "` header; nine rows (a second allele of the same ORF, a renamed ORF, a non-gene feature, a retired ORF, a padded copy of row 1, a blank Systematic Name, an AMBIGUOUS gene, blank and 0.0 cells) give eight records in row-major then condition order; `model_dump()` equality for a deletion row and a ts-allele-on-galactose row plus the reference and publication; the drop log as whole JSON (5 of 9 strains kept, 6 cells dropped); one reference over members 0 to 7; all 13 doses with units, 26 C as a derivation, `n_samples` 3 screens; refusals with exact messages (no genome, a missing condition column, a missing sheet, no mirror file, a sha256 mismatch naming both digests, `download` hashing an existing raw file); `deposit_raw_mirror` with a frozen time.

Findings: the sha256 pin is checked only inside `download` (lines 595-622) and a failed check leaves the copied bytes in `raw/`, so the next construction builds all 8 records from the unverified file; a refused deposit has already created `<mirror>/data/` (516, 518); a blank Systematic Name is resolved and logged as `"nan"` (770); a repeated row is stored twice with no ledger entry (769-821); an AMBIGUOUS drop keeps no candidate list.
