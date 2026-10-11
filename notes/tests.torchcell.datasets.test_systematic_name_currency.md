---
id: rmqiaza8op4c7zxhn0zj9od
title: Test_systematic_name_currency
desc: ''
updated: 1791701417338
created: 1791701417338
---

## 2026.10.11 - Tests of the SGA systematic-name currency check (#893, #897)

Module under test: [[torchcell.datasets.systematic_name_currency]].

- On a stub genome (live YAL042C-A, YAR042W, YAL001C and the blocked reading frame YER109C; YAL043C-A renamed, YAR044W merged into a gene, YER108C merged into a non-gene locus, YAR037W retired, YZZ999W an unpinned alias): `apply` keeps the index, maps the renamed name, refuses every row naming a refused ORF, and the ledger dump is compared whole; counts add up across two `apply` calls (the Dmf Kuzmin 2020 shape).
- Refusals: an unpinned non-current name raises `not_current_unpinned`; a blank name cell raises; construction raises `pin_rule_contradicted` when the genome contradicts a retired or merged refusal, `pin_not_a_candidate` when a rename's alias is gone, and refuses a non-rename rule or a name both renamed and refused.
- Model validation of `PinnedSystematicNameRefusal` and `SystematicNameLedger`, and the table shape: 16 renames, 28 merges (the four into non-gene loci quote the R64 GFF), 7 retirements.
- Data-gated (`DATA_ROOT` with the genomes tier): every pin's file hashes to its pinned sha256 and its quote is a substring of a decoded `locus.history[].note` (SGD locus JSON) or of the R64-4-1 GFF.
