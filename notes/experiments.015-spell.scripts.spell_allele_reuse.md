---
id: cbabfmntfn5ssisqpei56ax
title: Spell_allele_reuse
desc: ''
updated: 1791149705316
created: 1791149705316
---

## 2026.10.04 - Costanzo and Kuzmin conditional alleles, and their reuse in SPELL

Collects the temperature-sensitive (ts) and DAmP alleles of Costanzo 2016 (Data File S1 strain table, strain ids ending `_tsq`, `_tsa`, `_damp`) and Kuzmin 2020 (Table S1 array strains ending `_tsa`), then searches SPELL column headers for the same allele names and for other non-deletion alleles of the same genes.

Measured on the 2026.10.04 run:

- Costanzo 2016: 1,959 ts strains, 1,301 distinct ts allele names, 866 genes; 819 DAmP strains, 819 genes.
- Kuzmin 2020: 189 ts array strains, 189 genes; 50 of its allele names are not in the Costanzo table. Queries are all deletions and there are no DAmP strains.
- Union: 1,351 ts allele names over 866 genes; 819 DAmP alleles.
- SPELL headers naming the same allele: 95 conditions in 9 studies, 8 alleles (`cdc28-4`, `cdc15-2`, `scc2-4`, `esa1-L254P`, `prp22-1`, `cdc23-1`, `pre1-1`, `pob3-7`).
- SPELL headers naming a different non-deletion allele of a gene that has a ts or DAmP allele: 252 conditions in 33 studies, 35 genes. This tier has false positives (`eco1-Apr2011` is a date) and was not cleaned.
- No DAmP allele name appears in a SPELL header.

Provenance gap: the Costanzo raw zip under `$DATA_ROOT/data/torchcell/smf_costanzo2016/raw` is truncated on the Mac, so the strain table is read from an older local copy pinned by sha256 `3b9d3351...`. It has not been compared with the copy the loader extracts on the build machine. Kuzmin 2018 is not on the Mac and is not covered. Kuzmin 2020 Table S3 (pilot screen) was not read.

Used by [[experiments.015-spell.publications]].
