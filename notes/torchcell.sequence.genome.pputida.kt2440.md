---
id: n3tzqr9yqfbcu59nm5mspn7
title: Kt2440
desc: ''
updated: 1791374717380
created: 1791374717380
---


## 2026.10.07 - P. putida KT2440 genome

Step 3 of [[plan.bacteria-ontology-genome]]. Module:
`torchcell/sequence/genome/pputida/kt2440.py`; shared bacterial layer:
[[torchcell.sequence.genome.bacterial]]; tier: [[scripts.provision_bacterial_genomes]].
Tests: [[tests.torchcell.sequence.genome.pputida.test_kt2440]].

### Design

- `PPutidaKT2440Genome` binds `KT2440_ASSEMBLY` (GCA_000007565.2, replicon `AE015451.2`,
  chromosome key 1) and builds `PPutidaKT2440Gene`s. GenBank first, GFF3 `data.db` cache,
  name resolution and the GO route are the shared layer's.
- Locus tags: `KT2440_LOCUS_TAG_PATTERN = PP_(?:\d{4}|tm?\d{2}|mr\d{2}|r\d{2}|(?:5|16|23)S[A-Z])`.
  All 5,786 GenBank tags match it: 5,621 numbered, plus 165 named RNA tags (75 `PP_tNN`
  tRNAs, 67 `PP_mrNN`, 21 `PP_{5,16,23}S[A-G]` rRNAs, `PP_r01`, `PP_tm01`). Construction
  refuses a tag outside the pattern, so the namespace stays disjoint from the E. coli and
  yeast patterns (D1).
- The GenBank file carries no `gene_synonym` for this assembly, so the synonym layer is
  empty; `/gene` symbols exist for 2,084 of 5,786 loci. RefSeq's `PP_RS` tags resolve
  through the RefSeq GFF's `old_locus_tag` (5,561 of 5,719 RefSeq genes carry one).
- GO: the EBI GOA proteome file `109.P_putida_KT2440.goa`, column 11 tokens fully
  matching the host pattern. Tokens such as `PP4186` (no underscore) are not read.

### Measured counts (GilaHyper, 2026-10-07, from the deposited tier)

| quantity | value |
|---|---|
| GenBank gene features | 5,786 |
| gene set (non-pseudo) / pseudogenes | 5,729 / 57 |
| coding CDS = proteins re-keyed to locus tags | 5,564 |
| CDS whose translation differs from the protein | 1 (PP_0489, fdoG, Sec) |
| GOA rows / NOT rows / rows without a tag | 25,276 / 0 / 0 |
| tags reached / genes with GO / pseudogenes with GO | 3,912 / 3,912 / 0 |
| distinct GO terms | 2,619 |

3,912 equals the plan's measurement. Every term is live in go-basic 2026-07-26.

### resolve_gene_name round trips (tier)

`PP_0002`, `PP_16SA`, `PP_23SB`, `PP_t01`, `PP_tm01`, `PP_mr01` CURRENT; `parA` (gene
symbol) and `PP_RS00010` (RefSeq locus tag) RENAMED to `PP_0002`; `PP_9999`, `b0002` and
`ECK0002` RETIRED. A symbol shared by two genes (`asd`: PP_1989, PP_1992) is AMBIGUOUS.

### Cache root and network

Default root `data/pputida/kt2440/genome`, seeded on GilaHyper at
`/scratch/projects/torchcell-scratch/data/pputida/kt2440/genome` on 2026-10-07 (GFF
sha256 prefix `703bff9433dfd5d4`, trusted on reopen). Construction reads only tier
members; the tests build it with every network entry point raising.
