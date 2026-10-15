---
id: y55dbs29he902ub8frtp5o1
title: construct-span-parts
desc: ''
updated: 1792086719809
created: 1792086719809
---

## 2026.10.15

- [x] `construct-span-parts` worktree (lane T of the ontology unification): `ConstructSpan` now holds the ORDERED PARTS of a GenBank location plus the claim they make, so **17 of 17 Cachera maps index and 0 are blocked** (was 6 and 11), and the paper's pBTX2 is in the store. `SpanComposition` is `simple | origin_spanning | joined`, grounded in the INSDC `join` / `order` grammar, and `origin_spanning` is a separate claim from `joined` because one is a single interval whose coordinates wrapped and the other is two intervals with sequence between them ([[torchcell.datamodels.construct]], [[torchcell.sequence.plasmid]], [[torchcell.sequence.construct.registry]]).
- [x] Round trip proved on the bytes: **570 of 570** features of all 17 maps regenerate, from the stored parts and strand alone, the exact location string their own file wrote, and every compound span's canonical INSDC spelling re-parses through Biopython into the same parts ([[experiments.036-dataset-fixes-before-kg-build.scripts.cachera2023_construct_span_parts]]).
- [x] What the envelope used to say, measured: `pBTX002.gb`'s `pUC\ori` is **666 bp**, not the 12,096 bp of the whole plasmid, and each `AmpR` CDS is **861 bp** (beta-lactamase), not the ~4,894 bp of its plasmid.
- [x] The 9 `pBTX0xx` gene-targeting vectors the paper body never names are named from SUPPLEMENTARY TABLE S2 of `Supplementary Material.docx`, a member of the sha256-pinned SI container, extracted and re-hashed in memory on every provisioning run ([[scripts.provision_construct_sets]]). 8 names from `paper.md`, 9 from the table, 0 typed gaps.
- [x] Cachera's host placement: the XII-5 coordinate stays a typed gap and the pBTX1 insert span stays a typed gap, with the disagreement recorded rather than resolved. pBTX2 DOES admit an exact contig-side interval (1529..10798, 9,270 bp, from the paper's "NotI excised insert of pBTX2" and the two GCGGCCGC sites the bytes carry at 1527 and 10797, which is the paper's own "9.3 kb"), but that is pBTX2's coordinate space and all four records point at pBTX1, which carries no NotI site ([[torchcell.datasets.scerevisiae.cachera2023]]).
- [x] Still refused, each with its own typed reason on `GenBankLocationError.refusal`: INSDC `order`, a fuzzy bound, a between-bases site, a mixed-strand join, a remote accession reference, a backwards join, a wrap on a linear replicon, an unparsable location ([[torchcell.sequence.plasmid]]).
- [x] Schema-change record `database/schema_changes/2026-10-15-construct-span-holds-the-parts-of-a-compound-location.json`, `rebuild_scope: full_kg_rebuild`, 3 served datasets. The breaking change is authorized by the standing Path U rulings; **the span shape itself has not been reviewed by the owner** and is put forward with its measurements.
